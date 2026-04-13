#include "radon_template_matching.hpp"
#include <cmath>
#include <numeric>
#include <algorithm>

// =============================================================================
// Radon Transform Primitives
// =============================================================================

static std::vector<cv::Point> getLinePoints(const cv::Mat &image, int angle, int length) {
    int h = image.rows, w = image.cols;
    double angle_rad = angle * CV_PI / 180.0;

    int cx = w / 2, cy = h / 2;
    int start_x = static_cast<int>(cx - length / 2 * cos(angle_rad));
    int start_y = static_cast<int>(cy + length / 2 * sin(angle_rad));
    int end_x = static_cast<int>(cx + length / 2 * cos(angle_rad));
    int end_y = static_cast<int>(cy - length / 2 * sin(angle_rad));

    std::vector<cv::Point> points(length);
    for (int i = 0; i < length; i++) {
        float t = static_cast<float>(i) / (length - 1);
        points[i] = cv::Point(
            static_cast<int>(start_x * (1 - t) + end_x * t),
            static_cast<int>(start_y * (1 - t) + end_y * t));
    }
    return points;
}

static std::vector<uint64_t> perpendicularSum(const cv::Mat &image,
                                               const std::vector<cv::Point> &points,
                                               int width, uint8_t corner_mean) {
    int h = image.rows, w = image.cols;
    const uint8_t *data = image.data;

    // Find perpendicular direction
    cv::Point2f perp(0, 0);
    for (const auto &p : points) {
        perp = cv::Point2f(-(p.y - h / 2), p.x - w / 2);
        float norm = std::sqrt(perp.x * perp.x + perp.y * perp.y);
        if (norm != 0) { perp /= norm; break; }
    }

    // Fixed-point for speed
    const int SHIFT = 16, SCALE = 1 << SHIFT;
    int pvx = static_cast<int>(perp.x * SCALE);
    int pvy = static_cast<int>(perp.y * SCALE);
    int half = width / 2;

    std::vector<uint64_t> sums(points.size());
    for (size_t i = 0; i < points.size(); i++) {
        int bx = (points[i].x << SHIFT) - pvx * half;
        int by = (points[i].y << SHIFT) - pvy * half;
        uint64_t sum = 0;
        for (int j = 0; j < width; j++) {
            int px = bx >> SHIFT, py = by >> SHIFT;
            if ((unsigned)px < (unsigned)w && (unsigned)py < (unsigned)h)
                sum += data[py * w + px];
            else
                sum += corner_mean;
            bx += pvx;
            by += pvy;
        }
        sums[i] = sum;
    }
    return sums;
}

cv::Mat applyGaussianWindow(const cv::Mat &image, double sigma) {
    int h = image.rows, w = image.cols;
    cv::Mat window(h, w, CV_32F);
    double sigma2 = 2.0 * sigma * sigma;
    for (int y = 0; y < h; y++) {
        double ny = 2.0 * y / (h - 1) - 1.0;
        for (int x = 0; x < w; x++) {
            double nx = 2.0 * x / (w - 1) - 1.0;
            window.at<float>(y, x) = static_cast<float>(std::exp(-(nx * nx + ny * ny) / sigma2));
        }
    }
    cv::Mat float_img;
    image.convertTo(float_img, CV_32F);
    cv::Mat result;
    cv::multiply(float_img, window, result);
    result.convertTo(result, CV_8U);
    return result;
}

cv::Mat radonTransformFloat(const cv::Mat &image) {
    int h = image.rows, w = image.cols;
    int line_length = static_cast<int>(std::sqrt(h * h + w * w));

    // Corner pixels mean
    double cm = (cv::mean(image.row(0))[0] + cv::mean(image.row(h - 1))[0] +
                 cv::mean(image.col(0))[0] + cv::mean(image.col(w - 1))[0]) / 4.0;
    uint8_t corner_mean = static_cast<uint8_t>(cm);

    cv::Mat sinogram(360, line_length, CV_32F, cv::Scalar(0));

    #pragma omp parallel for schedule(dynamic)
    for (int angle = 0; angle < 360; angle++) {
        auto points = getLinePoints(image, angle, line_length);
        auto sums = perpendicularSum(image, points, line_length, corner_mean);
        float *row = sinogram.ptr<float>(angle);
        for (int j = 0; j < line_length; j++)
            row[j] = static_cast<float>(sums[j]);
    }
    return sinogram;
}

// =============================================================================
// Sinogram Core Extraction
// =============================================================================

std::vector<cv::Mat> extractSinogramCore(const cv::Mat &sinogram,
                                          int template_height, int template_width) {
    int row_len = sinogram.cols;
    int center = row_len / 2;
    std::vector<cv::Mat> cores(360);

    for (int angle = 0; angle < 360; angle++) {
        double theta = angle * CV_PI / 180.0;
        int proj_width = static_cast<int>(
            std::abs(template_width * cos(theta)) + std::abs(template_height * sin(theta)));
        proj_width = std::max(proj_width, 4);
        int half = proj_width / 2;
        int start = std::max(0, center - half);
        int end = std::min(row_len, center + half);
        cores[angle] = sinogram(cv::Range(angle, angle + 1),
                                cv::Range(start, end)).clone();
    }
    return cores;
}

// =============================================================================
// Hough Voting Angle Detection
// =============================================================================

HoughResult detectAngleHough(const cv::Mat &sinogram_image,
                              const cv::Mat &sinogram_template,
                              int template_height, int template_width,
                              float score_threshold) {
    auto cores = extractSinogramCore(sinogram_template, template_height, template_width);
    int n_img = sinogram_image.cols;

    // Precompute 1D NCC for all (image_row, template_core) pairs
    // match_scores[i * 180 + j] = best NCC score for image row i, template row j
    std::vector<float> match_scores(360 * 180, 0.0f);
    std::vector<int> match_positions(360 * 180, 0);

    #pragma omp parallel for schedule(dynamic) collapse(2)
    for (int j = 0; j < 180; j++) {
        for (int i = 0; i < 360; i++) {
            if (cores[j].cols < 4 || cores[j].cols >= n_img)
                continue;
            cv::Mat img_row = sinogram_image.row(i);
            cv::Mat result;
            cv::matchTemplate(img_row, cores[j], result, cv::TM_CCOEFF_NORMED);

            double maxVal;
            cv::Point maxLoc;
            cv::minMaxLoc(result, nullptr, &maxVal, nullptr, &maxLoc);

            int idx = i * 180 + j;
            match_scores[idx] = static_cast<float>(maxVal);
            match_positions[idx] = maxLoc.x;
        }
    }

    // Hough voting
    std::vector<double> accumulator(180, 0.0);
    for (int j = 0; j < 180; j++) {
        for (int i = 0; i < 360; i++) {
            float score = match_scores[i * 180 + j];
            if (score < score_threshold) continue;
            int alpha = ((i - j) % 180 + 180) % 180;
            accumulator[alpha] += score;
        }
    }

    int detected_alpha = static_cast<int>(
        std::max_element(accumulator.begin(), accumulator.end()) - accumulator.begin());

    // Position estimation via sinusoidal fitting
    std::vector<double> offsets, weights;
    std::vector<std::array<double, 2>> A_rows;

    for (int theta_img = 0; theta_img < 360; theta_img++) {
        int theta_tmpl = ((theta_img - detected_alpha) % 360 + 360) % 360;
        if (theta_tmpl >= 180) continue;

        int idx = theta_img * 180 + theta_tmpl;
        float score = match_scores[idx];
        if (score < score_threshold) continue;

        int pos = match_positions[idx];
        int core_len = cores[theta_tmpl].cols;
        double offset = pos + core_len / 2.0 - n_img / 2.0;

        double theta_rad = theta_img * CV_PI / 180.0;
        offsets.push_back(offset);
        weights.push_back(std::max(score, 0.0f));
        A_rows.push_back({cos(theta_rad), sin(theta_rad)});
    }

    int dx = 0, dy = 0;
    int n = static_cast<int>(offsets.size());
    if (n >= 2) {
        // Weighted least squares: A'WA x = A'Wb
        double a00 = 0, a01 = 0, a11 = 0, b0 = 0, b1 = 0;
        for (int k = 0; k < n; k++) {
            double w = weights[k];
            double c = A_rows[k][0], s = A_rows[k][1];
            a00 += w * c * c;
            a01 += w * c * s;
            a11 += w * s * s;
            b0 += w * c * offsets[k];
            b1 += w * s * offsets[k];
        }
        double det = a00 * a11 - a01 * a01;
        if (std::abs(det) > 1e-10) {
            dx = static_cast<int>(std::round((a11 * b0 - a01 * b1) / det));
            dy = static_cast<int>(std::round((a00 * b1 - a01 * b0) / det));
        }
    }

    return {detected_alpha, dx, dy, accumulator};
}

// =============================================================================
// Brute-force 2D NCC (for comparison)
// =============================================================================

BruteForceResult detectAngleBruteForce(const cv::Mat &image, const cv::Mat &templ) {
    int th = templ.rows, tw = templ.cols;
    double best_score = -1;
    int best_angle = 0;
    cv::Point best_pos(0, 0);

    for (int angle = 0; angle < 360; angle++) {
        cv::Mat M = cv::getRotationMatrix2D(cv::Point2f(tw / 2.0f, th / 2.0f), angle, 1.0);
        cv::Mat rotated;
        cv::warpAffine(templ, rotated, M, cv::Size(tw, th), cv::INTER_LINEAR,
                        cv::BORDER_REFLECT_101);

        cv::Mat result;
        cv::matchTemplate(image, rotated, result, cv::TM_CCOEFF_NORMED);

        double maxVal;
        cv::Point maxLoc;
        cv::minMaxLoc(result, nullptr, &maxVal, nullptr, &maxLoc);

        if (maxVal > best_score) {
            best_score = maxVal;
            best_angle = angle;
            best_pos = maxLoc;
        }
    }

    return {best_angle, best_pos, best_score};
}
