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

// =============================================================================
// 2-step detection: HF profile + 2D NCC refinement
// =============================================================================

std::vector<double> computeHFProfile(const cv::Mat &img_row,
                                      const cv::Mat &core,
                                      double cutoff_ratio) {
    int L = img_row.cols;
    int nc = core.cols;
    int nr = L - nc + 1;
    if (nr <= 0) return std::vector<double>(L, 1e30);

    // Zero-pad core to length L
    cv::Mat core_padded = cv::Mat::zeros(1, L, CV_64F);
    for (int i = 0; i < nc; i++)
        core_padded.at<double>(0, i) = core.at<float>(0, i);

    cv::Mat img_f;
    img_row.convertTo(img_f, CV_64F);

    // DFT (real→complex)
    cv::Mat A, B;
    cv::dft(img_f, A, cv::DFT_COMPLEX_OUTPUT);
    cv::dft(core_padded, B, cv::DFT_COMPLEX_OUTPUT);

    // Highpass mask
    int cutoff = std::max(1, (int)(L * cutoff_ratio));

    // Apply HP mask and compute A_hf, B_hf
    // DFT output: [0..L-1] complex pairs stored as (re, im) interleaved in 2-channel Mat
    cv::Mat A_hf = A.clone();
    cv::Mat B_hf = B.clone();
    // Zero out low frequencies [0, cutoff) and [L-cutoff+1, L)
    for (int k = 0; k < cutoff; k++) {
        A_hf.at<cv::Vec2d>(0, k) = {0, 0};
        B_hf.at<cv::Vec2d>(0, k) = {0, 0};
    }
    for (int k = L - cutoff + 1; k < L; k++) {
        A_hf.at<cv::Vec2d>(0, k) = {0, 0};
        B_hf.at<cv::Vec2d>(0, k) = {0, 0};
    }

    // const = Σ|A_HF|²/L + Σ|B_HF|²/L
    double sum_A2 = 0, sum_B2 = 0;
    for (int k = 0; k < L; k++) {
        auto a = A_hf.at<cv::Vec2d>(0, k);
        auto b = B_hf.at<cv::Vec2d>(0, k);
        sum_A2 += a[0] * a[0] + a[1] * a[1];
        sum_B2 += b[0] * b[0] + b[1] * b[1];
    }
    double constant = sum_A2 / L + sum_B2 / L;

    // cross_hf = A_hf * conj(B_hf)
    cv::Mat cross_hf(1, L, CV_64FC2);
    for (int k = 0; k < L; k++) {
        auto a = A_hf.at<cv::Vec2d>(0, k);
        auto b = B_hf.at<cv::Vec2d>(0, k);
        // (a_re + j*a_im) * (b_re - j*b_im)
        cross_hf.at<cv::Vec2d>(0, k) = {
            a[0] * b[0] + a[1] * b[1],
            a[1] * b[0] - a[0] * b[1]
        };
    }

    // IDFT → real part
    cv::Mat cross_spatial;
    cv::dft(cross_hf, cross_spatial, cv::DFT_INVERSE | cv::DFT_SCALE);

    // E_HF(p) = const - 2/L * Re(cross_spatial[p])
    std::vector<double> hf_energy(nr);
    for (int p = 0; p < nr; p++) {
        double re = cross_spatial.at<cv::Vec2d>(0, p)[0];
        hf_energy[p] = constant - 2.0 / L * re;
    }

    return hf_energy;
}


cv::Point findPositionByHFProfile(const cv::Mat &sinogram_image,
                                   const std::vector<cv::Mat> &cores,
                                   int alpha, int n_img, int center_t,
                                   const std::vector<double> &cos_t,
                                   const std::vector<double> &sin_t,
                                   int max_dx, int max_dy) {
    // Compute HF profiles for valid rows
    struct RowProfile {
        int row_idx;
        std::vector<double> profile;
        int core_len;
    };
    std::vector<RowProfile> profiles;

    for (int i = 0; i < 360; i++) {
        int j = ((i - alpha) % 360 + 360) % 360;
        if (j >= 180) continue;
        const cv::Mat &core = cores[j];
        if (core.cols < 4 || core.cols >= n_img) continue;

        cv::Mat img_row = sinogram_image.row(i);
        auto prof = computeHFProfile(img_row, core);
        profiles.push_back({i, std::move(prof), core.cols});
    }

    if (profiles.empty()) return {0, 0};

    int n_rows = (int)profiles.size();

    // Coarse search (step=2)
    int best_dx = 0, best_dy = 0;
    double best_e = 1e30;

    for (int dx = -max_dx; dx <= max_dx; dx += 2) {
        for (int dy = -max_dy; dy <= max_dy; dy += 2) {
            double total = 0;
            for (int k = 0; k < n_rows; k++) {
                const auto &rp = profiles[k];
                double offset = dx * cos_t[rp.row_idx] - dy * sin_t[rp.row_idx];
                int p = (int)std::round(center_t + offset - rp.core_len / 2.0);
                p = std::clamp(p, 0, (int)rp.profile.size() - 1);
                total += rp.profile[p];
            }
            double avg = total / n_rows;
            if (avg < best_e) {
                best_e = avg;
                best_dx = dx;
                best_dy = dy;
            }
        }
    }

    // Fine search (step=1, ±2px)
    int fine_dx = best_dx, fine_dy = best_dy;
    double fine_e = best_e;
    for (int dx = best_dx - 2; dx <= best_dx + 2; dx++) {
        if (std::abs(dx) > max_dx) continue;
        for (int dy = best_dy - 2; dy <= best_dy + 2; dy++) {
            if (std::abs(dy) > max_dy) continue;
            double total = 0;
            for (int k = 0; k < n_rows; k++) {
                const auto &rp = profiles[k];
                double offset = dx * cos_t[rp.row_idx] - dy * sin_t[rp.row_idx];
                int p = (int)std::round(center_t + offset - rp.core_len / 2.0);
                p = std::clamp(p, 0, (int)rp.profile.size() - 1);
                total += rp.profile[p];
            }
            double avg = total / n_rows;
            if (avg < fine_e) {
                fine_e = avg;
                fine_dx = dx;
                fine_dy = dy;
            }
        }
    }

    return {fine_dx, fine_dy};
}


DetectionResult refineByNCC(const cv::Mat &image, const cv::Mat &templ,
                             int coarse_angle, int coarse_dx, int coarse_dy,
                             int angle_range, int pos_range) {
    int th = templ.rows, tw = templ.cols;
    int img_h = image.rows, img_w = image.cols;
    int cy = img_h / 2, cx = img_w / 2;

    int best_a = coarse_angle;
    int best_dx = coarse_dx, best_dy = coarse_dy;
    double best_score = -1e30;

    for (int a = coarse_angle - angle_range; a <= coarse_angle + angle_range; a++) {
        int am = ((a % 360) + 360) % 360;
        cv::Point2f center(tw / 2.0f, th / 2.0f);
        cv::Mat M = cv::getRotationMatrix2D(center, (double)am, 1.0);

        cv::Mat tmpl_rot, mask_img;
        cv::warpAffine(templ, tmpl_rot, M, {tw, th},
                        cv::INTER_LINEAR, cv::BORDER_CONSTANT, 0);
        cv::Mat ones = cv::Mat::ones(th, tw, CV_8U) * 255;
        cv::warpAffine(ones, mask_img, M, {tw, th},
                        cv::INTER_LINEAR, cv::BORDER_CONSTANT, 0);

        // Precompute template masked values
        cv::Mat mask_bool;
        cv::compare(mask_img, 128, mask_bool, cv::CMP_GT);
        int n_valid = cv::countNonZero(mask_bool);
        if (n_valid < 10) continue;

        cv::Mat tmpl_f;
        tmpl_rot.convertTo(tmpl_f, CV_64F);
        double tm_sum = 0, tm_sq = 0;
        std::vector<cv::Point> mask_pts;
        for (int y = 0; y < th; y++) {
            for (int x = 0; x < tw; x++) {
                if (mask_bool.at<uchar>(y, x)) {
                    mask_pts.push_back({x, y});
                    tm_sum += tmpl_f.at<double>(y, x);
                }
            }
        }
        double tm_mean = tm_sum / mask_pts.size();
        double tm_energy = 0;
        std::vector<double> tm_centered(mask_pts.size());
        for (size_t k = 0; k < mask_pts.size(); k++) {
            double v = tmpl_f.at<double>(mask_pts[k].y, mask_pts[k].x) - tm_mean;
            tm_centered[k] = v;
            tm_energy += v * v;
        }
        if (tm_energy < 1e-10) continue;

        for (int dx = coarse_dx - pos_range; dx <= coarse_dx + pos_range; dx++) {
            for (int dy = coarse_dy - pos_range; dy <= coarse_dy + pos_range; dy++) {
                int y1 = cy - th / 2 + dy;
                int x1 = cx - tw / 2 + dx;
                if (y1 < 0 || y1 + th > img_h || x1 < 0 || x1 + tw > img_w)
                    continue;

                cv::Mat region_f;
                image(cv::Rect(x1, y1, tw, th)).convertTo(region_f, CV_64F);

                double rm_sum = 0;
                for (auto &pt : mask_pts)
                    rm_sum += region_f.at<double>(pt.y, pt.x);
                double rm_mean = rm_sum / mask_pts.size();

                double cross = 0, rm_energy = 0;
                for (size_t k = 0; k < mask_pts.size(); k++) {
                    double rv = region_f.at<double>(mask_pts[k].y, mask_pts[k].x) - rm_mean;
                    cross += rv * tm_centered[k];
                    rm_energy += rv * rv;
                }
                double denom = std::sqrt(rm_energy * tm_energy);
                if (denom < 1e-10) continue;
                double ncc = cross / denom;
                if (ncc > best_score) {
                    best_score = ncc;
                    best_a = am;
                    best_dx = dx;
                    best_dy = dy;
                }
            }
        }
    }

    return {best_a, best_dx, best_dy, best_score};
}


DetectionResult detectByHFAndNCC(const cv::Mat &image, const cv::Mat &templ,
                                  const cv::Mat &sinogram_image,
                                  const std::vector<cv::Mat> &cores,
                                  int n_img, int angle_step_coarse) {
    int th = templ.rows, tw = templ.cols;
    int img_h = image.rows, img_w = image.cols;
    int center_t = n_img / 2;
    int max_dx = (img_w - tw) / 2 - 2;
    int max_dy = (img_h - th) / 2 - 2;

    // Precompute cos/sin tables
    std::vector<double> cos_t(360), sin_t(360);
    for (int i = 0; i < 360; i++) {
        double rad = i * CV_PI / 180.0;
        cos_t[i] = std::cos(rad);
        sin_t[i] = std::sin(rad);
    }

    // Step 1: Coarse position estimation for each candidate angle
    struct Candidate {
        int alpha, dx, dy;
    };
    std::vector<Candidate> candidates;

    #pragma omp parallel for schedule(dynamic)
    for (int alpha = 0; alpha < 360; alpha += angle_step_coarse) {
        cv::Point pos = findPositionByHFProfile(
            sinogram_image, cores, alpha, n_img, center_t,
            cos_t, sin_t, max_dx, max_dy);
        #pragma omp critical
        candidates.push_back({alpha, pos.x, pos.y});
    }

    // Step 2: NCC refinement for each candidate, pick best
    DetectionResult best = {0, 0, 0, -1e30};

    for (const auto &c : candidates) {
        auto r = refineByNCC(image, templ, c.alpha, c.dx, c.dy,
                              angle_step_coarse, 3);
        if (r.score > best.score) best = r;
    }

    return best;
}
