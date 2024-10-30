#include <opencv2/opencv.hpp>
#include <cmath>
#include <vector>
#include <iostream>
#include <numeric>

std::vector<cv::Point> getLinePoints(const cv::Mat &image, int angle, int length) {
    int h = image.rows, w = image.cols;
    cv::Point center(w / 2, h / 2);

    double angle_rad = angle * CV_PI / 180.0;

    int start_x = static_cast<int>(center.x - length / 2 * cos(angle_rad));
    int start_y = static_cast<int>(center.y + length / 2 * sin(angle_rad));
    int end_x = static_cast<int>(center.x + length / 2 * cos(angle_rad));
    int end_y = static_cast<int>(center.y - length / 2 * sin(angle_rad));

    std::vector<cv::Point> line_points(length);
    for (int i = 0; i < length; i++) {
        float t = static_cast<float>(i) / (length - 1);
        int x = static_cast<int>(start_x * (1 - t) + end_x * t);
        int y = static_cast<int>(start_y * (1 - t) + end_y * t);
        line_points[i] = cv::Point(x, y);
    }
    return line_points;
}

std::vector<uint64_t> perpendicularSum(const cv::Mat &image, const std::vector<cv::Point> &points, int width, uint8_t corner_pixels_mean) {
    int h = image.rows, w = image.cols;
    uint8_t* data_ptr = image.data;
    
    size_t points_size = points.size();
    std::vector<uint64_t> sums(points_size);

    // 固定された垂直ベクトルの計算
    cv::Point2f perp_vec(0, 0);
    for (const auto &point : points) {
        perp_vec = cv::Point2f(-(point.y - h / 2), point.x - w / 2);
        float norm = sqrt(perp_vec.x * perp_vec.x + perp_vec.y * perp_vec.y);
        
        if (norm != 0) {
            perp_vec.x /= norm;
            perp_vec.y /= norm;
            break;
        }
    }

    // 各ポイントの処理を並列化
    for (int i = 0; i < points_size; i++) {
        const auto &point = points[i];
        uint64_t sum = 0;

        // 各ポイント周りのサム計算
        for (int j = -width / 2; j < width / 2; j++) {
            int perp_x = static_cast<int>(point.x + perp_vec.x * j);
            int perp_y = static_cast<int>(point.y + perp_vec.y * j);

            // 境界チェック
            if (perp_x >= 0 && perp_x < w && perp_y >= 0 && perp_y < h) {
                sum += data_ptr[perp_y * w + perp_x];
            } else {
                sum += corner_pixels_mean;
            }
        }
        sums[i] = sum;
    }

    return sums;
}

cv::Mat radonTransform(const cv::Mat &image) {
    int h = image.rows, w = image.cols;
    int line_length = static_cast<int>(sqrt(h * h + w * w));
    int perpendicular_width = line_length;

    cv::Scalar top_mean = cv::mean(image.row(0));
    cv::Scalar bottom_mean = cv::mean(image.row(h - 1));
    cv::Scalar left_mean = cv::mean(image.col(0));
    cv::Scalar right_mean = cv::mean(image.col(w - 1));

    cv::Scalar corner_pixels_mean = (top_mean + bottom_mean + left_mean + right_mean) / 4;

    std::vector<std::vector<uint64_t>> angle_sums(180);
    uint64_t max_value = 0;

    for (int angle = 0; angle < 180; angle++) {
        std::vector<cv::Point> line_points = getLinePoints(image, angle, line_length);
        std::vector<uint64_t> sums = perpendicularSum(image, line_points, perpendicular_width, static_cast<uint8_t>(corner_pixels_mean[0]));
        angle_sums[angle] = sums;
        max_value = std::max(max_value, *max_element(sums.begin(), sums.end()));
    }

    cv::Mat normalized_image(180, line_length, CV_32S, cv::Scalar(0));
    uint32_t* data_ptr = reinterpret_cast<uint32_t*>(normalized_image.data);
    for (int i = 0; i < angle_sums.size(); i++) {
        for (int j = 0; j < angle_sums[i].size(); j++) {
            data_ptr[i * line_length + j] = static_cast<uint32_t>(angle_sums[i][j] / (float)max_value * 255);
        }
    }

    return normalized_image;
}

cv::Mat radonFFT(const cv::Mat &image) {
    cv::Mat fft_image;
    image.convertTo(fft_image, CV_32F);  // 浮動小数点に変換

    // 結果を保存するための行列を作成
    cv::Mat magnitude_image = cv::Mat::zeros(image.size(), CV_32F);

    for (int i = 0; i < image.rows; i++) {
        // 各行を取り出してDFTを実行
        cv::Mat row = fft_image.row(i);
        cv::Mat complex_row;
        cv::dft(row, complex_row, cv::DFT_COMPLEX_OUTPUT);

        // 複素数を分離して振幅を計算
        cv::Mat planes[2];
        cv::split(complex_row, planes);
        cv::magnitude(planes[0], planes[1], magnitude_image.row(i));  // 振幅を計算して保存
    }

    // 振幅に対して対数変換
    magnitude_image += cv::Scalar::all(1);
    cv::log(magnitude_image, magnitude_image);

    return magnitude_image;
}

cv::Mat centerPasteImage(const cv::Mat &wide_img, const cv::Mat &narrow_img) {
    // 広い画像の高さと幅
    int wide_h = wide_img.rows;
    int wide_w = wide_img.cols;

    // 狭い画像の高さと幅
    int narrow_h = narrow_img.rows;
    int narrow_w = narrow_img.cols;

    // 広い画像と同じサイズの黒い背景画像を生成
    cv::Mat black_background = cv::Mat::zeros(wide_h, wide_w, narrow_img.type());

    // 狭い画像を中央に配置するためのオフセットを計算
    int x_offset = (wide_w - narrow_w) / 2;
    int y_offset = (wide_h - narrow_h) / 2;

    // 黒い背景に狭い画像を貼り付け
    narrow_img.copyTo(black_background(cv::Rect(x_offset, y_offset, narrow_w, narrow_h)));

    return black_background;
}

std::vector<float> matchTemplateOneLine(const cv::Mat &image_line, const cv::Mat &template_line) {
    int iteration_range = image_line.cols - template_line.cols;
    std::vector<float> diff_std_list;

    cv::Mat template_temp = template_line.clone();
    template_temp -= cv::mean(template_temp)[0];

    for (int i = 0; i < iteration_range; ++i) {
        // 対象画像の部分領域を取得
        cv::Mat target_img_temp = image_line.colRange(i, i + template_line.cols).clone();

        // 平均値を引いて標準化
        target_img_temp -= cv::mean(target_img_temp)[0];

        // 差分を計算
        cv::Mat diff = target_img_temp - template_temp;

        // 標準偏差を計算
        cv::Scalar stddev;
        cv::meanStdDev(diff, cv::noArray(), stddev);
        diff_std_list.push_back(static_cast<float>(stddev[0]));
    }

    return diff_std_list;
}

void matchTemplateRotatable(const cv::Mat &image, const cv::Mat &templ) {
    int rows = templ.rows, cols = templ.cols;
    
    // ガウシアンウィンドウを作成
    cv::Mat gaussian_window(rows, cols, CV_32F);
    float sigma = 0.3f;
    for (int y = 0; y < rows; y++) {
        for (int x = 0; x < cols; x++) {
            float norm_x = (2.0f * x / (cols - 1)) - 1.0f;
            float norm_y = (2.0f * y / (rows - 1)) - 1.0f;
            gaussian_window.at<float>(y, x) = exp(-(norm_x * norm_x + norm_y * norm_y) / (2 * sigma * sigma));
        }
    }

    // 画像に対応するガウシアンウィンドウを作成
    rows = image.rows;
    cols = image.cols;
    cv::Mat gaussian_window_for_image(rows, cols, CV_32F);
    sigma = 0.5f;
    for (int y = 0; y < rows; y++) {
        for (int x = 0; x < cols; x++) {
            float norm_x = (2.0f * x / (cols - 1)) - 1.0f;
            float norm_y = (2.0f * y / (rows - 1)) - 1.0f;
            gaussian_window_for_image.at<float>(y, x) = exp(-(norm_x * norm_x + norm_y * norm_y) / (2 * sigma * sigma));
        }
    }
    
    cv::Mat templ_float, image_float;
    templ.convertTo(templ_float, CV_32F);
    image.convertTo(image_float, CV_32F);
    cv::Mat gaussian_windowed_templ = templ_float.mul(gaussian_window);
    cv::Mat gaussian_windowed_image = image_float.mul(gaussian_window_for_image);
    gaussian_windowed_templ.convertTo(gaussian_windowed_templ, CV_8UC1);
    gaussian_windowed_image.convertTo(gaussian_windowed_image, CV_8UC1);

    // ラドン変換を適用
    cv::Mat radon_transformed_template = radonTransform(templ);
    cv::Mat radon_transformed_template_for_fft = radonTransform(gaussian_windowed_templ);

    cv::Mat radon_transformed_image = radonTransform(image);
    cv::Mat radon_transformed_image_for_fft = radonTransform(gaussian_windowed_image);

    cv::Mat radon_transformed_template_float, radon_transformed_image_float;
    radon_transformed_template.convertTo(radon_transformed_template_float, CV_32F);
    radon_transformed_image.convertTo(radon_transformed_image_float, CV_32F);

    cv::Mat radon_transformed_template_for_fft2 = centerPasteImage(radon_transformed_image, radon_transformed_template_for_fft);

    // FFTを適用
    cv::Mat fft_template = radonFFT(radon_transformed_template_for_fft2);

    cv::Mat fft_image = radonFFT(radon_transformed_image_for_fft);
    cv::Mat fft_image_combined;
    cv::vconcat(fft_image, fft_image, fft_image_combined);

    // FFTのマッチング
    std::vector<float> diff_mean_list(180);
    for (int i = 0; i < 180; i++) {
        cv::Mat diff = abs(fft_image_combined(cv::Rect(0, i, fft_template.cols, 180)) - fft_template);
        cv::Scalar mean_diff = mean(diff);
        diff_mean_list[i] = static_cast<float>(mean_diff[0]);
    }
    auto min_iter = min_element(diff_mean_list.begin(), diff_mean_list.end());
    int min_index = std::distance(diff_mean_list.begin(), min_iter);

    int detect_angle = min_index;
    int angle_option1 = 180 - detect_angle;
    if (angle_option1 >= 180) angle_option1 -= 180;
    
    // 選択肢となるテンプレート行を取得
    cv::Mat template_option1 = radon_transformed_template_float.row(angle_option1).clone();
    cv::Mat template_option2;
    cv::flip(template_option1, template_option2, 1);
    cv::Mat target_img = radon_transformed_image_float.row(0).clone();

    int angle_option1_90 = angle_option1 + 90;
    if (angle_option1_90 >= 180) angle_option1_90 -= 180;

    cv::Mat template_option1_90 = radon_transformed_template_float.row(angle_option1_90).clone();
    cv::Mat template_option2_90;
    cv::flip(template_option1_90, template_option2_90, 1);
    cv::Mat target_img_90 = radon_transformed_image_float.row(90).clone();

    // マッチング
    std::vector<float> result = matchTemplateOneLine(target_img, template_option1);
    auto min_val_iter = min_element(result.begin(), result.end());
    int min_val_index = std::distance(result.begin(), min_val_iter);
    float min_val = result[min_val_index];

    result = matchTemplateOneLine(target_img, template_option2);
    auto min_val2_iter = min_element(result.begin(), result.end());
    int min_val2_index = std::distance(result.begin(), min_val2_iter);
    float min_val2 = result[min_val2_index];

    result = matchTemplateOneLine(target_img_90, template_option1_90);
    auto min_val_90_iter = min_element(result.begin(), result.end());
    int min_val_90_index = std::distance(result.begin(), min_val_90_iter);
    float min_val_90 = result[min_val_90_index];

    result = matchTemplateOneLine(target_img_90, template_option2_90);
    auto min_val2_90_iter = min_element(result.begin(), result.end());
    int min_val2_90_index = std::distance(result.begin(), min_val2_90_iter);
    float min_val2_90 = result[min_val2_90_index];

    int dx_option1 = min_val_index + template_option1.cols / 2 - target_img.cols / 2;
    int dy_option1 = -(min_val_90_index + template_option1.cols / 2 - target_img.cols / 2);

    int dx_option2 = min_val2_index + template_option2.cols / 2 - target_img.cols / 2;
    int dy_option2 = -(min_val2_90_index + template_option2.cols / 2 - target_img.cols / 2);

    int target_angle, dx, dy;
    if (min_val + min_val_90 < min_val2 + min_val2_90) {
        target_angle = detect_angle;
        dx = dx_option1;
        dy = dy_option1;
    } else {
        target_angle = detect_angle + 180;
        dx = dx_option2;
        dy = dy_option2;
    }

    std::cout << "Detected: angle = " << target_angle << ", dx = " << dx << ", dy = " << dy << std::endl;
}

