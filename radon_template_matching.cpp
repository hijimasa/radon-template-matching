#include <opencv2/opencv.hpp>
#include <cmath>
#include <vector>
#include <iostream>
#include <numeric>
#include <chrono>

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

    // 最適化1: 固定小数点演算を使用（16ビットシフト）
    const int SHIFT = 16;
    const int SCALE = 1 << SHIFT;
    int perp_vec_x_fixed = static_cast<int>(perp_vec.x * SCALE);
    int perp_vec_y_fixed = static_cast<int>(perp_vec.y * SCALE);

    int half_width = width / 2;
    const int corner_mean_i = static_cast<int>(corner_pixels_mean);

    // メモリアクセスパターンの改善
    const int w_minus_1 = w - 1;
    const int h_minus_1 = h - 1;

    // 各ポイントの処理
    for (int i = 0; i < points_size; i++) {
        const cv::Point &point = points[i];

        // 開始座標を固定小数点で計算
        int base_x_fixed = (point.x << SHIFT) - perp_vec_x_fixed * half_width;
        int base_y_fixed = (point.y << SHIFT) - perp_vec_y_fixed * half_width;

        uint64_t sum = 0;

        // 内側ループ: 固定小数点演算で高速化
        int curr_x_fixed = base_x_fixed;
        int curr_y_fixed = base_y_fixed;

        for (int j = 0; j < width; j++) {
            int perp_x = curr_x_fixed >> SHIFT;
            int perp_y = curr_y_fixed >> SHIFT;

            // 境界チェック（符号なし整数への変換で高速化）
            // perp_x >= 0 && perp_x < w は (unsigned)perp_x < (unsigned)w と等価
            if (static_cast<unsigned>(perp_x) <= static_cast<unsigned>(w_minus_1) &&
                static_cast<unsigned>(perp_y) <= static_cast<unsigned>(h_minus_1)) {
                sum += data_ptr[perp_y * w + perp_x];
            } else {
                sum += corner_mean_i;
            }

            curr_x_fixed += perp_vec_x_fixed;
            curr_y_fixed += perp_vec_y_fixed;
        }

        sums[i] = sum;
    }

    return sums;
}

// angle_step: 角度ステップ（デフォルト1度）
cv::Mat radonTransform(const cv::Mat &image, int angle_step = 1) {
    auto radon_start = std::chrono::high_resolution_clock::now();

    int h = image.rows, w = image.cols;
    int line_length = static_cast<int>(sqrt(h * h + w * w));
    int perpendicular_width = line_length;

    // 角度数を計算
    int num_angles = 180 / angle_step;

    std::cout << "    [Radon] Image size: " << h << "x" << w
              << ", line_length: " << line_length
              << ", perpendicular_width: " << perpendicular_width << std::endl;
    std::cout << "    [Radon] Angle step: " << angle_step << " degrees, num_angles: " << num_angles << std::endl;
    std::cout << "    [Radon] Total pixels per angle: " << line_length << " x " << perpendicular_width
              << " = " << (long long)line_length * perpendicular_width << " pixels" << std::endl;

    auto prep_start = std::chrono::high_resolution_clock::now();
    cv::Scalar top_mean = cv::mean(image.row(0));
    cv::Scalar bottom_mean = cv::mean(image.row(h - 1));
    cv::Scalar left_mean = cv::mean(image.col(0));
    cv::Scalar right_mean = cv::mean(image.col(w - 1));

    cv::Scalar corner_pixels_mean = (top_mean + bottom_mean + left_mean + right_mean) / 4;
    auto prep_end = std::chrono::high_resolution_clock::now();
    std::cout << "    [Radon] Preparation (mean calculation): "
              << std::chrono::duration_cast<std::chrono::milliseconds>(prep_end - prep_start).count()
              << " ms" << std::endl;

    std::vector<std::vector<uint64_t>> angle_sums(num_angles);
    uint64_t max_value = 0;

    auto angle_loop_start = std::chrono::high_resolution_clock::now();
    long long total_getLinePoints_time = 0;
    long long total_perpendicularSum_time = 0;
    long long total_maxElement_time = 0;

    for (int i = 0; i < num_angles; i++) {
        int angle = i * angle_step;

        auto getline_start = std::chrono::high_resolution_clock::now();
        std::vector<cv::Point> line_points = getLinePoints(image, angle, line_length);
        auto getline_end = std::chrono::high_resolution_clock::now();
        total_getLinePoints_time += std::chrono::duration_cast<std::chrono::microseconds>(getline_end - getline_start).count();

        auto perpsum_start = std::chrono::high_resolution_clock::now();
        std::vector<uint64_t> sums = perpendicularSum(image, line_points, perpendicular_width, static_cast<uint8_t>(corner_pixels_mean[0]));
        auto perpsum_end = std::chrono::high_resolution_clock::now();
        total_perpendicularSum_time += std::chrono::duration_cast<std::chrono::microseconds>(perpsum_end - perpsum_start).count();

        angle_sums[i] = sums;

        auto maxelem_start = std::chrono::high_resolution_clock::now();
        max_value = std::max(max_value, *max_element(sums.begin(), sums.end()));
        auto maxelem_end = std::chrono::high_resolution_clock::now();
        total_maxElement_time += std::chrono::duration_cast<std::chrono::microseconds>(maxelem_end - maxelem_start).count();
    }
    auto angle_loop_end = std::chrono::high_resolution_clock::now();

    std::cout << "    [Radon] getLinePoints total: " << total_getLinePoints_time / 1000.0 << " ms" << std::endl;
    std::cout << "    [Radon] perpendicularSum total: " << total_perpendicularSum_time / 1000.0 << " ms" << std::endl;
    std::cout << "    [Radon] max_element total: " << total_maxElement_time / 1000.0 << " ms" << std::endl;
    std::cout << "  [Radon] Angle loop (" << num_angles << " iterations): "
              << std::chrono::duration_cast<std::chrono::milliseconds>(angle_loop_end - angle_loop_start).count()
              << " ms" << std::endl;

    auto normalize_start = std::chrono::high_resolution_clock::now();
    cv::Mat normalized_image(num_angles, line_length, CV_32S, cv::Scalar(0));
    uint32_t* data_ptr = reinterpret_cast<uint32_t*>(normalized_image.data);
    for (int i = 0; i < angle_sums.size(); i++) {
        for (int j = 0; j < angle_sums[i].size(); j++) {
            data_ptr[i * line_length + j] = static_cast<uint32_t>(angle_sums[i][j] / (float)max_value * 255);
        }
    }
    auto normalize_end = std::chrono::high_resolution_clock::now();
    std::cout << "    [Radon] Normalization: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(normalize_end - normalize_start).count()
              << " ms" << std::endl;

    auto radon_end = std::chrono::high_resolution_clock::now();
    std::cout << "  [Radon] Total time: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(radon_end - radon_start).count()
              << " ms" << std::endl;

    return normalized_image;
}

cv::Mat radonFFT(const cv::Mat &image) {
    auto fft_start = std::chrono::high_resolution_clock::now();

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

    auto fft_end = std::chrono::high_resolution_clock::now();
    std::cout << "  [FFT] Total time: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(fft_end - fft_start).count()
              << " ms" << std::endl;

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
    auto total_start = std::chrono::high_resolution_clock::now();

    int rows = templ.rows, cols = templ.cols;

    auto gaussian_start = std::chrono::high_resolution_clock::now();
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
    auto gaussian_end = std::chrono::high_resolution_clock::now();
    std::cout << "[1] Gaussian window creation: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(gaussian_end - gaussian_start).count()
              << " ms" << std::endl;

    // ラドン変換を適用
    // FFT用は粗い角度ステップで高速化（5度刻み）
    const int COARSE_ANGLE_STEP = 5;
    const int FINE_ANGLE_STEP = 1;

    std::cout << "[2] Radon transform - template (original, fine):" << std::endl;
    cv::Mat radon_transformed_template = radonTransform(templ, FINE_ANGLE_STEP);
    std::cout << "[3] Radon transform - template (windowed, coarse for FFT):" << std::endl;
    cv::Mat radon_transformed_template_for_fft = radonTransform(gaussian_windowed_templ, COARSE_ANGLE_STEP);

    std::cout << "[4] Radon transform - image (original, fine):" << std::endl;
    cv::Mat radon_transformed_image = radonTransform(image, FINE_ANGLE_STEP);
    std::cout << "[5] Radon transform - image (windowed, coarse for FFT):" << std::endl;
    cv::Mat radon_transformed_image_for_fft = radonTransform(gaussian_windowed_image, COARSE_ANGLE_STEP);

    cv::Mat radon_transformed_template_float, radon_transformed_image_float;
    radon_transformed_template.convertTo(radon_transformed_template_float, CV_32F);
    radon_transformed_image.convertTo(radon_transformed_image_float, CV_32F);

    // FFT用: coarse角度同士で処理
    cv::Mat radon_transformed_template_for_fft2 = centerPasteImage(radon_transformed_image_for_fft, radon_transformed_template_for_fft);

    // FFTを適用
    std::cout << "[6] FFT - template:" << std::endl;
    cv::Mat fft_template = radonFFT(radon_transformed_template_for_fft2);

    std::cout << "[7] FFT - image:" << std::endl;
    cv::Mat fft_image = radonFFT(radon_transformed_image_for_fft);
    cv::Mat fft_image_combined;
    cv::vconcat(fft_image, fft_image, fft_image_combined);

    // FFTのマッチング（粗い角度ステップで高速化）
    auto fft_matching_start = std::chrono::high_resolution_clock::now();
    int num_coarse_angles = 180 / COARSE_ANGLE_STEP;
    std::vector<float> diff_mean_list(num_coarse_angles);
    for (int i = 0; i < num_coarse_angles; i++) {
        cv::Mat diff = abs(fft_image_combined(cv::Rect(0, i, fft_template.cols, num_coarse_angles)) - fft_template);
        cv::Scalar mean_diff = mean(diff);
        diff_mean_list[i] = static_cast<float>(mean_diff[0]);
    }
    auto min_iter = min_element(diff_mean_list.begin(), diff_mean_list.end());
    int min_index = std::distance(diff_mean_list.begin(), min_iter);
    auto fft_matching_end = std::chrono::high_resolution_clock::now();
    std::cout << "[8] FFT matching (" << num_coarse_angles << " iterations, coarse): "
              << std::chrono::duration_cast<std::chrono::milliseconds>(fft_matching_end - fft_matching_start).count()
              << " ms" << std::endl;

    // 粗い角度から実際の角度に変換
    int detect_angle_coarse = min_index * COARSE_ANGLE_STEP;
    std::cout << "    Detected coarse angle: " << detect_angle_coarse << " degrees" << std::endl;

    int angle_option1 = 180 - detect_angle_coarse;
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
    auto matching_start = std::chrono::high_resolution_clock::now();
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
    auto matching_end = std::chrono::high_resolution_clock::now();
    std::cout << "[9] 1D template matching (4 iterations): "
              << std::chrono::duration_cast<std::chrono::milliseconds>(matching_end - matching_start).count()
              << " ms" << std::endl;

    int dx_option1 = min_val_index + template_option1.cols / 2 - target_img.cols / 2;
    int dy_option1 = -(min_val_90_index + template_option1.cols / 2 - target_img.cols / 2);

    int dx_option2 = min_val2_index + template_option2.cols / 2 - target_img.cols / 2;
    int dy_option2 = -(min_val2_90_index + template_option2.cols / 2 - target_img.cols / 2);

    int target_angle, dx, dy;
    if (min_val + min_val_90 < min_val2 + min_val2_90) {
        target_angle = detect_angle_coarse;
        dx = dx_option1;
        dy = dy_option1;
    } else {
        target_angle = detect_angle_coarse + 180;
        dx = dx_option2;
        dy = dy_option2;
    }

    auto total_end = std::chrono::high_resolution_clock::now();
    std::cout << "\n========================================" << std::endl;
    std::cout << "[TOTAL] matchTemplateRotatable: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(total_end - total_start).count()
              << " ms" << std::endl;
    std::cout << "========================================\n" << std::endl;
    std::cout << "Detected: angle = " << target_angle << ", dx = " << dx << ", dy = " << dy << std::endl;
}

