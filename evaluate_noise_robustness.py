"""
Noise Robustness Evaluation for Radon Template Matching
ラドンテンプレートマッチングのノイズ耐性評価

Compare three methods:
1. Conventional: FFT amplitude spectrum difference
2. Phase Correlation (POC): Phase-only correlation
3. SNR-weighted: Signal-to-noise ratio based weighting
"""

import matplotlib
matplotlib.use('Agg')  # ヘッドレスバックエンド

import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
from numba import jit
import time

# Import from main module (Hough voting algorithm)
from radon_template_matching import (
    getLinePoints, perpendicularSum, radonTransformFloat,
    matchTemplateOneLineNCC, detectAngleHough, detectPosition
)


# =============================================================================
# Legacy functions (kept here for comparison evaluation only)
# =============================================================================

@jit('uint8[:,:](uint8[:,:])', nopython=True, cache=True)
def radonTransform(image):
    """uint8ラドン変換（旧手法の比較評価用）"""
    h, w = image.shape
    line_length = int(math.sqrt(h*h+w*w))
    perpendicular_width = int(math.sqrt(h*h+w*w))
    angle_sums = []
    max_value = np.uint64(0)
    for angle in range(360):
        line_points = getLinePoints(image, np.int32(angle), np.int32(line_length))
        sums = perpendicularSum(image, line_points, perpendicular_width)
        angle_sums.append(sums)
        if max_value < max(sums):
            max_value = max(sums)
    normalized_image = np.zeros((360, line_length), dtype=np.uint32)
    for i, sums in enumerate(angle_sums):
        for j, value in enumerate(sums):
            normalized_image[i, j] = (value / max_value) * 255
    return normalized_image.astype(np.uint8)


def centerPasteImage(wide_img, narrow_img):
    """中心にペーストした画像を生成（旧手法の比較評価用）"""
    wide_h, wide_w = wide_img.shape
    narrow_h, narrow_w = narrow_img.shape
    black_background = np.zeros((wide_h, wide_w), dtype=np.uint8)
    x_offset = (wide_w - narrow_w) // 2
    y_offset = (wide_h - narrow_h) // 2
    black_background[y_offset:y_offset + narrow_h, x_offset:x_offset + narrow_w] = narrow_img
    return black_background


@jit('float32[:](float32[:], float32[:])', nopython=True, cache=True)
def matchTemplateOneLine(image_line, template_line):
    """旧1Dマッチング（比較評価用）"""
    iteration_range = image_line.shape[0] - template_line.shape[0]
    template_temp = template_line
    template_temp_normalized = template_temp - np.mean(template_line)
    diff_std_list = np.zeros(iteration_range, dtype=np.float32)
    for i in range(iteration_range):
        target_img_temp = image_line[i:i+template_line.shape[0]]
        if not np.max(target_img_temp) == 0:
            target_img_temp = (target_img_temp / np.mean(target_img_temp) * np.mean(template_temp)).astype(np.float32)
        target_img_temp = target_img_temp - np.mean(target_img_temp)
        diff = target_img_temp - template_temp_normalized
        diff_std_list[i] = np.std(diff)
    return diff_std_list


# =============================================================================
# FFT Methods for Comparison
# =============================================================================

def radonFFT_conventional(image):
    """
    従来手法: 振幅スペクトルのみ（対数変換付き）
    """
    fft_result = []
    for row in image:
        fft_row = np.fft.fft(row)
        fft_magnitude = np.abs(fft_row)
        fft_result.append(fft_magnitude)
    
    fft_image = np.array(fft_result, dtype=np.float32)
    fft_image_normalized = np.log(fft_image + 1)
    return fft_image_normalized


def radonFFT_complex(image):
    """
    複素数FFT結果を返す（位相相関用）
    """
    fft_result = []
    for row in image:
        fft_row = np.fft.fft(row.astype(np.float32))
        fft_result.append(fft_row)
    return np.array(fft_result, dtype=np.complex64)


def radonFFT_with_power(image):
    """
    振幅スペクトルとパワースペクトルを返す（SNR重み付け用）
    """
    fft_result_magnitude = []
    fft_result_power = []
    for row in image:
        fft_row = np.fft.fft(row.astype(np.float32))
        magnitude = np.abs(fft_row)
        power = magnitude ** 2
        fft_result_magnitude.append(magnitude)
        fft_result_power.append(power)
    
    magnitude_image = np.array(fft_result_magnitude, dtype=np.float32)
    power_image = np.array(fft_result_power, dtype=np.float32)
    magnitude_normalized = np.log(magnitude_image + 1)
    
    return magnitude_normalized, power_image


# =============================================================================
# Angle Detection Methods
# =============================================================================

def detect_angle_conventional(fft_image, fft_template_half):
    """
    従来手法: 単純な振幅差分
    """
    diff_mean_list = []
    for i in range(180):
        diff = fft_image[i:i+180, :] - fft_template_half
        diff_mean_list.append(np.mean(np.abs(diff)))
    
    min_index = np.argmin(diff_mean_list)
    return min_index, diff_mean_list


def detect_angle_phase_correlation(fft_image_complex, fft_template_complex):
    """
    手法3: 位相相関（Phase Only Correlation）
    各角度シフトに対して、行ごとの位相相関の和を計算
    """
    correlation_list = []
    num_angles = fft_template_complex.shape[0]  # 180
    
    for shift in range(180):
        corr_sum = 0.0
        for j in range(num_angles):
            # 対応する行同士の位相相関
            row1 = fft_image_complex[(shift + j) % 360, :]
            row2 = fft_template_complex[j, :]
            
            # クロスパワースペクトル
            cross_power = row1 * np.conj(row2)
            # 正規化（位相のみを抽出）
            magnitude = np.abs(cross_power)
            cross_power_normalized = cross_power / (magnitude + 1e-10)
            
            # 逆FFTしてピーク値を取得
            correlation = np.abs(np.fft.ifft(cross_power_normalized))
            corr_sum += np.max(correlation)
        
        correlation_list.append(corr_sum)
    
    # 最大相関を持つシフト量
    max_index = np.argmax(correlation_list)
    return max_index, correlation_list


def detect_angle_poc_bandlimited(fft_image_complex, fft_template_complex, cutoff_ratio=0.3):
    """
    手法5: 帯域制限POC（Band-limited Phase Only Correlation）
    高周波成分をカットして低周波のみで位相相関
    cutoff_ratio: 使用する周波数帯域の割合（0.0〜1.0）
    """
    correlation_list = []
    num_angles = fft_template_complex.shape[0]  # 180
    num_freqs = fft_template_complex.shape[1]
    
    # カットオフ周波数（低周波側のみ使用）
    cutoff = int(num_freqs * cutoff_ratio)
    if cutoff < 2:
        cutoff = 2
    
    for shift in range(180):
        corr_sum = 0.0
        for j in range(num_angles):
            # 対応する行同士の位相相関
            row1 = fft_image_complex[(shift + j) % 360, :cutoff]
            row2 = fft_template_complex[j, :cutoff]
            
            # クロスパワースペクトル
            cross_power = row1 * np.conj(row2)
            # 正規化（位相のみを抽出）
            magnitude = np.abs(cross_power)
            cross_power_normalized = cross_power / (magnitude + 1e-10)
            
            # 逆FFTしてピーク値を取得
            correlation = np.abs(np.fft.ifft(cross_power_normalized, n=num_freqs))
            corr_sum += np.max(correlation)
        
        correlation_list.append(corr_sum)
    
    # 最大相関を持つシフト量
    max_index = np.argmax(correlation_list)
    return max_index, correlation_list


def detect_angle_poc_lowpass_weighted(fft_image_complex, fft_template_complex, sigma_ratio=0.2):
    """
    手法6: 低周波重み付きPOC（Gaussian Low-pass Weighted POC）
    ガウシアン重みで低周波を重視（閾値不要、滑らかな減衰）
    sigma_ratio: ガウシアンの標準偏差（周波数軸長に対する比率）
    """
    correlation_list = []
    num_angles = fft_template_complex.shape[0]  # 180
    num_freqs = fft_template_complex.shape[1]
    
    # ガウシアン重み（低周波ほど重い）
    freq_axis = np.arange(num_freqs)
    sigma = num_freqs * sigma_ratio
    gaussian_weights = np.exp(-freq_axis**2 / (2 * sigma**2))
    
    for shift in range(180):
        corr_sum = 0.0
        for j in range(num_angles):
            # 対応する行同士の位相相関
            row1 = fft_image_complex[(shift + j) % 360, :]
            row2 = fft_template_complex[j, :]
            
            # クロスパワースペクトル
            cross_power = row1 * np.conj(row2)
            # 正規化（位相のみを抽出）
            magnitude = np.abs(cross_power)
            cross_power_normalized = cross_power / (magnitude + 1e-10)
            
            # ガウシアン重み付け
            weighted_cross = cross_power_normalized * gaussian_weights
            
            # 逆FFTしてピーク値を取得
            correlation = np.abs(np.fft.ifft(weighted_cross))
            corr_sum += np.max(correlation)
        
        correlation_list.append(corr_sum)
    
    # 最大相関を持つシフト量
    max_index = np.argmax(correlation_list)
    return max_index, correlation_list


def detect_angle_poc_adaptive(image, template, fft_image_complex, fft_template_complex, 
                               radon_func, fft_complex_func, center_paste_func,
                               contrast_threshold=0.6):
    """
    手法7: 適応的POC（Adaptive POC）
    - コントラスト比が閾値以下の場合のみ線形正規化を適用
    - それ以外は素のPOCを使用
    
    contrast_threshold: コントラスト比の閾値（image_std / template_std）
    """
    img_std = np.std(image)
    templ_std = np.std(template)
    contrast_ratio = img_std / (templ_std + 1e-10)
    
    if contrast_ratio < contrast_threshold:
        # コントラストが低い場合：線形正規化を適用してから再計算
        img_mean = np.mean(image)
        templ_mean = np.mean(template)
        
        # 線形正規化
        normalized = (image.astype(np.float32) - img_mean) / (img_std + 1e-10) * templ_std + templ_mean
        normalized = np.clip(normalized, 0, 255).astype(np.uint8)
        
        # ガウシアンウィンドウ
        rows, cols = template.shape
        sigma = 0.3
        x = np.linspace(-1, 1, cols)
        y = np.linspace(-1, 1, rows)
        x, y = np.meshgrid(x, y)
        gaussian_window = np.exp(-(x**2 + y**2) / (2 * sigma**2))
        
        rows, cols = normalized.shape
        sigma = 0.5
        x = np.linspace(-1, 1, cols)
        y = np.linspace(-1, 1, rows)
        x, y = np.meshgrid(x, y)
        gaussian_window_for_image = np.exp(-(x**2 + y**2) / (2 * sigma**2))
        
        # 正規化画像でラドン変換
        radon_image_for_fft = radon_func((normalized * gaussian_window_for_image).astype(np.uint8))
        radon_template_for_fft = radon_func((template * gaussian_window).astype(np.uint8))
        radon_template_for_fft2 = center_paste_func(radon_image_for_fft, radon_template_for_fft)
        
        # FFT
        fft_image_complex_new = fft_complex_func(radon_image_for_fft)
        fft_template_complex_new = fft_complex_func(radon_template_for_fft2)
        fft_template_complex_half = fft_template_complex_new[0:180, :]
        fft_image_complex_combined = np.vstack([fft_image_complex_new, fft_image_complex_new])
        
        used_normalization = True
    else:
        # コントラストが十分：素のPOCを使用
        fft_template_complex_half = fft_template_complex[0:180, :]
        fft_image_complex_combined = np.vstack([fft_image_complex, fft_image_complex])
        used_normalization = False
    
    # 位相相関
    correlation_list = []
    num_angles = fft_template_complex_half.shape[0]
    
    for shift in range(180):
        corr_sum = 0.0
        for j in range(num_angles):
            row1 = fft_image_complex_combined[(shift + j) % 360, :]
            row2 = fft_template_complex_half[j, :]
            
            cross_power = row1 * np.conj(row2)
            magnitude = np.abs(cross_power)
            cross_power_normalized = cross_power / (magnitude + 1e-10)
            
            correlation = np.abs(np.fft.ifft(cross_power_normalized))
            corr_sum += np.max(correlation)
        
        correlation_list.append(corr_sum)
    
    max_index = np.argmax(correlation_list)
    return max_index, correlation_list, used_normalization, contrast_ratio


def detect_angle_snr_weighted(fft_image, fft_template_half, power_template):
    """
    手法4: SNRベース重み付け
    テンプレートのパワースペクトルと推定ノイズパワーから重みを計算
    """
    # ノイズパワー推定: 高周波成分（後ろ1/4）の平均パワー
    noise_power = np.mean(power_template[:, -power_template.shape[1]//4:]) + 1e-10
    
    # SNRベースの重み（テンプレートのパワーに基づく）
    signal_power = np.mean(power_template, axis=0)  # 角度方向に平均
    weights = signal_power / (signal_power + noise_power)
    weights = weights / (np.sum(weights) + 1e-10)  # 正規化
    
    # 重みを対数スケールに対応させる
    weights_log = np.log(weights + 1)
    weights_log = weights_log / (np.sum(weights_log) + 1e-10)
    
    diff_mean_list = []
    for i in range(180):
        diff = fft_image[i:i+180, :] - fft_template_half
        weighted_diff = np.abs(diff) * weights_log
        diff_mean_list.append(np.mean(weighted_diff))
    
    min_index = np.argmin(diff_mean_list)
    return min_index, diff_mean_list


# =============================================================================
# Contrast Normalization Functions
# =============================================================================

def normalize_contrast_to_template(image, template):
    """
    対象画像のコントラストをテンプレートに合わせる
    標準偏差ベースの正規化
    """
    img_mean = np.mean(image)
    img_std = np.std(image) + 1e-10
    
    templ_mean = np.mean(template)
    templ_std = np.std(template) + 1e-10
    
    # テンプレートと同じ平均・標準偏差に正規化
    normalized = (image.astype(np.float32) - img_mean) / img_std * templ_std + templ_mean
    normalized = np.clip(normalized, 0, 255).astype(np.uint8)
    
    return normalized


def histogram_equalization(image):
    """ヒストグラム均等化"""
    return cv2.equalizeHist(image)


def clahe_normalization(image, clip_limit=2.0, tile_grid_size=(8, 8)):
    """CLAHE（適応的ヒストグラム均等化）"""
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    return clahe.apply(image)


def match_histogram(image, template):
    """
    対象画像のヒストグラムをテンプレートに合わせる（Histogram Matching）
    """
    # 累積分布関数を計算
    def compute_cdf(img):
        hist, _ = np.histogram(img.flatten(), bins=256, range=(0, 256))
        cdf = hist.cumsum()
        cdf = cdf / cdf[-1]  # 正規化
        return cdf
    
    img_cdf = compute_cdf(image)
    templ_cdf = compute_cdf(template)
    
    # ルックアップテーブルを作成
    lookup = np.zeros(256, dtype=np.uint8)
    for i in range(256):
        j = 0
        while j < 255 and templ_cdf[j] < img_cdf[i]:
            j += 1
        lookup[i] = j
    
    # マッピング適用
    matched = lookup[image]
    return matched


# =============================================================================
# Noise Generation Functions
# =============================================================================

def add_salt_pepper_noise(image, amount=0.05):
    """ごま塩ノイズを追加"""
    noisy = image.copy()
    h, w = image.shape
    
    # Salt (white)
    num_salt = int(amount * h * w / 2)
    coords = [np.random.randint(0, i, num_salt) for i in [h, w]]
    noisy[coords[0], coords[1]] = 255
    
    # Pepper (black)
    num_pepper = int(amount * h * w / 2)
    coords = [np.random.randint(0, i, num_pepper) for i in [h, w]]
    noisy[coords[0], coords[1]] = 0
    
    return noisy


def add_gaussian_noise(image, mean=0, std=25):
    """ガウシアンノイズを追加"""
    noise = np.random.normal(mean, std, image.shape)
    noisy = image.astype(np.float32) + noise
    noisy = np.clip(noisy, 0, 255).astype(np.uint8)
    return noisy


def adjust_brightness(image, delta=50):
    """明るさを変更"""
    adjusted = image.astype(np.float32) + delta
    adjusted = np.clip(adjusted, 0, 255).astype(np.uint8)
    return adjusted


def adjust_contrast(image, factor=0.5):
    """コントラストを変更"""
    mean_val = np.mean(image)
    adjusted = (image.astype(np.float32) - mean_val) * factor + mean_val
    adjusted = np.clip(adjusted, 0, 255).astype(np.uint8)
    return adjusted


def add_combined_noise(image, sp_amount=0.02, gauss_std=15, brightness_delta=30):
    """複合ノイズ"""
    noisy = add_salt_pepper_noise(image, sp_amount)
    noisy = add_gaussian_noise(noisy, 0, gauss_std)
    noisy = adjust_brightness(noisy, brightness_delta)
    return noisy


# =============================================================================
# Evaluation Functions
# =============================================================================

def prepare_fft_data(image, template):
    """FFTに必要なデータを準備"""
    rows, cols = template.shape
    sigma = 0.3
    x = np.linspace(-1, 1, cols)
    y = np.linspace(-1, 1, rows)
    x, y = np.meshgrid(x, y)
    gaussian_window = np.exp(-(x**2 + y**2) / (2 * sigma**2))

    rows, cols = image.shape
    sigma = 0.5
    x = np.linspace(-1, 1, cols)
    y = np.linspace(-1, 1, rows)
    x, y = np.meshgrid(x, y)
    gaussian_window_for_image = np.exp(-(x**2 + y**2) / (2 * sigma**2))

    radon_template = radonTransform(template)
    radon_template_for_fft = radonTransform((template * gaussian_window).astype(np.uint8))
    
    radon_image = radonTransform(image)
    radon_image_for_fft = radonTransform((image * gaussian_window_for_image).astype(np.uint8))
    
    radon_template_for_fft2 = centerPasteImage(radon_image, radon_template_for_fft)
    
    return radon_image, radon_template, radon_image_for_fft, radon_template_for_fft2


def evaluate_angle_detection(image, template, true_angle, use_contrast_norm=False):
    """
    3つの手法で角度検出精度を評価
    use_contrast_norm: True の場合、コントラスト正規化を適用
    """
    # コントラスト正規化（オプション）
    if use_contrast_norm:
        # 線形正規化（位相保存）を使用
        image_processed = normalize_contrast_to_template(image, template)
    else:
        image_processed = image
    
    # FFTデータ準備
    radon_image, radon_template, radon_image_for_fft, radon_template_for_fft2 = prepare_fft_data(image_processed, template)
    
    results = {}
    
    # 1. 従来手法
    fft_image_conv = radonFFT_conventional(radon_image_for_fft)
    fft_template_conv = radonFFT_conventional(radon_template_for_fft2)
    fft_template_half_conv = fft_template_conv[0:180, :]
    fft_image_combined_conv = np.vstack([fft_image_conv, fft_image_conv])
    
    detected_angle_conv, scores_conv = detect_angle_conventional(fft_image_combined_conv, fft_template_half_conv)
    results['conventional'] = {
        'detected_angle': detected_angle_conv,
        'error': min(abs(detected_angle_conv - true_angle), 
                     abs(detected_angle_conv - true_angle + 180),
                     abs(detected_angle_conv - true_angle - 180)),
        'scores': scores_conv
    }
    
    # 2. 位相相関
    fft_image_complex = radonFFT_complex(radon_image_for_fft)
    fft_template_complex = radonFFT_complex(radon_template_for_fft2)
    fft_template_complex_half = fft_template_complex[0:180, :]
    fft_image_complex_combined = np.vstack([fft_image_complex, fft_image_complex])
    
    detected_angle_poc, scores_poc = detect_angle_phase_correlation(fft_image_complex_combined, fft_template_complex_half)
    results['phase_correlation'] = {
        'detected_angle': detected_angle_poc,
        'error': min(abs(detected_angle_poc - true_angle), 
                     abs(detected_angle_poc - true_angle + 180),
                     abs(detected_angle_poc - true_angle - 180)),
        'scores': scores_poc
    }
    
    # 3. SNR重み付け
    fft_image_snr, _ = radonFFT_with_power(radon_image_for_fft)
    fft_template_snr, power_template = radonFFT_with_power(radon_template_for_fft2)
    fft_template_half_snr = fft_template_snr[0:180, :]
    power_template_half = power_template[0:180, :]
    fft_image_combined_snr = np.vstack([fft_image_snr, fft_image_snr])
    
    detected_angle_snr, scores_snr = detect_angle_snr_weighted(fft_image_combined_snr, fft_template_half_snr, power_template_half)
    results['snr_weighted'] = {
        'detected_angle': detected_angle_snr,
        'error': min(abs(detected_angle_snr - true_angle), 
                     abs(detected_angle_snr - true_angle + 180),
                     abs(detected_angle_snr - true_angle - 180)),
        'scores': scores_snr
    }
    
    # 4. 帯域制限POC（高周波カット）
    detected_angle_bl, scores_bl = detect_angle_poc_bandlimited(fft_image_complex_combined, fft_template_complex_half, cutoff_ratio=0.3)
    results['poc_bandlimited'] = {
        'detected_angle': detected_angle_bl,
        'error': min(abs(detected_angle_bl - true_angle), 
                     abs(detected_angle_bl - true_angle + 180),
                     abs(detected_angle_bl - true_angle - 180)),
        'scores': scores_bl
    }
    
    # 5. 低周波重み付きPOC（ガウシアン重み）
    detected_angle_lp, scores_lp = detect_angle_poc_lowpass_weighted(fft_image_complex_combined, fft_template_complex_half, sigma_ratio=0.2)
    results['poc_lowpass'] = {
        'detected_angle': detected_angle_lp,
        'error': min(abs(detected_angle_lp - true_angle), 
                     abs(detected_angle_lp - true_angle + 180),
                     abs(detected_angle_lp - true_angle - 180)),
        'scores': scores_lp
    }
    
    # 6. 適応的POC（低コントラスト時のみ線形正規化）
    detected_angle_adapt, scores_adapt, used_norm, contrast_ratio = detect_angle_poc_adaptive(
        image, template, fft_image_complex, fft_template_complex,
        radonTransform, radonFFT_complex, centerPasteImage,
        contrast_threshold=0.6
    )
    results['poc_adaptive'] = {
        'detected_angle': detected_angle_adapt,
        'error': min(abs(detected_angle_adapt - true_angle),
                     abs(detected_angle_adapt - true_angle + 180),
                     abs(detected_angle_adapt - true_angle - 180)),
        'scores': scores_adapt,
        'used_normalization': used_norm,
        'contrast_ratio': contrast_ratio
    }

    # 7. Hough voting (adaptive contrast normalization + sinogram core matching)
    img_for_hough = image
    cr = np.std(image.astype(np.float32)) / (np.std(template.astype(np.float32)) + 1e-10)
    if cr < 0.6:
        im_mean = np.mean(image.astype(np.float32))
        im_std = np.std(image.astype(np.float32))
        tm_mean = np.mean(template.astype(np.float32))
        tm_std = np.std(template.astype(np.float32))
        img_for_hough = np.clip(
            (image.astype(np.float32) - im_mean) / (im_std + 1e-10) * tm_std + tm_mean,
            0, 255).astype(np.uint8)

    sino_img_h = radonTransformFloat(img_for_hough)
    sino_tmpl_h = radonTransformFloat(template)
    th, tw = template.shape
    detected_angle_hough, _, _, scores_hough = detectAngleHough(sino_img_h, sino_tmpl_h, th, tw)
    results['ncc_sinogram'] = {
        'detected_angle': detected_angle_hough,
        'error': min(abs(detected_angle_hough - true_angle),
                     abs(detected_angle_hough - true_angle + 180),
                     abs(detected_angle_hough - true_angle - 180)),
        'scores': scores_hough
    }

    return results


def create_test_image(template, target_size, true_angle, true_dx, true_dy):
    """テスト画像を生成（テンプレートを回転・移動して配置）"""
    h, w = target_size
    th, tw = template.shape
    
    # 黒い背景画像
    image = np.zeros((h, w), dtype=np.uint8)
    
    # テンプレートを回転
    center = (tw // 2, th // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, -true_angle, 1.0)
    rotated_template = cv2.warpAffine(template, rotation_matrix, (tw, th))
    
    # 配置位置を計算
    cx = w // 2 + true_dx
    cy = h // 2 + true_dy
    
    x1 = max(0, cx - tw // 2)
    y1 = max(0, cy - th // 2)
    x2 = min(w, cx + tw // 2)
    y2 = min(h, cy + th // 2)
    
    tx1 = max(0, tw // 2 - cx)
    ty1 = max(0, th // 2 - cy)
    tx2 = tx1 + (x2 - x1)
    ty2 = ty1 + (y2 - y1)
    
    if x2 > x1 and y2 > y1:
        image[y1:y2, x1:x2] = rotated_template[ty1:ty2, tx1:tx2]
    
    return image


def run_evaluation(template, noise_configs, true_angle=30, true_dx=20, true_dy=-15, num_trials=5, compare_contrast_norm=False):
    """
    全ノイズ条件で評価を実行
    """
    # テスト画像サイズ（テンプレートより大きく）
    target_size = (template.shape[0] * 2, template.shape[1] * 2)
    
    # クリーン画像を生成
    clean_image = create_test_image(template, target_size, true_angle, true_dx, true_dy)
    
    all_results = {}
    
    for noise_name, noise_func in noise_configs.items():
        print(f"\nEvaluating: {noise_name}")
        
        # 全手法のエラーを記録
        method_errors = {
            'conventional': [],
            'phase_correlation': [],
            'poc_adaptive': [],      # 適応的POC
            'snr_weighted': [],
            'ncc_sinogram': [],      # 改良NCC手法
        }
        normalization_used = []
        contrast_ratios = []
        
        for trial in range(num_trials):
            # ノイズを追加
            if noise_func is None:
                noisy_image = clean_image.copy()
            else:
                noisy_image = noise_func(clean_image)
            
            # 評価
            results = evaluate_angle_detection(noisy_image, template, true_angle, use_contrast_norm=False)
            
            for method in method_errors:
                method_errors[method].append(results[method]['error'])
            
            # 適応的POCの追加情報
            normalization_used.append(results['poc_adaptive']['used_normalization'])
            contrast_ratios.append(results['poc_adaptive']['contrast_ratio'])
        
        # 統計を計算
        all_results[noise_name] = {
            'clean_image': clean_image,
            'noisy_image': noisy_image if noise_func else clean_image,
            'mean_errors': {m: np.mean(e) for m, e in method_errors.items()},
            'std_errors': {m: np.std(e) for m, e in method_errors.items()},
            'all_errors': method_errors,
            'norm_used_ratio': np.mean(normalization_used),
            'mean_contrast_ratio': np.mean(contrast_ratios)
        }
        
        norm_indicator = "✓" if all_results[noise_name]['norm_used_ratio'] > 0.5 else ""
        print(f"  Conventional:        {all_results[noise_name]['mean_errors']['conventional']:.2f} ± {all_results[noise_name]['std_errors']['conventional']:.2f} deg")
        print(f"  POC (full):          {all_results[noise_name]['mean_errors']['phase_correlation']:.2f} ± {all_results[noise_name]['std_errors']['phase_correlation']:.2f} deg")
        print(f"  POC (adaptive):      {all_results[noise_name]['mean_errors']['poc_adaptive']:.2f} ± {all_results[noise_name]['std_errors']['poc_adaptive']:.2f} deg {norm_indicator}")
        print(f"  SNR Weighted:        {all_results[noise_name]['mean_errors']['snr_weighted']:.2f} ± {all_results[noise_name]['std_errors']['snr_weighted']:.2f} deg")
        print(f"  Improved POC:        {all_results[noise_name]['mean_errors']['ncc_sinogram']:.2f} ± {all_results[noise_name]['std_errors']['ncc_sinogram']:.2f} deg")
        if norm_indicator:
            print(f"    → Contrast ratio: {all_results[noise_name]['mean_contrast_ratio']:.3f}, Normalization applied")
    
    return all_results


def plot_results(results, save_path=None):
    """結果を可視化"""
    noise_names = list(results.keys())
    methods = ['conventional', 'phase_correlation', 'poc_adaptive', 'snr_weighted', 'ncc_sinogram']
    method_labels = ['Conventional', 'POC (full)', 'POC (adaptive)', 'SNR Weighted', 'Improved POC']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

    # エラーバープロット
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    
    # 左: 平均誤差の比較
    ax1 = axes[0]
    x = np.arange(len(noise_names))
    width = 0.2
    
    for i, (method, label, color) in enumerate(zip(methods, method_labels, colors)):
        means = [results[n]['mean_errors'][method] for n in noise_names]
        stds = [results[n]['std_errors'][method] for n in noise_names]
        ax1.bar(x + i * width, means, width, label=label, color=color, yerr=stds, capsize=2)
    
    ax1.set_xlabel('Noise Type')
    ax1.set_ylabel('Angle Detection Error (degrees)')
    ax1.set_title('Angle Detection Error Comparison')
    ax1.set_xticks(x + width * 1.5)
    ax1.set_xticklabels(noise_names, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    
    # 右: ノイズ画像サンプル
    ax2 = axes[1]
    num_samples = min(4, len(noise_names))
    for i, noise_name in enumerate(noise_names[:num_samples]):
        ax_sub = fig.add_axes([0.55 + (i % 2) * 0.22, 0.55 - (i // 2) * 0.45, 0.18, 0.35])
        ax_sub.imshow(results[noise_name]['noisy_image'], cmap='gray')
        ax_sub.set_title(noise_name, fontsize=9)
        ax_sub.axis('off')
    
    axes[1].axis('off')
    axes[1].set_title('Sample Noisy Images')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\nPlot saved to: {save_path}")
    
    plt.show()


def plot_detailed_comparison(results, noise_type, save_path=None):
    """特定のノイズタイプに対する詳細比較"""
    fig, axes = plt.subplots(2, 5, figsize=(22, 8))
    
    # 上段: 画像表示
    axes[0, 0].imshow(results[noise_type]['clean_image'], cmap='gray')
    axes[0, 0].set_title('Clean Image')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(results[noise_type]['noisy_image'], cmap='gray')
    axes[0, 1].set_title(f'Noisy Image ({noise_type})')
    axes[0, 1].axis('off')
    
    # 各手法の誤差分布
    methods = ['conventional', 'phase_correlation', 'poc_adaptive', 'snr_weighted', 'ncc_sinogram']
    method_labels = ['Conventional', 'POC (full)', 'POC (adaptive)', 'SNR Weight', 'Improved POC']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    all_errors = results[noise_type]['all_errors']
    
    ax_box = axes[0, 2]
    bp = ax_box.boxplot([all_errors[m] for m in methods], labels=method_labels, patch_artist=True)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    ax_box.set_ylabel('Error (degrees)')
    ax_box.set_title('Error Distribution')
    ax_box.tick_params(axis='x', rotation=45)
    ax_box.grid(axis='y', alpha=0.3)
    
    # 上段右: サマリー
    axes[0, 3].axis('off')
    summary_text = f"Noise Type: {noise_type}\n\n"
    for method, label in zip(methods, method_labels):
        mean_err = results[noise_type]['mean_errors'][method]
        std_err = results[noise_type]['std_errors'][method]
        summary_text += f"{label}: {mean_err:.2f}° ± {std_err:.2f}°\n"
    axes[0, 3].text(0.1, 0.5, summary_text, fontsize=9, verticalalignment='center',
                    family='monospace', transform=axes[0, 3].transAxes)
    axes[0, 3].set_title('Summary')

    axes[0, 4].axis('off')

    # 下段: 各ノイズタイプでの比較
    noise_names = list(results.keys())
    
    for i, (method, label, color) in enumerate(zip(methods, method_labels, colors)):
        ax = axes[1, i]
        means = [results[n]['mean_errors'][method] for n in noise_names]
        stds = [results[n]['std_errors'][method] for n in noise_names]
        
        x = np.arange(len(noise_names))
        ax.bar(x, means, color=color, alpha=0.7, yerr=stds, capsize=3)
        ax.set_xlabel('Noise Type')
        ax.set_ylabel('Error (degrees)')
        ax.set_title(label)
        ax.set_xticks(x)
        ax.set_xticklabels(noise_names, rotation=45, ha='right', fontsize=7)
        ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Radon Template Matching - Noise Robustness Evaluation")
    print("=" * 60)
    
    # テンプレート画像を読み込み
    template_path = "figs/template.jpg"
    template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
    
    if template is None:
        print(f"Error: Could not load template from {template_path}")
        print("Creating synthetic template...")
        # 合成テンプレートを生成
        template = np.zeros((100, 100), dtype=np.uint8)
        cv2.rectangle(template, (10, 20), (90, 80), 200, -1)
        cv2.circle(template, (50, 50), 20, 100, -1)
    
    print(f"Template size: {template.shape}")
    
    # リサイズ（計算時間短縮のため）
    max_size = 128
    if max(template.shape) > max_size:
        scale = max_size / max(template.shape)
        template = cv2.resize(template, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        print(f"Resized template to: {template.shape}")
    
    # ノイズ設定
    noise_configs = {
        'Clean': None,
        'Salt&Pepper 5%': lambda img: add_salt_pepper_noise(img, 0.05),
        'Salt&Pepper 10%': lambda img: add_salt_pepper_noise(img, 0.10),
        'Gaussian σ=25': lambda img: add_gaussian_noise(img, 0, 25),
        'Gaussian σ=50': lambda img: add_gaussian_noise(img, 0, 50),
        'Brightness +50': lambda img: adjust_brightness(img, 50),
        'Brightness -50': lambda img: adjust_brightness(img, -50),
        'Contrast 0.5x': lambda img: adjust_contrast(img, 0.5),
        'Contrast 0.3x': lambda img: adjust_contrast(img, 0.3),
        'Combined': lambda img: add_combined_noise(img, 0.03, 20, 25),
    }
    
    # 評価実行（適応的POCを評価）
    print("\n" + "=" * 80)
    print("Running evaluation: POC (full) vs POC (adaptive with contrast normalization)")
    print("=" * 80)
    
    results = run_evaluation(
        template, 
        noise_configs, 
        true_angle=35,  # 真の回転角度
        true_dx=15,     # 真のx方向移動
        true_dy=-10,    # 真のy方向移動
        num_trials=5,   # 試行回数
    )
    
    # 結果をプロット
    print("\n" + "=" * 80)
    print("Generating plots...")
    print("=" * 80)
    
    plot_results(results, save_path="figs/noise_robustness_comparison.png")
    plot_detailed_comparison(results, 'Contrast 0.5x', save_path="figs/detailed_comparison.png")
    
    # サマリーテーブル（4手法比較）
    print("\n" + "=" * 100)
    print("Summary Table: POC (full) vs POC (adaptive)")
    print("=" * 100)
    print(f"{'Noise Type':<18} {'Conventional':>12} {'POC(full)':>12} {'POC(adaptive)':>14} {'SNR Weight':>12} {'Improved POC':>14} {'Norm?':>6}")
    print("-" * 115)
    for noise_name in noise_configs:
        conv = results[noise_name]['mean_errors']['conventional']
        poc = results[noise_name]['mean_errors']['phase_correlation']
        poc_adapt = results[noise_name]['mean_errors']['poc_adaptive']
        snr = results[noise_name]['mean_errors']['snr_weighted']
        ncc = results[noise_name]['mean_errors']['ncc_sinogram']
        norm_used = "✓" if results[noise_name].get('norm_used_ratio', 0) > 0.5 else ""
        print(f"{noise_name:<18} {conv:>10.2f}° {poc:>10.2f}° {poc_adapt:>12.2f}° {snr:>10.2f}° {ncc:>12.2f}° {norm_used:>6}")

    # 平均性能
    print("-" * 115)
    avg_conv = np.mean([results[n]['mean_errors']['conventional'] for n in noise_configs])
    avg_poc = np.mean([results[n]['mean_errors']['phase_correlation'] for n in noise_configs])
    avg_poc_adapt = np.mean([results[n]['mean_errors']['poc_adaptive'] for n in noise_configs])
    avg_snr = np.mean([results[n]['mean_errors']['snr_weighted'] for n in noise_configs])
    avg_ncc = np.mean([results[n]['mean_errors']['ncc_sinogram'] for n in noise_configs])
    print(f"{'Average':<18} {avg_conv:>10.2f}° {avg_poc:>10.2f}° {avg_poc_adapt:>12.2f}° {avg_snr:>10.2f}° {avg_ncc:>12.2f}°")
    
    # POC改善率
    print("\n" + "=" * 100)
    print("POC (full) vs POC (adaptive) - Improvement Analysis")
    print("=" * 100)
    print(f"{'Noise Type':<18} {'POC(full)':>10} {'POC(adapt)':>12} {'Change':>10} {'Best':>15}")
    print("-" * 100)
    for noise_name in noise_configs:
        poc = results[noise_name]['mean_errors']['phase_correlation']
        poc_adapt = results[noise_name]['mean_errors']['poc_adaptive']
        improvement = poc - poc_adapt
        arrow = "↑" if improvement > 0 else ("↓" if improvement < 0 else "=")
        best = "POC(adaptive)" if poc_adapt <= poc else "POC(full)"
        best_val = min(poc, poc_adapt)
        print(f"{noise_name:<18} {poc:>8.2f}° {poc_adapt:>10.2f}° {arrow} {abs(improvement):>6.2f}°   {best} ({best_val:.2f}°)")
    
    print("\n" + "=" * 100)
    print("Evaluation complete!")
    print("=" * 100)
