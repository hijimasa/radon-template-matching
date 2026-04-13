import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
from numba import jit

"""
Radon transform related functions
ラドン変換関連関数
"""
@jit('int32[:,:](uint8[:,:],int32,int32)', nopython=True, cache=True)
def getLinePoints(image, angle, length):
    """
    Obtain a point to draw a straight line at a specified angle from the image center
    画像中心から指定された角度で直線を描くためのポイントを取得
    :param image: Input image / 入力画像
    :param angle: Angle in degrees / 角度（度単位）
    :param length: Length of straight line / 直線の長さ
    :return: List of points on a straight line / 直線上のポイントのリスト
    """
    h, w = image.shape
    center = (w // 2, h // 2)

    angle_rad = np.deg2rad(angle)

    # 直線の始点を計算
    start_x = int(center[0] - length / 2 * np.cos(angle_rad))
    start_y = int(center[1] + length / 2 * np.sin(angle_rad))  # 画像座標系はy軸が下向き

    # 直線の終点を計算
    end_x = int(center[0] + length / 2 * np.cos(angle_rad))
    end_y = int(center[1] - length / 2 * np.sin(angle_rad))  # 画像座標系はy軸が下向き

    # 直線上のポイントを計算
    line_points = np.empty((length, 2), dtype=np.int32)
    for i in range(length):
        t = i / (length - 1)  # Coefficients for linear interpolation / 線形補間のための係数
        x = int(start_x * (1 - t) + end_x * t)
        y = int(start_y * (1 - t) + end_y * t)
        line_points[i] = (x, y)

    return line_points

@jit('uint64[:](uint8[:,:],int32[:,:],int32)', nopython=True, cache=True)
def perpendicularSum(image, points, width):
    """
    At each point on the specified line, calculate the sum of pixel values perpendicular to the center of that point
    指定された直線の各点において、その点を中心に垂直な方向の画素値の合計を計算
    :param image: Input image / 入力画像
    :param points: List of points on a straight line / 直線上のポイントのリスト
    :param width: Vertical width / 垂直方向に取る幅
    :return: Total list of pixel values / 画素値の合計リスト
    """
    sums = []
    h, w = image.shape

    top_row_mean = np.mean(image[0,:])
    bottom_row_mean = np.mean(image[-1,:])
    left_column_mean = np.mean(image[:,0])
    right_column_mean = np.mean(image[:,-1])
    corner_pixels_mean = np.uint64((top_row_mean +  bottom_row_mean + left_column_mean + right_column_mean) / 4.0)

    perp_vec = np.array([0.0, 0.0], dtype=np.float32)
    for point in points:
        x, y = point

        # 垂直方向のベクトル（直線の方向の90度回転）
        perp_vec = np.array([-(y - h//2), x - w//2], dtype=np.float32)
        norm = math.sqrt(perp_vec[0]*perp_vec[0]+perp_vec[1]*perp_vec[1])

        # ノルムがゼロの場合はスキップ
        if not norm == 0:
            perp_vec = perp_vec / norm  # 正規化
            break

    for point in points:
        x, y = point
        sum_value = np.uint64(0)

        for i in range(-width//2, width//2):
            perp_x = int(x + perp_vec[0] * i)
            perp_y = int(y + perp_vec[1] * i)

            # 画像範囲内かチェック
            if 0 <= perp_x < w and 0 <= perp_y < h:
                sum_value += image[perp_y, perp_x]
            else:
                sum_value += corner_pixels_mean

        # Pythonの標準sum関数を使用
        sums.append(sum_value)

    sums = np.array(sums, dtype=np.uint64)
    return sums

@jit('uint8[:,:](uint8[:,:])', nopython=True, cache=True)
def radonTransform(image):
    """
    Transform the input image into a Radon transformed sinograph with the vertical and horizontal axes corresponding to the angle and the sum of projected pixel values, respectively
    入力画像を縦軸は角度、横軸は投影された画素値の合計にそれぞれ対応させたラドン変換シノグラフに変換する
    :param image: Input image / 入力画像
    :return: Radon transformed sinograph / ラドン変換シノグラフ
    """
    h, w = image.shape

    # 直線の長さと垂直方向の幅を設定
    line_length = int(math.sqrt(h*h+w*w))  # 直線の長さ
    perpendicular_width = int(math.sqrt(h*h+w*w))  # 垂直方向の幅

    # 0度から360度までの垂直方向の画素値の合計を計算
    angle_sums = []
    max_value = np.uint64(0)

    for angle in range(360):
        line_points = getLinePoints(image, np.int32(angle), np.int32(line_length))
        sums = perpendicularSum(image, line_points, perpendicular_width)
        angle_sums.append(sums)
        if max_value < max(sums):
            max_value = max(sums)

    # 0度から360度までの角度ごとの合計画素値を正規化
    normalized_image = np.zeros((360, line_length), dtype=np.uint32)

    # 正規化して2D画像に変換
    for i, sums in enumerate(angle_sums):
        for j, value in enumerate(sums):
            normalized_image[i, j] = (value / max_value) * 255  # 最大値に基づいて255で正規化

    return normalized_image.astype(np.uint8)

def radonFFT(image):
    """
    FFT the results at each angle of the radon transform sinograph and return the results
    ラドン変換シノグラフの各角度における結果をFFTして結果を返す
    :param image: Input image (Radon transformed sinograph) / 入力画像（ラドン変換シノグラフ）
    :return: Output image (Vertical axis: Angle, Horizontal axis: Frequency) / 出力画像（縦軸：角度、横軸：周波数）
    """
    # 各行に対して1次元FFTを実行し、その振幅スペクトルを生成
    fft_result = []
    for row in image:
        fft_row = np.fft.fft(row)  # 1次元FFT
        fft_magnitude = np.abs(fft_row)  # 振幅スペクトル（複素数の絶対値）
        fft_result.append(fft_magnitude)

    # FFT結果を2次元配列に変換
    fft_image = np.array(fft_result, dtype=np.float32)

    # 振幅スペクトルを可視化するために正規化
    fft_image_normalized = np.log(fft_image + 1)  # 値の範囲を小さくするために対数変換

    return fft_image_normalized


"""
Auxiliary function
補助関数
"""
def centerPasteImage(wide_img, narrow_img):
    """
    Generate an image that is the same width as the wider image with a narrower image pasted in the center
    広い幅の画像と同じ幅で中心に狭い幅の画像を貼り付けた画像を生成する
    :param wide_img: Wide image / 幅の広い画像
    :param narrow_img: Narrow image / 幅の狭い画像
    :return: Output image / 出力画像
    """
    # 画像サイズを取得 (高さ, 幅)
    wide_h, wide_w = wide_img.shape
    narrow_h, narrow_w = narrow_img.shape

    # 幅が広い方の画像と同じサイズの黒い画像を生成
    black_background = np.zeros((wide_h, wide_w), dtype=np.uint8)

    # 幅の狭い画像を中心に配置するための座標計算
    x_offset = (wide_w - narrow_w) // 2
    y_offset = (wide_h - narrow_h) // 2

    # 黒い画像に幅の狭い画像を貼り付け
    black_background[y_offset:y_offset + narrow_h, x_offset:x_offset + narrow_w] = narrow_img

    return black_background

@jit('float32[:](float32[:], float32[:])', nopython=True, cache=True)
def matchTemplateOneLine(image_line, template_line):
    """
    Matching images one row at a time given
    与えられた１行づつの画像をマッチングする
    :param image: One line of the target image / 対象画像の１行
    :return: One line of the template image / テンプレート画像の１行
    """
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

"""
Drawing funcitions
描画関数
"""
def drawRotatedRectangleOnImage(image, center, width, height, angle, color=(255, 0, 0), thickness=8):
    """
    Draw a rectangle rotated by a given angle
    所定の角度だけ回転させた四角形を描画する
    :param image: Input image / 入力画像
    :param center: Center position of the rectangle to be drawn / 描画する四角形の中心位置
    :param width: Width of the rectangle to be drawn / 描画する四角形の幅
    :param height: Height of the rectangle to be drawn / 描画する四角形の高さ
    :param angle: Amount of rotation (in degrees) of the rectangle to be drawn / 描画する四角形の回転量(度単位)
    :param color: Color / 描画する色
    :param thickness: Thickness / 描画する幅
    :return: Output image / 出力画像
    """
    cx, cy = center

    rectangle = np.array([
        [-width / 2, -height / 2],
        [width / 2, -height / 2],
        [width / 2, height / 2],
        [-width / 2, height / 2]
    ])

    theta = np.radians(angle)
    rotation_matrix = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta), np.cos(theta)]
    ])

    rotated_rectangle = np.dot(rectangle, rotation_matrix) + np.array([cx, cy])
    rotated_rectangle = rotated_rectangle.astype(int)

    # 頂点をOpenCVの形式に変換
    points = rotated_rectangle.reshape((-1, 1, 2))

    cv2.polylines(image, [points], isClosed=True, color=color, thickness=thickness)

    return image

"""
Improved matching functions
改良されたマッチング関数
"""
def radonTransformFloat(image):
    """
    Radon transform with float32 precision (no uint8 quantization loss).
    uint8量子化損失のないfloat32精度のラドン変換
    :param image: Input image / 入力画像
    :return: Float32 sinogram (360 x line_length) / float32サイノグラム
    """
    h, w = image.shape
    line_length = int(math.sqrt(h * h + w * w))
    perpendicular_width = int(math.sqrt(h * h + w * w))

    sinogram = np.zeros((360, line_length), dtype=np.float32)

    for angle in range(360):
        line_points = getLinePoints(image, np.int32(angle), np.int32(line_length))
        sums = perpendicularSum(image, line_points, perpendicular_width)
        sinogram[angle, :] = sums.astype(np.float32)

    return sinogram


def matchTemplateOneLineNCC(image_line, template_line):
    """
    1D normalized cross-correlation using OpenCV matchTemplate.
    OpenCV matchTemplateを用いた1次元正規化相互相関
    :param image_line: 1D array of target sinogram row / 対象サイノグラムの1行
    :param template_line: 1D array of template sinogram row / テンプレートサイノグラムの1行
    :return: NCC result array / NCC結果配列
    """
    if len(template_line) >= len(image_line):
        return np.array([0.0], dtype=np.float32)
    img = image_line.astype(np.float32).reshape(1, -1)
    tmpl = template_line.astype(np.float32).reshape(1, -1)
    result = cv2.matchTemplate(img, tmpl, cv2.TM_CCOEFF_NORMED)
    return result[0]


def detectAnglePOC(sinogram_image, sinogram_template):
    """
    Detect rotation angle using Phase-Only Correlation (POC) in sinogram domain.
    サイノグラム領域での位相限定相関法(POC)による回転角度検出

    Key improvements over the original FFT magnitude difference approach:
    1. Float32 sinogram (no uint8 quantization loss)
    2. Center-padding for proper frequency alignment
    3. Phase-based matching is sensitive to shape details (not just energy)

    Input sinograms should be computed from Gaussian-windowed images
    to reduce edge artifacts (critical for POC phase accuracy).

    :param sinogram_image: Float sinogram of windowed target image (360 x L_img)
    :param sinogram_template: Float sinogram of windowed template (360 x L_tmpl)
    :return: (detected_angle, scores_array)
    """
    n_img = sinogram_image.shape[1]
    n_tmpl = sinogram_template.shape[1]

    # Center-pad template sinogram to match image sinogram width
    pad_left = (n_img - n_tmpl) // 2
    sinogram_tmpl_padded = np.zeros((360, n_img), dtype=np.float32)
    sinogram_tmpl_padded[:, pad_left:pad_left + n_tmpl] = sinogram_template

    # Compute complex FFTs
    fft_image = np.zeros((360, n_img), dtype=np.complex128)
    fft_template = np.zeros((360, n_img), dtype=np.complex128)
    for i in range(360):
        fft_image[i] = np.fft.fft(sinogram_image[i].astype(np.float64))
        fft_template[i] = np.fft.fft(sinogram_tmpl_padded[i].astype(np.float64))

    # POC angle detection
    scores = np.zeros(180, dtype=np.float64)
    for alpha in range(180):
        corr_sum = 0.0
        for theta in range(180):
            row_img = fft_image[(theta + alpha) % 360]
            row_tmpl = fft_template[theta]

            # Cross-power spectrum (phase only)
            cross_power = row_img * np.conj(row_tmpl)
            cross_power_normalized = cross_power / (np.abs(cross_power) + 1e-10)

            # IFFT peak = correlation strength
            correlation = np.abs(np.fft.ifft(cross_power_normalized))
            corr_sum += np.max(correlation)

        scores[alpha] = corr_sum
    return int(np.argmax(scores)), scores


def detectAngleNCC(sinogram_image, sinogram_template):
    """
    Detect rotation angle using NCC sum in sinogram domain.
    サイノグラム行ごとのNCCスコア合計による回転角度検出（補助手法）

    :param sinogram_image: Float sinogram of target image (360 x L_img)
    :param sinogram_template: Float sinogram of template (360 x L_tmpl)
    :return: (detected_angle, scores_array)
    """
    scores = np.zeros(180, dtype=np.float64)
    for alpha in range(180):
        total = 0.0
        for theta in range(180):
            img_row = sinogram_image[(theta + alpha) % 360]
            tmpl_row = sinogram_template[theta]
            ncc = matchTemplateOneLineNCC(img_row, tmpl_row)
            if len(ncc) > 0:
                total += np.max(ncc)
        scores[alpha] = total
    return int(np.argmax(scores)), scores


def detectAngleSinusoidalWarp(sinogram_image, sinogram_template,
                              dx_range=None, dy_range=None, dx_step=2, dy_step=2,
                              match_method=cv2.TM_CCORR_NORMED):
    """
    Detect rotation angle and position using sinusoidal warp matching.
    サイノグラム正弦波ワープマッチングによる回転角度・位置の同時推定

    Principle: A point at (dx, dy) in the image traces a sinusoid
    offset(theta) = dx*cos(theta) + dy*sin(theta) in the sinogram.
    By warping the template sinogram with this sinusoidal offset and
    measuring the match quality, we jointly estimate (alpha, dx, dy).

    All sinogram rows are constrained to follow the same sinusoidal pattern,
    so only the correct (alpha, dx, dy) produces consistently high NCC across
    all angles.

    :param sinogram_image: Float sinogram of target image (360 x L_img)
    :param sinogram_template: Float sinogram of template (360 x L_tmpl)
    :param dx_range: (min, max) range for dx search. Default: auto from sinogram sizes
    :param dy_range: (min, max) range for dy search. Default: same as dx_range
    :param dx_step: Step size for dx search
    :param dy_step: Step size for dy search
    :param match_method: OpenCV match method (default: TM_CCORR_NORMED).
                         TM_CCORR_NORMED recommended: no mean subtraction, robust to zero background.
    :return: (detected_angle, dx, dy, best_score, scores_per_alpha)
    """
    n_img = sinogram_image.shape[1]
    n_tmpl = sinogram_template.shape[1]
    n_ncc = n_img - n_tmpl + 1  # NCC result length

    if n_ncc <= 1:
        return 0, 0, 0, 0.0, np.zeros(180)

    # Auto-determine search range from sinogram dimensions
    max_offset = n_ncc // 2
    if dx_range is None:
        dx_range = (-max_offset, max_offset)
    if dy_range is None:
        dy_range = (-max_offset, max_offset)

    # Precompute NCC for all (image_row, template_row) pairs
    # ncc_all[i, j] = NCC(sinogram_image[i], sinogram_template[j])
    # NCC result length = n_img - n_tmpl + 1
    n_ncc = n_img - n_tmpl + 1
    if n_ncc <= 0:
        return 0, 0, 0, 0.0, np.zeros(180)

    # For SSD methods (TM_SQDIFF*), lower = better, so negate for consistent argmax
    invert = match_method in (cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED)

    ncc_all = np.zeros((360, 360, n_ncc), dtype=np.float32)
    for i in range(360):
        img_row = sinogram_image[i].astype(np.float32).reshape(1, -1)
        for j in range(360):
            tmpl_row = sinogram_template[j].astype(np.float32).reshape(1, -1)
            result = cv2.matchTemplate(img_row, tmpl_row, match_method)
            ncc_all[i, j, :] = -result[0] if invert else result[0]

    # Precompute cos/sin for theta = 0..179
    thetas = np.arange(180)
    cos_theta = np.cos(np.deg2rad(thetas))  # (180,)
    sin_theta = np.sin(np.deg2rad(thetas))  # (180,)

    # Center offset: when dx=dy=0, template is at center of image sinogram
    center_offset = n_ncc // 2

    # Search over (alpha, dx, dy)
    dx_vals = np.arange(dx_range[0], dx_range[1] + 1, dx_step)
    dy_vals = np.arange(dy_range[0], dy_range[1] + 1, dy_step)
    # Precompute offset matrix: offsets[dx_idx, dy_idx, theta]
    offsets = (dx_vals[:, None, None] * cos_theta[None, None, :] +
               dy_vals[None, :, None] * sin_theta[None, None, :])
    offsets = np.round(offsets).astype(np.int32) + center_offset
    offsets = np.clip(offsets, 0, n_ncc - 1)  # (n_dx, n_dy, 180)

    scores_per_alpha = np.zeros(180, dtype=np.float64)
    best_score = -np.inf
    best_alpha = 0
    best_dx_idx = 0
    best_dy_idx = 0

    for alpha in range(180):
        # Build NCC matrix for this alpha: ncc_matrix[theta, pos]
        # image row = theta, template row = (theta - alpha) % 360
        tmpl_indices = (thetas - alpha) % 360
        ncc_matrix = ncc_all[thetas, tmpl_indices, :]  # (180, n_ncc)

        # Vectorized lookup: for all (dx, dy), gather NCC at sinusoidal offsets
        # gathered[dx_idx, dy_idx, theta] = ncc_matrix[theta, offsets[dx_idx, dy_idx, theta]]
        gathered = ncc_matrix[thetas[None, None, :], offsets]  # (n_dx, n_dy, 180)
        total_scores = gathered.sum(axis=2)  # (n_dx, n_dy)

        alpha_best = total_scores.max()
        scores_per_alpha[alpha] = alpha_best

        if alpha_best > best_score:
            best_score = alpha_best
            best_alpha = alpha
            idx = np.unravel_index(total_scores.argmax(), total_scores.shape)
            best_dx_idx, best_dy_idx = idx

    # Resolve 180-degree ambiguity: also check alpha + 180
    # (done by the caller or by checking the score landscape)
    detected_angle = best_alpha
    detected_dx = int(dx_vals[best_dx_idx])
    detected_dy = int(dy_vals[best_dy_idx])

    return detected_angle, detected_dx, detected_dy, float(best_score), scores_per_alpha


def extractSinogramCore(sinogram, template_height, template_width):
    """
    Extract the core (content-bearing) region of each sinogram row.
    各サイノグラム行からテンプレート本体に対応する中央領域を抽出する

    The full sinogram row has length = diagonal of the image.
    The actual template projection occupies only the central portion,
    the rest is edge padding (corner_pixels_mean fill).

    :param sinogram: Sinogram array (360 x L)
    :param template_height: Template image height
    :param template_width: Template image width
    :return: List of 360 core arrays (varying length per angle for accuracy,
             or fixed length for simplicity)
    """
    row_len = sinogram.shape[1]
    center = row_len // 2
    cores = []
    for angle in range(360):
        theta_rad = np.deg2rad(angle)
        # Projection width depends on angle:
        # width(θ) = |w·cos(θ)| + |h·sin(θ)|
        proj_width = int(abs(template_width * np.cos(theta_rad)) +
                         abs(template_height * np.sin(theta_rad)))
        proj_width = max(proj_width, 4)  # minimum size
        half = proj_width // 2
        start = max(0, center - half)
        end = min(row_len, center + half)
        cores.append(sinogram[angle, start:end].copy())
    return cores


def detectAngleHough(sinogram_image, sinogram_template,
                     template_height, template_width, score_threshold=0.5):
    """
    Detect rotation angle using Hough-like voting on sinogram core matching.
    サイノグラムコアマッチングのハフ投票による回転角度検出

    Algorithm:
    1. Extract core (content-only) region from each template sinogram row
    2. For each (image_row, template_core) pair, find best 1D match
    3. Each strong match votes for rotation α = (θ_image - θ_template) % 180
    4. Accumulator peak = detected rotation
    5. For the best α, fit sinusoidal position model to get (dx, dy)

    The core extraction removes edge padding artifacts that corrupt
    correlation scores. The Hough voting naturally handles the
    angle-position coupling without exhaustive 3D search.

    :param sinogram_image: Float sinogram of target image (360 x L_img)
    :param sinogram_template: Float sinogram of template (360 x L_tmpl)
    :param template_height: Template image height (for core extraction)
    :param template_width: Template image width (for core extraction)
    :param score_threshold: Minimum NCC score for a vote to count
    :return: (detected_angle, dx, dy, accumulator)
    """
    # Step 1: Extract template sinogram cores
    cores = extractSinogramCore(sinogram_template, template_height, template_width)

    n_img = sinogram_image.shape[1]

    # Step 2: Precompute 1D matching for all (image_row, template_core) pairs
    # Store best match position and score for each pair
    # Only compute for θ_tmpl in [0, 180) and θ_img in [0, 360)
    # because sinogram has 180° symmetry in template
    match_scores = np.zeros((360, 180), dtype=np.float32)
    match_positions = np.zeros((360, 180), dtype=np.int32)

    for j in range(180):  # template angles
        core = cores[j]
        if len(core) < 4 or len(core) >= n_img:
            continue
        tmpl_2d = core.astype(np.float32).reshape(1, -1)

        for i in range(360):  # image angles
            img_2d = sinogram_image[i].astype(np.float32).reshape(1, -1)
            result = cv2.matchTemplate(img_2d, tmpl_2d, cv2.TM_CCOEFF_NORMED)
            ncc = result[0]
            best_pos = int(np.argmax(ncc))
            best_score = float(ncc[best_pos])
            match_scores[i, j] = best_score
            match_positions[i, j] = best_pos

    # Step 3: Hough voting for rotation angle
    accumulator = np.zeros(180, dtype=np.float64)

    for j in range(180):  # template angle
        for i in range(360):  # image angle
            score = match_scores[i, j]
            if score < score_threshold:
                continue
            # This match implies rotation α = (θ_i - θ_j) % 180
            alpha = (i - j) % 180
            accumulator[alpha] += score

    detected_alpha = int(np.argmax(accumulator))

    # Step 4: For the detected α, collect match positions and fit (dx, dy)
    cos_theta = np.cos(np.deg2rad(np.arange(360)))
    sin_theta = np.sin(np.deg2rad(np.arange(360)))

    offsets = []
    weights = []
    A_rows = []

    for theta_img in range(360):
        theta_tmpl = (theta_img - detected_alpha) % 360
        if theta_tmpl >= 180:
            continue  # only use primary half
        score = match_scores[theta_img, theta_tmpl]
        if score < score_threshold:
            continue

        pos = match_positions[theta_img, theta_tmpl]
        core_len = len(cores[theta_tmpl])
        # Convert NCC position to offset from image center
        offset = pos + core_len // 2 - n_img // 2

        offsets.append(offset)
        weights.append(max(score, 0.0))
        A_rows.append([cos_theta[theta_img], sin_theta[theta_img]])

    # Fit: offset = dx*cos(θ) + dy*sin(θ)
    dx, dy = 0.0, 0.0
    if len(offsets) >= 2:
        offsets_arr = np.array(offsets, dtype=np.float64)
        weights_arr = np.array(weights, dtype=np.float64)
        A = np.array(A_rows, dtype=np.float64)
        W = np.diag(weights_arr)
        try:
            AtWA = A.T @ W @ A
            AtWb = A.T @ W @ offsets_arr
            params = np.linalg.solve(AtWA, AtWb)
            dx, dy = params[0], params[1]
        except np.linalg.LinAlgError:
            pass

    return detected_alpha, int(round(dx)), int(round(dy)), accumulator


def detectPosition(sinogram_image, sinogram_template, detect_angle, num_angles=36):
    """
    Estimate template position using multi-angle weighted least squares.
    複数角度の重み付き最小二乗法でテンプレート位置を推定する

    At each angle theta, the sinogram offset of the template is:
        offset(theta) = dx * cos(theta) - dy * sin(theta)
    We measure this offset via 1D NCC at multiple angles and solve for (dx, dy)
    using weighted least squares (weights = NCC scores).

    :param sinogram_image: Float sinogram of target image / 対象画像のfloatサイノグラム
    :param sinogram_template: Float sinogram of template / テンプレートのfloatサイノグラム
    :param detect_angle: Detected rotation angle / 検出された回転角度
    :param num_angles: Number of angles for position estimation / 位置推定に使用する角度数
    :return: (dx, dy, total_score) / (dx, dy, 総合スコア)
    """
    angles_deg = np.linspace(0, 180, num_angles, endpoint=False)

    offsets = []
    weights = []
    A_rows = []

    for theta_deg in angles_deg:
        theta = int(theta_deg) % 360
        # Template angle corresponding to image angle theta
        tmpl_theta = (360 - detect_angle + theta) % 360

        img_row = sinogram_image[theta]
        tmpl_row = sinogram_template[tmpl_theta]
        ncc = matchTemplateOneLineNCC(img_row, tmpl_row)
        if len(ncc) == 0:
            continue

        best_pos = int(np.argmax(ncc))
        best_score = max(float(np.max(ncc)), 0.0)
        offset = best_pos + len(tmpl_row) // 2 - len(img_row) // 2

        offsets.append(offset)
        weights.append(best_score)
        theta_rad = np.deg2rad(theta_deg)
        A_rows.append([np.cos(theta_rad), -np.sin(theta_rad)])

    offsets = np.array(offsets, dtype=np.float64)
    weights = np.array(weights, dtype=np.float64)
    A = np.array(A_rows, dtype=np.float64)
    W = np.diag(weights)

    try:
        AtWA = A.T @ W @ A
        AtWb = A.T @ W @ offsets
        params = np.linalg.solve(AtWA, AtWb)
        dx, dy = params[0], params[1]
    except np.linalg.LinAlgError:
        dx, dy = 0.0, 0.0

    total_score = float(np.sum(weights))
    return dx, dy, total_score


"""
Template Matching Processing Body
テンプレートマッチング処理本体
"""
def matchTemplateRotatable(image, template):
    """
    Rotation-invariant template matching using Radon transform + POC.
    ラドン変換とPOCを用いた回転不変テンプレートマッチング

    Algorithm:
    1. Adaptive contrast normalization (if contrast ratio < 0.6)
    2. Gaussian windowing + uint8 Radon transform (uint8 quantization provides noise robustness)
    3. POC-based angle detection (replaces original FFT magnitude difference)
    4. Multi-angle NCC least squares position estimation

    Limitations:
    - Template should cover a significant portion (>40%) of the target image area
    - For small templates in large scenes, use a pre-localization step first

    :param image: Target image / 対象画像
    :param template: Template image / テンプレート画像
    :return: (target_angle, dx, dy) / (検出角度, dx, dy)
    """
    # Step 0: Adaptive contrast normalization
    contrast_ratio = np.std(image.astype(np.float32)) / (np.std(template.astype(np.float32)) + 1e-10)
    if contrast_ratio < 0.6:
        img_mean = np.mean(image.astype(np.float32))
        img_std = np.std(image.astype(np.float32))
        tmpl_mean = np.mean(template.astype(np.float32))
        tmpl_std = np.std(template.astype(np.float32))
        image_proc = (image.astype(np.float32) - img_mean) / (img_std + 1e-10) * tmpl_std + tmpl_mean
        image_proc = np.clip(image_proc, 0, 255).astype(np.uint8)
    else:
        image_proc = image

    # Step 1: Gaussian windows for edge tapering (critical for POC phase accuracy)
    rows, cols = template.shape
    x = np.linspace(-1, 1, cols)
    y = np.linspace(-1, 1, rows)
    x, y = np.meshgrid(x, y)
    gaussian_template = np.exp(-(x**2 + y**2) / (2 * 0.3**2))

    rows, cols = image_proc.shape
    x = np.linspace(-1, 1, cols)
    y = np.linspace(-1, 1, rows)
    x, y = np.meshgrid(x, y)
    gaussian_image = np.exp(-(x**2 + y**2) / (2 * 0.5**2))

    # Step 2: Radon transforms
    # - Windowed + uint8 for POC angle detection (uint8 provides noise robustness)
    # - Raw float for NCC position estimation (full precision for sub-pixel matching)
    radon_image_for_poc = radonTransform((image_proc * gaussian_image).astype(np.uint8))
    radon_template_for_poc = radonTransform((template * gaussian_template).astype(np.uint8))
    radon_template_for_poc_padded = centerPasteImage(radon_image_for_poc, radon_template_for_poc)

    sinogram_image_raw = radonTransformFloat(image_proc)
    sinogram_template_raw = radonTransformFloat(template)

    # Step 3: POC angle detection (on uint8 windowed sinograms)
    fft_image_complex = np.array([np.fft.fft(radon_image_for_poc[i].astype(np.float32))
                                   for i in range(360)], dtype=np.complex64)
    fft_template_complex = np.array([np.fft.fft(radon_template_for_poc_padded[i].astype(np.float32))
                                      for i in range(360)], dtype=np.complex64)

    angle_scores = np.zeros(180, dtype=np.float64)
    for alpha in range(180):
        corr_sum = 0.0
        for theta in range(180):
            row_img = fft_image_complex[(theta + alpha) % 360]
            row_tmpl = fft_template_complex[theta]
            cross_power = row_img * np.conj(row_tmpl)
            cross_power_normalized = cross_power / (np.abs(cross_power) + 1e-10)
            correlation = np.abs(np.fft.ifft(cross_power_normalized))
            corr_sum += np.max(correlation)
        angle_scores[alpha] = corr_sum
    detect_angle = int(np.argmax(angle_scores))

    # Step 4: Resolve 180-degree ambiguity and estimate position using NCC
    dx1, dy1, score1 = detectPosition(sinogram_image_raw, sinogram_template_raw, detect_angle)
    dx2, dy2, score2 = detectPosition(sinogram_image_raw, sinogram_template_raw, detect_angle + 180)

    if score1 >= score2:
        target_angle = detect_angle
        dx, dy = int(round(dx1)), int(round(dy1))
    else:
        target_angle = detect_angle + 180
        dx, dy = int(round(dx2)), int(round(dy2))

    print("detected: angle = ", target_angle, ", dx = ", dx, ", dy = ", dy)

    # Visualization
    # Template image / テンプレート画像
    plt.subplot(3, 3, 1)
    plt.imshow(template, cmap='gray')
    plt.title('Template')
    plt.colorbar()

    # Radon sinogram of template / テンプレートのサイノグラム
    plt.subplot(3, 3, 2)
    plt.imshow(sinogram_template_raw, cmap='gray', aspect='auto')
    plt.title('Sinogram (Template)')
    plt.colorbar()

    # Angle detection scores / 角度検出スコア
    plt.subplot(3, 3, 3)
    plt.plot(range(180), angle_scores)
    plt.axvline(x=detect_angle % 180, color='r', linestyle='--',
                label=f'best={detect_angle}')
    plt.title('Angle POC Scores')
    plt.xlabel('Angle shift (deg)')
    plt.ylabel('Sum of POC peaks')
    plt.legend()

    # Target image / 対象画像
    plt.subplot(3, 3, 4)
    plt.imshow(image, cmap='gray')
    plt.title('Target')
    plt.colorbar()

    # Radon sinogram of target / 対象画像のサイノグラム
    plt.subplot(3, 3, 5)
    plt.imshow(sinogram_image_raw, cmap='gray', aspect='auto')
    plt.title('Sinogram (Target)')
    plt.colorbar()

    # Position NCC at detected angle (theta=0) / 検出角度での位置NCC
    tmpl_angle_for_0 = (360 - target_angle) % 360
    img_row_0 = sinogram_image_raw[0]
    tmpl_row_0 = sinogram_template_raw[tmpl_angle_for_0]
    ncc_result = matchTemplateOneLineNCC(img_row_0, tmpl_row_0)
    plt.subplot(3, 3, 6)
    plt.plot(ncc_result)
    best_pos = np.argmax(ncc_result)
    plt.axvline(x=best_pos, color='r', linestyle='--', label=f'pos={best_pos}')
    plt.title('Position NCC (theta=0)')
    plt.xlabel('Position')
    plt.ylabel('NCC')
    plt.legend()

    # Matching result / マッチング結果
    draw_color = (0, 0, 0)
    result_image = drawRotatedRectangleOnImage(
        image,
        (dx + image.shape[1] // 2, dy + image.shape[0] // 2),
        template.shape[1], template.shape[0],
        target_angle, draw_color)

    plt.subplot(3, 3, 7)
    plt.imshow(result_image, cmap='gray')
    plt.title('Matched Image')
    plt.colorbar()

    plt.tight_layout()
    plt.show()

    return target_angle, dx, dy

"""
Sample
使用例
"""
if __name__ == "__main__":
    # Read template images / テンプレート画像を読み込む
    #image_path = 'figs/test_image6.jpg'
    image_path = 'figs/template.jpg'
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    #image = 255 - image

    # Read target images / 対象画像を読み込む
    #image_path2 = 'figs/test_image_large.jpg'
    image_path2 = 'figs/target.jpg'
    image2 = cv2.imread(image_path2, cv2.IMREAD_GRAYSCALE)
    #image2 = 255 - image2

    matchTemplateRotatable(image2, image)
