import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
import time
from numba import jit


# =============================================================================
# Radon Transform Primitives / ラドン変換基本関数
# =============================================================================

@jit('int32[:,:](uint8[:,:],int32,int32)', nopython=True, cache=True)
def getLinePoints(image, angle, length):
    """
    画像中心から指定角度で直線を描くためのポイントを取得
    :param image: 入力画像
    :param angle: 角度（度単位）
    :param length: 直線の長さ
    :return: 直線上のポイントのリスト
    """
    h, w = image.shape
    center = (w // 2, h // 2)
    angle_rad = np.deg2rad(angle)

    start_x = int(center[0] - length / 2 * np.cos(angle_rad))
    start_y = int(center[1] + length / 2 * np.sin(angle_rad))
    end_x = int(center[0] + length / 2 * np.cos(angle_rad))
    end_y = int(center[1] - length / 2 * np.sin(angle_rad))

    line_points = np.empty((length, 2), dtype=np.int32)
    for i in range(length):
        t = i / (length - 1)
        x = int(start_x * (1 - t) + end_x * t)
        y = int(start_y * (1 - t) + end_y * t)
        line_points[i] = (x, y)

    return line_points


@jit('uint64[:](uint8[:,:],int32[:,:],int32)', nopython=True, cache=True)
def perpendicularSum(image, points, width):
    """
    直線上の各点において、垂直方向の画素値の合計を計算
    :param image: 入力画像
    :param points: 直線上のポイントのリスト
    :param width: 垂直方向に取る幅
    :return: 画素値の合計リスト
    """
    sums = []
    h, w = image.shape

    top_row_mean = np.mean(image[0, :])
    bottom_row_mean = np.mean(image[-1, :])
    left_column_mean = np.mean(image[:, 0])
    right_column_mean = np.mean(image[:, -1])
    corner_pixels_mean = np.uint64(
        (top_row_mean + bottom_row_mean + left_column_mean + right_column_mean) / 4.0)

    perp_vec = np.array([0.0, 0.0], dtype=np.float32)
    for point in points:
        x, y = point
        perp_vec = np.array([-(y - h // 2), x - w // 2], dtype=np.float32)
        norm = math.sqrt(perp_vec[0] * perp_vec[0] + perp_vec[1] * perp_vec[1])
        if not norm == 0:
            perp_vec = perp_vec / norm
            break

    for point in points:
        x, y = point
        sum_value = np.uint64(0)
        for i in range(-width // 2, width // 2):
            perp_x = int(x + perp_vec[0] * i)
            perp_y = int(y + perp_vec[1] * i)
            if 0 <= perp_x < w and 0 <= perp_y < h:
                sum_value += image[perp_y, perp_x]
            else:
                sum_value += corner_pixels_mean
        sums.append(sum_value)

    sums = np.array(sums, dtype=np.uint64)
    return sums


def radonTransformFloat(image):
    """
    float32精度のラドン変換（量子化損失なし）
    :param image: 入力画像 (uint8)
    :return: float32サイノグラム (360 x line_length)
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



# =============================================================================
# Angle Detection: Hough Voting / 角度検出：ハフ投票
# =============================================================================

def extractSinogramCore(sinogram, template_height, template_width):
    """
    サイノグラム行からテンプレート本体の投影領域のみを抽出する。
    端部の填充（corner_pixels_mean）を除外し、純粋なテンプレート投影を得る。

    :param sinogram: サイノグラム (360 x L)
    :param template_height: テンプレートの高さ
    :param template_width: テンプレートの幅
    :return: 360個のcore配列のリスト（角度により長さが異なる）
    """
    row_len = sinogram.shape[1]
    center = row_len // 2
    cores = []
    for angle in range(360):
        theta_rad = np.deg2rad(angle)
        # 投影幅は角度に依存: width(θ) = |w·cos(θ)| + |h·sin(θ)|
        proj_width = int(abs(template_width * np.cos(theta_rad)) +
                         abs(template_height * np.sin(theta_rad)))
        proj_width = max(proj_width, 4)
        half = proj_width // 2
        start = max(0, center - half)
        end = min(row_len, center + half)
        cores.append(sinogram[angle, start:end].copy())
    return cores


def detectAngleHough(sinogram_image, sinogram_template,
                     template_height, template_width, score_threshold=0.5):
    """
    サイノグラムコアのハフ投票による回転角度・位置の同時検出。

    原理:
      画像上の1点(dx,dy)はサイノグラム上で正弦波 offset(θ)=dx·cos(θ)+dy·sin(θ) を描く。
      テンプレートのコア（本体投影のみ）を画像サイノグラムの各行と1Dマッチングし、
      強い一致ごとに回転角 α=(θ_image - θ_template)%180 に投票する。
      全行が同一の正弦波パラメータに拘束されるため、投票の集中により正しい角度が検出される。

    :param sinogram_image: 対象画像のfloatサイノグラム (360 x L_img)
    :param sinogram_template: テンプレートのfloatサイノグラム (360 x L_tmpl)
    :param template_height: テンプレートの高さ
    :param template_width: テンプレートの幅
    :param score_threshold: 投票に必要な最小NCC値
    :return: (detected_angle, dx, dy, accumulator)
    """
    # Step 1: テンプレートサイノグラムのコア抽出
    cores = extractSinogramCore(sinogram_template, template_height, template_width)
    n_img = sinogram_image.shape[1]

    # Step 2: 全(画像行, テンプレートコア)ペアの1Dマッチング
    match_scores = np.zeros((360, 180), dtype=np.float32)
    match_positions = np.zeros((360, 180), dtype=np.int32)

    for j in range(180):  # テンプレート角度
        core = cores[j]
        if len(core) < 4 or len(core) >= n_img:
            continue
        tmpl_2d = core.astype(np.float32).reshape(1, -1)

        for i in range(360):  # 画像角度
            img_2d = sinogram_image[i].astype(np.float32).reshape(1, -1)
            result = cv2.matchTemplate(img_2d, tmpl_2d, cv2.TM_CCOEFF_NORMED)
            ncc = result[0]
            best_pos = int(np.argmax(ncc))
            match_scores[i, j] = float(ncc[best_pos])
            match_positions[i, j] = best_pos

    # Step 3: ハフ投票
    accumulator = np.zeros(180, dtype=np.float64)
    for j in range(180):
        for i in range(360):
            score = match_scores[i, j]
            if score < score_threshold:
                continue
            alpha = (i - j) % 180
            accumulator[alpha] += score

    detected_alpha = int(np.argmax(accumulator))

    # Step 4: 検出角度での位置推定（正弦波フィッティング）
    cos_theta = np.cos(np.deg2rad(np.arange(360)))
    sin_theta = np.sin(np.deg2rad(np.arange(360)))

    offsets = []
    weights = []
    A_rows = []

    for theta_img in range(360):
        theta_tmpl = (theta_img - detected_alpha) % 360
        if theta_tmpl >= 180:
            continue
        score = match_scores[theta_img, theta_tmpl]
        if score < score_threshold:
            continue

        pos = match_positions[theta_img, theta_tmpl]
        core_len = len(cores[theta_tmpl])
        offset = pos + core_len // 2 - n_img // 2

        offsets.append(offset)
        weights.append(max(score, 0.0))
        A_rows.append([cos_theta[theta_img], sin_theta[theta_img]])

    dx, dy = 0.0, 0.0
    if len(offsets) >= 2:
        offsets_arr = np.array(offsets, dtype=np.float64)
        weights_arr = np.array(weights, dtype=np.float64)
        A = np.array(A_rows, dtype=np.float64)
        W = np.diag(weights_arr)
        try:
            params = np.linalg.solve(A.T @ W @ A, A.T @ W @ offsets_arr)
            dx, dy = params[0], params[1]
        except np.linalg.LinAlgError:
            pass

    return detected_alpha, int(round(dx)), int(round(dy)), accumulator


# =============================================================================
# Position Estimation / 位置推定
# =============================================================================

def matchTemplateOneLineNCC(image_line, template_line):
    """
    1次元正規化相互相関（cv2.matchTemplate wrapper）
    :param image_line: 対象サイノグラムの1行
    :param template_line: テンプレートサイノグラムの1行
    :return: NCC結果配列
    """
    if len(template_line) >= len(image_line):
        return np.array([0.0], dtype=np.float32)
    img = image_line.astype(np.float32).reshape(1, -1)
    tmpl = template_line.astype(np.float32).reshape(1, -1)
    result = cv2.matchTemplate(img, tmpl, cv2.TM_CCOEFF_NORMED)
    return result[0]


def detectPosition(sinogram_image, sinogram_template, detect_angle, num_angles=36):
    """
    多角度の重み付き最小二乗法でテンプレート位置を推定する。

    各角度θでのサイノグラムオフセットは offset(θ) = dx·cos(θ) - dy·sin(θ) に従う。
    複数角度でNCCによりオフセットを計測し、(dx, dy) を最小二乗法で推定する。

    :param sinogram_image: 対象画像のfloatサイノグラム
    :param sinogram_template: テンプレートのfloatサイノグラム
    :param detect_angle: 検出された回転角度
    :param num_angles: 位置推定に使用する角度数
    :return: (dx, dy, total_score)
    """
    angles_deg = np.linspace(0, 180, num_angles, endpoint=False)

    offsets = []
    weights = []
    A_rows = []

    for theta_deg in angles_deg:
        theta = int(theta_deg) % 360
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
        params = np.linalg.solve(A.T @ W @ A, A.T @ W @ offsets)
        dx, dy = params[0], params[1]
    except np.linalg.LinAlgError:
        dx, dy = 0.0, 0.0

    total_score = float(np.sum(weights))
    return dx, dy, total_score


# =============================================================================
# HF Energy Refinement / 高周波エネルギー精密推定
# =============================================================================

def computeHFProfile(img_row, core, cutoff_ratio=1/8):
    """
    FFTシフト定理で全位置のHFエネルギーを一括計算。

    画像行からコアを位置pで引いた残差のHFエネルギー:
      E_HF(p) = const - (2/L)·Re(IFFT(A_HF · conj(B_HF))[p])

    :param img_row: 画像サイノグラムの1行 (L,)
    :param core: テンプレートサイノグラムのコア (nc,), nc < L
    :param cutoff_ratio: ハイパスフィルタのカットオフ比率
    :return: 各位置のHFエネルギー (L - nc + 1,)
    """
    L = len(img_row)
    nc = len(core)
    nr = L - nc + 1
    if nr <= 0:
        return np.full(L, np.inf)

    core_padded = np.zeros(L, dtype=np.float64)
    core_padded[:nc] = core.astype(np.float64)

    A = np.fft.fft(img_row.astype(np.float64))
    B = np.fft.fft(core_padded)

    c = max(1, int(L * cutoff_ratio))
    hf_mask = np.ones(L, dtype=np.float64)
    hf_mask[:c] = 0
    hf_mask[-c + 1:] = 0

    A_hf = A * hf_mask
    B_hf = B * hf_mask

    const = np.sum(np.abs(A_hf) ** 2) / L + np.sum(np.abs(B_hf) ** 2) / L
    cross_hf = A_hf * np.conj(B_hf)
    cross_spatial = np.real(np.fft.ifft(cross_hf))
    hf_energy = const - 2.0 / L * cross_spatial

    return hf_energy[:nr]


def precomputeNCCHFData(sinogram_image, cores, cutoff_ratio=1/8):
    """
    NCC-HFプロファイル用の事前計算。
    HPF適用済み画像行のFFT・累積和と、HPF適用済みコアのFFT・統計量を計算する。
    """
    L = sinogram_image.shape[1]
    c_L = max(1, int(L * cutoff_ratio))
    hf_mask_L = np.ones(L, dtype=np.float64)
    hf_mask_L[:c_L] = 0
    hf_mask_L[-c_L + 1:] = 0

    # 画像行: HPF → FFT保持, 累積和 (ローカル平均・分散用)
    img_hf_fft = np.zeros((360, L), dtype=np.complex128)
    img_cumsum = np.zeros((360, L + 1), dtype=np.float64)
    img_cumsum_sq = np.zeros((360, L + 1), dtype=np.float64)
    for i in range(360):
        F = np.fft.fft(sinogram_image[i].astype(np.float64))
        F *= hf_mask_L
        img_hf_fft[i] = F
        row_hf = np.real(np.fft.ifft(F))
        img_cumsum[i, 1:] = np.cumsum(row_hf)
        img_cumsum_sq[i, 1:] = np.cumsum(row_hf ** 2)

    # コア: HPF (コア長基準) → ゼロパディングしてFFT, 平均・分散
    core_hf_fft = [None] * 180
    core_hf_mean = np.zeros(180, dtype=np.float64)
    core_hf_energy = np.zeros(180, dtype=np.float64)  # nc * var
    nc_list = np.zeros(180, dtype=np.int32)
    valid = np.zeros(180, dtype=bool)

    for j in range(180):
        core = cores[j]
        nc = len(core)
        nc_list[j] = nc
        if nc < 4 or nc >= L:
            continue
        c_nc = max(1, int(nc * cutoff_ratio))
        Fc = np.fft.fft(core.astype(np.float64))
        Fc[:c_nc] = 0
        Fc[-c_nc + 1:] = 0
        core_hf_real = np.real(np.fft.ifft(Fc))
        core_hf_mean[j] = np.mean(core_hf_real)
        core_hf_energy[j] = np.sum((core_hf_real - core_hf_mean[j]) ** 2)
        # L長にゼロパディングしてFFT（相互相関用）
        padded = np.zeros(L, dtype=np.float64)
        padded[:nc] = core_hf_real
        core_hf_fft[j] = np.fft.fft(padded)
        valid[j] = True

    return {
        'img_hf_fft': img_hf_fft,
        'img_cumsum': img_cumsum,
        'img_cumsum_sq': img_cumsum_sq,
        'core_hf_fft': core_hf_fft,
        'core_hf_mean': core_hf_mean,
        'core_hf_energy': core_hf_energy,
        'nc_list': nc_list,
        'valid': valid,
        'L': L,
    }


def precomputeHFData(sinogram_image, cores, cutoff_ratio=1/8):
    """
    全行のforward FFTとHPフィルタ適用を事前計算する。
    画像行のFFTは角度αに依存せず、コアのFFTはテンプレート角度jにのみ依存するため、
    ループ外で1回だけ計算すればよい。

    :param sinogram_image: ターゲットのサイノグラム (360 x L)
    :param cores: テンプレートのサイノグラムコア (360個, 0-179のみ使用)
    :param cutoff_ratio: ハイパスフィルタのカットオフ比率
    :return: 事前計算データの辞書
    """
    L = sinogram_image.shape[1]
    c = max(1, int(L * cutoff_ratio))

    # HPマスク（全行共通）
    hf_mask = np.ones(L, dtype=np.float64)
    hf_mask[:c] = 0
    hf_mask[-c + 1:] = 0

    # 画像行のFFT → HPフィルタ適用 (360行)
    A_hf = []
    sum_A2 = np.zeros(360, dtype=np.float64)
    for i in range(360):
        A = np.fft.fft(sinogram_image[i].astype(np.float64))
        a_hf = A * hf_mask
        A_hf.append(a_hf)
        sum_A2[i] = np.sum(np.abs(a_hf) ** 2) / L

    # テンプレートコアのFFT → HPフィルタ適用 (180行)
    B_hf = [None] * 180
    sum_B2 = np.zeros(180, dtype=np.float64)
    nc_list = np.zeros(180, dtype=np.int32)
    for j in range(180):
        core = cores[j]
        nc = len(core)
        nc_list[j] = nc
        if nc < 4 or nc >= L:
            continue
        core_padded = np.zeros(L, dtype=np.float64)
        core_padded[:nc] = core.astype(np.float64)
        B = np.fft.fft(core_padded)
        b_hf = B * hf_mask
        B_hf[j] = b_hf
        sum_B2[j] = np.sum(np.abs(b_hf) ** 2) / L

    # 行列形式に変換（fancy indexingの高速化）
    A_hf_matrix = np.array(A_hf)         # (360, L)
    B_hf_matrix = np.zeros((180, L), dtype=np.complex128)
    B_hf_valid = np.zeros(180, dtype=bool)
    for j in range(180):
        if B_hf[j] is not None:
            B_hf_matrix[j] = B_hf[j]
            B_hf_valid[j] = True

    return {
        'A_hf_matrix': A_hf_matrix,
        'B_hf_matrix': B_hf_matrix,
        'B_hf_valid': B_hf_valid,
        'sum_A2': sum_A2,
        'sum_B2': sum_B2,
        'L': L,
        'nc_list': nc_list,
        # 後方互換のために残す
        'A_hf': A_hf,
        'B_hf': B_hf,
    }


def findPositionByHFProfile(sino_img, cores, alpha, n_img, center_t,
                            cos_t, sin_t, max_dx, max_dy, hf_data=None):
    """
    角度αで全行のHFプロファイルを計算し、正弦波パスで最良(dx, dy)を推定。

    行の対応: image[i] ↔ template[(i - α) % 360]

    :return: (best_dx, best_dy, best_energy)
    """
    # HFプロファイル計算
    if hf_data is not None:
        L = hf_data['L']
        # 有効な(i, j)ペアを収集
        valid_i = []
        valid_j = []
        for i in range(360):
            j = (i - alpha) % 360
            if j >= 180:
                continue
            if hf_data['B_hf'][j] is None:
                continue
            nc = int(hf_data['nc_list'][j])
            if L - nc + 1 <= 0:
                continue
            valid_i.append(i)
            valid_j.append(j)

        if len(valid_i) == 0:
            return 0, 0

        n_rows = len(valid_i)
        nc_arr = hf_data['nc_list'][valid_j]

        # バッチ: cross_hf行列を構築 → 一括IFFT
        A_mat = np.array([hf_data['A_hf'][i] for i in valid_i])  # (n_rows, L)
        B_mat = np.array([hf_data['B_hf'][j] for j in valid_j])  # (n_rows, L)
        constants = hf_data['sum_A2'][valid_i] + hf_data['sum_B2'][valid_j]  # (n_rows,)

        cross_hf_mat = A_mat * np.conj(B_mat)                     # (n_rows, L)
        cross_spatial = np.real(np.fft.ifft(cross_hf_mat, axis=1)) # (n_rows, L) 一括IFFT

        # E_HF(p) = const - 2/L * Re(cross_spatial[p])
        prof_matrix_full = constants[:, None] - 2.0 / L * cross_spatial  # (n_rows, L)

        # 有効範囲外をinfで埋めたプロファイル行列を構築
        prof_lens = L - nc_arr + 1
        max_prof_len = int(np.max(prof_lens))
        prof_matrix = np.full((n_rows, max_prof_len), np.inf, dtype=np.float64)
        for k in range(n_rows):
            prof_matrix[k, :prof_lens[k]] = prof_matrix_full[k, :prof_lens[k]]

        row_indices = valid_i
        nc_list = nc_arr
        cos_vals = cos_t[row_indices]
        sin_vals = sin_t[row_indices]
    else:
        profiles = {}
        for i in range(360):
            j = (i - alpha) % 360
            if j >= 180:
                continue
            core = cores[j]
            if len(core) < 4 or len(core) >= n_img:
                continue
            prof = computeHFProfile(sino_img[i], core)
            profiles[i] = (prof, len(core))

        if len(profiles) == 0:
            return 0, 0

        row_indices = sorted(profiles.keys())
        prof_list = [profiles[i][0] for i in row_indices]
        nc_list = np.array([profiles[i][1] for i in row_indices])
        cos_vals = cos_t[row_indices]
        sin_vals = sin_t[row_indices]
        prof_lens = np.array([len(p) for p in prof_list])
        n_rows = len(row_indices)

        max_prof_len = max(prof_lens)
        prof_matrix = np.full((n_rows, max_prof_len), np.inf, dtype=np.float64)
        for k in range(n_rows):
            prof_matrix[k, :prof_lens[k]] = prof_list[k]

    def eval_positions_batch(dx_arr, dy_arr):
        # 全(dx, dy)を一括評価: (n_dx, n_dy, n_rows) のブロードキャスト演算
        # offsets[d, e, k] = dx_arr[d]*cos_vals[k] - dy_arr[e]*sin_vals[k]
        offsets = (dx_arr[:, None, None] * cos_vals[None, None, :]
                   - dy_arr[None, :, None] * sin_vals[None, None, :])
        positions = np.round(center_t + offsets - nc_list[None, None, :] / 2.0).astype(np.intp)
        np.clip(positions, 0, (prof_lens - 1)[None, None, :], out=positions)

        # 各(dx, dy)でのHFエネルギー合計
        row_idx = np.arange(n_rows)[None, None, :]
        energies = prof_matrix[row_idx, positions]   # (n_dx, n_dy, n_rows)
        totals = np.mean(energies, axis=2)           # (n_dx, n_dy)

        min_flat = np.argmin(totals)
        di, dj = np.unravel_index(min_flat, totals.shape)
        return int(dx_arr[di]), int(dy_arr[dj]), totals[di, dj]

    # 粗い位置探索 (step=2)
    dx_range = np.arange(-max_dx, max_dx + 1, 2)
    dy_range = np.arange(-max_dy, max_dy + 1, 2)
    best_dx, best_dy, _ = eval_positions_batch(dx_range, dy_range)

    # 精密探索 (step=1, ±2px)
    dx_fine = np.arange(max(best_dx - 2, -max_dx), min(best_dx + 3, max_dx + 1))
    dy_fine = np.arange(max(best_dy - 2, -max_dy), min(best_dy + 3, max_dy + 1))
    fine_dx, fine_dy, fine_e = eval_positions_batch(dx_fine, dy_fine)

    return fine_dx, fine_dy, fine_e


def findPositionByNCCHF(sino_img, cores, alpha, n_img, center_t,
                        cos_t, sin_t, max_dx, max_dy, ncc_data=None):
    """
    NCC-HF版: HPF適用済みサイノグラムのローカルNCC最大化で(dx, dy)を推定。

    HFエネルギー版と異なり、コア領域のみの正規化相関を使うため
    背景ベースラインの影響を受けない。

    :return: (best_dx, best_dy, best_ncc_score)
    """
    if ncc_data is None:
        return 0, 0, -np.inf

    L = ncc_data['L']
    # 有効な(i, j)ペアを収集
    valid_i = []
    valid_j = []
    for i in range(360):
        j = (i - alpha) % 360
        if j >= 180:
            continue
        if not ncc_data['valid'][j]:
            continue
        nc = int(ncc_data['nc_list'][j])
        if L - nc + 1 <= 0:
            continue
        valid_i.append(i)
        valid_j.append(j)

    if len(valid_i) == 0:
        return 0, 0, -np.inf

    n_rows = len(valid_i)
    nc_arr = ncc_data['nc_list'][valid_j].astype(np.float64)

    # バッチ相互相関: IFFT(img_hf_fft[i] * conj(core_hf_fft[j]))
    A_mat = ncc_data['img_hf_fft'][valid_i]                        # (n_rows, L)
    B_mat = np.array([ncc_data['core_hf_fft'][j] for j in valid_j])  # (n_rows, L)
    cross_spatial = np.real(np.fft.ifft(A_mat * np.conj(B_mat), axis=1))  # (n_rows, L)

    # ローカル統計量 (cumsum から window size nc_arr[k] で計算)
    cs = ncc_data['img_cumsum']      # (360, L+1)
    css = ncc_data['img_cumsum_sq']  # (360, L+1)
    core_mean = ncc_data['core_hf_mean'][valid_j]    # (n_rows,)
    core_energy = ncc_data['core_hf_energy'][valid_j]  # (n_rows,)

    # NCCプロファイル行列: ncc[k][p] = (cross[p]/nc - μ_img(p)*μ_core) / sqrt(σ²_img(p) * Σ(core-μ)²/nc)
    prof_lens = (L - nc_arr + 1).astype(int)
    max_prof_len = int(np.max(prof_lens))
    prof_matrix = np.full((n_rows, max_prof_len), -np.inf, dtype=np.float64)

    for k in range(n_rows):
        i_idx = valid_i[k]
        nc = int(nc_arr[k])
        nr = int(prof_lens[k])
        # ローカル平均・分散
        local_sum = cs[i_idx, nc:nc + nr] - cs[i_idx, :nr]
        local_sum_sq = css[i_idx, nc:nc + nr] - css[i_idx, :nr]
        local_mean = local_sum / nc
        local_var = local_sum_sq / nc - local_mean ** 2
        local_var = np.maximum(local_var, 0)  # 数値誤差対策
        # NCC
        denom = np.sqrt(local_var * core_energy[k])
        numerator = cross_spatial[k, :nr] / nc - local_mean * core_mean[k]
        safe = denom > 1e-10
        prof_matrix[k, :nr] = np.where(safe, numerator / np.where(safe, denom, 1), 0)

    cos_vals = cos_t[valid_i]
    sin_vals = sin_t[valid_i]

    def eval_positions_batch(dx_arr, dy_arr):
        offsets = (dx_arr[:, None, None] * cos_vals[None, None, :]
                   - dy_arr[None, :, None] * sin_vals[None, None, :])
        positions = np.round(center_t + offsets - nc_arr[None, None, :] / 2.0).astype(np.intp)
        np.clip(positions, 0, (prof_lens - 1)[None, None, :].astype(np.intp), out=positions)
        row_idx = np.arange(n_rows)[None, None, :]
        ncc_vals = prof_matrix[row_idx, positions]
        totals = np.mean(ncc_vals, axis=2)
        max_flat = np.argmax(totals)
        di, dj = np.unravel_index(max_flat, totals.shape)
        return int(dx_arr[di]), int(dy_arr[dj]), totals[di, dj]

    # 粗い位置探索 (step=2)
    dx_range = np.arange(-max_dx, max_dx + 1, 2)
    dy_range = np.arange(-max_dy, max_dy + 1, 2)
    best_dx, best_dy, _ = eval_positions_batch(dx_range, dy_range)

    # 精密探索 (step=1, ±2px)
    dx_fine = np.arange(max(best_dx - 2, -max_dx), min(best_dx + 3, max_dx + 1))
    dy_fine = np.arange(max(best_dy - 2, -max_dy), min(best_dy + 3, max_dy + 1))
    fine_dx, fine_dy, fine_score = eval_positions_batch(dx_fine, dy_fine)

    return fine_dx, fine_dy, fine_score


def refineByNCC(image, template, coarse_angle, coarse_dx, coarse_dy,
                angle_range=3, pos_range=3):
    """
    粗推定(α, dx, dy)の近傍で画像空間の2D NCCによる精密化。

    :return: (angle, dx, dy, ncc_score)
    """
    th, tw = template.shape
    img_h, img_w = image.shape
    cy, cx = img_h // 2, img_w // 2

    best_a = coarse_angle
    best_dx = coarse_dx
    best_dy = coarse_dy
    best_score = -np.inf

    for a in range(coarse_angle - angle_range, coarse_angle + angle_range + 1):
        am = a % 360
        M = cv2.getRotationMatrix2D((tw // 2, th // 2), am, 1.0)
        tmpl_rot = cv2.warpAffine(
            template, M, (tw, th),
            borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        mask = cv2.warpAffine(
            np.ones((th, tw), dtype=np.uint8) * 255,
            M, (tw, th),
            borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        m = mask > 128
        if np.sum(m) < 10:
            continue
        tmpl_masked = tmpl_rot.astype(np.float64)[m]
        tm = tmpl_masked - tmpl_masked.mean()
        tm_energy = np.sum(tm ** 2)
        if tm_energy < 1e-10:
            continue

        for dx in range(coarse_dx - pos_range, coarse_dx + pos_range + 1):
            for dy in range(coarse_dy - pos_range, coarse_dy + pos_range + 1):
                y1 = cy - th // 2 + dy
                x1 = cx - tw // 2 + dx
                if y1 < 0 or y1 + th > img_h or x1 < 0 or x1 + tw > img_w:
                    continue
                region = image[y1:y1 + th, x1:x1 + tw].astype(np.float64)[m]
                rm = region - region.mean()
                denom = np.sqrt(np.sum(rm ** 2) * tm_energy)
                if denom < 1e-10:
                    continue
                ncc = np.sum(rm * tm) / denom
                if ncc > best_score:
                    best_score = ncc
                    best_a = am
                    best_dx = dx
                    best_dy = dy

    return best_a, best_dx, best_dy, best_score


def detectByHFAndNCC(image, template, sinogram_image, cores, n_img,
                     angle_step_coarse=3):
    """
    2段階テンプレートマッチング:
      Step 1: 各角度αで行ごとHFプロファイル（FFTシフト定理）を
              正弦波パスで集約し、(α, dx, dy) を粗推定
      Step 2: 各候補の近傍で画像空間の2D NCCによる精密化

    :param image: ターゲット画像 (グレースケール)
    :param template: テンプレート画像 (グレースケール)
    :param sinogram_image: ターゲットのサイノグラム
    :param cores: テンプレートのサイノグラムコア（extractSinogramCoreで取得）
    :param n_img: サイノグラムの列数
    :param angle_step_coarse: 角度粗探索のステップ（度）
    :return: (angle, dx, dy, ncc_score)
    """
    th, tw = template.shape
    img_h, img_w = image.shape

    cos_t = np.cos(np.deg2rad(np.arange(360)))
    sin_t = np.sin(np.deg2rad(np.arange(360)))
    center_t = n_img // 2
    max_dx = (img_w - tw) // 2 - 2
    max_dy = (img_h - th) // 2 - 2

    # FFT事前計算（全行のforward FFTを1回だけ実行）
    t0 = time.time()
    hf_data = precomputeHFData(sinogram_image, cores)
    t_precomp = time.time() - t0
    print(f"  HF precompute: {t_precomp:.3f}s (540 FFTs)")

    # Step 1: 各角度で位置粗推定（HFエネルギースコア付き）
    t0 = time.time()
    candidates = []
    for alpha in range(0, 360, angle_step_coarse):
        dx_est, dy_est, hf_energy = findPositionByHFProfile(
            sinogram_image, cores, alpha, n_img, center_t,
            cos_t, sin_t, max_dx, max_dy, hf_data=hf_data)
        candidates.append((alpha, dx_est, dy_est, hf_energy))
    t_step1 = time.time() - t0
    n_angles = 360 // angle_step_coarse
    print(f"  HF position search: {t_step1:.3f}s ({n_angles} angles)")

    # HFエネルギーで上位K候補に絞り込み（低エネルギー=良い一致）
    top_k = 20
    candidates.sort(key=lambda c: c[3])
    candidates = candidates[:top_k]

    # Step 2: 上位候補の近傍で2D NCC精密化、最良を選択
    t0 = time.time()
    best = (0, 0, 0, -np.inf)
    for ca, cdx, cdy, _ in candidates:
        a, dx, dy, score = refineByNCC(
            image, template, ca, cdx, cdy,
            angle_range=angle_step_coarse, pos_range=3)
        if score > best[3]:
            best = (a, dx, dy, score)
    t_step2 = time.time() - t0
    print(f"  NCC refinement: {t_step2:.3f}s (top {top_k} candidates)")

    return best


# =============================================================================
# Drawing / 描画
# =============================================================================

def drawRotatedRectangleOnImage(image, center, width, height, angle,
                                color=(255, 0, 0), thickness=8):
    """
    所定の角度だけ回転させた四角形を描画する
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
    points = rotated_rectangle.astype(int).reshape((-1, 1, 2))
    cv2.polylines(image, [points], isClosed=True, color=color, thickness=thickness)
    return image


# =============================================================================
# Main Entry Point / メイン処理
# =============================================================================

def matchTemplateRotatable(image, template):
    """
    ラドン変換の2段階テンプレートマッチング

    処理フロー:
      1. 適応的コントラスト正規化（コントラスト比 < 0.6 の場合）
      2. テンプレートにガウシアン窓を適用、サイノグラム+コア計算
      3. Step 1: 行ごとHFプロファイル（FFTシフト定理）＋正弦波パス集約で
         各角度αに対する(dx, dy)を粗推定
      4. Step 2: 各候補の近傍で画像空間の2D NCCによる精密化
      5. 結果の可視化

    :param image: 対象画像 (グレースケール)
    :param template: テンプレート画像 (グレースケール)
    :return: (target_angle, dx, dy)
    """
    th, tw = template.shape
    img_h, img_w = image.shape

    # Step 1: 適応的コントラスト正規化
    cr = np.std(image.astype(np.float32)) / (np.std(template.astype(np.float32)) + 1e-10)
    if cr < 0.6:
        im_m = np.mean(image.astype(np.float32))
        im_s = np.std(image.astype(np.float32))
        tm_m = np.mean(template.astype(np.float32))
        tm_s = np.std(template.astype(np.float32))
        image_proc = np.clip(
            (image.astype(np.float32) - im_m) / (im_s + 1e-10) * tm_s + tm_m,
            0, 255).astype(np.uint8)
    else:
        image_proc = image

    # Step 2: テンプレートのサイノグラム+コア計算
    x = np.linspace(-1, 1, tw)
    y = np.linspace(-1, 1, th)
    x, y = np.meshgrid(x, y)
    gw = np.exp(-(x**2 + y**2) / (2 * 1.0**2))
    tmpl_windowed = (template.astype(np.float32) * gw).astype(np.uint8)

    # 画像のcorner_pixels_meanでキャンバスを充填（サイノグラム境界外の基準を統一）
    cm = int((np.mean(image_proc[0, :]) + np.mean(image_proc[-1, :]) +
              np.mean(image_proc[:, 0]) + np.mean(image_proc[:, -1])) / 4.0)
    tmpl_canvas = cv2.copyMakeBorder(
        tmpl_windowed,
        (img_h - th) // 2, img_h - th - (img_h - th) // 2,
        (img_w - tw) // 2, img_w - tw - (img_w - tw) // 2,
        cv2.BORDER_CONSTANT, value=cm)
    t_start = time.time()
    sinogram_image = radonTransformFloat(image_proc)
    sinogram_template = radonTransformFloat(tmpl_canvas)
    t_radon = time.time() - t_start
    print(f"Radon transform: {t_radon:.3f}s")

    cores = extractSinogramCore(sinogram_template, th, tw)
    n_img = sinogram_template.shape[1]

    # Step 3-4: 2段階検出（HFプロファイル粗推定 + 2D NCC精密化）
    t_detect = time.time()
    target_angle, dx, dy, ncc_score = detectByHFAndNCC(
        image_proc, template, sinogram_image, cores, n_img)
    t_detect = time.time() - t_detect

    print(f"Detection total: {t_detect:.3f}s")
    print(f"Overall: {time.time() - t_start:.3f}s")
    print("detected: angle =", target_angle, ", dx =", dx, ", dy =", dy,
          ", ncc =", f"{ncc_score:.4f}")

    # Step 5: 可視化
    plt.subplot(2, 3, 1)
    plt.imshow(template, cmap='gray')
    plt.title('Template')
    plt.colorbar()

    plt.subplot(2, 3, 2)
    plt.imshow(image, cmap='gray')
    plt.title('Target')
    plt.colorbar()

    draw_color = (0, 0, 0)
    result_image = drawRotatedRectangleOnImage(
        image,
        (dx + img_w // 2, dy + img_h // 2),
        tw, th, target_angle, draw_color)

    plt.subplot(2, 3, 3)
    plt.imshow(result_image, cmap='gray')
    plt.title('Matched Image')
    plt.colorbar()

    plt.tight_layout()
    plt.show()

    return target_angle, dx, dy


# =============================================================================
# Sample / 使用例
# =============================================================================

if __name__ == "__main__":
    image_path = 'figs/template.jpg'
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    image_path2 = 'figs/target.jpg'
    image2 = cv2.imread(image_path2, cv2.IMREAD_GRAYSCALE)

    matchTemplateRotatable(image2, image)
