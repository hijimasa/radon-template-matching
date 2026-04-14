import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
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


@jit(nopython=True, cache=True)
def _radonLineInterp(image, angle_rad, line_length, perp_width):
    """
    バイリニア補間によるラドン変換の1行計算。
    座標をfloatのまま保持し、int丸めによる角度依存の離散化誤差を排除する。
    これにより R{f_rotated}(θ,t) = R{f}(θ-α,t) の性質が離散でも成立する。

    :param image: 入力画像 (uint8, 2D)
    :param angle_rad: 角度（ラジアン）
    :param line_length: 直線の長さ
    :param perp_width: 垂直方向の積分幅
    :return: float64の投影プロファイル (line_length,)
    """
    h, w = image.shape
    cx = w / 2.0
    cy = h / 2.0
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)

    # 直線方向: (cos_a, -sin_a), 垂直方向: (sin_a, cos_a)
    sums = np.zeros(line_length, dtype=np.float64)

    for i in range(line_length):
        t = (i - line_length / 2.0 + 0.5)
        # 直線上の点（float座標）
        lx = cx + t * cos_a
        ly = cy - t * sin_a

        total = 0.0
        for j in range(-perp_width // 2, perp_width // 2):
            # 垂直方向のサンプル点（float座標）
            px = lx + j * sin_a
            py = ly + j * cos_a

            # バイリニア補間
            if 0 <= px < w - 1 and 0 <= py < h - 1:
                x0 = int(px)
                y0 = int(py)
                fx = px - x0
                fy = py - y0
                total += (image[y0, x0] * (1 - fx) * (1 - fy) +
                          image[y0, x0 + 1] * fx * (1 - fy) +
                          image[y0 + 1, x0] * (1 - fx) * fy +
                          image[y0 + 1, x0 + 1] * fx * fy)
            # 範囲外は0として扱う（corner_pixels_mean不使用）

        sums[i] = total
    return sums


def radonTransformInterp(image):
    """
    バイリニア補間によるラドン変換。
    角度間の離散化誤差が均一になり、行シフトによる回転表現が正確に成立する。

    :param image: 入力画像 (uint8)
    :return: float64サイノグラム (360 x line_length)
    """
    h, w = image.shape
    line_length = int(math.sqrt(h * h + w * w))
    perp_width = line_length

    sinogram = np.zeros((360, line_length), dtype=np.float64)
    for angle in range(360):
        angle_rad = np.deg2rad(angle)
        sinogram[angle, :] = _radonLineInterp(
            image, angle_rad, line_length, perp_width)

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

def isTemplateInside(angle, dx, dy, tw, th, img_h, img_w, margin=2):
    """
    回転後のテンプレートが画像内に完全に収まるか判定する。

    :param angle: 回転角度（度）
    :param dx: x方向オフセット（画像中心基準）
    :param dy: y方向オフセット（画像中心基準）
    :param tw: テンプレート幅
    :param th: テンプレート高さ
    :param img_h: 画像の高さ
    :param img_w: 画像の幅
    :param margin: 境界マージン（ピクセル）
    :return: True/False
    """
    corners = np.array([[-tw / 2, -th / 2], [tw / 2, -th / 2],
                        [tw / 2, th / 2], [-tw / 2, th / 2]])
    rad = np.deg2rad(angle)
    R = np.array([[np.cos(rad), -np.sin(rad)],
                  [np.sin(rad), np.cos(rad)]])
    rotated = corners @ R.T
    abs_x = rotated[:, 0] + img_w / 2.0 + dx
    abs_y = rotated[:, 1] + img_h / 2.0 + dy
    return (abs_x.min() >= margin and abs_x.max() <= img_w - margin and
            abs_y.min() >= margin and abs_y.max() <= img_h - margin)


def highpassEnergy(sinogram, cutoff_ratio=1/8):
    """
    サイノグラムの各行にハイパスフィルタを適用し、高周波エネルギーを計算。

    :param sinogram: (360, L) サイノグラム
    :param cutoff_ratio: 低周波カットオフ比率
    :return: 高周波エネルギー（スカラー）
    """
    sino = sinogram.astype(np.float64)
    fft_all = np.fft.fft(sino, axis=1)
    n = fft_all.shape[1]
    cutoff = max(1, int(n * cutoff_ratio))
    fft_all[:, :cutoff] = 0
    fft_all[:, -cutoff + 1:] = 0
    hf = np.real(np.fft.ifft(fft_all, axis=1))
    return float(np.sum(hf ** 2))


def makeTemplSinogram(template, angle, dx, dy, img_h, img_w):
    """
    指定姿勢でテンプレートを画像サイズのキャンバスに配置し、サイノグラムを計算。

    :param template: テンプレート画像 (グレースケール)
    :param angle: 回転角度（度）
    :param dx: x方向オフセット（画像中心基準）
    :param dy: y方向オフセット（画像中心基準）
    :param img_h: ターゲット画像の高さ
    :param img_w: ターゲット画像の幅
    :return: (360, L) サイノグラム
    """
    th, tw = template.shape
    M = cv2.getRotationMatrix2D((tw // 2, th // 2), -angle, 1.0)
    tmpl_rot = cv2.warpAffine(template, M, (tw, th),
                               borderMode=cv2.BORDER_REPLICATE)
    canvas = np.zeros((img_h, img_w), dtype=np.uint8)
    cy, cx = img_h // 2, img_w // 2
    y1 = cy - th // 2 + dy
    x1 = cx - tw // 2 + dx
    canvas[y1:y1 + th, x1:x1 + tw] = tmpl_rot
    return radonTransformFloat(canvas)


def makeTemplSinogramCentered(template, angle, img_h, img_w):
    """
    回転テンプレートを画像中央に配置したサイノグラムを計算。
    位置のオフセットは shiftSinogramPosition で後から適用する。

    :param template: テンプレート画像 (グレースケール)
    :param angle: 回転角度（度）
    :param img_h: ターゲット画像の高さ
    :param img_w: ターゲット画像の幅
    :return: (360, L) float32サイノグラム
    """
    th, tw = template.shape
    M = cv2.getRotationMatrix2D((tw // 2, th // 2), -angle, 1.0)
    tmpl_rot = cv2.warpAffine(template, M, (tw, th),
                               borderMode=cv2.BORDER_REPLICATE)
    canvas = np.zeros((img_h, img_w), dtype=np.uint8)
    cy, cx = img_h // 2, img_w // 2
    canvas[cy - th // 2:cy - th // 2 + th,
           cx - tw // 2:cx - tw // 2 + tw] = tmpl_rot
    return radonTransformFloat(canvas)


def shiftSinogramPosition(sinogram, dx, dy):
    """
    サイノグラムに平行移動 (dx, dy) を適用する。
    各行θをt方向に offset = dx·cos(θ) - dy·sin(θ) だけシフト（線形補間）。

    :param sinogram: (360, L) サイノグラム
    :param dx: x方向オフセット（画像中心基準）
    :param dy: y方向オフセット（画像中心基準）
    :return: シフト済みサイノグラム (360, L)
    """
    n_angles, L = sinogram.shape
    shifted = np.zeros((n_angles, L), dtype=np.float64)
    t_indices = np.arange(L, dtype=np.float64)
    for theta in range(n_angles):
        theta_rad = np.deg2rad(theta)
        offset = dx * np.cos(theta_rad) - dy * np.sin(theta_rad)
        src = t_indices - offset
        shifted[theta] = np.interp(src, t_indices, sinogram[theta].astype(np.float64),
                                   left=0.0, right=0.0)
    return shifted


def detectByHFEnergy(sinogram_image, template, img_h, img_w,
                     pos_step_coarse=2):
    """
    HFエネルギー最小化のみでテンプレートの姿勢 (angle, dx, dy) を検出する。
    ハフ投票には依存しない。

    最適化: 回転はラドン変換で計算（角度ごとに1回）、
    平行移動はサイノグラムのシフト演算で高速に評価。

    交互最適化による探索:
      Phase 1: 角度全探索 0-359°（dx=0, dy=0 固定、各角度でラドン変換1回）
      Phase 2: (dx, dy) 2Dスイープ（角度固定、シフト演算のみ）
      Phase 3: 角度を再探索（正しい位置でシフト評価）
      Phase 4: (dx, dy) を再探索（修正角度でシフト評価）
      Phase 5: 局所精密探索

    :param sinogram_image: ターゲット画像のサイノグラム
    :param template: テンプレート画像（元画像）
    :param img_h: ターゲット画像の高さ
    :param img_w: ターゲット画像の幅
    :param pos_step_coarse: Phase 2/4 の位置探索ステップ（ピクセル）
    :return: (best_angle, best_dx, best_dy, best_hf_energy)
    """
    th, tw = template.shape
    sino_img = sinogram_image.astype(np.float64)

    # 位置探索範囲の算出（テンプレートが画像内に収まる最大範囲）
    max_dx = (img_w - tw) // 2 - 2
    max_dy = (img_h - th) // 2 - 2

    def eval_hf_with_shift(sino_centered, angle, dx, dy):
        """中央配置サイノグラムを位置シフトしてHFエネルギーを計算"""
        if not isTemplateInside(angle, dx, dy, tw, th, img_h, img_w):
            return np.inf
        sino_shifted = shiftSinogramPosition(sino_centered, dx, dy)
        residual = sino_img - sino_shifted.astype(np.float64)
        return highpassEnergy(residual)

    # 全角度の中央配置サイノグラムを事前計算（ラドン変換360回）
    sino_per_angle = {}

    def get_sino_centered(angle):
        """角度ごとの中央配置サイノグラムを取得（キャッシュ付き）"""
        if angle not in sino_per_angle:
            sino_per_angle[angle] = makeTemplSinogramCentered(
                template, angle, img_h, img_w)
        return sino_per_angle[angle]

    def sweep_angle(dx, dy):
        """全角度 0-359° をスイープ（各角度のサイノグラムを位置シフト）"""
        ba, be = 0, np.inf
        for a in range(360):
            sc = get_sino_centered(a)
            e = eval_hf_with_shift(sc, a, dx, dy)
            if e < be:
                be = e
                ba = a
        return ba, be

    def sweep_position(angle, step):
        """(dx, dy) をシフト演算で2Dスイープ（ラドン変換なし）"""
        sc = get_sino_centered(angle)
        bd, bdy, be = 0, 0, np.inf
        for dx in range(-max_dx, max_dx + 1, step):
            for dy in range(-max_dy, max_dy + 1, step):
                e = eval_hf_with_shift(sc, angle, dx, dy)
                if e < be:
                    be = e
                    bd = dx
                    bdy = dy
        return bd, bdy, be

    # Phase 1: 角度全探索（位置は中央固定）
    best_a, best_e = sweep_angle(0, 0)

    # Phase 2: 位置スイープ（シフト演算のみ、ラドン変換なし）
    best_dx, best_dy, best_e = sweep_position(best_a, pos_step_coarse)

    # Phase 3: 角度を再探索（正しい位置でシフト評価）
    best_a, best_e = sweep_angle(best_dx, best_dy)

    # Phase 4: 位置を再探索（修正された角度で）
    best_dx, best_dy, best_e = sweep_position(best_a, pos_step_coarse)

    # Phase 5: 局所精密探索（±3度, ±pos_step_coarseピクセル）
    fine_a, fine_dx, fine_dy = best_a, best_dx, best_dy
    for a in range(best_a - 3, best_a + 4):
        a_mod = a % 360
        sc = get_sino_centered(a_mod)
        for dx in range(best_dx - pos_step_coarse, best_dx + pos_step_coarse + 1):
            for dy in range(best_dy - pos_step_coarse, best_dy + pos_step_coarse + 1):
                e = eval_hf_with_shift(sc, a_mod, dx, dy)
                if e < best_e:
                    best_e = e
                    fine_a = a_mod
                    fine_dx = dx
                    fine_dy = dy

    return fine_a, fine_dx, fine_dy, best_e


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
    残差サイノグラムのHFエネルギー最小化による回転不変テンプレートマッチング

    処理フロー:
      1. 適応的コントラスト正規化（コントラスト比 < 0.6 の場合）
      2. ターゲット画像のサイノグラム計算
      3. HFエネルギー最小化で姿勢 (angle, dx, dy) を検出
         Phase 1: 角度全探索 0-359°
         Phase 2: 位置2D探索
         Phase 3: 局所精密探索
      4. 結果の可視化

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

    # Step 2: ターゲット画像のサイノグラム計算
    sinogram_image = radonTransformFloat(image_proc)

    # Step 3: HFエネルギー最小化による姿勢検出
    target_angle, dx, dy, hf_energy = detectByHFEnergy(
        sinogram_image, template, img_h, img_w)

    print("detected: angle =", target_angle, ", dx =", dx, ", dy =", dy)

    # Step 4: 可視化
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
