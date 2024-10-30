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
Template Matching Processing Body
テンプレートマッチング処理本体
"""
def matchTemplateRotatable(image, template):
    """
    Template matching for rotation using Radon transforms.
    ラドン変換を用いて回転にも対応できるテンプレートマッチングを行う
    :param image: Target image / 対象画像
    :param image: Template image / テンプレート画像
    :return: None / なし
    """
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


    radon_transformed_template = radonTransform(template)
    radon_transformed_template_for_fft = radonTransform((template * gaussian_window).astype(np.uint8))
    
    radon_transformed_image = radonTransform(image)
    radon_transformed_image_for_fft = radonTransform((image * gaussian_window_for_image).astype(np.uint8))
    
    radon_transformed_template_for_fft2 = centerPasteImage(radon_transformed_image, radon_transformed_template_for_fft)

    fft_template = radonFFT(radon_transformed_template_for_fft2)
    fft_template_half = fft_template[0:180,:]

    fft_image = radonFFT(radon_transformed_image_for_fft)

    # Perform matching of FFT results
    # FFT結果のマッチングを実行
    diff_mean_list = []
    for i in range(180):
        diff = fft_image[i:i+180,:] - fft_template_half
        diff_mean_list.append(np.mean(np.abs(diff)))
    min_value = min(diff_mean_list)
    min_index = diff_mean_list.index(min_value)

    detect_angle = min_index
    angle_option1 = 360 - detect_angle
    if angle_option1 >= 360:
        angle_option1 = angle_option1 - 360
    angle_option2 = 360 - detect_angle - 180
    if angle_option2 >= 360:
        angle_option2 = angle_option2 - 360
    template_option1 = radon_transformed_template[angle_option1, :].astype(np.float32)
    template_option2 = radon_transformed_template[angle_option2, :].astype(np.float32)
    target_img = radon_transformed_image[0, :].astype(np.float32)

    angle_option1_90 = 360 - detect_angle + 90
    if angle_option1_90 >= 360:
        angle_option1_90 -= 360
    angle_option2_90 = 360 - detect_angle - 180 + 90
    if angle_option2_90 >= 360:
        angle_option2_90 -= 360
    template_option1_90 = radon_transformed_template[angle_option1_90, :].astype(np.float32)
    template_option2_90 = radon_transformed_template[angle_option2_90, :].astype(np.float32)
    target_img_90 = radon_transformed_image[90, :].astype(np.float32)

    result = matchTemplateOneLine(target_img, template_option1)
    min_val = np.min(result)
    min_val_index = np.argmin(result)

    result = matchTemplateOneLine(target_img, template_option2)
    min_val2 = np.min(result)
    min_val2_index = np.argmin(result)

    result = matchTemplateOneLine(target_img_90, template_option1_90)
    min_val_90 = np.min(result)
    min_val_90_index = np.argmin(result)

    result = matchTemplateOneLine(target_img_90, template_option2_90)
    min_val2_90 = np.min(result)
    min_val2_90_index = np.argmin(result)

    dx_option1 = min_val_index + template_option1.shape[0] // 2 - target_img.shape[0] // 2
    dy_option1 = -(min_val_90_index + template_option1.shape[0] // 2 - target_img.shape[0] // 2)

    dx_option2 = min_val2_index + template_option1.shape[0] // 2 - target_img.shape[0] // 2
    dy_option2 = -(min_val2_90_index + template_option1.shape[0] // 2 - target_img.shape[0] // 2)

    if min_val + min_val_90 < min_val2 + min_val2_90:
        target_angle = detect_angle
        dx = dx_option1
        dy = dy_option1
    else:
        target_angle = detect_angle + 180
        dx = dx_option2
        dy = dy_option2

    print("detected: angle = ", target_angle, ", dx = ", dx, ", dy = ", dy)

    # Template image / テンプレート画像
    plt.subplot(3, 3, 1)
    plt.imshow(template, cmap='gray')
    plt.title('Template')
    plt.colorbar()

    # Radon Transformed template image / テンプレート画像のradon変換結果
    plt.subplot(3, 3, 2)
    plt.imshow(radon_transformed_template, cmap='gray')
    plt.title('Radon Transformed Template')
    plt.colorbar()

    # FFT result of radon Transformed template image / テンプレート画像のradon変換のFFT結果
    plt.subplot(3, 3, 3)
    plt.imshow(fft_template, cmap='gray')
    plt.title('FFT of Radon Transformed Template')
    plt.colorbar()

    # Target image / 対象画像
    plt.subplot(3, 3, 4)
    plt.imshow(image, cmap='gray')
    plt.title('Target')
    plt.colorbar()

    # Radon Transformed target image / 対象画像のradon変換結果
    plt.subplot(3, 3, 5)
    plt.imshow(radon_transformed_image, cmap='gray')
    plt.title('Radon Transformed Target')
    plt.colorbar()

    # FFT result of radon Transformed target image / 対象画像のradon変換のFFT結果
    plt.subplot(3, 3, 6)
    plt.imshow(fft_image, cmap='gray')
    plt.title('FFT of Radon Transformed Target')
    plt.colorbar()

    #draw_color = color=(255, 255, 255)
    draw_color = color=(0, 0, 0)
    result_image = drawRotatedRectangleOnImage(image, (dx + image.shape[1]//2, dy + image.shape[0]//2), template.shape[1], template.shape[0], target_angle, draw_color)

    # Matching result / マッチング結果
    plt.subplot(3, 3, 7)
    plt.imshow(result_image, cmap='gray')
    plt.title('Matched Image')
    plt.colorbar()

    plt.tight_layout()
    plt.show()
    
"""
Sample
使用例
"""
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

