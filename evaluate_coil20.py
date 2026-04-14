"""
COIL-20 Benchmark Evaluation for Radon Template Matching
COIL-20データセットを用いたラドンテンプレートマッチングのベンチマーク評価

Usage:
  python evaluate_coil20.py              # NCC-HF (default)
  python evaluate_coil20.py --method ncchf
  python evaluate_coil20.py --method hough
"""

import matplotlib
matplotlib.use('Agg')

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import time
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed

from radon_template_matching import (
    radonTransformFloat, extractSinogramCore,
    detectAngleHough,
    detectByNCCHF,
)


# グローバル変数で手法を制御 (ProcessPoolExecutor のサブプロセスに渡すため)
_METHOD = 'ncchf'


def create_test_image(template, true_angle, border_ratio=0.5):
    """Create test image: black background with rotated template centered."""
    th, tw = template.shape
    frame_h = int(th * (1 + border_ratio * 2))
    frame_w = int(tw * (1 + border_ratio * 2))
    image = np.zeros((frame_h, frame_w), dtype=np.uint8)

    center_t = (tw // 2, th // 2)
    M = cv2.getRotationMatrix2D(center_t, true_angle, 1.0)
    rotated = cv2.warpAffine(template, M, (tw, th),
                              borderMode=cv2.BORDER_REFLECT_101)

    y0 = (frame_h - th) // 2
    x0 = (frame_w - tw) // 2
    image[y0:y0+th, x0:x0+tw] = rotated
    return image


def adaptive_contrast_norm(image, template):
    """適応的コントラスト正規化"""
    cr = np.std(image.astype(np.float32)) / (np.std(template.astype(np.float32)) + 1e-10)
    if cr < 0.6:
        im_m = np.mean(image.astype(np.float32))
        im_s = np.std(image.astype(np.float32))
        tm_m = np.mean(template.astype(np.float32))
        tm_s = np.std(template.astype(np.float32))
        return np.clip(
            (image.astype(np.float32) - im_m) / (im_s + 1e-10) * tm_s + tm_m,
            0, 255).astype(np.uint8)
    return image


def apply_noise(image, noise_name, noise_params):
    """ノイズ適用"""
    if noise_name == 'Gaussian':
        noise = np.random.normal(0, noise_params, image.shape)
        return np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    elif noise_name == 'Brightness':
        return np.clip(image.astype(np.float32) + noise_params, 0, 255).astype(np.uint8)
    elif noise_name == 'Contrast':
        mean_val = np.mean(image)
        return np.clip((image.astype(np.float32) - mean_val) * noise_params + mean_val,
                       0, 255).astype(np.uint8)
    return image


def angle_error(detected, true_angle):
    """360度考慮の角度誤差"""
    return min(abs(detected - true_angle),
               abs(detected - true_angle + 360),
               abs(detected - true_angle - 360))


def detect_hough(image_proc, template):
    """Hough Voting による角度検出"""
    th, tw = template.shape
    sino_img = radonTransformFloat(image_proc)
    sino_tmpl = radonTransformFloat(template)
    detect_angle, _, _, _ = detectAngleHough(sino_img, sino_tmpl, th, tw)
    # 180度曖昧性 → 2候補返す
    return detect_angle, detect_angle + 180


def detect_ncchf(image_proc, template):
    """NCC-HF による角度+位置検出"""
    th, tw = template.shape
    img_h, img_w = image_proc.shape

    x = np.linspace(-1, 1, tw)
    y = np.linspace(-1, 1, th)
    xx, yy = np.meshgrid(x, y)
    gw = np.exp(-(xx**2 + yy**2) / 2.0)
    tmpl_windowed = (template.astype(np.float32) * gw).astype(np.uint8)

    cm = int((np.mean(image_proc[0, :]) + np.mean(image_proc[-1, :]) +
              np.mean(image_proc[:, 0]) + np.mean(image_proc[:, -1])) / 4.0)
    tmpl_canvas = cv2.copyMakeBorder(
        tmpl_windowed,
        (img_h - th) // 2, img_h - th - (img_h - th) // 2,
        (img_w - tw) // 2, img_w - tw - (img_w - tw) // 2,
        cv2.BORDER_CONSTANT, value=cm)

    sino_img = radonTransformFloat(image_proc)
    sino_tmpl = radonTransformFloat(tmpl_canvas)
    cores = extractSinogramCore(sino_tmpl, th, tw)
    n_img = sino_tmpl.shape[1]

    ncc_angle, _, _, _ = detectByNCCHF(
        sino_img, cores, n_img, th, tw, img_h, img_w, verbose=False)
    return (ncc_angle,)


def evaluate_single(args):
    """Evaluate a single (object, angle, noise) combination."""
    obj_path, true_angle, noise_name, noise_params = args

    template = cv2.imread(obj_path, cv2.IMREAD_GRAYSCALE)
    if template is None:
        return None

    test_image = create_test_image(template, true_angle)
    if noise_name != 'Clean':
        test_image = apply_noise(test_image, noise_name, noise_params)
    image_proc = adaptive_contrast_norm(test_image, template)

    # 手法選択
    if _METHOD == 'hough':
        candidates = detect_hough(image_proc, template)
    else:
        candidates = detect_ncchf(image_proc, template)

    # 候補から最小誤差を選択
    best_err = 360
    best_angle = candidates[0]
    for cand in candidates:
        err = angle_error(cand, true_angle)
        if err < best_err:
            best_err = err
            best_angle = cand

    return {
        'obj': os.path.basename(obj_path),
        'true_angle': true_angle,
        'detected': best_angle,
        'error': best_err,
        'noise': noise_name,
    }


def ensure_coil20_dataset(dataset_dir="datasets/coil-20/coil-20-proc"):
    """Download and extract COIL-20 dataset if not present."""
    if os.path.isdir(dataset_dir) and len(glob.glob(os.path.join(dataset_dir, "*.png"))) > 0:
        return dataset_dir

    import urllib.request
    import zipfile

    url = "https://www.cs.columbia.edu/CAVE/databases/SLAM_coil-20_coil-100/coil-20/coil-20-proc.zip"
    zip_path = os.path.join("datasets", "coil-20-proc.zip")
    extract_dir = os.path.join("datasets", "coil-20")

    os.makedirs(extract_dir, exist_ok=True)

    print(f"Downloading COIL-20 dataset from {url} ...")
    urllib.request.urlretrieve(url, zip_path)
    print(f"Extracting to {extract_dir} ...")
    with zipfile.ZipFile(zip_path, 'r') as z:
        z.extractall(extract_dir)
    os.remove(zip_path)
    print("COIL-20 dataset ready.")
    return dataset_dir


def init_worker(method):
    """ProcessPoolExecutor のワーカー初期化"""
    global _METHOD
    _METHOD = method


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='COIL-20 Benchmark')
    parser.add_argument('--method', choices=['ncchf', 'hough'], default='ncchf',
                        help='Detection method (default: ncchf)')
    args = parser.parse_args()
    method = args.method

    method_label = 'NCC-HF' if method == 'ncchf' else 'Hough Voting'
    print("=" * 70)
    print(f"COIL-20 Benchmark: {method_label}")
    print("=" * 70)

    dataset_dir = ensure_coil20_dataset()

    files = sorted(glob.glob(os.path.join(dataset_dir, "obj*__0.png")))
    print(f"Found {len(files)} objects")

    test_angles = [0, 10, 20, 30, 45, 60, 90, 120, 150, 170]

    noise_configs = [
        ('Clean', None),
        ('Gaussian', 25),
        ('Brightness', 50),
        ('Contrast', 0.5),
    ]

    tasks = []
    for obj_path in files:
        for true_angle in test_angles:
            for noise_name, noise_params in noise_configs:
                tasks.append((obj_path, true_angle, noise_name, noise_params))

    n_total = len(tasks)
    print(f"Total tests: {n_total} ({len(files)} objects x {len(test_angles)} angles x {len(noise_configs)} conditions)")

    results = []
    t0 = time.time()
    n_workers = min(os.cpu_count() or 4, 8)
    print(f"Using {n_workers} workers")

    with ProcessPoolExecutor(max_workers=n_workers,
                             initializer=init_worker,
                             initargs=(method,)) as executor:
        futures = {executor.submit(evaluate_single, task): task for task in tasks}
        done_count = 0
        for future in as_completed(futures):
            res = future.result()
            if res is not None:
                results.append(res)
            done_count += 1
            if done_count % 40 == 0:
                elapsed = time.time() - t0
                eta = elapsed / done_count * (n_total - done_count)
                print(f"  Progress: {done_count}/{n_total} ({elapsed:.0f}s elapsed, ~{eta:.0f}s remaining)")

    total_time = time.time() - t0
    print(f"\nCompleted {len(results)} tests in {total_time:.1f}s")

    # === 結果分析 ===
    print("\n" + "=" * 70)
    print(f"RESULTS: {method_label}")
    print("=" * 70)

    summary = {}
    for noise_name, _ in noise_configs:
        errs = [r['error'] for r in results if r['noise'] == noise_name]
        if not errs:
            continue
        mean_e = np.mean(errs)
        std_e = np.std(errs)
        median_e = np.median(errs)
        r5 = np.mean([e <= 5 for e in errs]) * 100
        r10 = np.mean([e <= 10 for e in errs]) * 100

        summary[noise_name] = {
            'mean': mean_e, 'std': std_e, 'median': median_e,
            'rate5': r5, 'rate10': r10,
        }

        print(f"\n  {noise_name}:")
        print(f"    Mean error: {mean_e:.1f} +/- {std_e:.1f}  (median: {median_e:.1f})")
        print(f"    Success: <=5={r5:.0f}%  <=10={r10:.0f}%")

    # 角度別
    print(f"\n{'Angle':>6}", end='')
    for noise_name, _ in noise_configs:
        print(f"  {noise_name:>12}", end='')
    print()
    print("-" * 60)
    for angle in test_angles:
        print(f"{angle:>5}d", end='')
        for noise_name, _ in noise_configs:
            ae = [r['error'] for r in results
                  if r['noise'] == noise_name and r['true_angle'] == angle]
            print(f"  {np.mean(ae):>10.1f}d", end='')
        print()

    # サマリーテーブル
    print(f"\n{'Condition':<15} {'Mean':>8} {'Median':>8} {'<=5':>7} {'<=10':>7}")
    print("-" * 50)
    for name in summary:
        s = summary[name]
        print(f"{name:<15} {s['mean']:>6.1f}   {s['median']:>6.1f}   "
              f"{s['rate5']:>5.0f}%  {s['rate10']:>5.0f}%")

    # === プロット ===
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    for noise_name, _ in noise_configs:
        errs_by_angle = []
        for a in test_angles:
            ae = [r['error'] for r in results
                  if r['noise'] == noise_name and r['true_angle'] == a]
            errs_by_angle.append(np.mean(ae) if ae else 0)
        ax.plot(test_angles, errs_by_angle, 'o-', label=noise_name, markersize=4)
    ax.axhline(y=5, color='green', linestyle='--', alpha=0.5, label='5 deg threshold')
    ax.set_xlabel('True rotation angle (degrees)')
    ax.set_ylabel('Mean error (degrees)')
    ax.set_title(f'{method_label}: Error by Rotation Angle')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)

    ax = axes[1]
    noise_names = [n for n, _ in noise_configs]
    x = np.arange(len(noise_names))
    w = 0.3
    ax.bar(x - w/2, [summary.get(n, {}).get('rate5', 0) for n in noise_names],
           w, label='<=5 deg', color='steelblue', alpha=0.8)
    ax.bar(x + w/2, [summary.get(n, {}).get('rate10', 0) for n in noise_names],
           w, label='<=10 deg', color='coral', alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(noise_names, rotation=20, ha='right')
    ax.set_ylabel('Success Rate (%)')
    ax.set_title(f'{method_label}: Success Rate')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    out_path = f"figs/coil20_benchmark_{method}.png"
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to {out_path}")
