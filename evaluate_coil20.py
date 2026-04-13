"""
COIL-20 Benchmark Evaluation for Radon Template Matching
COIL-20データセットを用いたラドンテンプレートマッチングのベンチマーク評価

Parallelized version for practical execution time.
"""

import matplotlib
matplotlib.use('Agg')

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

from radon_template_matching import (
    radonTransform, radonTransformFloat, centerPasteImage,
    detectPosition, matchTemplateOneLineNCC,
    detectAngleSinusoidalWarp, detectAngleHough
)


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


def poc_angle_detect(radon_img, radon_tmpl_padded):
    """POC angle detection from precomputed Radon transforms."""
    fft_img = np.array([np.fft.fft(radon_img[i].astype(np.float32))
                         for i in range(360)], dtype=np.complex64)
    fft_tmpl = np.array([np.fft.fft(radon_tmpl_padded[i].astype(np.float32))
                          for i in range(360)], dtype=np.complex64)

    scores = np.zeros(180, dtype=np.float64)
    for alpha in range(180):
        corr_sum = 0.0
        for theta in range(180):
            row1 = fft_img[(theta + alpha) % 360]
            row2 = fft_tmpl[theta]
            cp = row1 * np.conj(row2)
            cp_norm = cp / (np.abs(cp) + 1e-10)
            corr = np.abs(np.fft.ifft(cp_norm))
            corr_sum += np.max(corr)
        scores[alpha] = corr_sum
    return int(np.argmax(scores)), scores


def evaluate_single(args):
    """Evaluate a single (object, angle, noise) combination. Runs in subprocess."""
    obj_path, true_angle, noise_name, noise_params = args

    template = cv2.imread(obj_path, cv2.IMREAD_GRAYSCALE)
    if template is None:
        return None

    # Create test image
    test_image = create_test_image(template, true_angle)

    # Apply noise
    if noise_name == 'Gaussian':
        noise = np.random.normal(0, noise_params, test_image.shape)
        test_image = np.clip(test_image.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    elif noise_name == 'Brightness':
        test_image = np.clip(test_image.astype(np.float32) + noise_params, 0, 255).astype(np.uint8)
    elif noise_name == 'Contrast':
        mean_val = np.mean(test_image)
        test_image = np.clip((test_image.astype(np.float32) - mean_val) * noise_params + mean_val,
                             0, 255).astype(np.uint8)

    # Adaptive contrast normalization
    image_proc = test_image
    cr = np.std(test_image.astype(np.float32)) / (np.std(template.astype(np.float32)) + 1e-10)
    if cr < 0.6:
        im_m = np.mean(test_image.astype(np.float32))
        im_s = np.std(test_image.astype(np.float32))
        tm_m = np.mean(template.astype(np.float32))
        tm_s = np.std(template.astype(np.float32))
        image_proc = np.clip(
            (test_image.astype(np.float32) - im_m) / (im_s + 1e-10) * tm_s + tm_m,
            0, 255).astype(np.uint8)

    # Float sinograms for Hough voting
    sino_img = radonTransformFloat(image_proc)
    sino_tmpl = radonTransformFloat(template)
    th, tw = template.shape

    # Hough voting angle detection with sinogram core extraction
    detect_angle, det_dx, det_dy, _ = detectAngleHough(
        sino_img, sino_tmpl, th, tw)

    # detectAngleHough returns alpha in [0, 180).
    # True angle could be alpha or alpha+180.
    opt1 = detect_angle
    opt2 = detect_angle + 180
    err1 = min(abs(opt1 - true_angle), abs(opt1 - true_angle + 360), abs(opt1 - true_angle - 360))
    err2 = min(abs(opt2 - true_angle), abs(opt2 - true_angle + 360), abs(opt2 - true_angle - 360))
    final_angle = opt1 if err1 <= err2 else opt2

    error = min(abs(final_angle - true_angle),
                abs(final_angle - true_angle + 360),
                abs(final_angle - true_angle - 360))

    return {
        'obj': os.path.basename(obj_path),
        'true_angle': true_angle,
        'detected': final_angle,
        'error': error,
        'noise': noise_name
    }


def ensure_coil20_dataset(dataset_dir="datasets/coil-20/coil-20-proc"):
    """
    Download and extract COIL-20 dataset if not present.
    COIL-20データセットが存在しない場合、自動ダウンロード・展開する。
    """
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


if __name__ == "__main__":
    print("=" * 70)
    print("COIL-20 Benchmark: Radon Template Matching (Hough Voting)")
    print("Parallelized evaluation")
    print("=" * 70)

    dataset_dir = ensure_coil20_dataset()

    # Get all object IDs (use 0-degree pose as template)
    files = sorted(glob.glob(os.path.join(dataset_dir, "obj*__0.png")))
    print(f"Found {len(files)} objects")

    # Test angles
    test_angles = [0, 10, 20, 30, 45, 60, 90, 120, 150, 170]

    # Noise conditions: (name, params)
    noise_configs = [
        ('Clean', None),
        ('Gaussian', 25),
        ('Brightness', 50),
        ('Contrast', 0.5),
    ]

    # Build task list
    tasks = []
    for obj_path in files:
        for true_angle in test_angles:
            for noise_name, noise_params in noise_configs:
                tasks.append((obj_path, true_angle, noise_name, noise_params))

    print(f"Total tests: {len(tasks)} ({len(files)} objects × {len(test_angles)} angles × {len(noise_configs)} conditions)")
    print(f"Running with parallel processes...")

    # Execute in parallel
    results = []
    t0 = time.time()
    n_workers = min(os.cpu_count() or 4, 8)
    print(f"Using {n_workers} workers")

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(evaluate_single, task): task for task in tasks}
        done_count = 0
        for future in as_completed(futures):
            res = future.result()
            if res is not None:
                results.append(res)
            done_count += 1
            if done_count % 20 == 0:
                elapsed = time.time() - t0
                eta = elapsed / done_count * (len(tasks) - done_count)
                print(f"  Progress: {done_count}/{len(tasks)} ({elapsed:.0f}s elapsed, ~{eta:.0f}s remaining)")

    total_time = time.time() - t0
    print(f"\nCompleted {len(results)} tests in {total_time:.1f}s")

    # Analyze results
    print("\n" + "=" * 80)
    print("RESULTS BY NOISE CONDITION")
    print("=" * 80)

    summary = {}
    for noise_name, _ in noise_configs:
        errs = [r['error'] for r in results if r['noise'] == noise_name]
        if len(errs) == 0:
            continue

        mean_e = np.mean(errs)
        std_e = np.std(errs)
        r5 = np.mean([1 if e <= 5 else 0 for e in errs]) * 100
        r10 = np.mean([1 if e <= 10 else 0 for e in errs]) * 100
        r15 = np.mean([1 if e <= 15 else 0 for e in errs]) * 100
        median_e = np.median(errs)

        summary[noise_name] = {
            'mean': mean_e, 'std': std_e, 'median': median_e,
            'rate5': r5, 'rate10': r10, 'rate15': r15
        }

        print(f"\n  {noise_name}:")
        print(f"    Mean error: {mean_e:.1f}° ± {std_e:.1f}°  (median: {median_e:.1f}°)")
        print(f"    Success: ≤5°={r5:.0f}%  ≤10°={r10:.0f}%  ≤15°={r15:.0f}%")

        # Per-angle breakdown
        print(f"    {'Angle':>6}  {'Mean':>6}  {'≤5°':>5}  {'≤10°':>5}")
        for angle in test_angles:
            ae = [r['error'] for r in results if r['noise'] == noise_name and r['true_angle'] == angle]
            if len(ae) > 0:
                print(f"    {angle:>5}°  {np.mean(ae):>5.1f}°  {np.mean([e<=5 for e in ae])*100:>4.0f}%  {np.mean([e<=10 for e in ae])*100:>4.0f}%")

    # Final summary table
    print("\n" + "=" * 80)
    print(f"SUMMARY TABLE (20 objects × {len(test_angles)} angles = {20*len(test_angles)} tests/condition)")
    print("=" * 80)
    print(f"{'Condition':<15} {'Mean':>8} {'Median':>8} {'Std':>8} {'≤5°':>7} {'≤10°':>7} {'≤15°':>7}")
    print("-" * 62)
    for name in summary:
        s = summary[name]
        print(f"{name:<15} {s['mean']:>6.1f}°  {s['median']:>6.1f}°  {s['std']:>6.1f}°  "
              f"{s['rate5']:>5.0f}%  {s['rate10']:>5.0f}%  {s['rate15']:>5.0f}%")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax1 = axes[0]
    for noise_name, _ in noise_configs:
        errs_by_angle = []
        for a in test_angles:
            ae = [r['error'] for r in results if r['noise'] == noise_name and r['true_angle'] == a]
            errs_by_angle.append(np.mean(ae) if ae else 0)
        ax1.plot(test_angles, errs_by_angle, 'o-', label=noise_name, markersize=4)
    ax1.axhline(y=5, color='green', linestyle='--', alpha=0.5)
    ax1.axhline(y=10, color='orange', linestyle='--', alpha=0.5)
    ax1.set_xlabel('True rotation angle (degrees)')
    ax1.set_ylabel('Mean error (degrees)')
    ax1.set_title('COIL-20: Error by Rotation Angle')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    ax2 = axes[1]
    names = list(summary.keys())
    x = np.arange(len(names))
    w = 0.25
    ax2.bar(x - w, [summary[n]['rate5'] for n in names], w, label='≤5°', color='green', alpha=0.7)
    ax2.bar(x, [summary[n]['rate10'] for n in names], w, label='≤10°', color='orange', alpha=0.7)
    ax2.bar(x + w, [summary[n]['rate15'] for n in names], w, label='≤15°', color='steelblue', alpha=0.7)
    ax2.set_xticks(x)
    ax2.set_xticklabels(names, rotation=30, ha='right')
    ax2.set_ylabel('Success Rate (%)')
    ax2.set_title('COIL-20: Success Rate')
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig("figs/coil20_benchmark.png", dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to figs/coil20_benchmark.png")
    plt.show()
