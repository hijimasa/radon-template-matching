"""
Multi-dataset Evaluation for Radon Hough Voting Template Matching
複数データセットでのラドンハフ投票テンプレートマッチングの評価

Datasets:
  1. COIL-20: 20 grayscale objects, 128x128 (auto-download)
  2. MPEG-7 CE-Shape-1: 70 binary shape classes, varied sizes (auto-download)
  3. Synthetic: Gaussian noise / texture backgrounds for robustness testing
"""

import matplotlib
matplotlib.use('Agg')

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import time
import urllib.request
import zipfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from PIL import Image

from radon_template_matching import radonTransformFloat, detectAngleHough, detectPosition


# =============================================================================
# Dataset Download
# =============================================================================

def ensure_coil20(base="datasets"):
    path = os.path.join(base, "coil-20", "coil-20-proc")
    if os.path.isdir(path) and len(glob.glob(os.path.join(path, "*.png"))) > 0:
        return path
    url = "https://www.cs.columbia.edu/CAVE/databases/SLAM_coil-20_coil-100/coil-20/coil-20-proc.zip"
    zp = os.path.join(base, "coil-20-proc.zip")
    os.makedirs(os.path.join(base, "coil-20"), exist_ok=True)
    print(f"  Downloading COIL-20...")
    urllib.request.urlretrieve(url, zp)
    with zipfile.ZipFile(zp, 'r') as z:
        z.extractall(os.path.join(base, "coil-20"))
    os.remove(zp)
    return path


def ensure_mpeg7(base="datasets"):
    path = os.path.join(base, "mpeg7", "original")
    if os.path.isdir(path) and len(glob.glob(os.path.join(path, "*.gif"))) > 0:
        return path
    url = "https://dabi.temple.edu/external/shape/MPEG7/MPEG7dataset.zip"
    zp = os.path.join(base, "mpeg7.zip")
    os.makedirs(os.path.join(base, "mpeg7"), exist_ok=True)
    print(f"  Downloading MPEG-7...")
    urllib.request.urlretrieve(url, zp)
    with zipfile.ZipFile(zp, 'r') as z:
        z.extractall(os.path.join(base, "mpeg7"))
    os.remove(zp)
    return path


# =============================================================================
# Test Image Creation
# =============================================================================

def create_test_image(template, true_angle, border_ratio=0.5, noise_func=None):
    """Create test: rotated template on black background with optional noise."""
    th, tw = template.shape
    fh = int(th * (1 + border_ratio * 2))
    fw = int(tw * (1 + border_ratio * 2))
    image = np.zeros((fh, fw), dtype=np.uint8)
    M = cv2.getRotationMatrix2D((tw // 2, th // 2), true_angle, 1.0)
    rotated = cv2.warpAffine(template, M, (tw, th), borderMode=cv2.BORDER_REFLECT_101)
    y0 = (fh - th) // 2
    x0 = (fw - tw) // 2
    image[y0:y0 + th, x0:x0 + tw] = rotated
    if noise_func:
        image = noise_func(image)
    return image


# =============================================================================
# Single Test Evaluation (for parallel execution)
# =============================================================================

def evaluate_single(args):
    """Evaluate one (template_path, angle, noise) combination."""
    tmpl_path, true_angle, noise_name, is_gif, target_size = args

    if is_gif:
        template = np.array(Image.open(tmpl_path).convert('L'))
    else:
        template = cv2.imread(tmpl_path, cv2.IMREAD_GRAYSCALE)

    if template is None:
        return None

    # Resize to target size
    if max(template.shape) > target_size:
        scale = target_size / max(template.shape)
        template = cv2.resize(template, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    elif max(template.shape) < target_size // 2:
        scale = (target_size // 2) / max(template.shape)
        template = cv2.resize(template, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

    # Noise function
    noise_func = None
    if noise_name == 'Gaussian':
        noise_func = lambda img: np.clip(
            img.astype(np.float32) + np.random.normal(0, 25, img.shape), 0, 255).astype(np.uint8)
    elif noise_name == 'Contrast':
        def contrast_noise(img):
            m = np.mean(img)
            return np.clip((img.astype(np.float32) - m) * 0.5 + m, 0, 255).astype(np.uint8)
        noise_func = contrast_noise

    image = create_test_image(template, true_angle, noise_func=noise_func)

    # Adaptive contrast normalization
    cr = np.std(image.astype(np.float32)) / (np.std(template.astype(np.float32)) + 1e-10)
    if cr < 0.6:
        im_m, im_s = np.mean(image.astype(np.float32)), np.std(image.astype(np.float32))
        tm_m, tm_s = np.mean(template.astype(np.float32)), np.std(template.astype(np.float32))
        image = np.clip(
            (image.astype(np.float32) - im_m) / (im_s + 1e-10) * tm_s + tm_m,
            0, 255).astype(np.uint8)

    th, tw = template.shape
    sino_img = radonTransformFloat(image)
    sino_tmpl = radonTransformFloat(template)
    det_a, dx, dy, _ = detectAngleHough(sino_img, sino_tmpl, th, tw)

    # 180-deg ambiguity: pick closer
    err1 = min(abs(det_a - true_angle), abs(det_a - true_angle + 360),
               abs(det_a - true_angle - 360))
    err2 = min(abs(det_a + 180 - true_angle), abs(det_a + 180 - true_angle + 360),
               abs(det_a + 180 - true_angle - 360))
    error = min(err1, err2)

    return {
        'file': os.path.basename(tmpl_path),
        'angle': true_angle,
        'detected': det_a if err1 <= err2 else det_a + 180,
        'error': error,
        'noise': noise_name
    }


# =============================================================================
# Dataset Evaluation
# =============================================================================

def evaluate_dataset(name, files, is_gif, target_size, test_angles, noise_configs, max_objects=None):
    """Run parallel evaluation on a dataset."""
    if max_objects:
        files = files[:max_objects]

    tasks = []
    for f in files:
        for angle in test_angles:
            for noise_name in noise_configs:
                tasks.append((f, angle, noise_name, is_gif, target_size))

    print(f"  {len(files)} templates × {len(test_angles)} angles × "
          f"{len(noise_configs)} conditions = {len(tasks)} tests")

    results = []
    t0 = time.time()
    n_workers = min(os.cpu_count() or 4, 8)

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(evaluate_single, t): t for t in tasks}
        done = 0
        for future in as_completed(futures):
            res = future.result()
            if res:
                results.append(res)
            done += 1
            if done % 50 == 0:
                elapsed = time.time() - t0
                eta = elapsed / done * (len(tasks) - done)
                print(f"    {done}/{len(tasks)} ({elapsed:.0f}s, ~{eta:.0f}s remaining)")

    elapsed = time.time() - t0
    print(f"  Completed in {elapsed:.1f}s")
    return results


def summarize(results, noise_configs, label):
    """Print summary table for one dataset."""
    print(f"\n  {'Condition':<15} {'Mean':>6} {'Median':>7} {'≤2°':>6} {'≤5°':>6} {'≤10°':>6}")
    print(f"  {'-'*50}")
    summary = {}
    for noise in noise_configs:
        errs = [r['error'] for r in results if r['noise'] == noise]
        if not errs:
            continue
        m = np.mean(errs)
        med = np.median(errs)
        r2 = sum(e <= 2 for e in errs) / len(errs) * 100
        r5 = sum(e <= 5 for e in errs) / len(errs) * 100
        r10 = sum(e <= 10 for e in errs) / len(errs) * 100
        summary[noise] = {'mean': m, 'median': med, 'r2': r2, 'r5': r5, 'r10': r10}
        print(f"  {noise:<15} {m:>5.1f}° {med:>5.0f}°  {r2:>4.0f}%  {r5:>4.0f}%  {r10:>4.0f}%")
    return summary


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("Multi-Dataset Benchmark: Radon Hough Voting Template Matching")
    print("=" * 70)

    test_angles = [0, 10, 20, 30, 45, 60, 90, 120, 150, 170]
    noise_configs = ['Clean', 'Gaussian', 'Contrast']

    all_summaries = {}

    # --- COIL-20 ---
    print("\n" + "=" * 70)
    print("[1/3] COIL-20 (20 grayscale objects, 128x128)")
    print("=" * 70)
    coil_path = ensure_coil20()
    coil_files = sorted(glob.glob(os.path.join(coil_path, "obj*__0.png")))
    coil_results = evaluate_dataset(
        "COIL-20", coil_files, is_gif=False, target_size=128,
        test_angles=test_angles, noise_configs=noise_configs)
    all_summaries['COIL-20'] = summarize(coil_results, noise_configs, "COIL-20")

    # --- MPEG-7 ---
    print("\n" + "=" * 70)
    print("[2/3] MPEG-7 CE-Shape-1 (70 binary shape classes)")
    print("=" * 70)
    mpeg7_path = ensure_mpeg7()
    mpeg7_all = sorted(glob.glob(os.path.join(mpeg7_path, "*.gif")))
    # Use first image per class as template
    seen_classes = set()
    mpeg7_files = []
    for f in mpeg7_all:
        cls = os.path.basename(f).rsplit('-', 1)[0]
        if cls not in seen_classes:
            seen_classes.add(cls)
            mpeg7_files.append(f)
    print(f"  {len(mpeg7_files)} classes selected (1 per class)")
    mpeg7_results = evaluate_dataset(
        "MPEG-7", mpeg7_files, is_gif=True, target_size=128,
        test_angles=test_angles, noise_configs=noise_configs)
    all_summaries['MPEG-7'] = summarize(mpeg7_results, noise_configs, "MPEG-7")

    # --- Synthetic (texture background) ---
    print("\n" + "=" * 70)
    print("[3/3] Synthetic (COIL-20 objects with Gaussian noise background)")
    print("=" * 70)
    # Create synthetic test: add Gaussian noise to background before placing template
    synth_tasks = []
    for f in coil_files:
        for angle in test_angles:
            synth_tasks.append((f, angle, 'NoisyBG', False, 128))

    def evaluate_noisy_bg(args):
        tmpl_path, true_angle, _, is_gif, target_size = args
        template = cv2.imread(tmpl_path, cv2.IMREAD_GRAYSCALE)
        if template is None:
            return None
        th, tw = template.shape
        fh, fw = th * 2, tw * 2
        # Gaussian noise background instead of black
        image = np.random.normal(50, 25, (fh, fw)).clip(0, 255).astype(np.uint8)
        M = cv2.getRotationMatrix2D((tw // 2, th // 2), true_angle, 1.0)
        rotated = cv2.warpAffine(template, M, (tw, th), borderMode=cv2.BORDER_REFLECT_101)
        y0, x0 = (fh - th) // 2, (fw - tw) // 2
        image[y0:y0 + th, x0:x0 + tw] = rotated

        sino_img = radonTransformFloat(image)
        sino_tmpl = radonTransformFloat(template)
        det_a, dx, dy, _ = detectAngleHough(sino_img, sino_tmpl, th, tw)

        err1 = min(abs(det_a - true_angle), abs(det_a - true_angle + 360),
                   abs(det_a - true_angle - 360))
        err2 = min(abs(det_a + 180 - true_angle), abs(det_a + 180 - true_angle + 360),
                   abs(det_a + 180 - true_angle - 360))
        return {
            'file': os.path.basename(tmpl_path),
            'angle': true_angle,
            'detected': det_a if err1 <= err2 else det_a + 180,
            'error': min(err1, err2),
            'noise': 'NoisyBG'
        }

    synth_results = []
    t0 = time.time()
    n_workers = min(os.cpu_count() or 4, 8)
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(evaluate_noisy_bg, t): t for t in synth_tasks}
        done = 0
        for future in as_completed(futures):
            res = future.result()
            if res:
                synth_results.append(res)
            done += 1
            if done % 50 == 0:
                elapsed = time.time() - t0
                print(f"    {done}/{len(synth_tasks)} ({elapsed:.0f}s)")
    print(f"  Completed in {time.time() - t0:.1f}s")
    all_summaries['Synthetic'] = summarize(synth_results, ['NoisyBG'], "Synthetic")

    # =================================================================
    # Final Summary
    # =================================================================
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    print(f"{'Dataset':<12} {'Condition':<12} {'Mean':>6} {'Median':>7} {'≤2°':>6} {'≤5°':>6} {'≤10°':>6}")
    print("-" * 58)
    for ds_name, summary in all_summaries.items():
        for noise, s in summary.items():
            print(f"{ds_name:<12} {noise:<12} {s['mean']:>5.1f}° {s['median']:>5.0f}°  "
                  f"{s['r2']:>4.0f}%  {s['r5']:>4.0f}%  {s['r10']:>4.0f}%")

    # =================================================================
    # Plot
    # =================================================================
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    for idx, (ds_name, summary) in enumerate(all_summaries.items()):
        ax = axes[idx]
        noises = list(summary.keys())
        x = np.arange(len(noises))
        r2 = [summary[n]['r2'] for n in noises]
        r5 = [summary[n]['r5'] for n in noises]
        r10 = [summary[n]['r10'] for n in noises]
        w = 0.25
        ax.bar(x - w, r2, w, label='≤2°', color='green', alpha=0.7)
        ax.bar(x, r5, w, label='≤5°', color='orange', alpha=0.7)
        ax.bar(x + w, r10, w, label='≤10°', color='steelblue', alpha=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels(noises, rotation=30, ha='right')
        ax.set_ylabel('Success Rate (%)')
        ax.set_title(ds_name)
        ax.set_ylim(0, 105)
        ax.legend(fontsize=8)
        ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig("figs/multi_dataset_benchmark.png", dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to figs/multi_dataset_benchmark.png")
    plt.show()
