"""
Realistic Evaluation for Radon Template Matching
現実的な画像ペアを用いたラドンテンプレートマッチングの評価

Test scenario:
- Target image: natural image (e.g., dog portrait)
- Template: cropped region from the target
- Evaluation: rotate the target by known angles, detect rotation angle and position
"""

import matplotlib
matplotlib.use('Agg')

import cv2
import numpy as np
import matplotlib.pyplot as plt
import time

from radon_template_matching import (
    radonTransformFloat, detectAnglePOC, detectPosition,
    matchTemplateOneLineNCC, drawRotatedRectangleOnImage
)
from evaluate_noise_robustness import (
    add_salt_pepper_noise, add_gaussian_noise,
    adjust_brightness, adjust_contrast, add_combined_noise
)


def create_realistic_test(target, template, true_angle, true_dx=0, true_dy=0,
                          margin_ratio=0.3):
    """
    Create a realistic test image by:
    1. Adding a reflected border to the template (natural context extension)
    2. Rotating the bordered image by true_angle

    This simulates the practical scenario: detecting the rotation of a known
    object in a slightly larger field of view with natural surrounding context.

    テンプレートに反射パディング（自然な周辺コンテキスト）を追加し、
    回転させた現実的なテスト画像を生成。
    """
    th, tw = template.shape

    # Add reflected border for natural context
    border = int(max(th, tw) * margin_ratio)
    bordered = cv2.copyMakeBorder(template, border, border, border, border,
                                   cv2.BORDER_REFLECT_101)

    # Rotate the bordered image
    bh, bw = bordered.shape
    center = (bw // 2, bh // 2)
    M = cv2.getRotationMatrix2D(center, -true_angle, 1.0)
    rotated = cv2.warpAffine(bordered, M, (bw, bh),
                              borderMode=cv2.BORDER_REFLECT_101)

    return rotated


def evaluate_angle_detection_realistic(image, template, true_angle, methods=None):
    """
    Evaluate angle detection on a realistic image pair.
    現実的な画像ペアで角度検出を評価
    """
    results = {}

    # Gaussian windows
    rows, cols = template.shape
    x = np.linspace(-1, 1, cols)
    y = np.linspace(-1, 1, rows)
    x, y = np.meshgrid(x, y)
    gw_tmpl = np.exp(-(x**2 + y**2) / (2 * 0.3**2))

    rows, cols = image.shape
    x = np.linspace(-1, 1, cols)
    y = np.linspace(-1, 1, rows)
    x, y = np.meshgrid(x, y)
    gw_img = np.exp(-(x**2 + y**2) / (2 * 0.5**2))

    # Float sinograms (windowed for POC, raw for position)
    sinogram_img_w = radonTransformFloat((image * gw_img).astype(np.uint8))
    sinogram_tmpl_w = radonTransformFloat((template * gw_tmpl).astype(np.uint8))
    sinogram_img_raw = radonTransformFloat(image)
    sinogram_tmpl_raw = radonTransformFloat(template)

    # POC angle detection
    t0 = time.time()
    detected_angle, scores = detectAnglePOC(sinogram_img_w, sinogram_tmpl_w)
    poc_time = time.time() - t0

    # Resolve 180° ambiguity
    dx1, dy1, score1 = detectPosition(sinogram_img_raw, sinogram_tmpl_raw, detected_angle)
    dx2, dy2, score2 = detectPosition(sinogram_img_raw, sinogram_tmpl_raw, detected_angle + 180)

    if score1 >= score2:
        final_angle = detected_angle
        dx, dy = dx1, dy1
    else:
        final_angle = detected_angle + 180
        dx, dy = dx2, dy2

    error = min(abs(final_angle - true_angle),
                abs(final_angle - true_angle + 360),
                abs(final_angle - true_angle - 360))

    results['improved_poc'] = {
        'detected_angle': final_angle,
        'error': error,
        'dx': dx, 'dy': dy,
        'time': poc_time
    }

    return results


def run_realistic_evaluation(target, template, angles, noise_configs=None):
    """
    Run evaluation across multiple rotation angles and noise conditions.
    """
    if noise_configs is None:
        noise_configs = {'Clean': None}

    all_results = {}

    for noise_name, noise_func in noise_configs.items():
        print(f"\n{'='*60}")
        print(f"Noise condition: {noise_name}")
        print(f"{'='*60}")

        angle_errors = []
        for true_angle in angles:
            # Create rotated test image
            rotated_target = create_realistic_test(target, template, true_angle)

            # Apply noise
            if noise_func is not None:
                rotated_target = noise_func(rotated_target)

            # Evaluate
            results = evaluate_angle_detection_realistic(
                rotated_target, template, true_angle)

            error = results['improved_poc']['error']
            detected = results['improved_poc']['detected_angle']
            angle_errors.append(error)

            status = "OK" if error <= 5 else ("WARN" if error <= 15 else "FAIL")
            print(f"  angle={true_angle:>4d}° → detected={detected:>4d}° "
                  f"error={error:>5.1f}° [{status}]")

        mean_error = np.mean(angle_errors)
        std_error = np.std(angle_errors)
        success_rate = np.mean([1 if e <= 5 else 0 for e in angle_errors]) * 100

        all_results[noise_name] = {
            'errors': angle_errors,
            'angles': angles,
            'mean_error': mean_error,
            'std_error': std_error,
            'success_rate': success_rate
        }

        print(f"\n  Summary: mean={mean_error:.2f}° ± {std_error:.2f}°, "
              f"success(≤5°)={success_rate:.0f}%")

    return all_results


def plot_realistic_results(all_results, angles, save_path=None):
    """Plot evaluation results."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: Error vs angle for each noise condition
    ax1 = axes[0]
    for noise_name, data in all_results.items():
        ax1.plot(angles, data['errors'], 'o-', label=noise_name, markersize=4)
    ax1.axhline(y=5, color='green', linestyle='--', alpha=0.5, label='5° threshold')
    ax1.set_xlabel('True rotation angle (degrees)')
    ax1.set_ylabel('Angle detection error (degrees)')
    ax1.set_title('Improved POC: Angle Detection Error')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Right: Summary bar chart
    ax2 = axes[1]
    names = list(all_results.keys())
    means = [all_results[n]['mean_error'] for n in names]
    stds = [all_results[n]['std_error'] for n in names]
    rates = [all_results[n]['success_rate'] for n in names]

    x = np.arange(len(names))
    bars = ax2.bar(x, means, yerr=stds, capsize=5, color='steelblue', alpha=0.7)
    ax2.set_xticks(x)
    ax2.set_xticklabels(names, rotation=45, ha='right')
    ax2.set_ylabel('Mean angle error (degrees)')
    ax2.set_title('Mean Error by Noise Condition')
    ax2.grid(axis='y', alpha=0.3)

    # Add success rate labels
    for i, (bar, rate) in enumerate(zip(bars, rates)):
        ax2.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + stds[i] + 1,
                 f'{rate:.0f}%', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\nPlot saved to: {save_path}")
    plt.show()


if __name__ == "__main__":
    print("=" * 60)
    print("Realistic Evaluation: Radon Template Matching")
    print("=" * 60)

    # Load images
    target = cv2.imread("figs/target.jpg", cv2.IMREAD_GRAYSCALE)
    template = cv2.imread("figs/template.jpg", cv2.IMREAD_GRAYSCALE)

    if target is None or template is None:
        print("Error: Could not load images")
        exit(1)

    # Resize for reasonable computation time
    max_tmpl = 128
    if max(template.shape) > max_tmpl:
        scale = max_tmpl / max(template.shape)
        template = cv2.resize(template, None, fx=scale, fy=scale,
                              interpolation=cv2.INTER_AREA)

    max_target = 256
    if max(target.shape) > max_target:
        scale = max_target / max(target.shape)
        target = cv2.resize(target, None, fx=scale, fy=scale,
                            interpolation=cv2.INTER_AREA)

    print(f"Target size: {target.shape}")
    print(f"Template size: {template.shape}")

    # Test angles
    angles = [0, 10, 20, 30, 45, 60, 75, 90, 120, 135, 150, 170]

    # Noise conditions
    noise_configs = {
        'Clean': None,
        'Gaussian σ=25': lambda img: add_gaussian_noise(img, 0, 25),
        'Brightness +50': lambda img: adjust_brightness(img, 50),
        'Contrast 0.5x': lambda img: adjust_contrast(img, 0.5),
        'Combined': lambda img: add_combined_noise(img, 0.02, 15, 25),
    }

    # Run evaluation
    results = run_realistic_evaluation(target, template, angles, noise_configs)

    # Summary table
    print("\n" + "=" * 80)
    print("SUMMARY TABLE")
    print("=" * 80)
    print(f"{'Condition':<20} {'Mean Error':>12} {'Std':>8} {'Success(≤5°)':>14}")
    print("-" * 60)
    for name, data in results.items():
        print(f"{name:<20} {data['mean_error']:>10.2f}° {data['std_error']:>6.2f}° "
              f"{data['success_rate']:>12.0f}%")

    # Plot
    plot_realistic_results(results, angles,
                          save_path="figs/realistic_evaluation.png")

    print("\nEvaluation complete!")
