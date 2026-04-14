#ifndef RADON_TEMPLATE_MATCHING_HPP
#define RADON_TEMPLATE_MATCHING_HPP

#include <opencv2/opencv.hpp>
#include <vector>

// Apply Gaussian window to an image (reduces crop boundary artifacts)
cv::Mat applyGaussianWindow(const cv::Mat &image, double sigma = 1.0);

// Radon transform (float sinogram, 360 angles)
cv::Mat radonTransformFloat(const cv::Mat &image);

// Extract sinogram core (content-only region per row)
std::vector<cv::Mat> extractSinogramCore(const cv::Mat &sinogram,
                                          int template_height, int template_width);

// Hough voting angle detection result
struct HoughResult {
    int angle;      // detected rotation angle (0-179)
    int dx, dy;     // estimated position offset
    std::vector<double> accumulator;  // 180-element vote scores
};

// Detect rotation angle via Hough voting on sinogram cores
HoughResult detectAngleHough(const cv::Mat &sinogram_image,
                              const cv::Mat &sinogram_template,
                              int template_height, int template_width,
                              float score_threshold = 0.5f);

// Brute-force 2D NCC angle detection (for comparison)
struct BruteForceResult {
    int angle;
    cv::Point position;
    double score;
};

BruteForceResult detectAngleBruteForce(const cv::Mat &image, const cv::Mat &templ);

// =====================================================================
// 2-step detection: HF profile + 2D NCC refinement
// =====================================================================

// Result of 2-step detection
struct DetectionResult {
    int angle;      // detected rotation angle (0-359, CCW positive)
    int dx, dy;     // position offset from image center
    double score;   // NCC score at detected pose
};

// Compute per-position HF energy profile using FFT shift theorem.
// Returns HF energy at each sliding position (length = L - core_len + 1).
std::vector<double> computeHFProfile(const cv::Mat &img_row,
                                      const cv::Mat &core,
                                      double cutoff_ratio = 1.0 / 8);

// Estimate (dx, dy) for a given angle alpha by aggregating HF profiles
// along sinusoidal paths.
cv::Point findPositionByHFProfile(const cv::Mat &sinogram_image,
                                   const std::vector<cv::Mat> &cores,
                                   int alpha, int n_img, int center_t,
                                   const std::vector<double> &cos_t,
                                   const std::vector<double> &sin_t,
                                   int max_dx, int max_dy);

// Refine (angle, dx, dy) in a local neighborhood using 2D NCC
// in image space.
DetectionResult refineByNCC(const cv::Mat &image,
                             const cv::Mat &templ,
                             int coarse_angle, int coarse_dx, int coarse_dy,
                             int angle_range = 3, int pos_range = 3);

// 2-step template matching:
//   Step 1: HF profile sinusoidal path → coarse (angle, dx, dy)
//   Step 2: 2D NCC refinement → precise (angle, dx, dy)
DetectionResult detectByHFAndNCC(const cv::Mat &image,
                                  const cv::Mat &templ,
                                  const cv::Mat &sinogram_image,
                                  const std::vector<cv::Mat> &cores,
                                  int n_img,
                                  int angle_step_coarse = 3);

#endif // RADON_TEMPLATE_MATCHING_HPP
