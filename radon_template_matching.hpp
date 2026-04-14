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

// =====================================================================
// NCC-HF detection (sinogram-space only, no image-space NCC needed)
// =====================================================================

// Precomputed data for NCC-HF detection
struct NCCHFData {
    int L;                                     // sinogram row length
    std::vector<cv::Mat> img_hf_fft;           // HPF'd image row FFTs (360 x L, complex)
    std::vector<std::vector<double>> img_cumsum;     // cumsum of HPF'd rows (360 x L+1)
    std::vector<std::vector<double>> img_cumsum_sq;  // cumsum of squared HPF'd rows
    std::vector<cv::Mat> core_hf_fft;          // HPF'd core FFTs padded to L (180, complex)
    std::vector<double> core_hf_mean;          // mean of HPF'd core (180)
    std::vector<double> core_hf_energy;        // Σ(core_hf - mean)² (180)
    std::vector<int> nc_list;                  // core lengths (180)
    std::vector<bool> valid;                   // whether core j is usable (180)
};

// Precompute FFTs, running sums, and core stats for NCC-HF
NCCHFData precomputeNCCHFData(const cv::Mat &sinogram_image,
                               const std::vector<cv::Mat> &cores,
                               double cutoff_ratio = 1.0 / 16);

// Find best (dx, dy) for a given angle using NCC-HF profiles
struct NCCHFResult {
    int dx, dy;
    double score;
};

NCCHFResult findPositionByNCCHF(const NCCHFData &data,
                                 int alpha, int center_t,
                                 const std::vector<double> &cos_t,
                                 const std::vector<double> &sin_t,
                                 int max_dx, int max_dy);

// Full NCC-HF detection: coarse angle search + fine refinement
DetectionResult detectByNCCHF(const cv::Mat &sinogram_image,
                               const std::vector<cv::Mat> &cores,
                               int n_img,
                               int template_height, int template_width,
                               int img_height, int img_width,
                               int angle_step_coarse = 3);

#endif // RADON_TEMPLATE_MATCHING_HPP
