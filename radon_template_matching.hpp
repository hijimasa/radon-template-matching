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

#endif // RADON_TEMPLATE_MATCHING_HPP
