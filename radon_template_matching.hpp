#ifndef RADON_TEMPLATE_MATCHING_HPP
#define RADON_TEMPLATE_MATCHING_HPP

#include <opencv2/opencv.hpp>
#include <vector>

cv::Mat radonTransform(const cv::Mat &image);
cv::Mat radonFFT(const cv::Mat &image);
void matchTemplateRotatable(const cv::Mat &image, const cv::Mat &templ);

#endif // RADON_TEMPLATE_MATCHING_HPP
