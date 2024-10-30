#include "radon_template_matching.hpp"
#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

int main() {
    string template_image_path = "../figs/template.jpg";
    Mat template_image = imread(template_image_path, IMREAD_GRAYSCALE);

    string target_image_path = "../figs/target.jpg";
    Mat target_image = imread(target_image_path, IMREAD_GRAYSCALE);

    if (template_image.empty() || target_image.empty()) {
        cerr << "Error loading images!" << endl;
        return -1;
    }

    matchTemplateRotatable(target_image, template_image);

    return 0;
}
