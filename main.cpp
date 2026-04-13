#include "radon_template_matching.hpp"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <chrono>

using namespace cv;
using namespace std;

int main() {
    // Load COIL-20 object as template
    string tmpl_path = "../datasets/coil-20/coil-20-proc/obj3__0.png";
    Mat templ = imread(tmpl_path, IMREAD_GRAYSCALE);
    if (templ.empty()) {
        cerr << "Error: cannot load " << tmpl_path << endl;
        cerr << "Run evaluate_coil20.py first to download the dataset." << endl;
        return -1;
    }

    int th = templ.rows, tw = templ.cols;
    int true_angle = 30;

    // Create test image: rotated template on black background (2x size)
    int fh = th * 2, fw = tw * 2;
    Mat image = Mat::zeros(fh, fw, CV_8UC1);
    Mat M = getRotationMatrix2D(Point2f(tw / 2.0f, th / 2.0f), true_angle, 1.0);
    Mat rotated;
    warpAffine(templ, rotated, M, Size(tw, th), INTER_LINEAR, BORDER_REFLECT_101);
    int y0 = (fh - th) / 2, x0 = (fw - tw) / 2;
    rotated.copyTo(image(Rect(x0, y0, tw, th)));

    cout << "Template: " << tw << "x" << th
         << ", Image: " << fw << "x" << fh
         << ", True angle: " << true_angle << " deg" << endl;
    cout << endl;

    // =================================================================
    // Method 1: Brute-force 2D NCC (360 rotations x 2D matchTemplate)
    // =================================================================
    {
        auto t0 = chrono::high_resolution_clock::now();
        auto result = detectAngleBruteForce(image, templ);
        auto t1 = chrono::high_resolution_clock::now();
        double ms = chrono::duration<double, milli>(t1 - t0).count();

        int err = min({abs(result.angle - true_angle),
                       abs(result.angle - true_angle + 360),
                       abs(result.angle - true_angle - 360)});

        cout << "[Brute-force 2D NCC]" << endl;
        cout << "  Time:     " << ms << " ms" << endl;
        cout << "  Detected: " << result.angle << " deg (error: " << err << " deg)" << endl;
        cout << "  Position: (" << result.position.x << ", " << result.position.y << ")" << endl;
        cout << endl;
    }

    // =================================================================
    // Method 2: Hough voting (Radon + sinogram core matching)
    // =================================================================
    {
        // Radon transforms
        auto t0 = chrono::high_resolution_clock::now();
        Mat sino_img = radonTransformFloat(image);
        Mat sino_tmpl = radonTransformFloat(templ);
        auto t_radon = chrono::high_resolution_clock::now();
        double ms_radon = chrono::duration<double, milli>(t_radon - t0).count();

        // Hough voting
        auto result = detectAngleHough(sino_img, sino_tmpl, th, tw);
        auto t1 = chrono::high_resolution_clock::now();
        double ms_hough = chrono::duration<double, milli>(t1 - t_radon).count();
        double ms_total = chrono::duration<double, milli>(t1 - t0).count();

        int err = min({abs(result.angle - true_angle),
                       abs(result.angle - true_angle + 180),
                       abs(result.angle - true_angle - 180)});

        cout << "[Hough voting (Radon)]" << endl;
        cout << "  Radon:    " << ms_radon << " ms" << endl;
        cout << "  Hough:    " << ms_hough << " ms" << endl;
        cout << "  Total:    " << ms_total << " ms" << endl;
        cout << "  Detected: " << result.angle << " deg (error: " << err << " deg)" << endl;
        cout << "  Position: dx=" << result.dx << ", dy=" << result.dy << endl;
        cout << endl;
    }

    // =================================================================
    // Benchmark: multiple sizes
    // =================================================================
    cout << "=== Scaling Benchmark ===" << endl;
    cout << "  Size       Brute(ms)   Hough(ms)   Ratio   Brute_err  Hough_err" << endl;
    cout << "  ---------------------------------------------------------------" << endl;

    for (int size : {64, 128, 200, 256}) {
        Mat t_resized, i_resized;
        resize(templ, t_resized, Size(size, size));
        int th2 = t_resized.rows, tw2 = t_resized.cols;
        int fh2 = th2 * 2, fw2 = tw2 * 2;

        Mat img2 = Mat::zeros(fh2, fw2, CV_8UC1);
        Mat M2 = getRotationMatrix2D(Point2f(tw2 / 2.0f, th2 / 2.0f), true_angle, 1.0);
        Mat rot2;
        warpAffine(t_resized, rot2, M2, Size(tw2, th2), INTER_LINEAR, BORDER_REFLECT_101);
        rot2.copyTo(img2(Rect((fw2 - tw2) / 2, (fh2 - th2) / 2, tw2, th2)));

        // Brute-force
        auto t0 = chrono::high_resolution_clock::now();
        auto bf = detectAngleBruteForce(img2, t_resized);
        auto t1 = chrono::high_resolution_clock::now();
        double ms_bf = chrono::duration<double, milli>(t1 - t0).count();
        int err_bf = min({abs(bf.angle - true_angle),
                          abs(bf.angle - true_angle + 360),
                          abs(bf.angle - true_angle - 360)});

        // Hough
        t0 = chrono::high_resolution_clock::now();
        Mat si = radonTransformFloat(img2);
        Mat st = radonTransformFloat(t_resized);
        auto hr = detectAngleHough(si, st, th2, tw2);
        t1 = chrono::high_resolution_clock::now();
        double ms_h = chrono::duration<double, milli>(t1 - t0).count();
        int err_h = min({abs(hr.angle - true_angle),
                         abs(hr.angle - true_angle + 180),
                         abs(hr.angle - true_angle - 180)});

        double ratio = ms_bf / ms_h;
        printf("  %3dx%-3d  %8.1f    %8.1f    %5.2fx     %2d          %2d\n",
               size, size, ms_bf, ms_h, ratio, err_bf, err_h);
    }

    return 0;
}
