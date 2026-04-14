#include "radon_template_matching.hpp"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <chrono>

using namespace cv;
using namespace std;

int main() {
    // Load template image
    string tmpl_path = "figs/template.jpg";
    Mat templ_raw = imread(tmpl_path, IMREAD_GRAYSCALE);
    if (templ_raw.empty()) {
        cerr << "Error: cannot load " << tmpl_path << endl;
        return -1;
    }
    // 128x128 にリサイズ (合成テスト用)
    Mat templ;
    resize(templ_raw, templ, Size(128, 128));

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
        Mat templ_windowed = applyGaussianWindow(templ, 1.0);
        Mat sino_tmpl = radonTransformFloat(templ_windowed);
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
    // Method 3: 2-step (HF profile + 2D NCC)
    // =================================================================
    {
        auto t0 = chrono::high_resolution_clock::now();

        // Prepare sinogram and cores
        Mat templ_windowed = applyGaussianWindow(templ, 1.0);
        int pad_top = (fh - th) / 2, pad_bottom = fh - th - pad_top;
        int pad_left = (fw - tw) / 2, pad_right = fw - tw - pad_left;
        Mat tmpl_canvas;
        copyMakeBorder(templ_windowed, tmpl_canvas,
                       pad_top, pad_bottom, pad_left, pad_right,
                       BORDER_CONSTANT, Scalar(0));
        Mat sino_img = radonTransformFloat(image);
        Mat sino_tmpl = radonTransformFloat(tmpl_canvas);
        auto cores = extractSinogramCore(sino_tmpl, th, tw);
        int n_img = sino_tmpl.cols;

        auto t_prep = chrono::high_resolution_clock::now();
        double ms_prep = chrono::duration<double, milli>(t_prep - t0).count();

        // 2-step detection
        auto result = detectByHFAndNCC(image, templ, sino_img, cores, n_img);
        auto t1 = chrono::high_resolution_clock::now();
        double ms_detect = chrono::duration<double, milli>(t1 - t_prep).count();
        double ms_total = chrono::duration<double, milli>(t1 - t0).count();

        int err = min({abs(result.angle - true_angle),
                       abs(result.angle - true_angle + 360),
                       abs(result.angle - true_angle - 360)});

        cout << "[2-step: HF profile + NCC]" << endl;
        cout << "  Prep:     " << ms_prep << " ms (Radon + cores)" << endl;
        cout << "  Detect:   " << ms_detect << " ms" << endl;
        cout << "  Total:    " << ms_total << " ms" << endl;
        cout << "  Detected: " << result.angle << " deg (error: " << err << " deg)" << endl;
        cout << "  Position: dx=" << result.dx << ", dy=" << result.dy << endl;
        cout << "  NCC:      " << result.score << endl;
        cout << endl;
    }

    // =================================================================
    // Method 4: NCC-HF (sinogram-space only, no image-space NCC)
    // =================================================================
    {
        auto t0 = chrono::high_resolution_clock::now();

        Mat templ_windowed = applyGaussianWindow(templ, 1.0);
        int pad_top = (fh - th) / 2, pad_bottom = fh - th - pad_top;
        int pad_left = (fw - tw) / 2, pad_right = fw - tw - pad_left;

        // corner_pixels_mean of image for canvas fill
        double cm = (mean(image.row(0))[0] + mean(image.row(fh-1))[0] +
                     mean(image.col(0))[0] + mean(image.col(fw-1))[0]) / 4.0;
        Mat tmpl_canvas;
        copyMakeBorder(templ_windowed, tmpl_canvas,
                       pad_top, pad_bottom, pad_left, pad_right,
                       BORDER_CONSTANT, Scalar(cm));
        Mat sino_img = radonTransformFloat(image);
        Mat sino_tmpl = radonTransformFloat(tmpl_canvas);
        auto cores = extractSinogramCore(sino_tmpl, th, tw);
        int n_img = sino_tmpl.cols;

        auto t_prep = chrono::high_resolution_clock::now();
        double ms_prep = chrono::duration<double, milli>(t_prep - t0).count();

        auto result = detectByNCCHF(sino_img, cores, n_img, th, tw, fh, fw);
        auto t1 = chrono::high_resolution_clock::now();
        double ms_detect = chrono::duration<double, milli>(t1 - t_prep).count();
        double ms_total = chrono::duration<double, milli>(t1 - t0).count();

        int err = min({abs(result.angle - true_angle),
                       abs(result.angle - true_angle + 360),
                       abs(result.angle - true_angle - 360)});

        cout << "[NCC-HF (sinogram-space only)]" << endl;
        cout << "  Prep:     " << ms_prep << " ms (Radon + cores)" << endl;
        cout << "  Detect:   " << ms_detect << " ms" << endl;
        cout << "  Total:    " << ms_total << " ms" << endl;
        cout << "  Detected: " << result.angle << " deg (error: " << err << " deg)" << endl;
        cout << "  Position: dx=" << result.dx << ", dy=" << result.dy << endl;
        cout << "  Score:    " << result.score << endl;
        cout << endl;
    }

    // =================================================================
    // Natural image test (figs/target.jpg + figs/template.jpg, 1/4 scale)
    // =================================================================
    {
        Mat nat_img = imread("figs/target.jpg", IMREAD_GRAYSCALE);
        Mat nat_tmpl = imread("figs/template.jpg", IMREAD_GRAYSCALE);
        if (!nat_img.empty() && !nat_tmpl.empty()) {
            resize(nat_img, nat_img, Size(nat_img.cols/4, nat_img.rows/4));
            resize(nat_tmpl, nat_tmpl, Size(nat_tmpl.cols/4, nat_tmpl.rows/4));
            int nth = nat_tmpl.rows, ntw = nat_tmpl.cols;
            int nih = nat_img.rows, niw = nat_img.cols;

            cout << "[Natural image test: " << niw << "x" << nih
                 << " + " << ntw << "x" << nth << "]" << endl;

            // Brute-force NCC
            auto t0 = chrono::high_resolution_clock::now();
            auto bf = detectAngleBruteForce(nat_img, nat_tmpl);
            auto t1 = chrono::high_resolution_clock::now();
            double ms_bf = chrono::duration<double, milli>(t1 - t0).count();
            int bf_dx = bf.position.x + ntw/2 - niw/2;
            int bf_dy = bf.position.y + nth/2 - nih/2;
            cout << "  BF-NCC:  " << ms_bf << " ms, angle=" << bf.angle
                 << ", dx=" << bf_dx << ", dy=" << bf_dy
                 << ", ncc=" << bf.score << endl;

            // NCC-HF
            t0 = chrono::high_resolution_clock::now();
            Mat tw_nat = applyGaussianWindow(nat_tmpl, 1.0);
            double cm2 = (mean(nat_img.row(0))[0] + mean(nat_img.row(nih-1))[0] +
                          mean(nat_img.col(0))[0] + mean(nat_img.col(niw-1))[0]) / 4.0;
            Mat canvas2;
            copyMakeBorder(tw_nat, canvas2,
                           (nih-nth)/2, nih-nth-(nih-nth)/2,
                           (niw-ntw)/2, niw-ntw-(niw-ntw)/2,
                           BORDER_CONSTANT, Scalar(cm2));
            Mat si2 = radonTransformFloat(nat_img);
            Mat st2 = radonTransformFloat(canvas2);
            auto cores2 = extractSinogramCore(st2, nth, ntw);
            auto r2 = detectByNCCHF(si2, cores2, si2.cols, nth, ntw, nih, niw);
            t1 = chrono::high_resolution_clock::now();
            double ms_ncc = chrono::duration<double, milli>(t1 - t0).count();
            cout << "  NCC-HF:  " << ms_ncc << " ms, angle=" << r2.angle
                 << ", dx=" << r2.dx << ", dy=" << r2.dy
                 << ", score=" << r2.score << endl;
            cout << endl;
        }
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
        Mat st = radonTransformFloat(applyGaussianWindow(t_resized, 1.0));
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
