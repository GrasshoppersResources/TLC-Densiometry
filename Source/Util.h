#pragma once

#include <opencv2/opencv.hpp>

#include <string>

#include <iostream>
#include <cstdlib> // for std::abort

#define ASSERT(condition)                                               \
    do {                                                                \
        if (!(condition)) {                                             \
            std::cerr << "Assertion failed: (" #condition "), "         \
                      << "function " << __func__                        \
                      << ", file " << __FILE__                          \
                      << ", line " << __LINE__ << "." << std::endl;     \
            std::abort();                                               \
        }                                                               \
    } while (false)

//shows an image in a window, resized if required
void show(const cv::Mat& image, const std::string& name = "image") {
    ASSERT(!image.empty());

    // Downscale if imageRaw is large
    int maxDimX = 2400;
    int maxDimY = 1200;

    // Resize only if scaling is needed
    cv::Mat resizedImage;
    if (image.cols > maxDimX || image.rows > maxDimY) {
        double scaleX = static_cast<double>(maxDimX) / image.cols;
        double scaleY = static_cast<double>(maxDimY) / image.rows;
        double scale = std::min(scaleX, scaleY); // Keep aspect ratio

        int newWidth = static_cast<int>(std::round(image.cols * scale));
        int newHeight = static_cast<int>(std::round(image.rows * scale));


        cv::resize(image, resizedImage, cv::Size(newWidth, newHeight), 0, 0, cv::INTER_AREA);
        resizedImage;
    }
    else
        resizedImage = image;

    //show image
    cv::imshow(name, resizedImage);
    cv::moveWindow(name, 100, 100);
    cv::waitKey(0);
    // Only destroy if still open
    if (cv::getWindowProperty(name, cv::WND_PROP_VISIBLE) >= 1)
        cv::destroyWindow(name);
}

//plots the curves of the density
void graph(cv::Mat density, const std::string& name = "graph")
{
    ASSERT(!density.empty());
    ASSERT(density.type() == CV_32FC3);

    const int width = 800;
    cv::Mat dbg = cv::Mat::zeros(cv::Size(width, density.rows), CV_8UC3);

    for (size_t y = 0; y < density.rows; y++)
    {
        dbg.at<cv::Vec3b>(y, (width / 2)) = cv::Vec3b(32, 32, 32);

        for (size_t c = 0; c < 3; c++)
        {
            int x = (density.at<cv::Vec3f>(y, 0)[c] * width);
            x = std::clamp(x, -width / 2, (width / 2) - 1);
            dbg.at<cv::Vec3b>(y, (width / 2) + x) = cv::Vec3b(255 * (c == 0), 255 * (c == 1), 255 * (c == 2));
        }
    }
    show(dbg, name);
}