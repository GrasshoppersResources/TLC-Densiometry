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
void show(const cv::Mat& image, const std::string& name = "image") 
{
    ASSERT(!image.empty());

    // Downscale if imageRaw is large
    int maxDimX = 2400;
    int maxDimY = 1000;

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
    cv::moveWindow(name, 10, 10);
    cv::waitKey(0);
    // Only destroy if still open
    if (cv::getWindowProperty(name, cv::WND_PROP_VISIBLE) >= 1)
        cv::destroyWindow(name);
}

void graph(const cv::Mat& density, const std::string& name = "Graph", int width = 800, int thickness = 2)
{
    CV_Assert(!density.empty());
    CV_Assert(density.type() == CV_32FC3);
    CV_Assert(density.cols == 1);  // Only 1D vertical input supported

    const int height = density.rows;
    const int centerX = width / 2;

    // Create black image
    cv::Mat image(height, width, CV_8UC3, cv::Scalar(0, 0, 0));

    // Draw vertical center line
    cv::line(image, cv::Point(centerX, 0), cv::Point(centerX, height - 1), cv::Scalar(32, 32, 32), 1);

    // Predefined BGR channel colors
    const cv::Scalar colors[3] = {
        {255, 0, 0},   // Blue
        {0, 255, 0},   // Green
        {0, 0, 255}    // Red
    };

    for (int y = 0; y < height - 1; ++y)
    {
        const cv::Vec3f& cur = density.at<cv::Vec3f>(y, 0);
        const cv::Vec3f& next = density.at<cv::Vec3f>(y + 1, 0);

        for (int c = 0; c < 3; ++c)
        {
            int x1 = std::clamp(static_cast<int>(cur[c] * width), -centerX, centerX - 1);
            int x2 = std::clamp(static_cast<int>(next[c] * width), -centerX, centerX - 1);

            cv::line(
                image,
                cv::Point(centerX + x1, y),
                cv::Point(centerX + x2, y + 1),
                colors[c],
                thickness,
                cv::LINE_AA
            );
        }
    }
    show(image, name);
}




void hist(cv::Mat signal, const std::string& name = "graph")
{
    cv::normalize(signal, signal, 0.0, 1.0, cv::NORM_MINMAX);
    cv::Mat dbgimg = cv::Mat::zeros(400, signal.cols, CV_32F);
    for (size_t x = 0; x < dbgimg.cols - 1; x++)
    {
        float y = (dbgimg.rows - 1) - (signal.at<float>(0, x) * (dbgimg.rows - 1));
        dbgimg.at<float>(y, x) = 1.0f;

        cv::line(dbgimg, cv::Point(x, y), cv::Point(x, (dbgimg.rows - 1)), cv::Scalar(1.0), 1);
    }

    show(dbgimg);
}
