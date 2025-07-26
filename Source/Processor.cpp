#include "Processor.h"

#include <iostream>
#include <filesystem>
#include <fstream>
#include <vector>
#include <string>
#include <algorithm>

#include "util.h"

//#define DEBUG_LOW //low level debugging
//#define DEBUG_HIGH //high level debugging
//#define DEBUG_SPEC //special debugging

//input function
#define INPUT_FOLDER "input\\"
#define INPUT_IMAGE_TYPE ".jpg"

//isolate function
#define BACKGROUND_DEPENDENT_PIXEL_NORMALIZATION

//density function
#define DENSITY_DETECTION_COMPLEX
//#define DENSITY_DETECTION_SIMPLE

//correct function
#define SCALING_FACTOR 220 //in percent

//output function
#define OUTPUT_FOLDER "output\\"
#define OUTPUT_IMAGE_TYPE ".jpg"

//class implementation ===========================================

Processor::Processor(std::string name) :
    name(name)
{
    read();
    crop();
    clean();
    toFloat();
    toLinear();
    isolate();
    density();
    correct();
    //toSRGB();
    toU8();
    write();
}

void Processor::read()
{
    std::string annotationPath = INPUT_FOLDER + name + ".txt";
    std::ifstream infile(annotationPath);
    if (!infile) {
        std::cerr << "Error: Cannot open file '" << annotationPath << "' for reading.\n";
        ASSERT(false);
    }
    std::string line;
    while (std::getline(infile, line)) {
        annotation.push_back(line);
    }
    if (infile.bad()) {
        std::cerr << "Error: I/O error while reading file.\n";
        ASSERT(false);
    }
#ifdef DEBUG_LOW
    for (auto line : annotation)
    {
        std::cout << line << std::endl;
    }
#endif
    std::string imagePath = INPUT_FOLDER + name + INPUT_IMAGE_TYPE;
    image = cv::imread(imagePath, cv::IMREAD_UNCHANGED);
    if (image.empty()) {
        std::cerr << "Error: Cannot open file '" << imagePath << "' for reading.\n";
        ASSERT(false);
    }
#ifdef DEBUG_HIGH
    show(image, "original image");
#endif
    //char c;
    //std::cin >> c;
}

void Processor::crop()
{
    ASSERT(!image.empty());
    ASSERT(image.type() == CV_8UC3);

    cv::Mat gray;
    // 1. grayscale conversion 
    cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);

    //2. adaptive threshold to find the plate
    cv::Mat morph;
    cv::adaptiveThreshold(gray, morph, 255, cv::ADAPTIVE_THRESH_MEAN_C, cv::THRESH_BINARY, 91, -5);
#ifdef DEBUG_LOW
    show(morph, "cropPlate Binary");
#endif

    // 3. Morphological closing to fill gaps
    cv::Mat closeKernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(7, 7));
    cv::morphologyEx(morph, morph, cv::MORPH_CLOSE, closeKernel);
#ifdef DEBUG_LOW
    show(morph, "cropPlate Closed");
#endif

    // 4. Morphological opening to remove noise
    cv::Mat openKernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(19, 19));
    cv::morphologyEx(morph, morph, cv::MORPH_OPEN, openKernel);
#ifdef DEBUG_LOW
    show(morph, "cropPlate Denoised");
#endif

    // 5. Find external contours
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(morph, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
#ifdef DEBUG_LOW
    cv::Mat debugContours = image.clone();
    cv::drawContours(debugContours, contours, -1,
        cv::Scalar(0, 255, 0), 2);
    show(debugContours, "cropPlate Contours");
#endif

    // 6. Select contour by area and fit rotated rectangle
    for (const auto& contour : contours) {
        if (contour.size() < 4) continue;
        std::vector<cv::Point> hull;
        cv::convexHull(contour, hull);
        double area = cv::contourArea(hull);
        int minPlateArea = 50000;
        if (area < minPlateArea) continue;

        int margin = 4;
        cv::RotatedRect rect = cv::minAreaRect(hull);
        // Shrink rect to avoid border artifacts
        cv::Size2f size(
            std::max(1.0f, rect.size.width - 2 * margin),
            std::max(1.0f, rect.size.height - 2 * margin));
        cv::RotatedRect shrunk(rect.center, size, rect.angle);

        // Draw shrunk rect for debug
#ifdef DEBUG_LOW
        cv::Mat debugRect = debugContours.clone();
        cv::Point2f pts[4]; shrunk.points(pts);
        for (int i = 0; i < 4; ++i)
            cv::line(debugRect, pts[i], pts[(i + 1) % 4], cv::Scalar(0, 0, 255), 2);
        show(debugRect, "cropPlate Shrinked Rect");
#endif

        // 7. Warp to upright rectangle
        cv::Point2f srcPts[4], dstPts[4];
        shrunk.points(srcPts);
        dstPts[0] = { 0, size.height - 1 };                 // BL
        dstPts[1] = { 0, 0 };                               // TL
        dstPts[2] = { size.width - 1, 0 };                  // TR
        dstPts[3] = { size.width - 1, size.height - 1 };    // BR

        cv::Mat M = cv::getPerspectiveTransform(srcPts, dstPts);
        cv::warpPerspective(image, image, M, size);

        // Auto-rotate if tilted >45°
        if (shrunk.angle > 45.0f)
            cv::rotate(image, image, cv::ROTATE_90_CLOCKWISE);
#ifdef DEBUG_HIGH
        show(image, "cropPlate Crop");
#endif
        break;  // stop after first valid plate
    }
}

void Processor::clean()
{
    cv::Mat gray;
    cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);

    cv::Mat morph;
    cv::adaptiveThreshold(gray, morph, 255, cv::ADAPTIVE_THRESH_MEAN_C, cv::THRESH_BINARY, 5, -3);
#ifdef DEBUG_LOW
    show(morph, "removeArtifacts Threshold");
#endif

    cv::Mat dilateKernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3));
    cv::morphologyEx(morph, morph, cv::MORPH_DILATE, dilateKernel);
#ifdef DEBUG_LOW
    show(morph, "removeArtifacts Dilate");
#endif

    cv::Mat closeKernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(7, 7));
    cv::morphologyEx(morph, morph, cv::MORPH_CLOSE, closeKernel);
#ifdef DEBUG_LOW
    show(morph, "removeArtifacts Close");
#endif

    /*cv::Mat erodeKernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5));
    cv::morphologyEx(morph, morph, cv::MORPH_OPEN, erodeKernel);
#ifdef DEBUG_HIGH
    show(imageCrop, "crop");
    show(morph, "open");
#endif*/
    
    cv::inpaint(image, morph, image, 13, cv::INPAINT_TELEA);
    cv::blur(image, image, cv::Size(7,7));

#ifdef DEBUG_HIGH
    show(image, "removeArtifacts Clean");
#endif
}

void Processor::isolate()
{
    ASSERT(!image.empty());
    ASSERT(!image.empty());
    ASSERT(image.type() == CV_32FC3);

    int backWidth = laneSpace() * 0.25;

    cv::Mat back;
    cv::Rect rect(blankPos(0) - (backWidth / 2), 0, backWidth, image.rows);
    back = image(rect).clone();
    for (size_t i = 1; i < blankCount(); i++)
    {
        rect = cv::Rect(blankPos(i) - (backWidth / 2), 0, backWidth, image.rows);
        cv::hconcat(back, image(rect).clone(), back);
    }

#ifdef DEBUG_LOW
    show(back, "signal back");
#endif

    cv::blur(back, back, cv::Size(backWidth, image.rows / 50));
    cv::resize(back, back, image.size(), 0, 0, cv::INTER_LINEAR);

#ifdef DEBUG_LOW
    show(back, "signal back");
#endif
    cv::blur(image, image, cv::Size(laneSpace() * 0.1, laneSpace() * 0.1));
    
    image = image - back;

    // Contrast enhancement based on background-subtracted fluorescence signal
#ifdef BACKGROUND_DEPENDENT_PIXEL_NORMALIZATION
    for (int y = 0; y < image.rows; ++y)
    {
        for (int x = 0; x < image.cols; ++x)
        {
            cv::Vec3f& ipixel = image.at<cv::Vec3f>(y, x);
            const cv::Vec3f& bpixel = back.at<cv::Vec3f>(y, x);

            for (int c = 0; c < 3; ++c)
            {
                float b = bpixel[c];
                float f = (ipixel[c] > 0.0f) ? (1.0f / (1.0f - b)) : (1.0f / b);
                ipixel[c] *= f;
            }
        }
    }
#endif

#ifdef DEBUG_HIGH
    show(image, "signal");
#endif
}

void Processor::density()
{
    ASSERT(!image.empty());
    ASSERT(image.type() == CV_32FC3);
    // Prepare output vector for each lane's density profile
    densities = std::vector<cv::Mat>(laneCount());

    int laneWidth = static_cast<int>(laneSpace() * 0.5); // Estimate lane width

    for (size_t i = 0; i < laneCount(); ++i)
    {
        // Define the rectangular region of the current lane
        int xCenter = lanePos(i);
        int xStart = xCenter - laneWidth / 2;
        cv::Rect laneRect(xStart, 0, laneWidth, image.rows);
        cv::Mat lane = image(laneRect).clone(); // Extract lane region

#ifdef DEBUG_LOW
        show(lane, "Lane");
#endif

        //Simple Density Average Intensity of lane
#ifdef DENSITY_DETECTION_SIMPLE
        cv::Mat densityAverage;
        cv::reduce(lane, densityAverage, 1, cv::REDUCE_AVG); // Average across width
        densities[i] = densityAverage;
#endif

#ifdef DENSITY_DETECTION_COMPLEX
        // Compute lane centerline based on max intensity column per row
        cv::Mat laneCenterPos = cv::Mat::zeros(lane.rows, 1, CV_16UC1); // Stores max x-pos per row (1 column)
        // Step 1: Find x-position of max intensity for each row
        for (int y = 0; y < lane.rows; ++y)
        {
            float maxIntensity = 0.0f;
            uint16_t maxPos = 0;
            for (int x = 2; x < lane.cols - 2; ++x)
            {
                const cv::Vec3f& pixel = lane.at<cv::Vec3f>(y, x);
                float intensity = cv::norm(pixel);  // Euclidean norm of RGB triplet

                if (intensity > maxIntensity)
                {
                    maxIntensity = intensity;
                    maxPos = static_cast<uint16_t>(x);
                }
            }
            laneCenterPos.at<uint16_t>(y, 0) = maxPos;
        }
        // Step 2: Smooth the centerline vertically to reduce jitter/noise
        int blurHeight = std::max(3, lane.rows / 5 | 1); // Ensure odd kernel size ≥ 3
        cv::blur(laneCenterPos, laneCenterPos, cv::Size(1, blurHeight));

        // Step 3: Sample five center pixels from each row
        cv::Mat densityCentral = cv::Mat::zeros(cv::Size(1, lane.rows), CV_32FC3);
        for (int y = 0; y < lane.rows; ++y)
        {
            int centerX = laneCenterPos.at<uint16_t>(y, 0);
            densityCentral.at<cv::Vec3f>(y, 0) = (
                lane.at<cv::Vec3f>(y, centerX - 2) * 0.1 +
                lane.at<cv::Vec3f>(y, centerX - 1) * 0.2 +
                lane.at<cv::Vec3f>(y, centerX + 0) * 0.4 +
                lane.at<cv::Vec3f>(y, centerX + 1) * 0.2 +
                lane.at<cv::Vec3f>(y, centerX + 2) * 0.1
                );
        }
        cv::blur(densityCentral, densityCentral, cv::Size(1, 3));
        densities[i] = densityCentral;
#endif

#ifdef DEBUG_LOW
        // Optional debug visualization of the centerline
        cv::Mat debugImage = lane.clone();
        for (int y = 0; y < lane.rows; ++y)
        {
            int centerX = laneCenterPos.at<uint16_t>(y, 0);
            debugImage.at<cv::Vec3f>(y, centerX) = cv::Vec3f(1.0f, 1.0f, 1.0f); // mark in white
        }

        show(debugImage, "Centerline Visualization");
#ifdef DENSITY_DETECTION_SIMPLE
        graph(densityAverage*2);
#endif
#ifdef DENSITY_DETECTION_COMPLEX
        graph(densityCentral, name + " " + annotation[i]);
#endif
#endif
    }
}

void Processor::correct()
{
    ASSERT(image.type() == CV_32FC3);
    //image = image * 3;
    //Euclidean redistribution of negative color channels
    for (int y = 0; y < image.rows; ++y)
    {
        for (int x = 0; x < image.cols; ++x)
        {
            cv::Vec3f& pixel = image.at<cv::Vec3f>(y, x);

            if (pixel[0] >= 0 && pixel[1] >= 0 && pixel[2] >= 0) //no color value is negative
                continue;

            if (pixel[0] <= 0 && pixel[1] <= 0 && pixel[2] <= 0) // all color values are negative
            {
                // Step 1: Invert sign to get into visible domain
                cv::Vec3f positive_pixel = -pixel;

                // Step 2: Convert to HLS
                cv::Mat rgb(1, 1, CV_32FC3, positive_pixel);
                cv::Mat hls;
                cv::cvtColor(rgb, hls, cv::COLOR_RGB2HLS); // OpenCV expects RGB, not BGR

                // Step 3: Invert Hue while keeping Lightness and Saturation
                cv::Vec3f& hls_pixel = hls.at<cv::Vec3f>(0, 0);
                hls_pixel[0] = std::fmod(hls_pixel[0] + 180.0f, 360.0f); // hue + 180 degrees

                // Step 4: Convert back to RGB
                cv::cvtColor(hls, rgb, cv::COLOR_HLS2RGB);
                pixel = rgb.at<cv::Vec3f>(0, 0);
            }

             // some color values are negative
            {
                // Clip negative components to zero
                cv::Vec3f clipped = pixel;
                for (int c = 0; c < 3; ++c)
                    if (clipped[c] < 0.0f)
                        clipped[c] = 0.0f;

                float original_norm = std::sqrt(pixel[0] * pixel[0] + pixel[1] * pixel[1] + pixel[2] * pixel[2]);
                float clipped_norm = std::sqrt(clipped[0] * clipped[0] + clipped[1] * clipped[1] + clipped[2] * clipped[2]);

                if (clipped_norm > 1e-6f)
                    pixel = clipped * (original_norm / clipped_norm); // Rescale to preserve energy
                else
                    pixel = cv::Vec3f(0.0f, 0.0f, 0.0f); // Totally dark if nothing left
            }

        }
    }

    //adjust scaling and soft clip the pixels
#if SCALING_FACTOR != 100
    image = image * (SCALING_FACTOR / 100.0);
    for (int y = 0; y < image.rows; ++y)
    {
        for (int x = 0; x < image.cols; ++x)
        {
            cv::Vec3f& pixel = image.at<cv::Vec3f>(y, x);
            for (int c = 0; c < 3; ++c)
            {
                float v = pixel[c];
                pixel[c] = (pixel[c]) / (1.0 + pixel[c]);
            }
        }
    }
#endif

    cv::blur(image, image, cv::Size(3, 3));

}

void Processor::write()
{
    ASSERT(!image.empty());
    ASSERT(image.type() == CV_8UC3);
    ASSERT(densities.size() > 0);
    ASSERT(annotation.size() > 0);

    //add annotation, show the image and write it
    int annoImageHeight = 50;
    cv::Mat anno(annoImageHeight, image.cols, CV_8UC3, cv::Scalar(0, 0, 0));
    anno.row(0).setTo(cv::Scalar(31, 31, 31)); // BGR: red
    double fontScale = 0.6;
    int fontThickness = 1;
    cv::HersheyFonts fontName = cv::FONT_HERSHEY_DUPLEX;
    for (int i = 0; i < annotation.size(); i++) 
    {
        const cv::Point position(lanePos(i), annoImageHeight / 2);
        int baseline = 0;
        cv::Size textSize = cv::getTextSize(annotation[i], fontName, fontScale, fontThickness, &baseline);
        cv::Point origin(lanePos(i) - textSize.width / 2, 25 + textSize.height / 2);
        cv::putText(anno, annotation[i], origin, fontName, fontScale, cv::Scalar(127,127,127), fontThickness);
    }
    cv::vconcat(image, anno, image);
    cv::imwrite(OUTPUT_FOLDER + name + OUTPUT_IMAGE_TYPE, image);

    //export csv
    for (size_t i = 0; i < densities.size(); i++)
    {
        std::ofstream file(OUTPUT_FOLDER + name + "__" + annotation[i] + ".csv");
        if (!file.is_open()) {
            std::cerr << "Error: Failed to create CSV file" << std::endl;
            ASSERT(false);
        }
        file << "B,G,R\n";
        for (size_t j = 0; j < densities[i].rows; j++)
        {
            file << densities[i].at<cv::Vec3f>(j, 0)[0] << ",";
            file << densities[i].at<cv::Vec3f>(j, 0)[1] << ",";
            file << densities[i].at<cv::Vec3f>(j, 0)[2] << std::endl;
        }
        file.close();
    }
    show(image, name);
}



//conversion subfunctions ===========================================

inline float sRGBToLinear(float V) {
    if (V <= 0.04045f)
        return V / 12.92f;
    else
        return std::pow((V + 0.055f) / 1.055f, 2.4f);
}

inline float linearToSRGB(float L) {
    if (L <= 0.0031308f)
        return L * 12.92f;
    else
        return 1.055f * std::pow(L, 1.0f / 2.4f) - 0.055f;
}

void Processor::toFloat()
{
    ASSERT(image.channels() == 3);
    image.convertTo(image, CV_32FC3, 1.0 / 255.0);
}

void Processor::toU8()
{
    ASSERT(image.channels() == 3);
    image.convertTo(image, CV_8UC3, 255.0);
}

void Processor::toLinear()
{
    ASSERT(image.type() == CV_32FC3);
    image.forEach<cv::Vec3f>([](cv::Vec3f& px, const int* /*pos*/) 
    {
        px[0] = sRGBToLinear(px[0]);  // B
        px[1] = sRGBToLinear(px[1]);  // G
        px[2] = sRGBToLinear(px[2]);  // R
    });
}

void Processor::toSRGB()
{
    ASSERT(image.type() == CV_32FC3);
    image.forEach<cv::Vec3f>([](cv::Vec3f& px, const int* )
    {
        px[0] = linearToSRGB(px[0]);
        px[1] = linearToSRGB(px[1]);
        px[2] = linearToSRGB(px[2]);
    });
}



//spacing subfunctions ===========================================

double Processor::laneSpace()
{
    ASSERT(!image.empty());
    ASSERT(blankCount() > 0);
    return double(image.cols) / (double)blankCount();
}

double Processor::lanePos(size_t i)
{
    ASSERT(laneCount() > 0);
    ASSERT(i <= laneCount());
    return ((double)(i + 1) * laneSpace());
}

size_t Processor::laneCount()
{
    return annotation.size();
}

double Processor::blankPos(size_t i)
{
    ASSERT(laneCount() > 0);
    ASSERT(i <= blankCount());
    return (laneSpace() * 0.5 + (double)i * laneSpace());
}

size_t Processor::blankCount()
{
    return annotation.size() + 1;
}

