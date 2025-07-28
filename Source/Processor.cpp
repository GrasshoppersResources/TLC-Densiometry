#include "Processor.h"

#include <iostream>
#include <filesystem>
#include <fstream>
#include <vector>
#include <string>
#include <algorithm>

#include "util.h"

#define DEBUG_SPEC //special debugging
//#define DEBUG_LOW //low level debugging
//#define DEBUG_HIGH //high level debugging


//input function
#define INPUT_FOLDER "input\\"
#define INPUT_IMAGE_TYPE ".jpg"

//isolate function
//#define BACKGROUND_DEPENDENT_PIXEL_NORMALIZATION

//density function
#define DENSITY_DETECTION_COMPLEX
//#define DENSITY_DETECTION_SIMPLE

//correct function
#define SCALING_FACTOR 280 //in percent

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
    mapping();
    //return;
    isolate();
    density();
    //idealize();
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


#ifdef DEBUG_LOW
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
    //grayscale conversion 
    cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);

    //adaptive threshold to find the plate
    cv::Mat morph;
    int thresholdSize = std::max(11, (int)((image.total() / 50000) | 1));
    cv::adaptiveThreshold(gray, morph, 255, cv::ADAPTIVE_THRESH_MEAN_C, cv::THRESH_BINARY, thresholdSize, -0);
#ifdef DEBUG_LOW
    show(morph, "cropPlate Binary");
#endif

    //Morphological opening to remove noise
    int openSize = std::max(3, (int)((image.total() / 600000) | 1));
    cv::Mat openKernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(openSize, openSize));
    cv::morphologyEx(morph, morph, cv::MORPH_OPEN, openKernel);
#ifdef DEBUG_LOW
    show(morph, "cropPlate Denoised");
#endif

    //Morphological closing to fill gaps
    int closeSize = std::max(3, (int)((image.total() / 200000) | 1));
    cv::Mat closeKernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(closeSize, closeSize));
    cv::morphologyEx(morph, morph, cv::MORPH_CLOSE, closeKernel);
#ifdef DEBUG_HIGH
    show(morph, "cropPlate Closed");
#endif

    //Find external contours
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(morph, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
#ifdef DEBUG_LOW
    cv::Mat debugContours = image.clone();
    cv::drawContours(debugContours, contours, -1,
        cv::Scalar(0, 255, 0), 2);
    show(debugContours, "cropPlate Contours");
#endif

    //find the biggest contour, thats the plate!
    double maxArea = 0;
    std::vector<cv::Point> maxHull;
    for (const auto& contour : contours)
    {
        std::vector<cv::Point> hull;
        cv::convexHull(contour, hull);
        double area = cv::contourArea(hull);
        if (area > maxArea)
        {
            maxArea = area;
            maxHull = hull;
        }
    }

    cv::RotatedRect rect = cv::minAreaRect(maxHull);
    // Shrink rect to avoid border artifacts
    int margin = (maxArea / 250000);
    cv::Size2f size(
        std::max(1.0f, rect.size.width - 2 * margin),
        std::max(1.0f, rect.size.height - 2 * margin));
    cv::RotatedRect shrunk(rect.center, size, rect.angle);

    // Draw shrunk rect for debug
#ifdef DEBUG_HIGH
    cv::Mat debugRect = image.clone();
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
#ifdef DEBUG_LOW
    show(image, "cropPlate Crop");
#endif

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

    cv::Mat closeKernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(11, 11));
    cv::morphologyEx(morph, morph, cv::MORPH_CLOSE, closeKernel);
#ifdef DEBUG_LOW
    show(morph, "removeArtifacts Close");
#endif

    cv::Mat erodeKernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3));
    cv::morphologyEx(morph, morph, cv::MORPH_OPEN, erodeKernel);
#ifdef DEBUG_LOW
    show(image, "open");
    show(morph, "open");
#endif
    
    cv::Mat dilateKernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(7, 7));
    cv::morphologyEx(morph, morph, cv::MORPH_DILATE, dilateKernel);
#ifdef DEBUG_LOW
    show(morph, "removeArtifacts Dilate");
#endif

#ifdef DEBUG_LOW
    show(image, "removeArtifacts Clean");
#endif

    cv::inpaint(image, morph, image, 11, cv::INPAINT_TELEA);

#ifdef DEBUG_LOW
    show(image, "removeArtifacts Clean");
#endif
}

void Processor::mapping()
{
    cv::Mat morph = image.clone();
    //blurring in a lane size dependant size
    cv::Size morphBlurSize = cv::Size(((int)laneWidth() / 2) | 1, (image.rows / 8) | 1);
    //get rid if noise
    cv::GaussianBlur(morph, morph, morphBlurSize, 0, 0);
    //find lanes by using sobel to detect areas of change in the color in a vertical direction
    cv::Sobel(morph, morph, CV_32F, 0, 1, 3); // vertical gradient (dy = 1)
    //get all changes positive and negative
    morph = cv::abs(morph);
    cv::GaussianBlur(morph, morph, morphBlurSize, 0, 0, cv::BORDER_REPLICATE);
#ifdef DEBUG_LOW
    show(morph * 100);
#endif
    //sobel in horizontal direction to get the exact position of the lanes (0 values)
    cv::Sobel(morph, morph, CV_32F, 1, 0, 3); // horizontal gradient (dx = 1)
    cv::GaussianBlur(morph, morph, morphBlurSize, 0, 0, cv::BORDER_REPLICATE);

    //mark the exact suspected lane positions and background positions
    cv::cvtColor(morph, morph, cv::COLOR_BGR2GRAY);
    cv::Mat marker;
    marker = cv::Mat::zeros(image.rows, image.cols, CV_32FC1);
    for (size_t x = 1; x < marker.cols - 1; x++)
    {
        for (size_t y = 1; y < marker.rows - 1; y++)
        {
            if (morph.at<float>(y, x) < 0 && morph.at<float>(y, x + 1) >= 0)
                marker.at<float>(y, x) = -1.f;

            if (morph.at<float>(y, x) > 0 && morph.at<float>(y, x + 1) <= 0)
                marker.at<float>(y, x) = 1.f;
        }
    }
#ifdef DEBUG_LOW
    show(marker);
#endif

    //blur the positions especially heavy blur n the vertical direction to compensate for missing sections
    cv::Size markerBlurSize = cv::Size(((int)laneWidth() / 2) | 1, (image.rows / 1) | 1);
    cv::GaussianBlur(marker, marker, markerBlurSize, 0, 0, cv::BORDER_CONSTANT);
    cv::normalize(marker, marker, -1.0, 1.0, cv::NORM_MINMAX);

#ifdef DEBUG_LOW
    show(marker);
 #endif

    //really find the lanes (required?)
    //cv::threshold(marker, marker, 0.5, 1.0, cv::THRESH_BINARY);
    //show(marker);

    //collapse to 1d signal
    cv::Mat signal;
    cv::Size signalBlurSize = cv::Size((int)laneWidth() | 1, 1);
    cv::reduce(marker, signal, 0, cv::REDUCE_SUM, CV_32F);
    cv::GaussianBlur(signal, signal, signalBlurSize, 0, 0, cv::BORDER_CONSTANT);

#ifdef DEBUG_LOW
    hist(signal);
#endif

    //find lanes and blanks
    const float* data = signal.ptr<float>(0);
    for (int i = 1; i < signal.cols - 1; ++i) {
        float prev = data[i - 1];
        float curr = data[i];
        float next = data[i + 1];

        if (curr > prev && curr >= next) {
            lanePos.push_back(i); // local max
        }
        else if (curr < prev && curr <= next) {
            spacePos.push_back(i); // local min
        }
    }
    ASSERT(lanePos.size() == annotation.size());
}

void Processor::isolate()
{
    ASSERT(!image.empty());
    ASSERT(!image.empty());
    ASSERT(image.type() == CV_32FC3);

    
#ifdef DEBUG_LOW
    /*cv::Mat tmp;
    tmp = image.clone();
    for (size_t i = 0; i < laneCount(); i++)
    {
        tmp.col(lanePos(i)).setTo(cv::Scalar(0, 0, 1));  // BGR format
    }
    for (size_t i = 0; i < blankCount(); i++)
    {
        tmp.col(blankPos(i)).setTo(cv::Scalar(1, 0, 0));  // BGR format
    }
    show(tmp);*/
#endif

    cv::Mat back = cv::Mat::zeros(image.rows, spacePos.size(), CV_32FC3); // Stores max x-pos per row (1 column)
    for (size_t i = 0; i < spacePos.size(); i++)
    {
        cv::Rect rect;
        if (i == 0)
        {
            int width = lanePos[i];
            rect = cv::Rect(0, 0, width, image.rows);
        }
        else if (i == spacePos.size()-1)
        {
            int width = image.cols - lanePos[i - 1];
            rect = cv::Rect(lanePos[i - 1], 0, width, image.rows);
        }
        else
        {
            int width = lanePos[i] - lanePos[i-1];
            rect = cv::Rect(lanePos[i - 1], 0, width, image.rows);
        }

        cv::Mat space;
        space = image(rect).clone();

        cv::Mat softSpace;
        cv::Size softSpaceBlurSize(((int)laneWidth() / 12)|1, (image.rows / 18)|1);
        cv::GaussianBlur(space, softSpace, softSpaceBlurSize, 0, 0, cv::BORDER_REPLICATE);
        //find lanes by using sobel to detect areas of change in the color in a vertical direction
        cv::Mat contrast;
        cv::Sobel(softSpace, contrast, CV_32F, 1, 1, 3); // vertical gradient (dy = 1)
        contrast = cv::abs(contrast);
        cv::Size contrastBlurSize(((int)laneWidth() / 4) | 1, (image.rows / 4) | 1);
        cv::GaussianBlur(contrast, contrast, contrastBlurSize, 0, 0);
        cv::normalize(contrast, contrast, 0, 1.0, cv::NORM_MINMAX);

#ifdef DEBUG_LOW
        show(contrast);
#endif
        cv::Mat spaceCenterPos = cv::Mat::zeros(contrast.rows, 1, CV_16UC1); // Stores max x-pos per row (1 column)
        // Step 1: Find x-position of max intensity for each row
        for (int y = 0; y < contrast.rows; ++y)
        {
            float minIntensity = FLT_MAX;
            uint16_t maxPos = 0;
            for (int x = contrast.cols / 4; x < contrast.cols - contrast.cols / 4; ++x)
            {
                const cv::Vec3f& pixel = contrast.at<cv::Vec3f>(y, x);
                float intensity = cv::norm(pixel, cv::NORM_L2);  // Euclidean norm of RGB triplet

                if (intensity < minIntensity)
                {
                    minIntensity = intensity;
                    maxPos = static_cast<uint16_t>(x);
                }
            }
            //if (minIntensity > 0.025)
            spaceCenterPos.at<uint16_t>(y, 0) = maxPos;
        }

        cv::Size spaceCenterBlurSize(1, (image.cols / 4) | 1);
        cv::GaussianBlur(spaceCenterPos, spaceCenterPos, spaceCenterBlurSize, 0, 0);

        cv::Size spaceBlurSize(((int)laneWidth() / 2) | 1, (image.rows / 100) | 1);
        cv::GaussianBlur(space, space, spaceBlurSize, 0, 0);

#ifdef DEBUG_LOW
        cv::Mat debugImage = space.clone();
        space = cv::abs(space);
        cv::normalize(debugImage, debugImage, 0, 1.0, cv::NORM_MINMAX);
        for (int y = 0; y < space.rows; ++y)
        {
            int centerX = spaceCenterPos.at<uint16_t>(y, 0);
            if (centerX > 1)
            {
                debugImage.at<cv::Vec3f>(y, centerX) = cv::Vec3f(1.0f, 1.0f, 1.0f); // mark in white
            }
        }
        show(debugImage, "Centerline Visualization");
#endif
        for (int y = 0; y < space.rows; ++y)
        {
            int centerX = spaceCenterPos.at<uint16_t>(y, 0);
            back.at<cv::Vec3f>(y, i) = space.at<cv::Vec3f>(y, centerX);
        }
    }

    cv::resize(back, back, image.size(), 0, 0, cv::INTER_LINEAR);
    cv::Size backBlurSize((int)laneWidth() | 1, (image.rows / 240) | 1);
    cv::GaussianBlur(back, back, backBlurSize, 0, 0);


    image = image - back;
    //cv::Size imageBlurSize((int)laneWidth() | 1, (image.rows / 240) | 1);
    //cv::GaussianBlur(image, image, cv::Size(33, 33), 0, 0);
    



    /*
    int backWidth = laneWidth() * 0.15;

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
    //cv::blur(image, image, cv::Size(laneWidth() * 0.1, laneWidth() * 0.1));
    image = image - back;
    cv::GaussianBlur(image, image, cv::Size(33, 33), 0, 0);

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
*/
    
}

void Processor::density()
{
    ASSERT(!image.empty());
    ASSERT(image.type() == CV_32FC3);
    // Prepare output vector for each lane's density profile
    densities = std::vector<cv::Mat>(lanePos.size());

    for (size_t i = 0; i < lanePos.size(); ++i)
    {
        cv::Rect rect(spacePos[i], 0, spacePos[i+1] - spacePos[i], image.rows);
        cv::Mat lane = image(rect).clone(); // Extract lane region

        //Simple Density Average Intensity of lane
#ifdef DENSITY_DETECTION_SIMPLE
        cv::Mat densityAverage;
        cv::reduce(lane, densityAverage, 1, cv::REDUCE_AVG); // Average across width
        densities[i] = densityAverage;
#endif

        // Compute lane centerline based on max intensity column per row
#ifdef DENSITY_DETECTION_COMPLEX

        //heavy blurring of the lane for reliable midline detection
        cv::Mat laneSoft = lane.clone();
        laneSoft = cv::abs(laneSoft);

        cv::Size laneSoftBlurSize(1,  (image.rows / 4) | 1);
        cv::GaussianBlur(laneSoft, laneSoft, laneSoftBlurSize, cv::BORDER_CONSTANT);

#ifdef DEBUG_LOW
        cv::Mat dbgImg;
        cv::normalize(laneSoft, dbgImg, 0.0, 1.0, cv::NORM_MINMAX);
        show(dbgImg, "Lane");
#endif

        cv::Mat laneCenterPos = cv::Mat::ones(laneSoft.rows, 1, CV_16UC1) * laneSoft.cols/2; // Stores max x-pos per row (1 column)
        // Step 1: Find x-position of max intensity for each row
        for (int y = 0; y < laneSoft.rows; ++y)
        {
            float maxIntensity = 0.0f;
            uint16_t maxPos = 0;
            for (int x = 3; x < laneSoft.cols - 3; ++x)
            {
                //const cv::Vec3f& pixel = lane.at<cv::Vec3f>(y, x);
                const cv::Vec3f& pixel = laneSoft.at<cv::Vec3f>(y, x + 0);
                float intensity = cv::norm(pixel, cv::NORM_L2);  // Euclidean norm of RGB triplet
                if (intensity > maxIntensity)
                {
                    maxIntensity = intensity;
                    maxPos = static_cast<uint16_t>(x);
                }
            }
            //if (maxIntensity > 0.05)
            laneCenterPos.at<uint16_t>(y, 0) = maxPos;
        }

        //Smooth the centerline vertically to reduce jitter/noise
        cv::Size laneCenterBlurSize(1, (image.rows / 4) | 1);
        cv::GaussianBlur(laneCenterPos, laneCenterPos, laneCenterBlurSize, cv::BORDER_CONSTANT);


#ifdef DEBUG_LOW
        cv::Mat debugImage = lane.clone();
        cv::normalize(debugImage, debugImage, 0, 1.0, cv::NORM_MINMAX);
        for (int y = 0; y < lane.rows; ++y)
        {
            int centerX = laneCenterPos.at<uint16_t>(y, 0);
            if (centerX > 1)
            {
                debugImage.at<cv::Vec3f>(y, centerX - 1) = cv::Vec3f(1.0f, 1.0f, 1.0f); // mark in white
                debugImage.at<cv::Vec3f>(y, centerX) = cv::Vec3f(1.0f, 1.0f, 1.0f); // mark in white
                debugImage.at<cv::Vec3f>(y, centerX + 1) = cv::Vec3f(1.0f, 1.0f, 1.0f); // mark in white
            }
        }
        show(debugImage, "Centerline Visualization");
#endif

        //denoise lane
        cv::Size laneDenoiseBlurSize((int)(laneWidth() / 3.0) | 1, (image.rows / 20) | 1);
        cv::GaussianBlur(lane, lane, laneDenoiseBlurSize, cv::BORDER_CONSTANT);

        // Sample five center pixels from each row
        cv::Mat densityCentral = cv::Mat::zeros(cv::Size(1, lane.rows), CV_32FC3);
        for (int y = 0; y < lane.rows; ++y)
        {
            int centerX = laneCenterPos.at<uint16_t>(y, 0);
            densityCentral.at<cv::Vec3f>(y, 0) = lane.at<cv::Vec3f>(y, centerX + 0);
        }
        //cv::GaussianBlur(densityCentral, densityCentral, cv::Size(1, 5), cv::BORDER_CONSTANT);
        densities[i] = densityCentral;
#endif


#ifdef DEBUG_LOW

#ifdef DENSITY_DETECTION_COMPLEX
        graph(densityCentral, name + " " + annotation[i]);
#endif

#ifdef DENSITY_DETECTION_SIMPLE
        graph(densityAverage*2);
#endif

#endif
    }
}

void Processor::idealize()
{
    cv::Mat black(image.rows, 3, CV_32FC3, cv::Scalar(0.0f, 0.0f, 0.0f));
    cv::Mat ideal = black;
    for (size_t i = 0; i < lanePos.size(); i++)
    {
        cv::hconcat(ideal, densities[i], ideal);
        cv::hconcat(ideal, densities[i], ideal);
        cv::hconcat(ideal, densities[i], ideal);
        cv::hconcat(ideal, black, ideal);
    }
    cv::resize(ideal, image, cv::Size(image.cols, image.rows), 0, 0, cv::INTER_LINEAR);
}

void Processor::correct()
{
    ASSERT(image.type() == CV_32FC3);
    cv::Size blurSize((image.total() / 40000)|1, (image.total() / 40000)|1);
    cv::GaussianBlur(image, image, blurSize, cv::BORDER_CONSTANT);

    //image = image * 3;
    //Euclidean redistribution of negative color channels
    for (int y = 0; y < image.rows; ++y)
    {
        for (int x = 0; x < image.cols; ++x)
        {
            cv::Vec3f& pixel = image.at<cv::Vec3f>(y, x);

            if (pixel[0] >= 0 && pixel[1] >= 0 && pixel[2] >= 0) //no color value is negative
                continue;

            //thats a bit mor etricky here
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
            if (pixel[0] < 0 || pixel[1] < 0 || pixel[2] < 0)
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
    image = image * ((double)SCALING_FACTOR / 100.0);
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
        const cv::Point position(lanePos[i], annoImageHeight / 2);
        int baseline = 0;
        cv::Size textSize = cv::getTextSize(annotation[i], fontName, fontScale, fontThickness, &baseline);
        cv::Point origin(lanePos[i] - textSize.width / 2, 25 + textSize.height / 2);
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

//TODO get rid of this function
double Processor::laneWidth()
{
    ASSERT(!image.empty());
    return double(image.cols) / (double)annotation.size();
}

/*double Processor::lanePos(size_t i)
{
    ASSERT(laneCount() > 0);
    ASSERT(i <= laneCount());
    return ((double)(i + 1) * laneWidth());
}*/

/*
size_t Processor::laneCount()
{
    return annotation.size();
}
*/

/*double Processor::blankPos(size_t i)
{
    ASSERT(laneCount() > 0);
    ASSERT(i <= blankCount());
    return (laneWidth() * 0.5 + (double)i * laneWidth());
}*/

/*
size_t Processor::blankCount()
{
    return annotation.size() + 1;
}
*/
