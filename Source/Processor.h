#pragma once

#include <string>
#include <vector>

#include <opencv2/opencv.hpp>

class Processor
{
public:
    Processor(std::string name);

private:
    void read();
    void crop();
    void clean();
    void isolate();
    void density();
    void correct();
    void write();

    //
    void toLinear();
    void toSRGB();
    void toFloat();
    void toU8();

    //
    double laneSpace();
    double lanePos(size_t i);
    size_t laneCount();
    double blankPos(size_t i);
    size_t blankCount();

    std::string name;

    std::vector<std::string> annotation;

    cv::Mat image;

    std::vector<cv::Mat> densities;
};