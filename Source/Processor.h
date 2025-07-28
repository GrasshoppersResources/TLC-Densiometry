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
    void toFloat();
    void toLinear();
    void mapping();
    void isolate();
    void density();
    void idealize();
    void correct();
    void write();

    //
    void toSRGB();

    void toU8();

    //
    double laneWidth();
    //double lanePos(size_t i);
    //size_t laneCount();
    //double blankPos(size_t i);
    //size_t blankCount();

    std::string name;
    std::vector<std::string> annotation;
    cv::Mat image;
    std::vector<cv::Mat> densities;

    std::vector<size_t> lanePos;
    std::vector<size_t> spacePos;
};
