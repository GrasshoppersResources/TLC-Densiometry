//OPENCV DOWNLOAD
// https://github.com/thommyho/Cpp-OpenCV-Windows-PreBuilts
// https://opencv.org/releases/

#include <filesystem>
#include <vector>

#include "Processor.h"

int main() 
{
    //iterate over jpg files in the input folder and create records from them
    std::filesystem::path inputDir = std::filesystem::current_path() / "input";
    for (const auto& entry : std::filesystem::directory_iterator(inputDir)) {
        if (entry.path().extension() == ".jpp" || entry.path().extension() == ".JPG")
        {
            std::string name = entry.path().filename().stem().string();
            Processor process(name);
        }
    }
    return 0;
}


