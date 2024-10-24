#include <iostream>
#include <opencv2/opencv.hpp> // Include OpenCV headers
#include "opti_vibe.h"

// Callback function for vibe intensity
void vibe_callback(double vibe_intensity) {
    std::cout << "Vibe Intensity: " << vibe_intensity << std::endl;
}

// Callback function for debug
void debug_callback(const cv::Mat& annotated_frame) {
    std::cout << "Debug Callback: Frame received" << std::endl;
    // You can add code here to display or process the annotated frame
}

int main() {
    OptiVibe optiVibe; // Instantiate OptiVibe object

    // Create a dummy frame (e.g., a 640x480 black image)
    cv::Mat frame = cv::Mat::zeros(480, 640, CV_8UC3);

    // Set a time value
    double time = 0.0;

    // Call the process_frame_debug method
    optiVibe.process_frame_debug(frame, time, vibe_callback, debug_callback);

    std::cout << "Hello, World!" << std::endl;
    return 0;
}
