// test.cpp

#include <iostream>
#include <fstream>
#include <opencv2/opencv.hpp> // Include OpenCV headers
#include "opti_vibe.h"


// Global variable to store the processed frame
cv::Mat processed_frame;

int main(int argc, char** argv) {
    OptiVibe optiVibe; // Instantiate OptiVibe object

    std::string video_src;
    std::string output_path;
    int stride = 1;

    if (argc > 1) {
        video_src = "/workspaces/TreehornCV/_video/" + std::string(argv[1]);
        output_path = "/workspaces/TreehornCV/_video/processed_" + std::string(argv[1]);
        if (argc > 2) {
            stride = std::stod(argv[2]); // Second argument is for how many frames to drop, or stride
        }
    } else {
        std::cerr << "Error: No input file provided." << std::endl;
        return -1;
    }

    cv::VideoCapture cap;

    // Open video source
    if (video_src == "0") {
        cap.open(0); // Open default webcam
    } else {
        cap.open(video_src); // Open video file
    }

    if (!cap.isOpened()) {
        std::cerr << "Error: Could not open video source." << std::endl;
        return -1;
    }

    // Get video properties
    int frame_width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    int frame_height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    double fps = cap.get(cv::CAP_PROP_FPS);
    if (fps <= 0.0)
        fps = 30.0; // Default to 30 FPS if unable to get FPS

    // Define the codec and create VideoWriter object
    cv::VideoWriter outputVideo;
    int codec = cv::VideoWriter::fourcc('X', 'V', 'I', 'D'); // Use 'XVID' for AVI
    outputVideo.open(output_path, codec, fps, cv::Size(frame_width, frame_height), true);

    if (!outputVideo.isOpened()) {
        std::cerr << "Error: Could not open the output video for write: " << output_path << std::endl;
        return -1;
    }

    // Open a CSV file to write target, velocity, and timestamp values
    std::ofstream csvFile("../_dev/target_velocity.csv");
    if (!csvFile.is_open()) {
        std::cerr << "Error: Could not open CSV file for writing." << std::endl;
        return -1;
    }
    csvFile << "Timestamp,Target,Velocity\n"; // Write CSV header

    // Define the stroker and debug_callback using std::function
    stroker_callback_t stroker_callback = [&csvFile, &cap](double target, double vel) {
        double timestamp = cap.get(cv::CAP_PROP_POS_MSEC) / 1000.0; // Get timestamp in seconds
        csvFile << timestamp << "," << target << "," << vel << "\n"; // Write timestamp, target, and velocity to CSV
    };

    debug_callback_t debug_callback = [](const cv::Mat& annotated_frame) {
        processed_frame = annotated_frame.clone();
    };

    int stride_count = 0;

    while (true) {
        cv::Mat frame;
        bool ret = cap.read(frame);
        if (!ret || frame.empty()) {
            std::cerr << "Error: Could not read frame or end of video." << std::endl;
            break;
        }

        // Get the current time in seconds
        double current_time = static_cast<double>(cv::getTickCount()) / cv::getTickFrequency();

        // Increment stride_count for each frame read
        stride_count++;

        // Stride: Process only when stride_count is a multiple of stride
        if (stride > 0 && stride_count % stride != 0) {
            continue;
        }

        // Reset processed_frame before processing
        processed_frame.release();

        // Call the new process method
        optiVibe.process_frame_stroker_debug(frame, current_time, stroker_callback, debug_callback);

        // Write the processed frame to the output video
        if (!processed_frame.empty()) {
            outputVideo.write(processed_frame);
        } else {
            // If processed_frame is empty, write the original frame
            outputVideo.write(frame);
        }

        // Since we're headless, we don't need to handle key presses
        // Optionally, you can implement a condition to break the loop if needed
    }

    cap.release();
    outputVideo.release();

    // Close the CSV file
    csvFile.close();

    std::cout << "Done" << std::endl;
    return 0;
}
