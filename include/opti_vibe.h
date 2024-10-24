#ifndef OPTI_VIBE_H
#define OPTI_VIBE_H

#include <opencv2/core.hpp>

class OptiVibe
{
public:
    // Constructor
    OptiVibe();

    // Destructor
    ~OptiVibe();

    // Callback function types
    typedef void (*vibe_callback_t)(double vibe_intensity);
    typedef void (*debug_callback_t)(const cv::Mat& annotated_frame);

    // Processing methods
    void process_frame(const cv::Mat& frame, double time, vibe_callback_t vibe_callback);
    void process_frame_debug(const cv::Mat& frame, double time, vibe_callback_t vibe_callback, debug_callback_t debug_callback);

private:
    // Private attributes
    double last_time;

    // Private methods
    std::pair<cv::Mat, double> compute_vibe_signal(const cv::Mat& frame, double time, bool debug);
};

#endif // OPTI_VIBE_H
