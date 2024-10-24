#include "opti_vibe.h"
#include <cmath>

double pi = 3.14159265358979323846;

OptiVibe::OptiVibe() : last_time(0.0) {}

OptiVibe::~OptiVibe() {}

void OptiVibe::process_frame(const cv::Mat& frame, double time, vibe_callback_t vibe_callback) {
    auto [processed_frame, vibe_signal] = compute_vibe_signal(frame, time, false);
    last_time = time;
    vibe_callback(vibe_signal);
}

void OptiVibe::process_frame_debug(const cv::Mat& frame, double time, vibe_callback_t vibe_callback, debug_callback_t debug_callback) {
    auto [processed_frame, vibe_signal] = compute_vibe_signal(frame, time, true);
    last_time = time;
    vibe_callback(vibe_signal);
    debug_callback(processed_frame);
}

std::pair<cv::Mat, double> OptiVibe::compute_vibe_signal(const cv::Mat& frame, double time, bool debug) {
    // TODO: Implement vibe signal computation
    double frequency = 0.1;
    double vibe_signal = (std::sin(2 * pi * frequency * time) + 1.0) / 2.0;
    return std::make_pair(frame, vibe_signal);
}
