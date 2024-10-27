// opti_vibe.h

#ifndef OPTI_VIBE_H
#define OPTI_VIBE_H

#include <opencv2/opencv.hpp>
#include <functional>

// Define the callback function types
typedef std::function<void(double)> vibe_callback_t;
typedef std::function<void(const cv::Mat&)> debug_callback_t;

class OptiVibe {
public:
    OptiVibe();
    ~OptiVibe();

    void process_frame(const cv::Mat& frame, double time, vibe_callback_t vibe_callback);
    void process_frame_debug(const cv::Mat& frame, double time, vibe_callback_t vibe_callback, debug_callback_t debug_callback);

private:
    // Private member variables
    int track_len;
    int detect_interval;
    std::vector<std::vector<std::tuple<int, cv::Point2f, cv::Point2f>>> tracks;
    int frame_idx;
    int next_id;
    double displacement_threshold_percentage;
    cv::Mat prev_gray;
    double signal;
    double speed;
    double signal_threshold;
    double last_time;

    // Private methods
    std::pair<cv::Mat, double> compute_vibe_signal(const cv::Mat& frame, double time, bool debug);
    cv::Mat convert_to_grayscale(const cv::Mat& frame, bool invert = false, bool sharpen = false);
    bool tracks_exist();
    void process_existing_tracks(const cv::Mat& frame_gray);
    std::pair<cv::Mat, std::vector<uchar>> calculate_optical_flow(const cv::Mat& p0, const cv::Mat& img0, const cv::Mat& img1);
    void update_tracks(const cv::Mat& p1, const std::vector<uchar>& good);
    void process_displacement(const cv::Mat& frame);
    std::pair<int, int> calculate_displacement(const std::vector<std::tuple<int, cv::Point2f, cv::Point2f>>& track, double x_threshold, double y_threshold);
    bool should_detect_new_features();
    void detect_new_features(const cv::Mat& frame_gray);
    cv::Mat create_feature_mask(const cv::Mat& frame_gray);
    void process_signal();
    cv::Mat annotate_frame(cv::Mat vis);
    cv::Scalar assign_track_color(const cv::Point2f& disp);
    void annotate_frame_with_point(cv::Mat& vis, float x, float y, const cv::Scalar& color);
    void annotate_frame_with_signal(cv::Mat& vis);
};

#endif // OPTI_VIBE_H
