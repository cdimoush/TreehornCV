// opti_vibe.cpp

#include "opti_vibe.h"
#include <cmath>

OptiVibe::OptiVibe()
{
    /*
    OptiVibe:
    - Class tracks motion (y-displacement) between frames and computes a vibe signal
    - The vibe signal is determined by evaluating a set of trackers in a rolling buffer

    Primary parameters:
    - disp_percentage: Percentage of frame height that a tracker's displacement must exceed to be considered significant
    - disp_threshold: The avg value of trackers must exceed this value (positive or negative) to update the vibe signal
    - acc: Amount that the vibe signal changes between frames. 1.0 is the maximum change

    Secondary parameters:
    - track_len: Number of frames to track a feature
    - detect_interval: Number of frames between feature detection refreshes
    */

    // Primary parameters
    disp_percentage = 0.01;
    disp_threshold = 0.2;
    acc = 0.075;
    max_vel = 1.0;

    // Secondary parameters
    track_len = 5;
    detect_interval = 5;

    // State variables
    next_id = 0;
    frame_idx = 0;
    signal_target = 0.0;
    signal_pos = 0.0;
    signal_vel = 0.0;
    last_time = 0.0;
}

OptiVibe::~OptiVibe()
{
    // No dynamic memory to release
}

void OptiVibe::process_frame_vibe(const cv::Mat& frame, double time, vibe_callback_t vibe_callback)
{
    auto [processed_frame, target, pos, vel] = compute_signal(frame, time, false);
    last_time = time;
    vibe_callback(pos);
}

void OptiVibe::process_frame_vibe_debug(const cv::Mat& frame, double time, vibe_callback_t vibe_callback, debug_callback_t debug_callback)
{
    auto [processed_frame, target, pos, vel] = compute_signal(frame, time, true);
    last_time = time;
    vibe_callback(pos);
    debug_callback(processed_frame);
}

void OptiVibe::process_frame_stroker(const cv::Mat& frame, double time, stroker_callback_t stroker_callback)
{
    auto [processed_frame, target, pos, vel] = compute_signal(frame, time, false);
    last_time = time;
    stroker_callback(target, vel);
}

void OptiVibe::process_frame_stroker_debug(const cv::Mat& frame, double time, stroker_callback_t stroker_callback, debug_callback_t debug_callback)
{
    auto [processed_frame, target, pos, vel] = compute_signal(frame, time, true);
    last_time = time;
    stroker_callback(target, vel);
    debug_callback(processed_frame);
}

std::tuple<cv::Mat, double, double, double> OptiVibe::compute_signal(const cv::Mat& frame, double time, bool debug)
{
    cv::Mat frame_gray = convert_to_grayscale(frame, true, true);
    cv::Mat processed_frame;

    if (!prev_gray.empty() && tracks_exist())
    {
        process_existing_tracks(frame_gray);
        process_displacement(frame_gray);
        process_signal();
        if (debug)
        {
            processed_frame = annotate_frame(frame.clone());
        }
        else
        {
            processed_frame = frame;
        }
    }
    else
    {
        processed_frame = frame;
    }

    if (should_detect_new_features())
    {
        detect_new_features(frame_gray);
    }

    frame_idx += 1;
    prev_gray = frame_gray;

    return std::make_tuple(processed_frame, signal_target, signal_pos, signal_vel);
}

cv::Mat OptiVibe::convert_to_grayscale(const cv::Mat& frame, bool invert, bool sharpen)
{
    cv::Mat gray;
    if (sharpen)
    {
        cv::GaussianBlur(frame, gray, cv::Size(0, 0), 3);
        cv::addWeighted(frame, 1.5, gray, -0.5, 0, gray);
    }
    else
    {
        gray = frame.clone();
    }
    cv::cvtColor(gray, gray, cv::COLOR_BGR2GRAY);
    if (invert)
    {
        cv::bitwise_not(gray, gray);
    }
    return gray;
}

bool OptiVibe::tracks_exist()
{
    return !tracks.empty();
}

void OptiVibe::process_existing_tracks(const cv::Mat& frame_gray)
{
    cv::Mat img0 = prev_gray;
    cv::Mat img1 = frame_gray;

    std::vector<cv::Point2f> p0;
    for (const auto& tr : tracks)
    {
        p0.push_back(std::get<1>(tr.back()));
    }

    cv::Mat p0_mat(p0);
    cv::Mat p1;
    std::vector<uchar> good;

    std::tie(p1, good) = calculate_optical_flow(p0_mat, img0, img1);

    update_tracks(p1, good);
}

std::pair<cv::Mat, std::vector<uchar>> OptiVibe::calculate_optical_flow(const cv::Mat& p0, const cv::Mat& img0, const cv::Mat& img1)
{
    // Parameters for lucas kanade optical flow
    std::vector<cv::Point2f> p0_vec;
    p0.copyTo(p0_vec);

    std::vector<cv::Point2f> p1_vec;
    std::vector<uchar> st;
    std::vector<float> err;

    cv::Size winSize(15, 15);
    int maxLevel = 2;
    cv::TermCriteria criteria(cv::TermCriteria::COUNT | cv::TermCriteria::EPS, 10, 0.03);

    // Calculate optical flow (forward)
    cv::calcOpticalFlowPyrLK(img0, img1, p0_vec, p1_vec, st, err, winSize, maxLevel, criteria);

    // Calculate optical flow (backward)
    std::vector<cv::Point2f> p0r_vec;
    std::vector<uchar> st_back;
    std::vector<float> err_back;

    cv::calcOpticalFlowPyrLK(img1, img0, p1_vec, p0r_vec, st_back, err_back, winSize, maxLevel, criteria);

    // Compute the difference between the original points and the back-tracked points
    std::vector<float> d;
    for (size_t i = 0; i < p0_vec.size(); ++i)
    {
        cv::Point2f diff = p0_vec[i] - p0r_vec[i];
        d.push_back(std::max(std::abs(diff.x), std::abs(diff.y)));
    }

    // Identify good points based on the difference
    std::vector<uchar> good;
    for (size_t i = 0; i < d.size(); ++i)
    {
        good.push_back(d[i] < 1);
    }

    cv::Mat p1_mat(p1_vec);
    return std::make_pair(p1_mat, good);
}

void OptiVibe::update_tracks(const cv::Mat& p1, const std::vector<uchar>& good)
{
    std::vector<std::vector<std::tuple<int, cv::Point2f, cv::Point2f>>> new_tracks;

    std::vector<cv::Point2f> p1_vec;
    p1.copyTo(p1_vec);

    for (size_t i = 0; i < tracks.size(); ++i)
    {
        if (!good[i])
            continue;

        auto tr = tracks[i];
        float x = p1_vec[i].x;
        float y = p1_vec[i].y;
        // Append new point with zero displacement
        tr.push_back(std::make_tuple(std::get<0>(tr[0]), cv::Point2f(x, y), cv::Point2f(0.0f, 0.0f)));
        if (tr.size() > static_cast<size_t>(track_len))
        {
            tr.erase(tr.begin() + 1);
        }
        new_tracks.push_back(tr);
    }

    tracks = new_tracks;
}

void OptiVibe::process_displacement(const cv::Mat& frame)
{
    int frame_height = frame.rows;
    int frame_width = frame.cols;

    double x_threshold = frame_width * disp_percentage;
    double y_threshold = frame_height * disp_percentage;

    for (auto& tr : tracks)
    {
        auto [x_disp, y_disp] = calculate_displacement(tr, x_threshold, y_threshold);
        // Store displacement in the track
        if (std::get<2>(tr.back()) == cv::Point2f(0.0f, 0.0f))
        {
            // Update the last point in the track with displacement
            std::get<2>(tr.back()) = cv::Point2f(static_cast<float>(x_disp), static_cast<float>(y_disp));
        }
    }
}

std::pair<int, int> OptiVibe::calculate_displacement(const std::vector<std::tuple<int, cv::Point2f, cv::Point2f>>& track, double x_threshold, double y_threshold)
{
    if (track.size() < 2)
        return std::make_pair(0, 0);

    double total_x_disp = 0.0;
    double total_y_disp = 0.0;

    for (size_t i = 2; i < track.size(); ++i)
    {
        total_x_disp += std::get<1>(track[i]).x - std::get<1>(track[i - 1]).x;
        total_y_disp += std::get<1>(track[i]).y - std::get<1>(track[i - 1]).y;
    }

    int x_disp = (total_x_disp > x_threshold) ? 1 : (total_x_disp < -x_threshold) ? -1 : 0;
    int y_disp = (total_y_disp > y_threshold) ? 1 : (total_y_disp < -y_threshold) ? -1 : 0;

    return std::make_pair(x_disp, y_disp);
}

bool OptiVibe::should_detect_new_features()
{
    return (frame_idx % detect_interval == 0);
}

void OptiVibe::detect_new_features(const cv::Mat& frame_gray)
{
    int maxCorners = 200;
    double qualityLevel = 0.05;
    double minDistance = 14;
    int blockSize = 14;

    cv::Mat mask = create_feature_mask(frame_gray);
    std::vector<cv::Point2f> p;
    cv::goodFeaturesToTrack(frame_gray, p, maxCorners, qualityLevel, minDistance, mask, blockSize);

    if (!p.empty())
    {
        for (const auto& pt : p)
        {
            // Initialize the track with displacement value of (0,0)
            std::vector<std::tuple<int, cv::Point2f, cv::Point2f>> tr;
            tr.push_back(std::make_tuple(next_id, pt, cv::Point2f(0.0f, 0.0f)));
            tracks.push_back(tr);
            next_id++;
        }
    }
}

cv::Mat OptiVibe::create_feature_mask(const cv::Mat& frame_gray)
{
    cv::Mat mask = cv::Mat::ones(frame_gray.size(), CV_8U) * 255;
    for (const auto& tr : tracks)
    {
        int x = static_cast<int>(std::get<1>(tr.back()).x);
        int y = static_cast<int>(std::get<1>(tr.back()).y);
        cv::circle(mask, cv::Point(x, y), 5, 0, -1);
    }
    return mask;
}

void OptiVibe::process_signal()
{
    if (tracks.empty())
        return;

    double total_y_disp = 0.0;
    int count = 0;

    for (const auto& tr : tracks)
    {
        cv::Point2f disp = std::get<2>(tr.back());
        double y_disp = disp.y;
        if (y_disp != 0)
        {
            total_y_disp += y_disp;
            count += 1;
        }
    }

    double average_y_disp = (count == 0) ? 0.0 : total_y_disp / count;

    // Handle sign
    double new_target = (average_y_disp > 0) ? 1.0 : 0.0;
    if (new_target != signal_target)
    {
        signal_vel = 0.0; // Reset velocity on sign flip
    }

    // Update signal
    signal_target = new_target;
    signal_vel = std::min(signal_vel + acc, max_vel);
    if (signal_target == 1.0)
    {
        signal_pos = std::min(signal_pos + signal_vel, 1.0);
    }
    else
    {
        signal_pos = std::max(signal_pos - signal_vel, 0.0);
    }
}

cv::Mat OptiVibe::annotate_frame(cv::Mat vis)
{
    // for (const auto& tr : tracks)
    // {
    //     float x = std::get<1>(tr.back()).x;
    //     float y = std::get<1>(tr.back()).y;
    //     cv::Point2f disp = std::get<2>(tr.back());
    //     cv::Scalar color = assign_track_color(disp);
    //     annotate_frame_with_point(vis, x, y, color);
    // }
    annotate_frame_with_signal(vis);
    return vis;
}

cv::Scalar OptiVibe::assign_track_color(const cv::Point2f& disp)
{
    int y_disp = static_cast<int>(disp.y);
    if (y_disp == 1)
    {
        return cv::Scalar(0, 0, 255); // Red for moving down
    }
    else if (y_disp == -1)
    {
        return cv::Scalar(255, 0, 0); // Blue for moving up
    }
    else
    {
        return cv::Scalar(0, 255, 0); // Green for not moving significantly
    }
}

void OptiVibe::annotate_frame_with_point(cv::Mat& vis, float x, float y, const cv::Scalar& color)
{
    cv::circle(vis, cv::Point(static_cast<int>(x), static_cast<int>(y)), 20, color, -1);
}

void OptiVibe::annotate_frame_with_signal(cv::Mat& vis)
{
    int height = vis.rows;
    int width = vis.cols;

    // Define percentages for dimensions and offsets
    double bar_width_pct = 0.1;
    double bar_height_pct = 0.8;
    double offset_right_pct = 0.02;

    // Calculate bar dimensions and positions
    int bar_width = static_cast<int>(width * bar_width_pct);
    int bar_height = static_cast<int>(height * bar_height_pct);
    int offset_right = static_cast<int>(width * offset_right_pct);
    int offset_top = (height - bar_height) / 2;

    int x1 = width - offset_right - bar_width;
    int y1 = offset_top;
    int x2 = width - offset_right;
    int y2 = y1 + bar_height;

    cv::Point top_left(x1, y1);
    cv::Point bottom_right(x2, y2);

    // Draw the blue bar
    cv::rectangle(vis, top_left, bottom_right, cv::Scalar(255, 0, 0), -1); // Blue color in BGR

    // Clamp signal values between 0 and 1
    double signal_target_clamped = std::max(0.0, std::min(signal_target, 1.0));
    double signal_pos_clamped = std::max(0.0, std::min(signal_pos, 1.0));
    double signal_vel_clamped = std::max(0.0, std::min(signal_vel, 1.0));

    // Calculate positions for signal_target and signal_pos
    int target_y = y1 + static_cast<int>(signal_target_clamped * bar_height);
    int pos_y = y1 + static_cast<int>(signal_pos_clamped * bar_height);

    // Draw the signal_target line
    cv::line(vis, cv::Point(x1, target_y), cv::Point(x2, target_y), cv::Scalar(255, 255, 255), 15); // White line

    // Define slider ring dimensions (small rectangle)
    int slider_height = static_cast<int>(bar_height * 0.1);
    int x1_slider = x1 - 0.1 * bar_width;
    int x2_slider = x2 + 0.1 * bar_width;

    int slider_y1 = pos_y - slider_height / 2;
    int slider_y2 = pos_y + slider_height / 2;

    // Ensure the slider stays within the bar
    slider_y1 = std::max(y1, slider_y1);
    slider_y2 = std::min(y2, slider_y2);

    cv::Point slider_top_left(x1_slider, slider_y1);
    cv::Point slider_bottom_right(x2_slider, slider_y2);

    // Compute the color of the slider ring based on signal_vel (0 = green, 1 = red)
    // Linearly interpolate between green and red
    int red = static_cast<int>(255 * signal_vel_clamped);
    int green = static_cast<int>(255 * (1.0 - signal_vel_clamped));
    int blue = 0;

    // Ensure the values are within [0, 255]
    red = std::min(255, std::max(0, red));
    green = std::min(255, std::max(0, green));

    cv::Scalar slider_color(blue, green, red); // BGR format
    cv::rectangle(vis, slider_top_left, slider_bottom_right, slider_color, -1);
}