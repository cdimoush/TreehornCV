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
    - displacement_threshold_percentage: Percentage of frame height that a tracker's displacement must exceed to be considered significant
    - speed: Amount that the vibe signal changes between frames. 1.0 is the maximum change
    - signal_threshold: The avg value of trackers must exceed this value (positive or negative) to update the vibe signal

    Secondary parameters:
    - track_len: Number of frames to track a feature
    - detect_interval: Number of frames between feature detection refreshes
    */

    // Primary parameters
    displacement_threshold_percentage = 0.01;
    speed = 0.1;
    signal_threshold = 0.2;

    // Secondary parameters
    track_len = 5;
    detect_interval = 5;

    // State variables
    next_id = 0;
    frame_idx = 0;
    signal = 0.0;
    last_time = 0.0;
}

OptiVibe::~OptiVibe()
{
    // No dynamic memory to release
}

void OptiVibe::process_frame(const cv::Mat& frame, double time, vibe_callback_t vibe_callback)
{
    auto [processed_frame, vibe_signal] = compute_vibe_signal(frame, time, false);
    last_time = time;
    vibe_callback(vibe_signal);
}

void OptiVibe::process_frame_debug(const cv::Mat& frame, double time, vibe_callback_t vibe_callback, debug_callback_t debug_callback)
{
    auto [processed_frame, vibe_signal] = compute_vibe_signal(frame, time, true);
    last_time = time;
    vibe_callback(vibe_signal);
    debug_callback(processed_frame);
}

std::pair<cv::Mat, double> OptiVibe::compute_vibe_signal(const cv::Mat& frame, double time, bool debug)
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

    return std::make_pair(processed_frame, signal);
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

    double x_threshold = frame_width * displacement_threshold_percentage;
    double y_threshold = frame_height * displacement_threshold_percentage;

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

    // Update the signal based on the average y-displacement and signal threshold
    if (std::abs(average_y_disp) > signal_threshold)
    {
        if (average_y_disp > 0)
        {
            signal = std::min(signal + speed, 1.0);
        }
        else
        {
            signal = std::max(signal - speed, 0.0);
        }
    }
}

cv::Mat OptiVibe::annotate_frame(cv::Mat vis)
{
    for (const auto& tr : tracks)
    {
        float x = std::get<1>(tr.back()).x;
        float y = std::get<1>(tr.back()).y;
        cv::Point2f disp = std::get<2>(tr.back());
        cv::Scalar color = assign_track_color(disp);
        annotate_frame_with_point(vis, x, y, color);
    }
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

    int bar_width = 100; 
    int bar_height = static_cast<int>(height * signal);

    // Adjust offsets for the bar
    int bar_offset_x = 20; // Offset from the right side
    int bar_offset_y = 20; // Offset from the bottom

    cv::Point top_left(width - bar_width - bar_offset_x, height - bar_height - bar_offset_y);
    cv::Point bottom_right(width - bar_offset_x, height - bar_offset_y);

    // Draw the bar
    cv::rectangle(vis, top_left, bottom_right, cv::Scalar(0, 0, 255), -1);

    // Add tick marks and labels
    int tick_height = 2;
    int tick_positions[] = {height - bar_offset_y, static_cast<int>(height * 0.75) - bar_offset_y, static_cast<int>(height * 0.5) - bar_offset_y, static_cast<int>(height * 0.25) - bar_offset_y, 0 - bar_offset_y};
    std::string tick_labels[] = {"0%", "25%", "50%", "75%", "100%"};

    for (int i = 0; i < 5; ++i)
    {
        int y_pos = tick_positions[i];
        cv::line(vis, cv::Point(width - bar_width - bar_offset_x - 5, y_pos), cv::Point(width - bar_offset_x + 5, y_pos), cv::Scalar(255, 255, 255), tick_height * 2);
        cv::putText(vis, tick_labels[i], cv::Point(width - bar_width - bar_offset_x - 100, y_pos + 5), cv::FONT_HERSHEY_COMPLEX_SMALL, 1.5, cv::Scalar(255, 255, 255), 1);
    }

    // Add label for the bar
    cv::putText(vis, "Vibe", cv::Point(width - bar_width - bar_offset_x - 50, height - bar_height - bar_offset_y - 10), cv::FONT_HERSHEY_COMPLEX, 1.5, cv::Scalar(255, 255, 255), 2);
}
