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
    displacement_threshold_percentage = 0.015;
    speed = 0.2;
    signal_threshold = 0.25;

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
    // Add header
    vis = add_header(vis);

    // Add title and subtitle
    vis = add_title(vis, "OhMiBod", "--Stroker AI--");

    // Draw the tracks as before
    for (const auto& tr : tracks)
    {
        float x = std::get<1>(tr.back()).x;
        float y = std::get<1>(tr.back()).y;
        cv::Point2f disp = std::get<2>(tr.back());
        cv::Scalar color = assign_track_color(disp);
        // annotate_frame_with_point(vis, x, y, color);
    }

    // Add the legend representing the vibe signal
    vis = add_legend(vis, signal);

    return vis;
}

cv::Mat OptiVibe::add_header(cv::Mat image, double header_thickness, cv::Scalar header_color)
{
    int height = image.rows;
    int width = image.cols;
    int header_thickness_px = static_cast<int>(height * header_thickness);
    cv::rectangle(image, cv::Point(0, 0), cv::Point(width, header_thickness_px), header_color, -1);
    // Add a white border to the header
    cv::rectangle(image, cv::Point(0, 0), cv::Point(width, header_thickness_px), cv::Scalar(255, 255, 255), 2);
    return image;
}

cv::Mat OptiVibe::add_title(cv::Mat image, std::string title, std::string subtitle, int title_font, double title_font_scale, cv::Scalar title_color, int title_thickness, int subtitle_font, double subtitle_font_scale, cv::Scalar subtitle_color, int subtitle_thickness)
{
    int height = image.rows;
    int width = image.cols;

    // Calculate the position for the title
    int baseLine = 0;
    cv::Size title_size = cv::getTextSize(title, title_font, title_font_scale, title_thickness, &baseLine);
    int title_x = (width - title_size.width) / 2;
    int title_y = title_size.height + 20; // 20 pixels from the top

    // Add the title to the image
    cv::putText(image, title, cv::Point(title_x, title_y), title_font, title_font_scale, title_color, title_thickness);

    // If a subtitle is provided, add it below the title
    if (!subtitle.empty())
    {
        cv::Size subtitle_size = cv::getTextSize(subtitle, subtitle_font, subtitle_font_scale, subtitle_thickness, &baseLine);
        int subtitle_x = (width - subtitle_size.width) / 2;
        int subtitle_y = title_y + title_size.height + 10; // 10 pixels below the title
        cv::putText(image, subtitle, cv::Point(subtitle_x, subtitle_y), subtitle_font, subtitle_font_scale, subtitle_color, subtitle_thickness);
    }

    return image;
}

cv::Mat OptiVibe::add_legend(cv::Mat image, double signal, double legend_width_scale, double legend_height_scale, cv::Scalar legend_color, cv::Scalar border_color, int border_thickness, double offset_top, double offset_right)
{
    int height = image.rows;
    int width = image.cols;
    int legend_width = static_cast<int>(width * legend_width_scale);
    int legend_height = static_cast<int>(height * legend_height_scale);

    // Calculate the top left corner position with offsets
    int top_left_x = width - legend_width - static_cast<int>(width * offset_right);
    int top_left_y = static_cast<int>(height * offset_top);

    // Draw the legend box
    cv::rectangle(image, cv::Point(top_left_x, top_left_y),
                  cv::Point(top_left_x + legend_width, top_left_y + legend_height),
                  legend_color, -1);

    // Add a border to the legend box
    cv::rectangle(image, cv::Point(top_left_x, top_left_y),
                  cv::Point(top_left_x + legend_width, top_left_y + legend_height),
                  border_color, border_thickness);

    // Draw the shaft
    int center_x = top_left_x + legend_width / 2;
    int center_y = top_left_y + legend_height / 2;
    int shaft_height_offset = legend_height / 4;
    int shaft_width_offset = legend_width / 4;

    cv::rectangle(image, cv::Point(center_x - shaft_width_offset, center_y - shaft_height_offset),
                  cv::Point(center_x + shaft_width_offset, center_y + shaft_height_offset),
                  cv::Scalar(0, 0, 0), -1);

    // Draw circles at the top and bottom of the shaft
    cv::circle(image, cv::Point(center_x, center_y - shaft_height_offset), shaft_width_offset, cv::Scalar(0, 0, 0), -1);
    cv::circle(image, cv::Point(static_cast<int>(center_x + 0.75 * shaft_width_offset), center_y + shaft_height_offset), shaft_width_offset, cv::Scalar(0, 0, 0), -1);
    cv::circle(image, cv::Point(static_cast<int>(center_x - 0.75 * shaft_width_offset), center_y + shaft_height_offset), shaft_width_offset, cv::Scalar(0, 0, 0), -1);

    // Draw the ring indicator based on the signal level
    int shaft_top_y = static_cast<int>(center_y - 0.5 * shaft_height_offset);
    int shaft_bottom_y = static_cast<int>(center_y + 0.5 * shaft_height_offset);
    int ring_height = static_cast<int>(0.3 * std::abs(shaft_bottom_y - shaft_top_y));
    int ring_width = static_cast<int>(1.25 * shaft_width_offset);

    // Map signal from [0,1] to position from shaft_top_y to shaft_bottom_y
    int ring_position = static_cast<int>(-1.2*ring_height + shaft_top_y + signal * (shaft_bottom_y - shaft_top_y));

    cv::rectangle(image, cv::Point(center_x - ring_width, ring_position),
                  cv::Point(center_x + ring_width, ring_position + ring_height),
                  cv::Scalar(255, 0, 0), -1);

    // // Add text to top center of the legend
    // std::string text = "Stroker\nSignal";
    // int baseLine = 0;
    // double font_scale = 1.0;
    // int thickness = 1;
    // cv::Size text_size = cv::getTextSize(text, cv::FONT_HERSHEY_DUPLEX, font_scale, thickness, &baseLine);
    // int text_x = static_cast<int>(top_left_x + legend_width * 0.5 - text_size.width * 0.5);
    // int text_y = static_cast<int>(top_left_y + legend_height * 0.05 + text_size.height);
    // int line_spacing = 10; // Space between lines

    // // Split the text into lines
    // std::istringstream iss(text);
    // std::string line;
    // int line_number = 0;
    // while (std::getline(iss, line, '\n'))
    // {
    //     cv::putText(image, line, cv::Point(text_x, text_y + line_number * (text_size.height + line_spacing)), cv::FONT_HERSHEY_DUPLEX, font_scale, cv::Scalar(255, 255, 255), thickness);
    //     line_number++;
    // }

    return image;
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