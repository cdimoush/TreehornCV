// Lucas-Kanade Tracker
// ====================
//
// This script demonstrates the Lucas-Kanade method for sparse optical flow tracking.
// It uses `goodFeaturesToTrack` for initializing points to track and employs back-tracking
// for match verification between frames.
//
// Usage:
// ------
// Run the script and optionally provide a video source and output path as arguments.
// If no source is provided, the default webcam will be used.
//
//     ./lk_track [<video_source>] [<output_path>]
//
// Example:
//     ./lk_track /path/to/video.mp4 /path/to/output.mp4
//
// Note:
// -----
// This version writes the output to a video file instead of displaying it with `imshow`.
//
// Keys:
// -----
// ESC - Exit the program (if applicable)

#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <string>

using namespace cv;
using namespace std;

// Parameters for Lucas-Kanade optical flow
Size winSize = Size(15, 15);  // Size of the search window at each pyramid level
int maxLevel = 2;              // 0-based maximal pyramid level number
TermCriteria criteria = TermCriteria(TermCriteria::COUNT | TermCriteria::EPS, 10, 0.03);  // Termination criteria

// Parameters for Shi-Tomasi corner detection used by `goodFeaturesToTrack`
int maxCorners = 100;     // Maximum number of corners to return
double qualityLevel = 0.05;   // Quality level for corner detection
double minDistance = 14;      // Minimum possible Euclidean distance between the returned corners
int blockSize = 14;         // Size of an average block for computing a derivative covariance matrix

class App
{
public:
    App(string video_src, string output_path);
    void run();
private:
    int track_len;                 // Maximum length of the trajectories
    int detect_interval;           // Interval to detect new features
    vector<vector<pair<int, Point2f>>> tracks;  // List to store the trajectories
    VideoCapture cam;              // Video capture object
    VideoWriter outputVideo;       // Video writer object
    int frame_idx;                 // Frame index counter
    int next_id;                   // Unique ID for each new point
    int y_displacement_threshold;  // Threshold for significant y-displacement
    Mat prev_gray;                 // Previous grayscale frame
    string outputPath;             // Output video path

    VideoCapture initialize_video_capture(string video_src);
    void release_resources();

    // Methods
    pair<bool, Mat> read_frame();
    Mat convert_to_grayscale(Mat frame, bool invert = false, bool sharpen = true);
    bool tracks_exist();
    void process_existing_tracks(Mat frame_gray, Mat& vis);
    pair<Mat, vector<uchar>> calculate_optical_flow(Mat img0, Mat img1, Mat p0);
    void update_tracks(Mat p1, vector<uchar> good);
    void update_visualization(Mat& vis);
    int calculate_y_displacement(vector<pair<int, Point2f>>& track);
    Scalar assign_track_color(int disp);
    void draw_feature_point(Mat& vis, float x, float y, Scalar color, int track_id);
    void draw_trajectory(Mat& vis, vector<pair<int, Point2f>>& track);
    void display_track_count(Mat& vis);
    bool should_detect_new_features();
    void detect_new_features(Mat frame_gray);
    Mat create_feature_mask(Mat frame_gray);
    bool handle_exit_condition();
};

App::App(string video_src, string output_path)
{
    track_len = 5;              // Maximum length of the trajectories
    detect_interval = 5;        // Interval to detect new features
    cam = initialize_video_capture(video_src);  // Video capture object
    frame_idx = 0;              // Frame index counter
    next_id = 0;                // Unique ID for each new point
    y_displacement_threshold = 10;  // Threshold for significant y-displacement
    outputPath = output_path;

    // Check if camera opened successfully
    if (!cam.isOpened())
    {
        cerr << "Unable to open video source!" << endl;
        exit(1);
    }

    // Initialize VideoWriter
    int codec = VideoWriter::fourcc('m', 'p', '4', 'v'); // MPEG-4 codec
    double fps = cam.get(CAP_PROP_FPS);
    if (fps == 0.0) fps = 30.0; // Default to 30 if can't get fps
    Size frame_size(cam.get(CAP_PROP_FRAME_WIDTH), cam.get(CAP_PROP_FRAME_HEIGHT));
    outputVideo.open(outputPath, codec, fps, frame_size, true);
    if (!outputVideo.isOpened())
    {
        cerr << "Unable to open the output video file for writing!" << endl;
        exit(1);
    }
}

VideoCapture App::initialize_video_capture(string video_src)
{
    if (video_src.size() == 1 && isdigit(video_src[0]))
    {
        int device_id = video_src[0] - '0';
        return VideoCapture(device_id);
    }
    else
    {
        return VideoCapture(video_src);
    }
}

void App::release_resources()
{
    cam.release();
    outputVideo.release();
    destroyAllWindows();
}

pair<bool, Mat> App::read_frame()
{
    Mat frame;
    bool ret = cam.read(frame);
    return make_pair(ret, frame);
}

Mat App::convert_to_grayscale(Mat frame, bool invert, bool sharpen)
{
    Mat gray = frame.clone();
    if (sharpen)
    {
        GaussianBlur(gray, gray, Size(0, 0), 3);
        Mat zeros = Mat::zeros(gray.size(), gray.type());
        addWeighted(gray, 1.5, zeros, 0, 0, gray);
    }
    cvtColor(frame, gray, COLOR_BGR2GRAY);
    if (invert)
    {
        bitwise_not(gray, gray);
    }
    return gray;
}

bool App::tracks_exist()
{
    return !tracks.empty();
}

void App::process_existing_tracks(Mat frame_gray, Mat& vis)
{
    Mat img0 = prev_gray;
    Mat img1 = frame_gray;
    vector<Point2f> p0;
    for (auto& tr : tracks)
    {
        p0.push_back(tr.back().second);
    }
    Mat p0_mat(p0);
    Mat p1;
    vector<uchar> good;
    tie(p1, good) = calculate_optical_flow(img0, img1, p0_mat);
    update_tracks(p1, good);
    update_visualization(vis);
}

pair<Mat, vector<uchar>> App::calculate_optical_flow(Mat img0, Mat img1, Mat p0)
{
    // Calculate optical flow (forward)
    Mat p1;
    vector<uchar> st1;
    vector<float> err1;
    calcOpticalFlowPyrLK(img0, img1, p0, p1, st1, err1, winSize, maxLevel, criteria);
    // Calculate optical flow (backward)
    Mat p0r;
    vector<uchar> st2;
    vector<float> err2;
    calcOpticalFlowPyrLK(img1, img0, p1, p0r, st2, err2, winSize, maxLevel, criteria);
    // Compute the difference between the original points and the back-tracked points
    Mat d;
    absdiff(p0, p0r, d);
    d = d.reshape(2, d.rows);
    vector<float> distances;
    for (int i = 0; i < d.rows; i++)
    {
        float dx = d.at<Vec2f>(i, 0)[0];
        float dy = d.at<Vec2f>(i, 0)[1];
        float dist = max(abs(dx), abs(dy));
        distances.push_back(dist);
    }
    vector<uchar> good;
    for (size_t i = 0; i < distances.size(); i++)
    {
        good.push_back(distances[i] < 1);
    }
    return make_pair(p1, good);
}

void App::update_tracks(Mat p1, vector<uchar> good)
{
    vector<vector<pair<int, Point2f>>> new_tracks;
    p1 = p1.reshape(2, p1.rows);
    for (size_t i = 0; i < tracks.size(); i++)
    {
        if (!good[i])
            continue;
        auto tr = tracks[i];
        float x = p1.at<Point2f>(i).x;
        float y = p1.at<Point2f>(i).y;
        tr.push_back(make_pair(tr[0].first, Point2f(x, y)));
        if (tr.size() > track_len)
        {
            tr.erase(tr.begin() + 1);
        }
        new_tracks.push_back(tr);
    }
    tracks = new_tracks;
}

void App::update_visualization(Mat& vis)
{
    for (auto& tr : tracks)
    {
        float x = tr.back().second.x;
        float y = tr.back().second.y;
        int disp = calculate_y_displacement(tr);
        Scalar color = assign_track_color(disp);
        draw_feature_point(vis, x, y, color, tr[0].first);
        draw_trajectory(vis, tr);
    }
    display_track_count(vis);
}

int App::calculate_y_displacement(vector<pair<int, Point2f>>& track)
{
    if (track.size() < 2)
        return 0;
    float total_disp = 0;
    for (size_t i = 2; i < track.size(); i++)
    {
        total_disp += track[i].second.y - track[i - 1].second.y;
    }
    if (abs(total_disp) > y_displacement_threshold)
    {
        return (total_disp > 0) ? 1 : -1;
    }
    return 0;
}

Scalar App::assign_track_color(int disp)
{
    if (disp == 1)
    {
        return Scalar(0, 0, 255); // Red
    }
    else if (disp == -1)
    {
        return Scalar(255, 0, 0); // Blue
    }
    else
    {
        return Scalar(0, 255, 0); // Green
    }
}

void App::draw_feature_point(Mat& vis, float x, float y, Scalar color, int track_id)
{
    circle(vis, Point2f(x, y), 6, color, -1); // Increase marker size by 3x (originally 2)
    putText(vis, to_string(track_id), Point(int(x) + 5, int(y) - 5),
            FONT_HERSHEY_SIMPLEX, 1.2, color, 3); // Increase font size and thickness by 3x (originally 0.4 and 1)
}

void App::draw_trajectory(Mat& vis, vector<pair<int, Point2f>>& track)
{
    vector<Point> pts;
    for (auto& pt : track)
    {
        pts.push_back(Point(int(pt.second.x), int(pt.second.y)));
    }
    polylines(vis, pts, false, Scalar(0, 255, 0), 3); // Increase trail thickness by 3x (originally 1)
}

void App::display_track_count(Mat& vis)
{
    string text = "Track count: " + to_string(tracks.size());
    putText(vis, text, Point(20, 20), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 0));
}

bool App::should_detect_new_features()
{
    return (frame_idx % detect_interval == 0);
}

void App::detect_new_features(Mat frame_gray)
{
    Mat mask = create_feature_mask(frame_gray);
    vector<Point2f> p;
    goodFeaturesToTrack(frame_gray, p, maxCorners, qualityLevel, minDistance, mask, blockSize);
    if (!p.empty())
    {
        for (auto& pt : p)
        {
            vector<pair<int, Point2f>> tr;
            tr.push_back(make_pair(next_id, pt));
            tracks.push_back(tr);
            next_id++;
        }
    }
}

Mat App::create_feature_mask(Mat frame_gray)
{
    Mat mask = Mat::zeros(frame_gray.size(), frame_gray.type());
    mask.setTo(255);
    for (auto& tr : tracks)
    {
        int x = int(tr.back().second.x);
        int y = int(tr.back().second.y);
        circle(mask, Point(x, y), 5, Scalar(0), -1);
    }
    return mask;
}

bool App::handle_exit_condition()
{
    // Since we're running headless, we don't check for key presses.
    // If you need to implement an exit condition, you can do so here.
    return false;
}

void App::run()
{
    while (true)
    {
        pair<bool, Mat> frame_pair = read_frame();
        bool ret = frame_pair.first;
        Mat frame = frame_pair.second;
        if (!ret || frame.empty())
            break;

        Mat frame_gray = convert_to_grayscale(frame, false, true);
        Mat vis = frame.clone();

        if (tracks_exist())
        {
            process_existing_tracks(frame_gray, vis);
        }

        if (should_detect_new_features())
        {
            detect_new_features(frame_gray);
        }

        frame_idx += 1;
        prev_gray = frame_gray;

        // Write the processed frame to the output video
        outputVideo.write(vis);

        if (handle_exit_condition())
            break;
    }
    release_resources();
}

int main(int argc, char** argv)
{
    string video_src = "/workspaces/TreehornCV/_video/ride_the_d.mp4";
    string output_path = "/workspaces/TreehornCV/_video/output.mp4";
    if (argc > 1)
        video_src = argv[1];
    if (argc > 2)
        output_path = argv[2];

    App app(video_src, output_path);
    app.run();

    cout << "Done" << endl;
    return 0;
}
