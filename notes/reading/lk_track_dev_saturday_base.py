#!/usr/bin/env python

'''
Lucas-Kanade Tracker
====================

This script demonstrates the Lucas-Kanade method for sparse optical flow tracking.
It uses `goodFeaturesToTrack` for initializing points to track and employs back-tracking
for match verification between frames.

Usage:
------
Run the script and optionally provide a video source as an argument. If no source is provided,
the default webcam will be used.

    python lk_track.py [<video_source>]

Keys:
-----
ESC - Exit the program
'''

import numpy as np
import cv2 as cv

# Parameters for Lucas-Kanade optical flow
lk_params = dict(
    winSize=(15, 15),  # Size of the search window at each pyramid level
    maxLevel=2,        # 0-based maximal pyramid level number
    criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03)  # Termination criteria
)

# Parameters for Shi-Tomasi corner detection used by `goodFeaturesToTrack`
feature_params = dict(
    maxCorners=100,     # Maximum number of corners to return
    qualityLevel=0.05,   # Quality level for corner detection
    minDistance=14,      # Minimum possible Euclidean distance between the returned corners
    blockSize=14         # Size of an average block for computing a derivative covariance matrix
)

class App:
    def __init__(self, video_src):
        self.track_len = 5              # Maximum length of the trajectories
        self.detect_interval = 5         # Interval to detect new features
        self.tracks = []                 # List to store the trajectories
        self.cam = self.initialize_video_capture(video_src)  # Video capture object
        self.frame_idx = 0               # Frame index counter
        self.next_id = 0                 # Unique ID for each new point
        self.y_displacement_threshold = 10  # Threshold for significant y-displacement
        self.prev_gray = None            # Previous grayscale frame

    def initialize_video_capture(self, video_src):
        return cv.VideoCapture(video_src)

    def release_resources(self):
        self.cam.release()
        cv.destroyAllWindows()

    def run(self):
        while True:
            ret, frame = self.read_frame()
            if not ret:
                break
            frame_gray = self.convert_to_grayscale(frame, invert=False, sharpen=True)
            vis = frame.copy()

            if self.tracks_exist():
                self.process_existing_tracks(frame_gray, vis)

            if self.should_detect_new_features():
                self.detect_new_features(frame_gray)

            self.frame_idx += 1
            self.prev_gray = frame_gray
            cv.imshow('Lucas-Kanade Tracker', vis)

            if self.handle_exit_condition():
                break

        self.release_resources()

    def read_frame(self):
        ret, frame = self.cam.read()
        return ret, frame

    def convert_to_grayscale(self, frame, invert=False, sharpen=False):
        gray = frame
        if sharpen:
            gray = cv.GaussianBlur(gray, (0, 0), 3)
            gray = cv.addWeighted(gray, 1.5, np.zeros(gray.shape, gray.dtype), 0, 0)
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        if invert:
            gray = cv.bitwise_not(gray)
        return gray

    def tracks_exist(self):
        return len(self.tracks) > 0

    def process_existing_tracks(self, frame_gray, vis):
        img0, img1 = self.prev_gray, frame_gray
        p0 = np.float32([tr[-1][1] for tr in self.tracks]).reshape(-1, 1, 2)
        p1, good = self.calculate_optical_flow(p0, img0, img1)
        self.update_tracks(p1, good)
        self.update_visualization(vis)

    def calculate_optical_flow(self, p0, img0, img1):
        # Calculate optical flow (forward)
        p1, st, err = cv.calcOpticalFlowPyrLK(img0, img1, p0, None, **lk_params)
        # Calculate optical flow (backward)
        p0r, st, err = cv.calcOpticalFlowPyrLK(img1, img0, p1, None, **lk_params)
        # Compute the difference between the original points and the back-tracked points
        d = abs(p0 - p0r).reshape(-1, 2).max(-1)
        # Identify good points based on the difference
        good = d < 1
        return p1, good

    def update_tracks(self, p1, good):
        new_tracks = []
        for tr, (x, y), good_flag in zip(self.tracks, p1.reshape(-1, 2), good):
            if not good_flag:
                continue
            tr.append((tr[0][0], (x, y)))  # Keep the ID and update the position
            if len(tr) > self.track_len:
                del tr[1]
            new_tracks.append(tr)
        self.tracks = new_tracks

    def update_visualization(self, vis):
        for tr in self.tracks:
            x, y = tr[-1][1]
            disp = self.calculate_y_displacement(tr)
            color = self.assign_track_color(disp)
            self.draw_feature_point(vis, x, y, color, tr[0][0])
            self.draw_trajectory(vis, tr)
        self.display_track_count(vis)

    def calculate_y_displacement(self, track):
        """
        IMPORTANT: The index 0 and 1 should not be used during the displacement calculation loop as they seem
        to coorespond to a point that is not moving or streaming with buffer. Start from index 2.
        """
        if len(track) < 2:
            return 0
        # Weird Behavior, Start from index 2
        total_disp = sum(track[i][1][1] - track[i-1][1][1] for i in range(2, len(track)))
        if abs(total_disp) > self.y_displacement_threshold:
            return 1 if total_disp > 0 else -1
        return 0

    def assign_track_color(self, disp):
        if disp == 1:
            color = (0, 0, 255)  # Red for moving down
        elif disp == -1:
            color = (255, 0, 0)  # Blue for moving up
        else:
            color = (0, 255, 0)  # Green for not moving significantly
        return color

    def draw_feature_point(self, vis, x, y, color, track_id):
        cv.circle(vis, (int(x), int(y)), 2, color, -1)
        cv.putText(vis, str(track_id), (int(x) + 5, int(y) - 5),
                   cv.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

    def draw_trajectory(self, vis, track):
        pts = np.int32([pt[1] for pt in track])
        cv.polylines(vis, [pts], False, (0, 255, 0))

    def display_track_count(self, vis):
        cv.putText(vis, f'Track count: {len(self.tracks)}', (20, 20),
                   cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0))

    def should_detect_new_features(self):
        return self.frame_idx % self.detect_interval == 0

    def detect_new_features(self, frame_gray):
        mask = self.create_feature_mask(frame_gray)
        p = cv.goodFeaturesToTrack(frame_gray, mask=mask, **feature_params)
        if p is not None:
            for x, y in np.float32(p).reshape(-1, 2):
                self.tracks.append([(self.next_id, (x, y))])
                self.next_id += 1

    def create_feature_mask(self, frame_gray):
        mask = np.zeros_like(frame_gray)
        mask[:] = 255
        for tr in self.tracks:
            x, y = np.int32(tr[-1][1])
            cv.circle(mask, (x, y), 5, 0, -1)
        return mask

    def handle_exit_condition(self):
        ch = cv.waitKey(1)
        if ch == 27:
            return True
        return False

def main():
    import sys
    try:
        video_src = sys.argv[1]
    except IndexError:
        video_src = 0

    App(video_src).run()
    print('Done')

if __name__ == '__main__':
    print(__doc__)
    main()
