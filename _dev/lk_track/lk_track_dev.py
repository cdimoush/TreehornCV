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
    qualityLevel=0.05,  # Quality level for corner detection
    minDistance=14,     # Minimum possible Euclidean distance between the returned corners
    blockSize=14        # Size of an average block for computing a derivative covariance matrix
)

class App:
    def __init__(self, video_src):
        self.track_len = 5
        self.detect_interval = 5
        self.tracks = []  # Simplified track structure
        self.cam = self.initialize_video_capture(video_src)
        self.frame_idx = 0
        self.next_id = 0
        self.displacement_threshold_percentage = 0.01
        self.prev_gray = None
        self.signal = 0
        self.speed = 0.1
        self.signal_threshold = 0.2

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
                self.process_existing_tracks(frame_gray)
                self.process_displacement(frame_gray)
                self.process_signal()
                self.annotate_frame(vis)

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
        if sharpen:
            frame = cv.GaussianBlur(frame, (0, 0), 3)
            frame = cv.addWeighted(frame, 1.5, np.zeros(frame.shape, frame.dtype), 0, 0)
        
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        if invert:
            gray = cv.bitwise_not(gray)
        return gray

    def tracks_exist(self):
        return len(self.tracks) > 0

    def process_existing_tracks(self, frame_gray):
        img0, img1 = self.prev_gray, frame_gray
        p0 = np.float32([tr[-1][1] for tr in self.tracks]).reshape(-1, 1, 2)
        p1, good = self.calculate_optical_flow(p0, img0, img1)
        self.update_tracks(p1, good)

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
            tr.append((tr[0][0], (x, y), (0, 0)))  # Initialize with zero displacement
            if len(tr) > self.track_len:
                del tr[1]
            new_tracks.append(tr)
        self.tracks = new_tracks

    def process_displacement(self, frame):
        frame_height, frame_width = frame.shape[:2]
        x_threshold = frame_width * self.displacement_threshold_percentage
        y_threshold = frame_height * self.displacement_threshold_percentage

        for tr in self.tracks:
            x_disp, y_disp = self.calculate_displacement(tr, x_threshold, y_threshold)
            # Store displacement in the track
            if len(tr[-1]) == 2:
                # Append displacement values to the last point in the track
                tr[-1] = tr[-1] + ((x_disp, y_disp),)
            else:
                # Update the displacement values
                tr[-1] = (tr[-1][0], tr[-1][1], (x_disp, y_disp))

    def calculate_displacement(self, track, x_threshold, y_threshold):
        """
        Calculate the horizontal and vertical displacement of a track.
        """
        if len(track) < 2:
            return 0, 0
        # Loop index MUST start at 2!!! Do not delete comment or change start index!!!
        total_x_disp = sum(track[i][1][0] - track[i-1][1][0] for i in range(2, len(track)))
        total_y_disp = sum(track[i][1][1] - track[i-1][1][1] for i in range(2, len(track)))
        x_disp = 1 if total_x_disp > x_threshold else -1 if total_x_disp < -x_threshold else 0
        y_disp = 1 if total_y_disp > y_threshold else -1 if total_y_disp < -y_threshold else 0
        return x_disp, y_disp

    def assign_track_color(self, disp):
        y_disp = disp[1]  # Only consider y displacement for color
        if y_disp == 1:
            color = (0, 0, 255)  # Red for moving down
        elif y_disp == -1:
            color = (255, 0, 0)  # Blue for moving up
        else:
            color = (0, 255, 0)  # Green for not moving significantly
        return color

    def should_detect_new_features(self):
        return self.frame_idx % self.detect_interval == 0

    def detect_new_features(self, frame_gray):
        mask = self.create_feature_mask(frame_gray)
        p = cv.goodFeaturesToTrack(frame_gray, mask=mask, **feature_params)
        if p is not None:
            for x, y in np.float32(p).reshape(-1, 2):
                # Initialize the track with displacement value of 0
                self.tracks.append([(self.next_id, (x, y), (0, 0))])
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
        
    def process_signal(self):
        """
        Compute the average y-displacement across all tracks and update the signal.
        """
        if not self.tracks:
            return

        total_y_disp = 0
        count = 0

        for tr in self.tracks:
            # Check if displacement data is available
            if len(tr[-1]) >= 3:
                y_disp = tr[-1][2][1]  # Get y-displacement from the last point in the track
                if y_disp != 0:
                    total_y_disp += y_disp
                    count += 1

        if count == 0:
            average_y_disp = 0
        else:
            average_y_disp = total_y_disp / count

        # Update the signal based on the average y-displacement and signal threshold
        if abs(average_y_disp) > self.signal_threshold:
            if average_y_disp > 0:
                self.signal = min(self.signal + self.speed, 1)
            else:
                self.signal = max(self.signal - self.speed, 0)

    def annotate_frame(self, vis):
        for tr in self.tracks:
            x, y = tr[-1][1]
            disp = tr[-1][2]
            color = self.assign_track_color(disp)
            self.annotate_frame_with_point(vis, x, y, color)
        self.annotate_frame_with_signal(vis)

    def annotate_frame_with_point(self, vis, x, y, color):
        cv.circle(vis, (int(x), int(y)), 5, color, -1)

    def annotate_frame_with_signal(self, vis):
        """
        Draw a vertical bar on the right-hand side of the screen.
        The height of the bar is proportional to the signal value.
        """
        height, width = vis.shape[:2]
        bar_width = 20  # Width of the bar
        bar_height = int(height * self.signal)  # Height of the bar based on the signal
        top_left = (width - bar_width - 10, height - bar_height)  # Top-left corner of the bar
        bottom_right = (width - 10, height)  # Bottom-right corner of the bar

        # Draw the bar
        cv.rectangle(vis, top_left, bottom_right, (0, 0, 255), -1)  # Red color

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
