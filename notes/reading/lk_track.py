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
    maxCorners=500,     # Maximum number of corners to return
    qualityLevel=0.3,   # Quality level for corner detection
    minDistance=7,      # Minimum possible Euclidean distance between the returned corners
    blockSize=7         # Size of an average block for computing a derivative covariance matrix
)

class App:
    def __init__(self, video_src):
        self.track_len = 10              # Maximum length of the trajectories
        self.detect_interval = 5         # Interval to detect new features
        self.tracks = []                 # List to store the trajectories
        self.cam = cv.VideoCapture(video_src)  # Video capture object
        self.frame_idx = 0               # Frame index counter

    def run(self):
        while True:
            _ret, frame = self.cam.read()  # Read a new frame
            if not _ret:
                break  # Exit if no frame is captured
            frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)  # Convert frame to grayscale
            vis = frame.copy()  # Copy of the frame for visualization

            if len(self.tracks) > 0:
                # Previous and current grayscale images
                img0, img1 = self.prev_gray, frame_gray
                # Get the last known positions of the tracked points
                p0 = np.float32([tr[-1] for tr in self.tracks]).reshape(-1, 1, 2)
                # Calculate optical flow (forward)
                p1, _st, _err = cv.calcOpticalFlowPyrLK(img0, img1, p0, None, **lk_params)
                # Calculate optical flow (backward)
                p0r, _st, _err = cv.calcOpticalFlowPyrLK(img1, img0, p1, None, **lk_params)
                # Compute the difference between the original points and the back-tracked points
                d = abs(p0 - p0r).reshape(-1, 2).max(-1)
                # Identify good points based on the difference
                good = d < 1
                new_tracks = []
                # Update the trajectories with the new points
                for tr, (x, y), good_flag in zip(self.tracks, p1.reshape(-1, 2), good):
                    if not good_flag:
                        continue
                    tr.append((x, y))
                    if len(tr) > self.track_len:
                        del tr[0]  # Keep the trajectory length under `track_len`
                    new_tracks.append(tr)
                    # Draw a small circle at the new point position
                    cv.circle(vis, (int(x), int(y)), 2, (0, 255, 0), -1)
                self.tracks = new_tracks
                # Draw the trajectories
                for tr in self.tracks:
                    cv.polylines(vis, [np.int32(tr)], False, (0, 255, 0))
                # Display the number of tracks
                cv.putText(vis, f'Track count: {len(self.tracks)}', (20, 20),
                           cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0))

            # Every `detect_interval` frames, detect new features to track
            if self.frame_idx % self.detect_interval == 0:
                # Create a mask to avoid re-detecting points close to existing ones
                mask = np.zeros_like(frame_gray)
                mask[:] = 255
                for x, y in [np.int32(tr[-1]) for tr in self.tracks]:
                    cv.circle(mask, (x, y), 5, 0, -1)
                # Detect new good features to track
                p = cv.goodFeaturesToTrack(frame_gray, mask=mask, **feature_params)
                if p is not None:
                    # Initialize new tracks with the detected points
                    for x, y in np.float32(p).reshape(-1, 2):
                        self.tracks.append([(x, y)])

            self.frame_idx += 1  # Increment the frame index
            self.prev_gray = frame_gray  # Store the current frame as previous
            cv.imshow('Lucas-Kanade Tracker', vis)  # Display the visualization

            # Exit when ESC key is pressed
            ch = cv.waitKey(1)
            if ch == 27:
                break

def main():
    import sys
    # Use the video source provided as a command-line argument; default to 0 (webcam)
    try:
        video_src = sys.argv[1]
    except IndexError:
        video_src = 0

    # Create an instance of the App and run it
    App(video_src).run()
    print('Done')
    cv.destroyAllWindows()

if __name__ == '__main__':
    print(__doc__)
    main()
