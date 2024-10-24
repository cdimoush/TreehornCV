# optical_flow.py

import cv2
import numpy as np
from collections import deque

class OpticalFlowLucasKanade:
    def __init__(self):
        # Parameters for ShiTomasi corner detection
        self.feature_params = dict(maxCorners=100,
                                   qualityLevel=0.05,
                                   minDistance=5,
                                   blockSize=5)

        # Parameters for Lucas-Kanade optical flow
        self.lk_params = dict(winSize=(15, 15),
                              maxLevel=2,
                              criteria=(cv2.TERM_CRITERIA_EPS |
                                        cv2.TERM_CRITERIA_COUNT, 10, 0.03))

        # Random colors for visualization
        self.color = np.random.randint(0, 255, (100, 3))

        # Text
        self.output_text = ""

        # Initialize variables
        self.old_gray = None
        self.p0 = None
        self.mask = None
        self.decay_factor = 0.9  # Decay factor for fading trails

        # Rolling buffer
        self.rolling_gain = deque(maxlen=5)
        
        # Stail Count
        self.stail_count = 0
        self.stail_threshold = 100

    def get_p0(self, frame, method="uniform"):
        """
        Returns the points to track
        """
        # Check method for `goodFeaturesToTrack` or `uniform`
        if method not in ["goodFeaturesToTrack", "uniform"]:
            raise ValueError("Invalid method. Use 'goodFeaturesToTrack' or 'uniform'.")

        if method == "goodFeaturesToTrack":
            return cv2.goodFeaturesToTrack(frame, mask=None,
                                           **self.feature_params)
        else:
            h, w = frame.shape
            # Define the grid step size
            n = 15  
            x_coords = np.linspace(0.15*w, 0.85*w, n)
            y_coords = np.linspace(0.15*h, 0.85*h, n)
            x_grid, y_grid = np.meshgrid(x_coords, y_coords)
            p0 = np.column_stack((x_grid.ravel(), y_grid.ravel())).astype(np.float32)
            return p0.reshape(-1, 1, 2)

    def analyze_velocity(self, good_old, good_new):
        """
        Analyzes the velocity to determine convergence or divergence.
        Returns "converge", "diverge", or "".
        """

        # Compute displacement vectors
        dx_dy = good_new - good_old  # Shape: (N, 2)

        print(dx_dy.shape)

        movement_threshold = 5
        # Filter outpoint with small displacement
        mag = np.linalg.norm(dx_dy, axis=1)
        if mag.mean() > movement_threshold:
            self.rolling_gain.append(1)
        else:
            self.rolling_gain.append(-1)

        return np.sign(np.mean(self.rolling_gain))

    def process_frame(self, frame):
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if self.old_gray is None:
            # First frame initialization
            self.old_gray = frame_gray.copy()
            self.p0 = self.get_p0(frame_gray)
            self.mask = np.zeros_like(frame)
            return frame, 0

        # Calculate optical flow
        p1, st, err = cv2.calcOpticalFlowPyrLK(self.old_gray, frame_gray,
                                               self.p0, None, **self.lk_params)
        
        # Check if points are found
        if p1 is not None and len(p1[st == 1]) > 5:
            # Select good points
            good_new = p1[st == 1]
            good_old = self.p0[st == 1].reshape(-1, 2)

            # Analyze velocity
            gain = self.analyze_velocity(good_old, good_new)

            # Fade the mask to create decaying trails
            self.mask = cv2.addWeighted(self.mask, self.decay_factor,
                                        np.zeros_like(self.mask), 0, 0)

            # Draw the tracks
            for i, (new, old) in enumerate(zip(good_new, good_old)):
                a, b = new.ravel()
                c, d = old.ravel()
                self.mask = cv2.line(self.mask, (int(a), int(b)), (int(c), int(d)),
                                     self.color[i % 100].tolist(), 2)
                frame = cv2.circle(frame, (int(a), int(b)), 5,
                                   self.color[i % 100].tolist(), -1)
            img = cv2.add(frame, self.mask)

            # Add text annotation to frame
            font = cv2.FONT_HERSHEY_SIMPLEX
            color = (255, 255, 255)
            cv2.putText(img, str(gain), (50, 50), font, 1,
                        color, 2, cv2.LINE_AA)

            # Update the previous frame and points
            self.old_gray = frame_gray.copy()
            self.p0 = good_new.reshape(-1, 1, 2)

            # Update stail count
            self.stail_count += 1
            if self.stail_count > self.stail_threshold:
                self.stail_count = 0
                self.p0 = self.get_p0(frame_gray)

            return img, gain
        else:
            # Re-initialize if no points are found
            self.old_gray = frame_gray.copy()
            self.p0 = self.get_p0(frame_gray)
            self.mask = np.zeros_like(frame)
            return frame, 0
