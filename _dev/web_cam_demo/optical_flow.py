# optical_flow.py

import cv2
import numpy as np

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

        # Stail Count
        self.stail_count = 0
        self.stail_threshold = 100

    def get_p0(self, frame, method="goodFeaturesToTrack"):
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
            n = 20  
            x_coords = np.linspace(0.25*w, 0.75*w, n)
            y_coords = np.linspace(0.25*h, 0.75*h, n)
            x_grid, y_grid = np.meshgrid(x_coords, y_coords)
            p0 = np.column_stack((x_grid.ravel(), y_grid.ravel())).astype(np.float32)
            return p0.reshape(-1, 1, 2)

    def analyze_velocity(self, good_old, good_new):
        """
        Analyzes the velocity to determine convergence or divergence.
        Returns "converge", "diverge", or "".
        """
        movement_threshold = 10

        # Compute displacement vectors
        dx_dy = good_new - good_old  # Shape: (N, 2)

        # Filter outpoint with small displacement
        mask = np.linalg.norm(dx_dy, axis=1) > movement_threshold
        points_above_threshold = len(dx_dy[mask])
        if points_above_threshold < 2:
            return ""
        
        dx_dy = dx_dy[mask]

        # Compute the reference center as the average of all p0 values
        center = np.mean(good_old, axis=0)

        # Compute position vectors relative to the center
        r_xy = good_old[mask] - center  # Shape: (N, 2)


        # Compute dot products
        dots = np.einsum('ij,ij->i', dx_dy, r_xy)

        # Compute average dot product
        average_dot = np.mean(dots)

        # Determine movement trend
        threshold = 0.2  # You may need to adjust this based on your data
        if average_dot > threshold:
            movement_trend = "diverge"
        elif average_dot < -threshold:
            movement_trend = "converge"
        else:
            movement_trend = ""

        return movement_trend

    def process_frame(self, frame):
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if self.old_gray is None:
            # First frame initialization
            self.old_gray = frame_gray.copy()
            self.p0 = self.get_p0(frame_gray)
            self.mask = np.zeros_like(frame)
            return frame

        # Calculate optical flow
        p1, st, err = cv2.calcOpticalFlowPyrLK(self.old_gray, frame_gray,
                                               self.p0, None, **self.lk_params)
        
        # Check if points are found
        if p1 is not None and len(p1[st == 1]) > 5:
            # Select good points
            good_new = p1[st == 1]
            good_old = self.p0[st == 1].reshape(-1, 2)

            # Analyze velocity
            movement_trend = self.analyze_velocity(good_old, good_new)

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
            if movement_trend != "":
                self.output_text = movement_trend


            color = (255, 255, 255)
            if self.output_text == 'converge':
                color = (0, 0, 255)
            cv2.putText(img, self.output_text, (50, 50), font, 1,
                        color, 2, cv2.LINE_AA)

            # Update the previous frame and points
            self.old_gray = frame_gray.copy()
            self.p0 = good_new.reshape(-1, 1, 2)

            # Update stail count
            self.stail_count += 1
            if self.stail_count > self.stail_threshold:
                self.stail_count = 0
                self.p0 = self.get_p0(frame_gray)

            return img
        else:
            # Re-initialize if no points are found
            self.old_gray = frame_gray.copy()
            self.p0 = self.get_p0(frame_gray)
            self.mask = np.zeros_like(frame)
            return frame
