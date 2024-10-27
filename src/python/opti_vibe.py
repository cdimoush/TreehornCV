# opti_vibe.py
import numpy as np
import cv2

class OptiVibe:
    def __init__(self):
        self.track_len = 5
        self.detect_interval = 5
        self.tracks = []
        self.frame_idx = 0
        self.next_id = 0
        self.displacement_threshold_percentage = 0.01
        self.prev_gray = None
        self.signal = 0
        self.speed = 0.1
        self.signal_threshold = 0.2
        self.last_time = 0.0

    def process_frame(self, frame, time, vibe_callback):
        processed_frame, vibe_signal = self._process_vibe_signal(frame, time, False)
        self.last_time = time
        vibe_callback(vibe_signal)

    def process_frame_debug(self, frame, time, vibe_callback, debug_callback):        
        processed_frame, vibe_signal = self._process_vibe_signal(frame, time, True)
        self.last_time = time
        vibe_callback(vibe_signal)
        debug_callback(processed_frame)

    def _process_vibe_signal(self, frame, time, debug):
        # Initialize processed_frame and vibe_signal
        vibe_signal = self.signal  # For now, we can set the vibe_signal to self.signal

        # Convert frame to grayscale
        frame_gray = self._convert_to_grayscale(frame, invert=False, sharpen=True)

        if self.prev_gray is not None and self.tracks_exist():
            self.process_existing_tracks(frame_gray)
            self.process_displacement(frame_gray)
            self.process_signal()
            if debug:
                processed_frame = self._annotate_frame(frame.copy())
            else:
                processed_frame = frame
        else:
            processed_frame = frame

        if self.should_detect_new_features():
            self.detect_new_features(frame_gray)

        self.frame_idx += 1
        self.prev_gray = frame_gray

        return processed_frame, self.signal

    def _convert_to_grayscale(self, frame, invert=False, sharpen=False):
        if sharpen:
            frame = cv2.GaussianBlur(frame, (0, 0), 3)
            frame = cv2.addWeighted(frame, 1.5, np.zeros(frame.shape, frame.dtype), 0, 0)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if invert:
            gray = cv2.bitwise_not(gray)
        return gray

    def tracks_exist(self):
        return len(self.tracks) > 0

    def process_existing_tracks(self, frame_gray):
        img0, img1 = self.prev_gray, frame_gray
        p0 = np.float32([tr[-1][1] for tr in self.tracks]).reshape(-1, 1, 2)
        p1, good = self.calculate_optical_flow(p0, img0, img1)
        self.update_tracks(p1, good)

    def calculate_optical_flow(self, p0, img0, img1):
        lk_params = dict(
            winSize=(10, 10),
            maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 5, 0.03)
        )
        # Calculate optical flow (forward)
        p1, st, err = cv2.calcOpticalFlowPyrLK(img0, img1, p0, None, **lk_params)
        # Calculate optical flow (backward)
        p0r, st, err = cv2.calcOpticalFlowPyrLK(img1, img0, p1, None, **lk_params)
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
        if len(track) < 2:
            return 0, 0
        # Loop index MUST start at 2!!!
        total_x_disp = sum(track[i][1][0] - track[i-1][1][0] for i in range(2, len(track)))
        total_y_disp = sum(track[i][1][1] - track[i-1][1][1] for i in range(2, len(track)))
        x_disp = 1 if total_x_disp > x_threshold else -1 if total_x_disp < -x_threshold else 0
        y_disp = 1 if total_y_disp > y_threshold else -1 if total_y_disp < -y_threshold else 0
        return x_disp, y_disp

    def should_detect_new_features(self):
        return self.frame_idx % self.detect_interval == 0

    def detect_new_features(self, frame_gray):
        feature_params = dict(
            maxCorners=100,
            qualityLevel=0.05,
            minDistance=14,
            blockSize=14
        )
        mask = self.create_feature_mask(frame_gray)
        p = cv2.goodFeaturesToTrack(frame_gray, mask=mask, **feature_params)
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
            cv2.circle(mask, (x, y), 5, 0, -1)
        return mask

    def process_signal(self):
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

    def _annotate_frame(self, vis):
        for tr in self.tracks:
            x, y = tr[-1][1]
            disp = tr[-1][2]
            color = self.assign_track_color(disp)
            self.annotate_frame_with_point(vis, x, y, color)
        self.annotate_frame_with_signal(vis)
        return vis

    def assign_track_color(self, disp):
        y_disp = disp[1]  # Only consider y displacement for color
        if y_disp == 1:
            color = (0, 0, 255)  # Red for moving down
        elif y_disp == -1:
            color = (255, 0, 0)  # Blue for moving up
        else:
            color = (0, 255, 0)  # Green for not moving significantly
        return color

    def annotate_frame_with_point(self, vis, x, y, color):
        cv2.circle(vis, (int(x), int(y)), 5, color, -1)

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
        cv2.rectangle(vis, top_left, bottom_right, (0, 0, 255), -1)  # Red color
