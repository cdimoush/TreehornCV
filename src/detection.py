import cv2
import numpy as np

class Detection:
    def __init__(self):
        # Initialize the background subtractor
        self.backSub = cv2.createBackgroundSubtractorMOG2(history=10, varThreshold=25, detectShadows=False)

    def detect_and_annotate(self, frame, mode='annotated'):
        # Apply background subtractor to get the foreground mask
        fg_mask = self.backSub.apply(frame)
        final_mask = fg_mask

        # Apply threshold to get binary image
        _, fg_mask = cv2.threshold(fg_mask, 127, 255, cv2.THRESH_BINARY)

        # Define morphological kernel
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

        # Remove noise
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel, iterations=2)

        # Fill small holes
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel, iterations=2)

        
        # final_mask = fg_mask
        threshold = 3
        # Y
        y_values = []
        for i in range(int(frame.shape[0]*0.5), frame.shape[0]):
            x_avg = final_mask[:, i].mean()
            if x_avg > threshold:
                y_values.append(i)

        if len(y_values) > 0:
            y_avg = np.mean(y_values) 
        else:
            y_avg = None

        return y_avg, final_mask
    

