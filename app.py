import cv2
import numpy as np
from src.detection import Detection
from time import time
# Logging
import logging
logging.basicConfig(level=logging.INFO)

# from src.plotter import Plotter

""" 
Debug Notes: During development, the plotter will be turned off to test the detection and tracking.
"""
MODE = 'annotated'  # Use 'annotated' or any other mode defined in the Detection class
VIDEO_SOURCE = '_video/ride_the_d_5x.mp4'  # Path to your video file

def annotate_frame(frame, y_buffer):
    """ 
    Draw horizontal red line at the half x value

    Draw Red Circle high or low
    """ 
    # Draw horizontal red line at the half x value
    cv2.line(frame, (0, frame.shape[0] // 2), (frame.shape[1], frame.shape[0] // 2), (0, 0, 255), 10)
    try:
        
        # Draw Red Circle high or low
        high = 3*frame.shape[0] // 4
        middle = frame.shape[0] // 2
        low = frame.shape[0] // 4

        min_val = min(y_buffer)
        max_val = max(y_buffer)
        mean_val = np.mean(y_buffer)
        current_value = np.mean(np.array(y_buffer)[-int(0.25*len(y_buffer)):])

        # Normalize the mean
        normalized_mean = (current_value - mean_val) / (mean_val)
        print(normalized_mean)

        if normalized_mean > 0.01:
            cv2.circle(frame, (frame.shape[1] // 2, high), 200, (0, 0, 255), -1)
        elif normalized_mean < -0.01:
            cv2.circle(frame, (frame.shape[1] // 2, low), 200, (0, 0, 255), -1)
        else:
            cv2.circle(frame, (frame.shape[1] // 2, middle), 200, (0, 0, 255), -1)
    except:
        cv2.circle(frame, (frame.shape[1] // 2, middle), 200, (0, 0, 255), -1)

    return frame

def main():
    # Initialize detection and plotting
    detector = Detection()
    # plotter = Plotter()

    # Load video
    cap = cv2.VideoCapture(VIDEO_SOURCE)

    # Create a resizable window
    cv2.namedWindow('TreehornCV', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('TreehornCV', 800, 600)  # Adjust size as needed

    # y buffer
    y_buffer = []
    buffer_size = 10

    start_time = time() 
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Detect and annotate frame
        y_avg, test = detector.detect_and_annotate(frame, mode=MODE)

        if y_avg is not None:
            y_buffer.append(y_avg)
            if len(y_buffer) > buffer_size:
                y_buffer.pop(0)
        frame = annotate_frame(test, y_buffer)

        # Display annotated frame
        cv2.imshow('TreehornCV', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    # Release resources
    cap.release()
    cv2.destroyAllWindows()
    # plotter.finalize()

if __name__ == '__main__':
    # Logging
    logging.info("Starting TreehornCV")
    try:
        main()
    except Exception as e:
        logging.error(f"Error in main: {e}")
        raise e