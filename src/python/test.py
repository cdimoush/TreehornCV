import cv2
import numpy as np
from opti_vibe import OptiVibe

def vibe_callback(vibe_signal):
    print(f"Vibe Signal: {vibe_signal}")

def debug_callback(annotated_frame):
    cv2.imshow('Debug Frame', annotated_frame)

def main():
    opti_vibe = OptiVibe()
    cap = cv2.VideoCapture(0)  # Open the default webcam

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        time = cv2.getTickCount() / cv2.getTickFrequency()  # Get the current time in seconds
        opti_vibe.process_frame_debug(frame, time, vibe_callback, debug_callback)

        if cv2.waitKey(1) == 27:  # ESC key to exit
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()

