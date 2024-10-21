# main.py

import cv2
from optical_flow import OpticalFlowLucasKanade

def main():
    # Open a connection to the webcam (0 is usually the default camera)
    cap = cv2.VideoCapture(0)

    # Check if the webcam is opened correctly
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    # Create an instance of the OpticalFlowLucasKanade class
    optical_flow = OpticalFlowLucasKanade()

    try:
        while True:
            # Capture frame-by-frame
            ret, frame = cap.read()

            # If frame reading was not successful, break the loop
            if not ret:
                print("Error: Could not read frame.")
                break

            # Process frame for optical flow
            processed_frame = optical_flow.process_frame(frame)

            # Display the resulting frame
            cv2.imshow('Webcam Stream with Optical Flow', processed_frame)

            # Exit streaming when 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Exiting streaming.")
                break

    except KeyboardInterrupt:
        # Handle any keyboard interrupts
        print("Streaming interrupted by user.")

    finally:
        # Release the capture and close any OpenCV windows
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    print("Starting webcam demo with optical flow")
    main()
