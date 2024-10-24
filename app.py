import threading
import queue
from src.python.test_gui import run_tkinter, run_opencv

# Shared flag to signal threads to stop
stop_event = threading.Event()

# Queue for communication between threads
data_queue = queue.Queue()

def main():
    # Example: Use a local video file instead of the webcam
    # video_source = "_video/ride_the_d_5x.mp4"  # Replace with your video file path
    # video_source = "_video/ride_the_d.mp4"  # Replace with your video file path
    video_source = 0

    # Create threads for Tkinter and OpenCV
    opencv_thread = threading.Thread(target=run_opencv, args=(stop_event, data_queue, video_source))
    opencv_thread.start()

    run_tkinter(data_queue)
    stop_event.set()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Keyboard interrupt received, exiting.")
        stop_event.set()
