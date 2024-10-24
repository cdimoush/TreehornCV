import tkinter as tk
import cv2
import time
import threading
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from collections import deque
from queue import Queue, Empty
from src.python.test_algo import OpticalFlowLucasKanade, TestAlgoData

# Constants (initial conditions)
WINDOW_SIZE = 50  # Number of points in the rolling window

def run_tkinter(data_queue):
    # Launch the WaveGUI
    root = tk.Tk()
    app = WaveGUI(root, data_queue)
    root.mainloop()

def run_opencv(stop_event, data_queue, video_source=0):
    """
    Run OpenCV to capture video from a specified source.

    :param stop_event: threading.Event to signal when to stop the thread.
    :param data_queue: Queue for communication between threads.
    :param video_source: Source for video capture. Default is 0 (webcam).
                         Can be a file path for a local video.
    """
    # Open a connection to the video source
    cap = cv2.VideoCapture(video_source)

    if not cap.isOpened():
        print(f"Error: Could not open video source {video_source}.")
        return
    
    # Create an instance of the OpticalFlowLucasKanade class
    optical_flow = OpticalFlowLucasKanade()

    try:
        while not stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame.")
                break

            # Resize the frame to be a quarter of the monitor size
            frame = cv2.resize(frame, (640, 360))  # Assuming a 2560x1440 monitor

            # Process the frame and put data into the queue
            processed_frame, test_algo_data = optical_flow.process_frame(frame)
            data_queue.put(test_algo_data)

            # Display the frame
            cv2.imshow('Video Stream', processed_frame)

            # Exit when 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                stop_event.set()
                break

    except KeyboardInterrupt:
        stop_event.set()

    finally:
        cap.release()
        cv2.destroyAllWindows()

class WaveGUI:
    def __init__(self, root, data_queue):
        self.root = root
        self.data_queue = data_queue
        self.start_time_s = time.time()
        self.root.title("Wave Equation Visualization")
        self.root.geometry("800x600")

        # Create a time series plot for each attribute
        self.figure, self.axes = plt.subplots(len(TestAlgoData.DATA_ATTRIBUTES), 1, figsize=(8, 6))
        
        # Initialize the rolling buffer for time and number of attributes in TestAlgoData
        self.t_vals = deque(maxlen=WINDOW_SIZE)
        self.attr_vals = {attr: deque(maxlen=WINDOW_SIZE) for attr in TestAlgoData.DATA_ATTRIBUTES}  # Initialize attr_vals
        self.attr_lines = {}
        self.attr_axes = {}
        for i, attr in enumerate(TestAlgoData.DATA_ATTRIBUTES):
            self.attr_axes[attr] = self.axes[i]
            self.attr_lines[attr], = self.attr_axes[attr].plot([], [], lw=2)

        # Adjust spacing between the plots
        self.figure.tight_layout(pad=3.0)

        # Create a canvas and attach it to the Tkinter window
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.root)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # Start the animation
        self.check_queue()
        self.update_plot()

    def update_plot(self):
        for attr in TestAlgoData.DATA_ATTRIBUTES:
            # print(f'{attr}: {self.attr_vals[attr]}')
            self.attr_lines[attr].set_data(self.t_vals, self.attr_vals[attr])


        # Setup Autoscale and Autorange
        for ax in self.axes:
            ax.relim()
            ax.autoscale()
            ax.autoscale_view(tight=True)

        # Redraw the canvas
        self.canvas.draw()

        # # Schedule the next update
        self.root.after(10, self.update_plot)

    def check_queue(self):
        try:
            while True:
                # Get data from the queue
                test_algo_data = self.data_queue.get_nowait()
                if test_algo_data is not None:
                    self.update_data_from_queue(test_algo_data)
        except Empty:
            pass

        # Schedule the next queue check
        self.root.after(10, self.check_queue)

    def update_data_from_queue(self, data):
        self.t_vals.append(time.time() - self.start_time_s)
        for attr in TestAlgoData.DATA_ATTRIBUTES:
            self.attr_vals[attr].append(data[attr])

