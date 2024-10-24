import tkinter as tk
import cv2
import threading
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from collections import deque
from queue import Queue, Empty
from src.python.free_fall import solve_free_fall
from src.python.algo import OpticalFlowLucasKanade

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
            processed_frame, gain = optical_flow.process_frame(frame)
            data_queue.put(gain)

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
        self.root.title("Wave Equation Visualization")
        self.root.geometry("800x600")

        # Initialize the rolling buffer
        self.t_vals = deque(maxlen=WINDOW_SIZE)
        self.theta_vals = deque(maxlen=WINDOW_SIZE)
        self.theta_dot_vals = deque(maxlen=WINDOW_SIZE)

        # Initial conditions
        self.theta = None
        self.theta_dot = None
        self.gain = -1
        self.t = 0

        # Populate initial values
        self.t_vals.append(self.t)
        self.theta_vals.append(self.theta)
        self.theta_dot_vals.append(self.theta_dot)

        # Create the plot with an additional subplot
        self.figure, (self.ax, self.ax2) = plt.subplots(2, 1, figsize=(8, 6))

        # Adjust spacing between the plots
        self.figure.tight_layout(pad=3.0)

        # Main plot for theta vs time
        self.line, = self.ax.plot([], [], lw=2)
        self.ax.set_xlabel('Time (s)')
        self.ax.set_ylabel('Theta (rad)')

        # Additional subplot for phase space
        self.ax2.set_title('Phase Space')
        self.ax2.set_xlabel('Theta')
        self.ax2.set_ylabel('Theta_dot')
        self.scatter, = self.ax2.plot([], [], 'bo')

        # Embed the plot in the Tkinter window
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.root)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

        # Start the animation
        self.check_queue()
        self.update_plot()

    def update_plot(self, factor=100):
        # Solve the differential equation
        for _ in range(factor):
            self.theta, self.theta_dot, dt = solve_free_fall(self.theta, self.theta_dot, self.gain)
            self.t += dt

        # Update the rolling buffer
        self.t_vals.append(self.t)
        self.theta_vals.append(self.theta)
        self.theta_dot_vals.append(self.theta_dot)

        print(f'theta: {self.theta}, theta_dot: {self.theta_dot}')

        # Update the plot data for the main plot
        self.line.set_data(self.t_vals, self.theta_vals)

        # Update the scatter plot with the current theta and theta_dot
        self.scatter.set_data([self.theta], [self.theta_dot])

        self.ax.relim()
        self.ax.set_autoscaley_on(False)  # Disable autoscaling for y-axis
        self.ax.autoscale_view(scalex=True, scaley=True)  # Only autoscale x-axis
        # self.ax.set_ylim(-np.pi, np.pi)

        self.ax2.relim()
        self.ax2.set_autoscaley_on(False)
        self.ax2.set_autoscalex_on(False)
        self.ax2.autoscale_view(scalex=True, scaley=True)
        # self.ax2.set_xlim(-2*np.pi, 2*np.pi)
        # self.ax2.set_ylim(-2*np.pi, 2*np.pi)

        # Redraw the canvas
        self.canvas.draw()

        # Schedule the next update
        self.root.after(int(dt * 1000), self.update_plot)

    def check_queue(self):
        try:
            while True:
                # Get data from the queue
                data = self.data_queue.get_nowait()
                # Process the data (e.g., update GUI elements)
                self.gain = data
        except Empty:
            pass

        # Schedule the next queue check
        self.root.after(100, self.check_queue)
