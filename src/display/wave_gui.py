import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from collections import deque

# Constants (initial conditions)
THETA_0 = np.pi/2  # Initial theta (in radians)
THETA_DOT_0 = 0  # Initial angular velocity (theta_dot)
DELTA_T = 0.001  # Time step size
WINDOW_SIZE = 50  # Number of points in the rolling window
GAIN_0 = 1

# Function to compute theta_double_dot = d2theta/dt^2 = sin(theta)
def get_theta_double_dot(theta, theta_dot, gain=1):
    return -gain*np.sin(theta)

# Function to numerically solve d2theta/dt^2 = sin(theta) using Euler's method
def solve_theta(theta, theta_dot, dt, gain):
    theta_double_dot = get_theta_double_dot(theta, theta_dot, gain)

    # Update theta and theta_dot using Euler's method
    theta += theta_dot * dt
    theta_dot += theta_double_dot * dt

    # Limits on theta
    if theta > np.pi/2:
        theta = np.pi/2
        theta_dot = 0
    elif theta < -np.pi/2:
        theta = -np.pi/2
        theta_dot = 0

    return theta, theta_dot

# Main application class
class WaveGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Wave Equation Visualization")
        self.root.geometry("800x600")

        # Initialize the rolling buffer
        self.t_vals = deque(maxlen=WINDOW_SIZE)
        self.theta_vals = deque(maxlen=WINDOW_SIZE)

        # Initialize the rolling buffer for key state
        self.key_state_vals = deque(maxlen=WINDOW_SIZE)

        # Initial conditions
        self.theta = THETA_0
        self.theta_dot = THETA_DOT_0
        self.t = 0

        # Initial key state
        self.key_state = False

        # Populate initial values
        self.t_vals.append(self.t)
        self.theta_vals.append(self.theta)
        self.key_state_vals.append(self.key_state)

        # Create the plot
        self.figure, self.ax = plt.subplots()
        self.line, = self.ax.plot([], [], lw=2)
        self.ax.set_xlabel('Time (s)')
        self.ax.set_ylabel('Theta (rad)')

        # Highlight line for key press
        self.highlight_line, = self.ax.plot([], [], lw=2, color='red', alpha=0.5)

        # Embed the plot in the Tkinter window
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.root)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

        # Bind key press and release events for the up arrow key
        self.root.bind("<KeyPress-Up>", self.press_gain)
        self.root.bind("<KeyRelease-Up>", self.release_gain)

        # Gain state
        self.gain = GAIN_0

        # Start the animation
        self.update_plot()

    def press_gain(self, event):
        # Set gain to 10x when the up arrow key is pressed
        self.gain = 10 * GAIN_0
        self.key_state = True

    def release_gain(self, event):
        # Revert gain to default when the up arrow key is released
        self.gain = GAIN_0
        self.key_state = False

    def update_plot(self, factor=100):
        # Solve the differential equation with the current gain
        for _ in range(factor):
            self.theta, self.theta_dot = solve_theta(self.theta, self.theta_dot, DELTA_T, self.gain)
            self.t += DELTA_T

        # Update the rolling buffer
        self.t_vals.append(self.t)
        self.theta_vals.append(self.theta)
        self.key_state_vals.append(self.key_state)

        # Update the plot data
        self.line.set_data(self.t_vals, self.theta_vals)

        # Highlight portions of the plot where the key is pressed)
        # highlight_t = [t for t, state in zip(self.t_vals, self.key_state_vals) if state]
        highlight_theta = [np.pi/2 if state else -np.pi/2 for state in self.key_state_vals]
        self.highlight_line.set_data(self.t_vals, highlight_theta)

        self.ax.relim()
        self.ax.set_autoscaley_on(False)  # Disable autoscaling for y-axis
        self.ax.autoscale_view(scalex=True, scaley=False)  # Only autoscale x-axis
        self.ax.set_ylim(-np.pi, np.pi)
        
        # Redraw the canvas
        self.canvas.draw()

        # Schedule the next update
        self.root.after(int(DELTA_T * 1000), self.update_plot)

    def stop_updates(self):
        self.root.quit()
        self.root.destroy()

# Main execution
if __name__ == "__main__":
    root = tk.Tk()
    app = WaveGUI(root)
    root.protocol("WM_DELETE_WINDOW", app.stop_updates)  # Ensure updates stop on close
    root.mainloop()
