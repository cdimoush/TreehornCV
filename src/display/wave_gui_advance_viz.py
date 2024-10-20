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
GAIN_0 = 0

# Function to compute theta_double_dot = d2theta/dt^2 = sin(theta)
def get_theta_double_dot(theta, theta_dot, gain=0):
    return -np.sin(theta) + gain

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

def create_vector_field(gain=0):
    # Limit the theta range to -pi/2 to pi/2
    theta = np.linspace(-np.pi/2, np.pi/2, 10)  # theta values from -pi/2 to pi/2
    theta_dot_vals = np.linspace(-10, 10, 10)     # theta_dot (dtheta/dt) values

    # Create a meshgrid for theta and theta_dot
    Theta, Theta_dot = np.meshgrid(theta, theta_dot_vals)

    # Compute the vector field using the get_theta_double_dot function
    U = Theta_dot  # dtheta/dt
    V = get_theta_double_dot(Theta, Theta_dot, gain)  # d2theta/dt^2 using the function

    return Theta, Theta_dot, U, V

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

        # Initialize the rolling buffer for theta_dot
        self.theta_dot_vals = deque(maxlen=WINDOW_SIZE)

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
        self.theta_dot_vals.append(self.theta_dot)

        # Create the plot with an additional subplot
        self.figure, (self.ax, self.ax2) = plt.subplots(2, 1, figsize=(8, 6))

        # Adjust spacing between the plots
        self.figure.tight_layout(pad=3.0)
        
        # Main plot for theta vs time
        self.line, = self.ax.plot([], [], lw=2)
        self.ax.set_xlabel('Time (s)')
        self.ax.set_ylabel('Theta (rad)')

        # Highlight line for up key press
        self.up_key_highlight_line, = self.ax.plot([], [], lw=2, color='red', alpha=0.5)

        # Highlight line for down key press
        self.down_key_highlight_line, = self.ax.plot([], [], lw=2, color='blue', alpha=0.5)

        # Additional subplot for vector field
        self.ax2.set_title('2nd Order Phase Space')
        self.ax2.set_xlabel('theta')
        self.ax2.set_ylabel('theta_dot')

        # Pre-calculate and plot the vector field
        self.vector_fields = [
            create_vector_field(gain) for gain in [0, 10, -10]
        ]
        self.plot_vector_field(self.ax2, self.vector_fields[0])

        # Initialize scatter plot for dynamic point
        self.scatter, = self.ax2.plot([THETA_0], [THETA_DOT_0], 'bo')

        # Embed the plot in the Tkinter window
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.root)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

        # Bind key press and release events for the up and down arrow keys
        self.root.bind("<KeyPress-Up>", self.press_up_gain)
        self.root.bind("<KeyRelease-Up>", self.release_up_gain)
        self.root.bind("<KeyPress-Down>", self.press_down_gain)
        self.root.bind("<KeyRelease-Down>", self.release_down_gain)

        # Gain state
        self.gain = GAIN_0
        # Previous gain state for comparison
        self.previous_gain = self.gain


        # Start the animation
        self.update_plot()

    def plot_vector_field(self, ax, vector_field):
        Theta, Theta_dot, U, V = vector_field

        # Plot the vector field using quiver
        return ax.quiver(Theta, Theta_dot, U, V, color='r')

    def press_up_gain(self, event):
        # Set gain to 10x when the up arrow key is pressed
        self.gain = 10
        self.key_state = True
        
    def release_up_gain(self, event):
        # Revert gain to default when the up arrow key is released
        self.gain = GAIN_0
        self.key_state = False

    def press_down_gain(self, event):
        # Set gain to -10x when the down arrow key is pressed
        self.gain = -10
        self.key_state = True
        
    def release_down_gain(self, event):
        # Revert gain to default when the down arrow key is released
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
        self.theta_dot_vals.append(self.theta_dot)

        # Update the plot data for the main plot
        self.line.set_data(self.t_vals, self.theta_vals)

        # Update the scatter plot with the current theta and theta_dot
        self.scatter.set_data([self.theta_vals[-1]], [self.theta_dot_vals[-1]])

        # Check if the gain has changed
        if self.gain != self.previous_gain:
            # Remove the old quiver plot
            self.ax2.clear()
            self.scatter, = self.ax2.plot([], [], 'bo')

            # Create and plot a new vector field
            if self.gain == 10:
                self.quiver = self.plot_vector_field(self.ax2, self.vector_fields[1])
            elif self.gain == -10:
                self.quiver = self.plot_vector_field(self.ax2, self.vector_fields[2])
            else:
                self.quiver = self.plot_vector_field(self.ax2, self.vector_fields[0])

            # Update the scatter plot position
            self.scatter.set_data([self.theta_vals[-1]], [self.theta_dot_vals[-1]])

            # Update the previous gain
            self.previous_gain = self.gain

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
