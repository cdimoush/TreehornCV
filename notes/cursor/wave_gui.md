# Wave GUI
This project will include a simple GUI for visualizing the numerical solution of a differential equation.

The differential equation will vary but it will always be second order and time dependent.

So d2_theta/dt2 = f(theta, d_theta/dt, t)

The solution will be a function of time.

The GUI will allow you to plot the solution on a graph.

## What to Use
Ideally you can use matplotlib and tkinter, but if this is going to be problematic for smooth animation or easy development then other libraries can be used.

## Requirements
The program will execute from a single python file. The program will continue to execute to close or keyboard interupt.

The program will have a rolling window of the solution.

## Features
I need to had functionality like buttons after the initial setup. Don't worry about this initially but will be required in the future so use this for choosing libraries.

## Concept Code
Here is the concept code for making a plot on a tkinter window. However this is NOT the solution. A rolling buffer of solution will be required. The rolling window will be used for updating the plot. You can use a data type from collections for the buffer. 

You also need to globally track the time. The time should continue to be updated by a time step.
```python
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk

# Constants (initial conditions)
THETA_0 = np.pi  # Initial theta (in radians)
THETA_DOT_0 = -1  # Initial angular velocity (theta_dot)
delta_t = 0.01  # Time step size

# Function to compute theta_double_dot = d2theta/dt^2 = sin(theta)
def get_theta_double_dot(theta, theta_dot):
    return np.sin(theta)

# Function to numerically solve d2theta/dt^2 = sin(theta) using Euler's method
def solve_theta(theta, theta_dot, dt)
=    theta_double_dot = get_theta_double_dot(theta, theta_dot)

    # Update theta and theta_dot using Euler's method
    theta += theta_dot * delta_t
    theta_dot += theta_double_dot * delta_t

    return theta, theta_dot

# Main application class
class WaveGUI:
    def __init__(self, root):
        # Initialize the main window using tkinter
        # (Code HERE)

        # Initialize the plot and data structures here
        # (Code HERE)

        # Start the animation
        self.update_plot()

    def update_plot(self):
        pass

    def stop_updates(self):
        pass

# Main execution
if __name__ == "__main__":
    root = tk.Tk()
    app = WaveGUI(root)
    root.protocol("WM_DELETE_WINDOW", app.stop_updates)  # Ensure updates stop on close
    root.mainloop()

```

## Planning
Make plans for the wholes in the whole app!

**Detailed Plan for Completing the Wave GUI Application**

1. **Import Necessary Modules**

   - **Core Libraries:**
     - `import numpy as np` for numerical computations.
     - `import matplotlib.pyplot as plt` for plotting.
     - `import tkinter as tk` for GUI components.
   - **Matplotlib Backend for Tkinter:**
     - `from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg` to embed Matplotlib plots in Tkinter.
   - **Rolling Buffer Implementation:**
     - `from collections import deque` for an efficient rolling window (buffer) data structure.
   - **Optional Libraries (if needed):**
     - `import time` for precise time control (if required).
     - `from scipy.integrate import odeint` or `solve_ivp` for more accurate numerical solvers than Euler's method.

2. **Define Global Variables and Initial Conditions**

   - **Initial Conditions:**
     - `THETA_0 = np.pi` — Initial angle (θ) in radians.
     - `THETA_DOT_0 = -1` — Initial angular velocity (θ̇).
   - **Time Variables:**
     - `delta_t = 0.01` — Time step size.
     - `t = 0` — Initialize global time variable.
   - **Rolling Window Parameters:**
     - `WINDOW_SIZE = 500` — Number of points to display in the rolling window.

3. **Implement the Numerical Solver Function**

   - **Define the Differential Equation Function:**
     - `def get_theta_double_dot(theta, theta_dot, t):`
       - Return `np.sin(theta)` or a more general `f(theta, theta_dot, t)`.
   - **Implement an Improved Numerical Solver:**
     - Replace Euler's method with a more accurate method like Runge-Kutta (e.g., `RK4`).
     - Alternatively, use `scipy.integrate.solve_ivp` for adaptive time-stepping and better accuracy.
   - **Modify `solve_theta` Function:**
     - Include time `t` as an argument.
     - Return updated `theta`, `theta_dot`, and incremented `t`.
     - Ensure consistency in units and scaling.

4. **Initialize the Rolling Buffer**

   - **Create Deques for Data Storage:**
     - `self.t_vals = deque(maxlen=WINDOW_SIZE)`
     - `self.theta_vals = deque(maxlen=WINDOW_SIZE)`
   - **Populate Deques with Initial Values:**
     - Append initial conditions to deques.
     - Ensure deques are thread-safe if using multi-threading.

5. **Initialize the Main Window Using Tkinter**

   - **Set Up the Root Window:**
     - `self.root = root` — Assign root window.
     - `self.root.title("Wave Equation Visualization")` — Set window title.
     - `self.root.geometry("800x600")` — Define window size.
   - **Configure Window Close Protocol:**
     - `self.root.protocol("WM_DELETE_WINDOW", self.stop_updates)`
   - **Plan Layout for Future Buttons:**
     - Use `tk.Frame` or `tk.PanedWindow` to separate plot area and controls.

6. **Initialize the Plot and Data Structures**

   - **Create Matplotlib Figure and Axes:**
     - `self.figure, self.ax = plt.subplots()`
     - Adjust figure size if necessary.
   - **Embed Figure in Tkinter Canvas:**
     - `self.canvas = FigureCanvasTkAgg(self.figure, master=self.root)`
     - `self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)`
   - **Initialize Plot Elements:**
     - `self.line, = self.ax.plot([], [], lw=2)` — Initialize an empty line plot.
     - Set plot labels: `self.ax.set_xlabel('Time (s)')`, `self.ax.set_ylabel('Theta (rad)')`
     - Set plot limits if necessary.

7. **Set Up the Animation Loop**

   - **Choose Update Mechanism:**
     - Use Tkinter's `after` method for periodic updates: `self.root.after(update_interval, self.update_plot)`
     - Define `update_interval` in milliseconds (e.g., `update_interval = int(delta_t * 1000)`).
   - **Handle Animation Timing:**
     - Ensure that the update rate matches the time step for accurate simulation speed.

8. **Implement the `update_plot` Method**

   - **Numerical Computation:**
     - Call `solve_theta` with current `theta`, `theta_dot`, and `t`.
     - Update `theta`, `theta_dot`, and `t` with returned values.
   - **Update Rolling Buffers:**
     - Append new `t` and `theta` to `self.t_vals` and `self.theta_vals`.
   - **Update Plot Data:**
     - `self.line.set_data(self.t_vals, self.theta_vals)`
     - Adjust axes limits: `self.ax.relim()`, `self.ax.autoscale_view()`
   - **Redraw Canvas:**
     - `self.canvas.draw()`
   - **Schedule Next Update:**
     - `self.root.after(update_interval, self.update_plot)`
   - **Add Error Handling:**
     - Use try-except blocks to handle exceptions and ensure smooth operation.

9. **Implement the `stop_updates` Method**

   - **Stop the Animation Loop:**
     - Set a flag `self.running = False` to indicate the loop should stop.
   - **Destroy the Window:**
     - `self.root.quit()` to exit the Tkinter main loop.
     - `self.root.destroy()` to close the window and release resources.
   - **Cancel Scheduled Calls:**
     - If possible, cancel any pending `after` calls.

10. **Handle Global Time Tracking**

    - **Increment Time:**
      - In `solve_theta`, increment `t` by `delta_t`.
    - **Use Time in Calculations:**
      - Pass `t` to `get_theta_double_dot` if `f` depends on time.
    - **Update Time Buffer:**
      - Append new `t` to `self.t_vals`.

11. **Plan for Future GUI Enhancements**

    - **Prepare Layout:**
      - Reserve space or create frames for future buttons and controls.
    - **Modularize Code:**
      - Encapsulate functionalities into methods to allow easy integration of new features.
    - **Event Handling:**
      - Define placeholder methods for button callbacks and other user interactions.

12. **Ensure Smooth Animation**

    - **Optimize Update Rate:**
      - Balance `delta_t` and `update_interval` for smooth visuals.
    - **Performance Improvements:**
      - Limit the amount of data plotted by using rolling buffers.
      - Avoid heavy computations in the main thread; consider multi-threading if necessary.
    - **Consider Alternative Libraries:**
      - If performance is inadequate, explore libraries like `PyQt5` with `matplotlib`, or `matplotlib`'s `FuncAnimation`.

13. **Consider Using Other Libraries if Necessary**

    - **Assess Matplotlib and Tkinter Limitations:**
      - If they hinder smooth animation or future development, consider:
        - **PyQt5/PySide2:** More advanced GUI features and better threading support.
        - **Bokeh or Plotly Dash:** For interactive web-based visualizations.
        - **Pygame or Vispy:** For high-performance graphics.

14. **Validate and Test the Numerical Solver**

    - **Accuracy Checks:**
      - Compare numerical results with known analytical solutions where possible.
    - **Adjust Solver Parameters:**
      - Refine `delta_t` or solver method for better accuracy.
    - **Stability Analysis:**
      - Ensure that the numerical method remains stable over time.

15. **Error Handling and Robustness**

    - **Implement Try-Except Blocks:**
      - Surround critical code sections with exception handling.
    - **User Interruption Handling:**
      - Gracefully handle keyboard interrupts or window closures.
    - **Validation of Inputs:**
      - Ensure that inputs to functions are valid and within expected ranges.

16. **Finalize the Main Execution Block**

    - **Entry Point Check:**
      - Use `if __name__ == "__main__":` to ensure proper execution when the script is run.
    - **Initialize the Application:**
      - Create the root window and instantiate `WaveGUI`.
    - **Start the Main Loop:**
      - Call `root.mainloop()` to start the Tkinter event loop.
    - **Ensure Clean Exit:**
      - Confirm that all resources are released upon closure.
