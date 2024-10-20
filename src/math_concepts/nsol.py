import numpy as np
import matplotlib.pyplot as plt

# Constants (initial conditions)
THETA_0 = np.pi # Initial theta (in radians)
THETA_DOT_0 = -1     # Initial angular velocity (theta_dot)
delta_t = 0.01       # Time step size
total_time = 10      # Total time for simulation

# Function to compute theta_double_dot = d2theta/dt^2 = sin(theta)
def get_theta_double_dot(theta, theta_dot):
    return np.sin(theta)

# Function to numerically solve d2theta/dt^2 = sin(theta) using Euler's method
def solve_theta(t):
    theta = THETA_0
    theta_dot = THETA_DOT_0

    # Time stepping
    for time in np.arange(0, t, delta_t):
        theta_double_dot = get_theta_double_dot(theta, theta_dot)

        # Update theta and theta_dot using Euler's method
        theta += theta_dot * delta_t
        theta_dot += theta_double_dot * delta_t

    return theta

# Time array for plotting
time_array = np.arange(0, total_time, delta_t)

# Solve for theta over time
theta_values = [solve_theta(t) for t in time_array]

# Plot results
plt.plot(time_array, theta_values, label='Theta (angular position)')
plt.xlabel('Time (s)')
plt.ylabel('Theta (radians)')
plt.title('Numerical Solution of d2theta/dt^2 = sin(theta)')
plt.legend()
plt.grid(True)
plt.show()
