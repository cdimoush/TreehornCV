import numpy as np
import matplotlib.pyplot as plt

# Define a function for the second derivative of theta with respect to time
def theta_dot_dot(theta, theta_dot):
    return np.sin(theta) - np.cos(theta)

# Function to plot the vector field for a given second-order differential equation
def plot_vector_field(theta_dot_dot_func):
    # Generate a grid of theta and theta_dot values for the plot
    theta = np.linspace(-2 * np.pi, 2 * np.pi, 20)  # theta values from -2pi to 2pi
    theta_dot_vals = np.linspace(-2, 2, 20)         # theta_dot (dtheta/dt) values

    # Create a meshgrid for theta and theta_dot
    Theta, Theta_dot = np.meshgrid(theta, theta_dot_vals)

    # Compute the vector field for the given second-order differential equation
    U = Theta_dot  # dtheta/dt
    V = theta_dot_dot_func(Theta, Theta_dot)  # d2theta/dt^2

    # Plot the vector field using quiver
    plt.quiver(Theta, Theta_dot, U, V, color='r')

    # Set axis labels and title
    plt.xlabel('theta')
    plt.ylabel('theta_dot')
    plt.title('2nd Order Phase Space')

    # Display the plot
    plt.show()

# Call the function with the desired second-order differential equation
plot_vector_field(theta_dot_dot)
