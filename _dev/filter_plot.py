import matplotlib.pyplot as plt
import numpy as np

# Read the data from the file
with open('y_displacements.txt', 'r') as file:
    y_displacements = [float(line.strip()) for line in file]

# Define the window size for the moving average
window_size = 2

# Compute the simple moving average
def moving_average(data, window_size):
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

# Apply the moving average filter
filtered_y_displacements = moving_average(y_displacements, window_size)

# Plot the raw and filtered data
plt.figure(figsize=(10, 5))
plt.plot(y_displacements, label='Raw Data', linestyle='--', color='gray')
plt.plot(range(window_size - 1, len(y_displacements)), filtered_y_displacements, label='Filtered Data (SMA)', color='blue')
plt.xlabel('Frame Index')
plt.ylabel('Average Y-Displacement')
plt.title('Average Y-Displacement Over Time (Raw vs. SMA Filtered)')
plt.legend()

plt.savefig('y_displacements.png')