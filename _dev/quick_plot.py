import matplotlib.pyplot as plt
import pandas as pd

def plot_target_velocity(csv_file_path):
    # Read the CSV file
    data = pd.read_csv(csv_file_path)

    # Plot the data
    plt.figure(figsize=(10, 6))
    plt.plot(data['Target'], label='Target', color='b')
    plt.plot(data['Velocity'], label='Velocity', color='r')

    # Add labels and title
    plt.xlabel('Frame Index')
    plt.ylabel('Value')
    plt.title('Target and Velocity over Time')
    plt.legend()

    # Save the plot to a file
    plt.savefig('target_velocity.png')

if __name__ == "__main__":
    csv_file_path = "../_dev/target_velocity.csv"  # Path to the CSV file
    plot_target_velocity(csv_file_path)
