import matplotlib.pyplot as plt
import pandas as pd

def calculate_distance(speed, duration, target):
    """
    Calculate the distance traveled based on speed and duration using the derived formula.
    speed: Normalized speed (0 to 1), always positive.
    duration: Duration in milliseconds.
    target: Target position (0 to 1), used to determine the direction of travel.
    """
    if speed == 0:
        return 0  # No distance traveled if speed is zero
    # Determine the sign based on target
    direction = 1 if (target - 0.5) >= 0 else -1
    mil = (speed / 250) ** -0.95
    distance = (duration * 0.9) / mil
    return distance * direction  # Apply direction based on target

def plot_target_velocity_and_position(csv_file_path):
    # Read the CSV file
    data = pd.read_csv(csv_file_path)

    # Ensure the DataFrame is sorted by Timestamp
    data = data.sort_values(by='Timestamp').reset_index(drop=True)

    # Calculate elapsed time between rows in milliseconds
    data['Elapsed'] = data['Timestamp'].diff().fillna(0) * 1000

    # Calculate average FPS
    average_fps = 1000 / data['Elapsed'].mean()

    # Initialize position column
    data['Position'] = 0.0

    # Iterate through the DataFrame to calculate position
    for i in range(1, len(data)):
        speed = data.loc[i, 'Velocity']
        elapsed = data.loc[i, 'Elapsed']
        target = data.loc[i, 'Target']
        distance = calculate_distance(speed, elapsed, target)
        new_position = data.loc[i - 1, 'Position'] + distance
        # Bound position between 0 and 1
        data.loc[i, 'Position'] = max(0, min(1, new_position))

    # Plot the data using Timestamp as the x-axis
    fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

    # Top plot: Target and Velocity
    axes[0].plot(data['Timestamp'], data['Target'], label='Target', color='b')
    axes[0].plot(data['Timestamp'], data['Velocity'], label='Velocity', color='r')
    axes[0].set_ylabel('Value')
    axes[0].set_title('Target and Velocity over Time')
    axes[0].legend()

    # Bottom plot: Position
    axes[1].plot(data['Timestamp'], data['Position'], label='Position', color='g')
    axes[1].set_xlabel('Timestamp (s)')
    axes[1].set_ylabel('Position (Distance)')
    axes[1].set_title('Position of the Stroker over Time')
    axes[1].legend()

    # Annotate average FPS on the bottom plot
    axes[1].annotate(f'Average FPS: {average_fps:.2f}', xy=(0.5, 0.1), xycoords='axes fraction',
                     fontsize=12, color='black', ha='center', va='center',
                     bbox=dict(facecolor='white', alpha=0.8))

    # Adjust layout and save the plot to a file
    plt.tight_layout()
    plt.savefig('target_velocity_position.png')

if __name__ == "__main__":
    csv_file_path = "../_dev/target_velocity.csv"  # Path to the CSV file
    plot_target_velocity_and_position(csv_file_path)
