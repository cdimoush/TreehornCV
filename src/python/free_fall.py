import numpy as np

# Constants for free fall with damping and gain
INITIAL_POSITION = 0.0  # Initial position (e.g., height in meters)
INITIAL_VELOCITY = 0.0  # Initial velocity (e.g., m/s)
GRAVITY = 1 # Acceleration
DELTA_T = 0.01  # Time step size
DAMPING_COEFFICIENT = 0.0  # Damping coefficient
FLOOR_POSITION = 0.0  # Position of the floor
CEILING_POSITION = 0.5  # Position of the ceiling

def solve_free_fall(position, velocity, gain, dt=DELTA_T):
    if position is None:
        position = INITIAL_POSITION
    if velocity is None:
        velocity = INITIAL_VELOCITY

    # Update velocity with damping and gain
    if np.sign(gain) == 1:
        acc = GRAVITY * 100
    else:
        acc = -GRAVITY

    velocity += (acc - DAMPING_COEFFICIENT * velocity) * dt
    # Update position
    position += velocity * dt

    # Check for floor and ceiling conditions
    if position <= FLOOR_POSITION:
        position = FLOOR_POSITION
        velocity = 0  # Stop the object or reverse velocity for a bounce

    if position >= CEILING_POSITION:
        position = CEILING_POSITION
        velocity = 0  # Stop the object or reverse velocity for a bounce

    # Return the updated position, velocity, and time step
    return position, velocity, dt
