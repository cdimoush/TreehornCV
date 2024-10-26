import numpy as np

# Constants for free fall with damping and gain
INITIAL_POSITION = 0.0  # Initial position (e.g., height in meters)
INITIAL_VELOCITY = 0.0  # Initial velocity (e.g., m/s)
GRAVITY = 10 # Acceleration
DELTA_T = 0.0001  # Time step size
DAMPING_COEFFICIENT = 0.0  # Damping coefficient
FLOOR_POSITION = 0.0  # Position of the floor
CEILING_POSITION = 50  # Position of the ceiling

def solve_free_fall(position, velocity, gain, time=DELTA_T, dt=DELTA_T, ceiling=CEILING_POSITION, floor=FLOOR_POSITION):
    if position is None:
        position = INITIAL_POSITION
    if velocity is None:
        velocity = INITIAL_VELOCITY

    ellapsed_time = 0
    while ellapsed_time < time:
        # Update velocity with damping and gain
        if np.sign(gain) > 0:
            acc = 10*GRAVITY * (1 - position / ceiling)
        else:
            acc = -GRAVITY 

        velocity += (acc - DAMPING_COEFFICIENT * velocity) * dt
        # Update position
        position += velocity * dt

        # Check for floor and ceiling conditions
        if position <= floor:
            position = floor
            velocity = 0  # Stop the object or reverse velocity for a bounce

        if position >= ceiling:
            position = ceiling
            velocity = 0  # Stop the object or reverse velocity for a bounce

        ellapsed_time += dt

    # Return the updated position, velocity, and time step
    return position, velocity, dt
