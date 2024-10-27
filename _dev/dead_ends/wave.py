import tkinter as tk
import cv2
import threading
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from collections import deque
from queue import Queue, Empty

# Constants (initial conditions)
THETA_0 = np.pi / 2  # Initial theta (in radians)
THETA_DOT_0 = 0  # Initial angular velocity (theta_dot)
DELTA_T = 0.001  # Time step size
GAIN_0 = 0


def solve_theta(theta, theta_dot, dt=DELTA_T, gain=GAIN_0):
    if theta is None:
        theta = THETA_0
    if theta_dot is None:
        theta_dot = THETA_DOT_0

    theta_double_dot = -np.sin(theta) + 25*gain

    # Update theta and theta_dot using Euler's method
    theta += theta_dot * dt
    theta_dot += theta_double_dot * dt

    # Limits on theta
    if theta > np.pi / 2:
        theta = np.pi / 2
        theta_dot = 0
    elif theta < -np.pi / 2:
        theta = -np.pi / 2
        theta_dot = 0

    return theta, theta_dot, dt