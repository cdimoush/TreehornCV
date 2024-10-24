# Project: TreehornCV
## Introduction

TreehornCV is a project designed to utilize a Python vision library to process a prerecorded video by tracking a person in each frame and marking their center of body mass. As the video plays, the application will output the video with this marker overlaid and simultaneously stream the person's X and Y positions to a real-time plot using tools like Matplotlib or Tkinter. The main challenge lies in selecting and implementing a suitable open-source vision library for object detection and tracking, given unfamiliarity with the vision aspect.

## Problem Statement

TreehornCV aims to create a concise yet powerful demonstration of real-time object tracking and data visualization. The project's objectives include:

- **Video Processing**: Loading and processing a prerecorded video frame by frame.
- **Person Detection**: Identifying the person in each frame using a vision library.
- **Position Tracking**: Calculating and marking the center of body mass of the detected person.
- **Data Streaming**: Outputting the video with the marker and streaming the person's X and Y positions to a real-time plot.
- **Code Simplicity**: Implementing the solution in under 50 lines of code to maintain simplicity and focus.

## Problem Breakdown

1. **Video Processing**:
   - Load the prerecorded video using a suitable Python library.
   - Process the video frame by frame for real-time analysis.

2. **Person Detection and Tracking**:
   - Utilize a vision library to detect the person in each frame.
   - Calculate the center of mass (X and Y coordinates) of the detected person.

3. **Marker Overlay**:
   - Overlay a marker (e.g., a circle or dot) on the video frame at the person's center of mass.
   - Update and display the video with the marker in real-time.

4. **Data Streaming and Visualization**:
   - Collect the X and Y positions of the person over time.
   - Stream these positions to a real-time plot using Matplotlib or Tkinter.

5. **User Interface Integration**:
   - Use familiar tools to create a graphical interface for displaying the video and plot.
   - Ensure the interface is user-friendly and displays updates in real-time.

## Vision Library Recommendation

To meet the project's requirements, especially focusing on simplicity and effectiveness, the following vision libraries are considered:

1. **OpenCV (Open Source Computer Vision Library)**:
   - **Pros**:
     - Widely used with extensive documentation and community support.
     - Comprehensive functionalities for image and video processing.
     - Offers various object detection methods suitable for this project.
     - Easy integration with Python and other libraries.
   - **Cons**:
     - May require a basic understanding of computer vision concepts for effective use.

2. **Background Subtraction**:
   - **Pros**:
     - Efficient for detecting moving objects in videos.
     - Lightweight and suitable for real-time applications on small devices.
     - No need for complex machine learning models.
   - **Cons**:
     - Requires a static camera for best results.
     - Sensitive to lighting changes and shadows.

**Recommendation**:

For TreehornCV, using **OpenCV with background subtraction** is recommended because:

- It provides a good balance between simplicity and functionality.
- Allows the implementation to remain concise (under 50 lines of code).
- Does not require deep learning frameworks, keeping dependencies minimal.

## Implementation Overview (Example Code)
*Note: This section contains example code it does not need to be used exactly in development of the project.*

### 1. Initialize the Background Subtractor

```python
import cv2

backSub = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=True)
```

### 2. Process Video Frames

- Load the video using OpenCV's `VideoCapture`.
- Iterate over each frame in a loop.

```python
cap = cv2.VideoCapture('video.mp4')

while True:
    ret, frame = cap.read()
    if not ret:
        break
    # Further processing goes here
```

### 3. Detect Moving Objects in Each Frame

- Apply background subtraction to detect moving objects.
- Calculate the center of mass for each detected object.

```python
fgMask = backSub.apply(frame)
contours, _ = cv2.findContours(fgMask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
for cnt in contours:
    if cv2.contourArea(cnt) > 500:
        x, y, w, h = cv2.boundingRect(cnt)
        center_x = x + w // 2
        center_y = y + h // 2
        # Draw marker and collect positions
```

### 4. Overlay Marker on Frame

- Draw a circle or marker at the center of mass on the frame.
- Display the annotated frame.

```python
cv2.circle(frame, (center_x, center_y), 5, (0, 255, 0), -1)
cv2.imshow('Frame', frame)
```

### 5. Stream Data to Real-Time Plot

- Collect the positions into a list for plotting.
- Use Matplotlib in interactive mode to update the plot in real-time.

```python
import matplotlib.pyplot as plt

positions = []

# Inside the loop, after calculating center_x and center_y
positions.append((center_x, center_y))

# After the loop, plot the positions
plt.ion()
for pos in positions:
    plt.scatter(pos[0], pos[1])
    plt.draw()
    plt.pause(0.001)
```
## File Structure
### src/detection.py: Handles all OpenCV detection and tracking operations, including person detection, calculating the center of mass, and overlaying markers on frames.
    Purpose: Consolidates person detection, tracking, and overlay functionalities into a single module for simplicity and efficiency.

    Key Components:

        Detection class:

            __init__(self): Initializes the background subtractor.

            detect_and_annotate(self, frame): Detects moving objects in the given frame, calculates the center of mass, draws a marker on the frame, and returns the annotated frame along with the center coordinates.

            Methods:

                detect_moving_objects(self, frame): Performs background subtraction on the frame.

                calculate_center(self, bbox): Calculates the center coordinates from a bounding box.

                draw_marker(self, frame, center): Draws a marker at the specified center on the frame.

### src/plotter.py: Manages real-time plotting of the person's positions using Matplotlib.

    Purpose: Handles the visualization of the tracked positions to keep plotting logic separate and organized.

    Key Components:

        Plotter class:

            __init__(self): Sets up the plotting environment in interactive mode.

            update(self, center): Updates the plot with the new center position.

            finalize(self): Displays the final plot after processing is complete.

### app.py: The main application script that orchestrates the entire process, including video processing.

    Purpose: Serves as the entry point, managing video capture, integrating detection and plotting modules, and controlling the overall workflow.

    Responsibilities:

        Initialize instances of Detection and Plotter.

        Load the video using OpenCV's VideoCapture.

        Implement the main loop to read frames, perform detection and annotation, update the plot, and display the video.

        Handle resource cleanup and graceful shutdown upon completion or interruption.

### requirements.txt: Lists all the dependencies required to run the project.

    Dependencies:

        opencv-python

        numpy

        matplotlib

### README.md: Provides documentation and instructions.

    Contents:

        Project overview.

        Setup instructions.

        How to run the application.

        Description of each module and their roles.
