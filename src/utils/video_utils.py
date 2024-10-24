import cv2

def speed_up_video(input_video_path: str, output_video_path: str, speed_factor: int) -> None:
    """
    Speeds up the video by skipping frames.

    :param input_video_path: Path to the input video file.
    :param output_video_path: Path to save the sped-up video.
    :param speed_factor: Factor by which to speed up the video. Must be greater than 1.
    """
    if speed_factor <= 1:
        raise ValueError("Speed factor must be greater than 1.")

    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video file: {input_video_path}")

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    # Create VideoWriter object
    out = cv2.VideoWriter(output_video_path, fourcc, fps / speed_factor, (width, height))

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Write every nth frame, where n is the speed factor
        if frame_count % speed_factor == 0:
            out.write(frame)

        frame_count += 1

    cap.release()
    out.release()
