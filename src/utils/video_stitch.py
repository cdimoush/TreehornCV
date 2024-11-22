import os
import ffmpeg

def stitch_videos_ffmpeg(directory, output_filename):
    # List all files in the directory
    files = os.listdir(directory)
    
    # Filter out files that contain "processed" in their name and have a .mp4 extension
    video_files = [f for f in files if "processed" in f and f.endswith('.mp4')]
    
    # Sort the video files to maintain a consistent order
    video_files.sort()
    
    # Create a text file with the list of video files
    with open('file_list.txt', 'w') as file_list:
        for video in video_files:
            file_list.write(f"file '{os.path.join(directory, video)}'\n")
    
    # Use ffmpeg to concatenate the videos
    output_path = os.path.join(directory, output_filename)
    ffmpeg.input('file_list.txt', format='concat', safe=0).output(output_path, c='copy').run()

    # Clean up the temporary file list
    os.remove('file_list.txt')

# Example usage
stitch_videos_ffmpeg(r'C:\Users\Conner\Home\Projects\TreehornCV\_video', 'dirty_demo.mp4')