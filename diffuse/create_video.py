"""
Video Creation Utility Module

This module provides functionality to create videos from a sequence of images.
It's particularly useful for creating animations from simulation frames.

Author: Agosh Saini - conact@agoshsaini.com
"""

import cv2 
import glob
import os

def create_video_from_images(input_dir="data", output_file="output.mp4", fps=30):
    """
    Create a video from a sequence of PNG images in a directory.
    
    Args:
        input_dir (str): Directory containing the input images
        output_file (str): Name of the output video file
        fps (int): Frames per second for the output video
        
    Returns:
        bool: True if video creation was successful, False otherwise
    """
    try:
        # Get list of images and sort them by modification time
        image_files = sorted(glob.glob(os.path.join(input_dir, '*.png')), 
                           key=os.path.getmtime)
        
        if not image_files:
            raise ValueError(f"No PNG images found in {input_dir}")
            
        # Read first image to get dimensions
        first_image = cv2.imread(image_files[0])
        if first_image is None:
            raise ValueError(f"Could not read image: {image_files[0]}")
            
        height, width, layers = first_image.shape
        size = (width, height)
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'MP4V')
        video_out = cv2.VideoWriter(output_file, fourcc, fps, size)
        
        # Add each image to the video
        for filename in image_files:
            img = cv2.imread(filename)
            if img is not None:
                video_out.write(img)
            else:
                print(f"Warning: Could not read image {filename}")
        
        # Release the video writer
        video_out.release()
        return True
        
    except Exception as e:
        print(f"Error creating video: {str(e)}")
        return False

if __name__ == "__main__":
    # Create video from default data directory
    success = create_video_from_images()
    if success:
        print("Video created successfully!")
    else:
        print("Failed to create video.")
