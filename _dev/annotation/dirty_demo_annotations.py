import cv2
import numpy as np

def add_all_annotations(image):
    image = add_header(image)
    image = add_title(image, "OhMiBod!", "--Stroker Demo--")
    image = add_legend(image)
    return image

def add_title(image, title, subtitle=None, title_font=cv2.FONT_HERSHEY_DUPLEX, 
              title_font_scale=3, title_color=(255, 255, 255), title_thickness=3, 
              subtitle_font=cv2.FONT_HERSHEY_DUPLEX, subtitle_font_scale=2, 
              subtitle_color=(255, 255, 255), subtitle_thickness=2):
    """
    Add a title and optional subtitle to the top of the image.
    """
    # Get the image dimensions
    height, width = image.shape[:2]
    
    # Calculate the position for the title
    title_size = cv2.getTextSize(title, title_font, title_font_scale, title_thickness)[0]
    title_x = (width - title_size[0]) // 2
    title_y = title_size[1] + 20  # 20 pixels from the top
    
    # Add the title to the image
    cv2.putText(image, title, (title_x, title_y), title_font, title_font_scale, title_color, title_thickness)
    
    # If a subtitle is provided, add it below the title
    if subtitle:
        subtitle_size = cv2.getTextSize(subtitle, subtitle_font, subtitle_font_scale, subtitle_thickness)[0]
        subtitle_x = (width - subtitle_size[0]) // 2
        subtitle_y = title_y + title_size[1] + 10  # 20 pixels below the title
        cv2.putText(image, subtitle, (subtitle_x, subtitle_y), subtitle_font, subtitle_font_scale, subtitle_color, subtitle_thickness)


    return image

def add_header(image, header_thickness=0.15, header_color=(153, 50, 204)):
    """
    Add a solid header to the top of the image, with white border.
    """
    height, width = image.shape[:2]
    header_thickness = int(height * header_thickness)
    cv2.rectangle(image, (0, 0), (width, header_thickness), header_color, -1)

    # Add a white border to the header
    cv2.rectangle(image, (0, 0), (width, header_thickness), (255, 255, 255), 2)

    return image

def add_legend(image, signal=1, legend_width_scale=0.3, legend_height_scale=0.5, 
               legend_color=(153, 50, 204), border_color=(255, 255, 255), border_thickness=2, offset_top=0.2, offset_right=0.05):
    """
    Add a legend box to the top right of the image.
    """
    height, width = image.shape[:2]
    legend_width = int(width * legend_width_scale)
    legend_height = int(height * legend_height_scale)
        
    # Calculate the top right corner position with offsets
    top_left_x = width - legend_width - int(width * offset_right)
    top_left_y = int(height * offset_top)
    
    # Draw the legend box
    cv2.rectangle(image, (top_left_x, top_left_y), 
                  (top_left_x + legend_width, top_left_y + legend_height), 
                  legend_color, -1)
    
    # Add a border to the legend box
    cv2.rectangle(image, (top_left_x, top_left_y), 
                  (top_left_x + legend_width, top_left_y + legend_height), 
                  border_color, border_thickness)
    
    # Add text to top center of the legend
    text = "Stroker Signal"
    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, 1.5, 4)[0]
    text_x = int(top_left_x + legend_width * 0.5 - text_size[0] * 0.5)
    text_y = int(top_left_y + legend_height * 0.05)
    cv2.putText(image, text, (text_x, text_y), cv2.FONT_HERSHEY_DUPLEX, 1.5, (255, 255, 255), 4)
    
    # Draw a rectangle that is red in the center of the legend box
    center_x = top_left_x + legend_width // 2
    center_y = top_left_y + legend_height // 2
    shaft_height_offset = legend_height // 6
    shaft_width_offset = legend_width // 6

    cv2.rectangle(image, (center_x - shaft_width_offset, center_y - shaft_height_offset), 
                  (center_x + shaft_width_offset, center_y + shaft_height_offset), 
                  (0, 0, 0), -1)
    
    # Draw a circle2 at the top and bottom of the shaft
    cv2.circle(image, (center_x, center_y - shaft_height_offset), shaft_width_offset, (0, 0, 0), -1)
    cv2.circle(image, (int(center_x + 0.75*shaft_width_offset), center_y + shaft_height_offset), shaft_width_offset, (0, 0, 0), -1)
    cv2.circle(image, (int(center_x - 0.75*shaft_width_offset), center_y + shaft_height_offset), shaft_width_offset, (0, 0, 0), -1)

    # Draw a rectangle that corresponds to the signal level (1: top of shaft, 0: bottom of shaft) this rectangle slides on the shaft
    shaft_top_y = int(center_y - 0.5*shaft_height_offset)
    shaft_bottom_y = int(center_y + 0.5*shaft_height_offset)
    ring_height = int(0.1 * abs(shaft_top_y - shaft_bottom_y))
    ring_position = int(shaft_top_y + signal * abs(shaft_top_y - shaft_bottom_y))

    cv2.rectangle(image, (center_x - shaft_width_offset, ring_position), 
                  (center_x + shaft_width_offset, ring_position + ring_height), 
                  (255, 0, 0), -1)
    

    return image




