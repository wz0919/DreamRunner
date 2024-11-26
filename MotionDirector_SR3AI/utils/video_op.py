from PIL import Image

import os
import numpy as np
from PIL import Image, ImageSequence, ImageDraw, ImageFont
import cv2
import imageio

def calculate_font_size_and_wrap_text(draw, text, width, max_height, font_path):
    """
    Calculate the maximum font size that fits within the given width and max_height
    and wrap the text to fit within the width.
    """
    font_size = 30
    font = ImageFont.truetype(font_path, font_size)
    text_lines = wrap_text(draw, text, font, width)

    while True:
        text_height = sum(draw.textbbox((0, 0), line, font=font)[3] for line in text_lines)
        if text_height < max_height:
            break
        font_size -= 1
        font = ImageFont.truetype(font_path, font_size)
        text_lines = wrap_text(draw, text, font, width)
    
    return font_size-2, text_lines

def wrap_text(draw, text, font, max_width):
    """
    Wrap text to fit within max_width
    """
    lines = []
    words = text.split()
    while words:
        line = []
        while words and draw.textbbox((0, 0), ' '.join(line + [words[0]]), font=font)[2] <= max_width:
            line.append(words.pop(0))
        lines.append(' '.join(line))
    return lines

def add_caption_below_frame(frame, caption_lines, font, original_height, caption_height):
    """
    Add caption below the frame by extending the frame height
    """
    original_width = frame.width
    new_height = original_height + caption_height
    new_frame = Image.new('RGB', (original_width, new_height), 'black')
    new_frame.paste(frame, (0, 0))
    
    draw = ImageDraw.Draw(new_frame)
    
    # Calculate text position
    y_text = original_height + (caption_height - sum(draw.textbbox((0, 0), line, font=font)[3] for line in caption_lines)) / 2
    for line in caption_lines:
        text_size = draw.textbbox((0, 0), line, font=font)[2:]
        text_x = (original_width - text_size[0]) / 2
        draw.text((text_x, y_text), line, font=font, fill="white")
        y_text += text_size[1]
    
    return new_frame

def add_caption_to_frames(frames, caption, font_path="../files/Arial.ttf"):
    """
    Add caption to each frame by extending the frame height
    """
    original_width, original_height = frames[0].size
    caption_height = original_height // 3
    new_height = original_height + caption_height

    draw = ImageDraw.Draw(frames[0])
    font_size, wrapped_caption = calculate_font_size_and_wrap_text(draw, caption, original_width, caption_height, font_path)
    font = ImageFont.truetype(font_path, font_size)
    
    return [add_caption_below_frame(frame, wrapped_caption, font, original_height, caption_height) for frame in frames]

def add_caption_in_middle(frame, caption_lines, font):
    """
    Add caption in the middle of the frame
    """
    original_width, original_height = frame.size
    draw = ImageDraw.Draw(frame)
    
    # Calculate total height of the caption
    total_caption_height = sum(draw.textbbox((0, 0), line, font=font)[3] for line in caption_lines)
    
    # Calculate starting Y position for the caption to be centered
    y_text = (original_height - total_caption_height) / 2
    for line in caption_lines:
        text_size = draw.textbbox((0, 0), line, font=font)[2:]
        text_x = (original_width - text_size[0]) / 2
        draw.text((text_x, y_text), line, font=font, fill="white")
        y_text += text_size[1]
    
    return frame

def add_caption_to_frames_middle(frames, caption, font_path="../files/Arial.ttf"):
    """
    Add caption to each frame by extending the frame height
    """
    original_width, original_height = frames[0].size
    caption_height = original_height

    draw = ImageDraw.Draw(frames[0])
    font_size, wrapped_caption = calculate_font_size_and_wrap_text(draw, caption, original_width, caption_height, font_path)
    font = ImageFont.truetype(font_path, font_size)
    
    return [add_caption_in_middle(frame, wrapped_caption, font) for frame in frames]
    
def export_to_video(video_frames, output_video_path, fps):
    video_writer = imageio.get_writer(output_video_path, fps=fps)
    for img in video_frames:
        video_writer.append_data(np.array(img))
    video_writer.close()
    
if __name__ == "__main__":
    os.chdir('python_scripts/')
    frames = '?'
    output_path = 'output_video.mp4'   

    export_to_video(frames, output_path, fps=8)
    # from diffusers.utils import export_to_video
    # export_to_video(frames,output_path,fps=8)
