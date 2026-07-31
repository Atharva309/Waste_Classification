import sys
import os
import io
import base64
import requests
import numpy as np
from PIL import Image, ImageDraw
import cv2

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from src.synthetic_detection import BeltLocalizer

import glob
import random

def get_random_sprite():
    sprites = glob.glob(os.path.join(os.path.dirname(os.path.dirname(__file__)), "outputs", "sprites", "*", "*.png"))
    if not sprites:
        raise FileNotFoundError("No sprites found in outputs/sprites")
    path = random.choice(sprites)
    label = os.path.basename(os.path.dirname(path))
    img = Image.open(path).convert("RGBA")
    return img, label

def create_debug_frame(frame_arr, proposals, mask_arr):
    # Convert mask to 3-channel for side-by-side
    mask_rgb = cv2.cvtColor(mask_arr, cv2.COLOR_GRAY2RGB)
    
    # Draw boxes on frame
    frame_pil = Image.fromarray(frame_arr)
    draw = ImageDraw.Draw(frame_pil)
    for tid, box in proposals:
        x1, y1, x2, y2 = box
        draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
        draw.text((x1, max(0, y1 - 15)), f"ID: {tid}", fill="red")
        
    frame_with_box = np.array(frame_pil)
    
    # Concatenate side by side
    debug_view = np.hstack((frame_with_box, mask_rgb))
    return debug_view

def test_sequence(output_gif, num_frames=30):
    frame_size = 640
    belt_color = (60, 60, 65)
    
    try:
        img, label = get_random_sprite()
    except Exception as e:
        print("Failed to get sprite from API. Is the server running? Fallback to a synthetic square.")
        img = Image.new('RGBA', (100, 100), (255, 0, 0, 255))
        label = "dummy"
        
    scale = np.random.uniform(0.15, 0.25)
    target_w = int(frame_size * scale)
    target_h = int(target_w * (img.height / img.width))
    img = img.resize((target_w, target_h))
    
    localizer = BeltLocalizer()
    
    frames_debug = []
    
    start_x = -target_w
    end_x = frame_size + target_w
    
    y = frame_size // 2 - target_h // 2
    
    for i in range(num_frames):
        # 1. Create moving frame
        bg = Image.new('RGB', (frame_size, frame_size), belt_color)
        
        # Add some noise to simulate camera grain/lighting drift
        noise = np.random.normal(0, 2, (frame_size, frame_size, 3))
        bg_arr = np.clip(np.array(bg) + noise, 0, 255).astype(np.uint8)
        bg = Image.fromarray(bg_arr)
        
        x = int(start_x + (end_x - start_x) * (i / float(num_frames - 1)))
        
        # Paste object
        bg.paste(img, (x, y), img)
        frame_arr = np.array(bg)
        
        # 2. Localize
        proposals, mask = localizer.update(frame_arr)
        
        # 3. Create Debug View
        debug_view = create_debug_frame(frame_arr, proposals, mask)
        frames_debug.append(Image.fromarray(debug_view))
        
        print(f"Frame {i:02d}: Proposals {proposals}")

    frames_debug[0].save(output_gif, save_all=True, append_images=frames_debug[1:], duration=100, loop=0)
    print(f"Saved {output_gif} - label: {label}")
    
    # Print tracking stats
    print("Tracking Stats:")
    for tid, track in localizer.tracks.items():
        print(f"  Track ID {tid}: Captured {len(track['crops'])} crops over sequence.")

if __name__ == '__main__':
    os.makedirs("outputs/investigations", exist_ok=True)
    for run in range(3):
        out = f"outputs/investigations/tracking_test_{run}.gif"
        test_sequence(out)
