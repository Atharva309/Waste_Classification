import cv2
import glob
import os
import shutil
import numpy as np

src_dir = "data/stage2_material/metal/"
dest_dir = "data/stage2_material/_excluded_label_crops/metal/"
os.makedirs(dest_dir, exist_ok=True)

files = glob.glob(os.path.join(src_dir, "*.jpg")) + glob.glob(os.path.join(src_dir, "*.png"))
moved = 0

for f in files:
    img = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
    if img is None: continue
    h, w = img.shape
    ch, cw = h//2, w//2
    cy, cx = h//2, w//2
    crop = img[cy-ch//2:cy+ch//2, cx-cw//2:cx+cw//2]
    
    blurred = cv2.GaussianBlur(crop, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    
    edge_density = np.sum(edges > 0) / (ch * cw)
    
    if edge_density > 0.08:
        # Move it
        shutil.move(f, os.path.join(dest_dir, os.path.basename(f)))
        moved += 1

print(f"Moved {moved} files to {dest_dir}.")
