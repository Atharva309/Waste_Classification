import os
import json
import cv2
import numpy as np
import subprocess

scratch_dir = "scratch/taco"
dest_base = "data/stage2_material"

if not os.path.exists(scratch_dir):
    print("Cloning TACO...")
    subprocess.run(["git", "clone", "https://github.com/pedropro/TACO", scratch_dir], check=True)

# Install requirements for download.py if needed (requests, etc)
subprocess.run(["venv/bin/pip", "install", "requests", "Pillow"], check=True)

# Run download script
python_exec = os.path.abspath("venv/bin/python")
subprocess.run([python_exec, "download.py", "--dataset_path", "data/annotations.json"], cwd=scratch_dir, check=True)

print("Processing annotations...")
with open(os.path.join(scratch_dir, "data/annotations.json"), "r") as f:
    coco = json.load(f)

# Build category mapping
cat_mapping = {
    "organic": ["Food waste"],
    "cardboard": ["Corrugated carton", "Drink carton", "Egg carton", "Meal carton", "Other carton", "Pizza box", "Toilet tube"],
    "ewaste": ["Battery"],
    "glass": ["Glass bottle", "Broken glass", "Glass cup", "Glass jar"],
    "metal": ["Aerosol", "Aluminium foil", "Aluminium blister pack", "Metal bottle cap", "Drink can", "Food Can", "Metal lid", "Pop tab", "Scrap metal"],
    "paper": ["Paper cup", "Magazine paper", "Tissues", "Wrapping paper", "Normal paper", "Paper bag", "Plastified paper bag", "Paper straw"],
    "plastic": ["Carded blister pack", "Clear plastic bottle", "Other plastic bottle", "Plastic bottle cap", "Disposable plastic cup", "Foam cup", "Other plastic cup", "Plastic lid", "Garbage bag", "Single-use carrier bag", "Polypropylene bag", "Produce bag", "Cereal bag", "Bread bag", "Plastic film", "Crisp packet", "Other plastic wrapper", "Retort pouch", "Spread tub", "Tupperware", "Disposable food container", "Foam food container", "Other plastic container", "Plastic gloves", "Plastic utensils", "Six pack rings", "Squeezable tube", "Plastic straw", "Styrofoam piece", "Other plastic"]
}
# Skip: Cigarette, Unlabeled litter, Shoe, Rope & strings

# Reverse mapping: supercategory/name -> our_class
taco_to_our = {}
id_to_name = {}
for cat in coco["categories"]:
    name = cat["name"]
    id_to_name[cat["id"]] = name
    for our_class, names in cat_mapping.items():
        if name in names:
            taco_to_our[cat["id"]] = our_class
            break

# Image lookup
img_dict = {img["id"]: img for img in coco["images"]}

copied = 0
for ann in coco["annotations"]:
    cat_id = ann["category_id"]
    if cat_id not in taco_to_our:
        continue
    
    our_class = taco_to_our[cat_id]
    img_info = img_dict[ann["image_id"]]
    img_path = os.path.join(scratch_dir, "data", img_info["file_name"])
    
    if not os.path.exists(img_path):
        continue
        
    # Read image
    img = cv2.imread(img_path)
    if img is None: continue
    
    # Get mask from polygon
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    for seg in ann["segmentation"]:
        # segmentation is a list of polygons [x1, y1, x2, y2, ...]
        if isinstance(seg, list):
            poly = np.array(seg).reshape((-1, 2)).astype(np.int32)
            cv2.fillPoly(mask, [poly], 255)
            
    # Apply mask and background color
    masked_img = np.zeros_like(img)
    # Use a neutral gray background for the cropped object to avoid pure black learning
    masked_img[:] = (128, 128, 128)
    masked_img[mask == 255] = img[mask == 255]
    
    # Crop to bounding box
    # bbox is [x, y, width, height]
    x, y, w, h = [int(v) for v in ann["bbox"]]
    x = max(0, x)
    y = max(0, y)
    w = max(1, w)
    h = max(1, h)
    
    crop = masked_img[y:y+h, x:x+w]
    if crop.size == 0: continue
    
    dest_dir = os.path.join(dest_base, our_class)
    os.makedirs(dest_dir, exist_ok=True)
    out_name = f"taco_{ann['image_id']}_{ann['id']}.jpg"
    
    cv2.imwrite(os.path.join(dest_dir, out_name), crop)
    copied += 1

print(f"Extracted {copied} object instances from TACO.")
