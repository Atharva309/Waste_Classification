# WARNING (see NOTES.md, "Sprite pool data leakage"): this script regenerates
# outputs/sprites/{glass,metal,plastic,cardboard,paper} directly from the raw,
# UNSPLIT scratch/taco/data/annotations.json -- it does not check
# outputs/stage2_split.json at all. Running this will reintroduce train-split
# images into the live webapp's test pool (72% of the pool was train data
# before this was caught). Use scripts/rebuild_sprites.py instead, which
# samples only from the test split.

import os
import json
import cv2
import numpy as np
import shutil

scratch_dir = "scratch/taco"
sprites_dir = "outputs/sprites"

taco_classes = ["glass", "metal", "plastic", "cardboard", "paper"]
for c in taco_classes:
    d = os.path.join(sprites_dir, c)
    if os.path.exists(d):
        shutil.rmtree(d)
    os.makedirs(d, exist_ok=True)

with open(os.path.join(scratch_dir, "data/annotations.json"), "r") as f:
    coco = json.load(f)

cat_mapping = {
    "cardboard": ["Corrugated carton", "Drink carton", "Egg carton", "Meal carton", "Other carton", "Pizza box", "Toilet tube"],
    "glass": ["Glass bottle", "Broken glass", "Glass cup", "Glass jar"],
    "metal": ["Aerosol", "Aluminium foil", "Aluminium blister pack", "Metal bottle cap", "Drink can", "Food Can", "Metal lid", "Pop tab", "Scrap metal"],
    "paper": ["Paper cup", "Magazine paper", "Tissues", "Wrapping paper", "Normal paper", "Paper bag", "Plastified paper bag", "Paper straw"],
    "plastic": ["Carded blister pack", "Clear plastic bottle", "Other plastic bottle", "Plastic bottle cap", "Disposable plastic cup", "Foam cup", "Other plastic cup", "Plastic lid", "Garbage bag", "Single-use carrier bag", "Polypropylene bag", "Produce bag", "Cereal bag", "Bread bag", "Plastic film", "Crisp packet", "Other plastic wrapper", "Retort pouch", "Spread tub", "Tupperware", "Disposable food container", "Foam food container", "Other plastic container", "Plastic gloves", "Plastic utensils", "Six pack rings", "Squeezable tube", "Plastic straw", "Styrofoam piece", "Other plastic"]
}

taco_to_our = {}
for cat in coco["categories"]:
    name = cat["name"]
    for our_class, names in cat_mapping.items():
        if name in names:
            taco_to_our[cat["id"]] = our_class
            break

img_dict = {img["id"]: img for img in coco["images"]}

generated = 0
for ann in coco["annotations"]:
    cat_id = ann["category_id"]
    if cat_id not in taco_to_our:
        continue
    
    our_class = taco_to_our[cat_id]
    img_info = img_dict[ann["image_id"]]
    img_path = os.path.join(scratch_dir, "data", img_info["file_name"])
    
    if not os.path.exists(img_path):
        continue
        
    img = cv2.imread(img_path)
    if img is None: continue
    
    img_rgba = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    
    for seg in ann["segmentation"]:
        if isinstance(seg, list):
            poly = np.array(seg).reshape((-1, 2)).astype(np.int32)
            cv2.fillPoly(mask, [poly], 255)
            
    img_rgba[:, :, 3] = mask
    
    x, y, w, h = [int(v) for v in ann["bbox"]]
    pad = 5
    x1, y1 = max(0, x-pad), max(0, y-pad)
    x2, y2 = min(img.shape[1], x+w+pad), min(img.shape[0], y+h+pad)
    
    crop = img_rgba[y1:y2, x1:x2]
    if crop.shape[0] == 0 or crop.shape[1] == 0: continue
    
    out_name = f"taco_sprite_{ann['image_id']}_{ann['id']}.png"
    cv2.imwrite(os.path.join(sprites_dir, our_class, out_name), crop)
    generated += 1

print(f"Generated {generated} alpha-transparent sprites from TACO.")
