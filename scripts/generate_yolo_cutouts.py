import os
import glob
import json
import random
import time
import cv2
import numpy as np
import PIL.Image
from scipy.ndimage import label
from ultralytics import FastSAM

TARGETS = {
    "cardboard": 1500,
    "ewaste": 1500,
    "glass": 1500,
    "metal": 1500,
    "paper": 1200,
    "plastic": 1500,
    "organic": 1500
}
# Keep a fallback for any unexpected class
def get_target(cls):
    return TARGETS.get(cls, 1500)

import os
BASE_OUTPUT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'outputs'))

# Update paths
STAGE1_SPLIT_PATH = os.path.join(BASE_OUTPUT, 'stage1_split.json')
STAGE2_SPLIT_PATH = os.path.join(BASE_OUTPUT, 'stage2_split.json')
OUT_DIR = os.path.join(BASE_OUTPUT, 'yolo_cutouts')
SPRITES_DIR = os.path.join(BASE_OUTPUT, 'sprites')


def load_train_splits():
    train_paths = {}
    with open(STAGE1_SPLIT_PATH) as f:
        d = json.load(f)
        for p, label in d.get("train", []):
            if "organic" in p.lower() or "O" in p:
                train_paths.setdefault("organic", []).append(p)
                
    with open(STAGE2_SPLIT_PATH) as f:
        d = json.load(f)
        for p, label in d.get("train", []):
            for cls in ["cardboard", "ewaste", "glass", "metal", "paper", "plastic"]:
                if f"/{cls}/" in p or f"\\{cls}\\" in p:
                    train_paths.setdefault(cls, []).append(p)
    return train_paths

def check_mask_quality(mask_img, class_name):
    if mask_img.sum() == 0:
        return False, None
        
    labeled_array, num_features = label(mask_img > 0)
    if num_features == 0:
        return False, None
        
    component_sizes = np.bincount(labeled_array.ravel())
    component_sizes[0] = 0
    largest_component_size = component_sizes.max()
    total_mask_area = np.sum(mask_img > 0)
    
    cc_ratio = largest_component_size / total_mask_area
    if cc_ratio < 0.90:
        return False, None
        
    largest_cc_idx = component_sizes.argmax()
    largest_cc_mask = (labeled_array == largest_cc_idx).astype(np.uint8)
    
    y, x = np.where(largest_cc_mask > 0)
    ymin, ymax = y.min(), y.max()
    xmin, xmax = x.min(), x.max()
    
    bbox_w = xmax - xmin + 1
    bbox_h = ymax - ymin + 1
    
    if min(bbox_w, bbox_h) < 40:
        return False, None
    if max(bbox_w, bbox_h) / min(bbox_w, bbox_h) > 3.0:
        return False, None
    
    bbox_area = bbox_w * bbox_h
    fill_ratio = largest_component_size / bbox_area
    
    img_area = mask_img.shape[0] * mask_img.shape[1]
    bbox_fraction = bbox_area / img_area
    
    if fill_ratio < 0.20:
        return False, None
        
    if fill_ratio > 0.97:
        if class_name in ["cardboard", "paper"] and bbox_fraction >= 0.40:
            pass
        else:
            return False, None
            
    return True, (xmin, ymin, xmax, ymax, largest_cc_mask)

def main():
    random.seed(42)
    os.makedirs(OUT_DIR, exist_ok=True)
    
    # Pre-load existing sprite basenames to avoid leakage
    existing_sprites = glob.glob(os.path.join(SPRITES_DIR, "**/*.png"), recursive=True)
    sprite_basenames = {os.path.basename(p).replace(".png", "") for p in existing_sprites}
    
    print("Loading FastSAM model...")
    # NOTE: outputs/FastSAM-s.pt was deleted in an earlier cleanup pass
    # (NOTES.md section 21, as a "base pretrained starting weight" no
    # longer referenced at the time). Using the bare model name instead of
    # a hardcoded path into outputs/ lets Ultralytics auto-download and
    # cache it the standard way (same mechanism YOLO("yolov8n.pt") already
    # relies on below in train_evaluate_yolo.py) instead of hard-failing on
    # a missing local file -- important for an unattended overnight run
    # where there's no one around to notice and re-run it manually.
    model = FastSAM("FastSAM-s.pt")
    
    train_paths = load_train_splits()
    
    start_time = time.time()
    generated_counts = {}
    
    classes = ["organic", "cardboard", "ewaste", "glass", "metal", "paper", "plastic"]
    
    for cls in classes:
        os.makedirs(os.path.join(OUT_DIR, cls), exist_ok=True)
        paths = train_paths.get(cls, [])
        random.shuffle(paths)
        
        success_count = 0
        
        print(f"Generating cutouts for {cls}...")
        for p in paths:
            if success_count >= get_target(cls):
                break
                
            basename = os.path.basename(p).replace(".jpg", "")
            if basename in sprite_basenames:
                continue
                
            try:
                # Need absolute path since the json has relative
                abs_p = os.path.abspath(p)
                if not os.path.exists(abs_p):
                    # fallback
                    abs_p = os.path.abspath(os.path.join("data", p.split("data/")[-1] if "data/" in p else p))
                if not os.path.exists(abs_p): continue
                
                # FastSAM predicts
                pil_img = PIL.Image.open(abs_p).convert("RGB")
                orig_w, orig_h = pil_img.size
                
                # FastSAM runs at 1024 or 640
                results = model(pil_img, device="mps", retina_masks=True, imgsz=1024, conf=0.4, iou=0.9, verbose=False)
                
                if len(results) == 0 or results[0].masks is None:
                    continue
                    
                # We just take the mask with the largest area, or iterate until we find one that passes
                masks_data = results[0].masks.data.cpu().numpy()
                
                # Mask output shape is (N, H, W). We resize back to orig
                # Sort masks by area descending
                areas = [m.sum() for m in masks_data]
                sorted_idx = np.argsort(areas)[::-1]
                
                for idx in sorted_idx:
                    mask = masks_data[idx]
                    mask_256 = (mask > 0.0).astype(np.uint8)
                    mask_orig = cv2.resize(mask_256, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
                    
                    # Check for background inversion
                    top = mask_orig[0, :]
                    bottom = mask_orig[-1, :]
                    left = mask_orig[:, 0]
                    right = mask_orig[:, -1]
                    border_sum = top.sum() + bottom.sum() + left.sum() + right.sum()
                    border_len = (mask_orig.shape[0] + mask_orig.shape[1]) * 2
                    coverage = border_sum / border_len
                    
                    if coverage > 0.40:
                        mask_orig = 1 - mask_orig
                        
                    is_good, bbox_info = check_mask_quality(mask_orig, cls)
                    if is_good:
                        xmin, ymin, xmax, ymax, largest_cc_mask = bbox_info
                        
                        sanity_diff = np.logical_and(largest_cc_mask == 1, mask_orig == 0).sum()
                        if sanity_diff > 0:
                            continue
                            
                        rgba = np.zeros((orig_h, orig_w, 4), dtype=np.uint8)
                        img_np = np.array(pil_img)
                        rgba[:, :, :3] = img_np
                        rgba[:, :, 3] = largest_cc_mask * 255
                        
                        cropped = rgba[ymin:ymax+1, xmin:xmax+1]
                        
                        out_path = os.path.join(OUT_DIR, cls, basename + ".png")
                        cv2.imwrite(out_path, cv2.cvtColor(cropped, cv2.COLOR_RGBA2BGRA))
                        success_count += 1
                        break # Successfully found a mask for this image
                        
            except Exception as e:
                pass
                
        generated_counts[cls] = success_count
        print(f"{cls}: Generated {success_count} cutouts.")
        
    print("\n--- Summary ---")
    for k, v in generated_counts.items():
        print(f"{k}: {v}")
    print(f"Total time: {time.time() - start_time:.1f}s")
    
if __name__ == "__main__":
    main()
