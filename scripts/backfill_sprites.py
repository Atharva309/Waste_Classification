# WARNING (see NOTES.md, "Sprite pool data leakage"): this script tops up
# sprite counts by globbing ALL images in the raw, unsplit source dirs
# (data/stage1_merged/O, data/stage2_material/<class>) with no train/test
# filtering. This is the script that caused organic (79% train) and ewaste
# (74% train) contamination in the live webapp's test pool. It also depends
# on outputs/sam2_models, which was deleted during project cleanup. Use
# scripts/rebuild_sprites.py instead, which samples only from the test split.

import os
import glob
import random
import time
import cv2
import numpy as np
import PIL.Image
import coremltools as ct
from scipy.ndimage import label

SPRITES_DIR = "outputs/sprites"
MODELS_DIR = "outputs/sam2_models"

TARGET_COUNTS = {
    "organic": 150,
    "cardboard": 114,
    "ewaste": 127,
    "glass": 143,
    "metal": 87,
    "paper": 102,
    "plastic": 274
}

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
    
    # NEW: Degenerate crop filter
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
    os.makedirs(SPRITES_DIR, exist_ok=True)
    
    print("Loading CoreML models...")
    img_enc = ct.models.MLModel(os.path.join(MODELS_DIR, "SAM2LargeImageEncoderFLOAT16.mlpackage"))
    prompt_enc = ct.models.MLModel(os.path.join(MODELS_DIR, "SAM2LargePromptEncoderFLOAT16.mlpackage"))
    mask_dec = ct.models.MLModel(os.path.join(MODELS_DIR, "SAM2LargeMaskDecoderFLOAT16.mlpackage"))
    
    points = np.array([[[512, 512]]], dtype=np.float32)
    labels = np.array([[1]], dtype=np.float32)
    
    for class_name, target in TARGET_COUNTS.items():
        class_dir = os.path.join(SPRITES_DIR, class_name)
        os.makedirs(class_dir, exist_ok=True)
        
        existing_sprites = glob.glob(os.path.join(class_dir, "*.png"))
        current_count = len(existing_sprites)
        
        if current_count >= target:
            print(f"{class_name}: Already at target {target} (has {current_count}).")
            continue
            
        print(f"{class_name}: Needs {target - current_count} more (has {current_count}/{target}).")
        
        if class_name == "organic":
            source_dir = "data/stage1_merged/O"
        else:
            source_dir = f"data/stage2_material/{class_name}"
            
        all_source_images = glob.glob(os.path.join(source_dir, "*.jpg"))
        random.shuffle(all_source_images)
        
        existing_basenames = set([os.path.basename(p).replace(".png", "") for p in existing_sprites])
        
        exclusions = [
            "cardboard_2853", "cardboard_1757", "cardboard_1812",
            "cardboard_1042", "cardboard_557", "cardboard_2750",
            "cardboard_1", "cardboard_1413", "paper_1059"
        ]
        existing_basenames.update(exclusions)
        
        success_count = 0
        needed = target - current_count
        
        for img_path in all_source_images:
            if success_count >= needed:
                break
                
            basename = os.path.basename(img_path).replace(".jpg", "")
            if basename in existing_basenames:
                continue
                
            # Run SAM2
            try:
                pil_img = PIL.Image.open(img_path).convert("RGB")
                orig_w, orig_h = pil_img.size
                
                img_resized = pil_img.resize((1024, 1024), PIL.Image.BILINEAR)
                img_out = img_enc.predict({"image": img_resized})
                
                prompt_out = prompt_enc.predict({"points": points, "labels": labels})
                
                dec_input = {
                    "image_embedding": img_out["image_embedding"],
                    "feats_s0": img_out["feats_s0"],
                    "feats_s1": img_out["feats_s1"],
                    "sparse_embedding": prompt_out["sparse_embeddings"],
                    "dense_embedding": prompt_out["dense_embeddings"],
                }
                mask_out = mask_dec.predict(dec_input)
                
                low_res_masks = mask_out["low_res_masks"]
                scores = mask_out["scores"]
                
                best_idx = np.argmax(scores[0])
                best_mask = low_res_masks[0, best_idx]
                mask_256 = (best_mask > 0.0).astype(np.uint8)
                
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
                
                is_good, bbox_info = check_mask_quality(mask_orig, class_name)
                if not is_good: continue
                
                xmin, ymin, xmax, ymax, largest_cc_mask = bbox_info
                
                sanity_diff = np.logical_and(largest_cc_mask == 1, mask_orig == 0).sum()
                if sanity_diff > 0:
                    continue
                
                rgba = np.zeros((orig_h, orig_w, 4), dtype=np.uint8)
                img_np = np.array(pil_img)
                rgba[:, :, :3] = img_np
                rgba[:, :, 3] = largest_cc_mask * 255
                
                cropped = rgba[ymin:ymax+1, xmin:xmax+1]
                
                out_path = os.path.join(class_dir, basename + ".png")
                cv2.imwrite(out_path, cv2.cvtColor(cropped, cv2.COLOR_RGBA2BGRA))
                
                success_count += 1
                existing_basenames.add(basename)
                if success_count % 10 == 0:
                    print(f"  ... backfilled {success_count}/{needed} {class_name}")
                    
            except Exception as e:
                pass
                
        print(f"{class_name}: Finished. Total now: {current_count + success_count}/{target}")
        
    print("Done backfilling all classes.")
    
if __name__ == "__main__":
    main()
