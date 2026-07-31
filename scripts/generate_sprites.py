import os
import json
import random
import time
import cv2
import numpy as np
import PIL.Image
import coremltools as ct
from scipy.ndimage import label, sum as ndsum

# Paths
STAGE1_SPLIT = "outputs/stage1_split.json"
STAGE2_SPLIT = "outputs/stage2_split.json"
SPRITES_DIR = "outputs/sprites"
MODELS_DIR = "outputs/sam2_models"

# Mask Heuristics
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
        
    # Get bbox of largest component
    largest_cc_idx = component_sizes.argmax()
    largest_cc_mask = (labeled_array == largest_cc_idx).astype(np.uint8)
    
    y, x = np.where(largest_cc_mask > 0)
    ymin, ymax = y.min(), y.max()
    xmin, xmax = x.min(), x.max()
    
    bbox_w = xmax - xmin + 1
    bbox_h = ymax - ymin + 1
    
    # Degenerate crop filter
    if min(bbox_w, bbox_h) < 40:
        return False, None
    if max(bbox_w, bbox_h) / min(bbox_w, bbox_h) > 3.0:
        return False, None
        
    bbox_area = bbox_w * bbox_h
    fill_ratio = largest_component_size / bbox_area
    
    img_area = mask_img.shape[0] * mask_img.shape[1]
    bbox_fraction = bbox_area / img_area
    
    # Check fill ratio
    if fill_ratio < 0.20:
        return False, None
        
    if fill_ratio > 0.97:
        if class_name in ["cardboard", "paper"] and bbox_fraction >= 0.40:
            pass # Relaxed
        else:
            return False, None
            
    return True, (xmin, ymin, xmax, ymax, largest_cc_mask)

def main():
    start_time = time.time()
    random.seed(42)
    os.makedirs(SPRITES_DIR, exist_ok=True)
    
    with open(STAGE1_SPLIT, "r") as f:
        stage1_split = json.load(f)
    organic_test = [item for item in stage1_split["test"] if item[1] == "O"]
    random.shuffle(organic_test)
    organic_test = organic_test[:150]
    
    with open(STAGE2_SPLIT, "r") as f:
        stage2_split = json.load(f)
    material_test = stage2_split["test"]
    
    all_images = organic_test + material_test
    print(f"Total images to process: {len(all_images)} (150 organic + {len(material_test)} material)")
    
    print("Loading CoreML models...")
    img_enc = ct.models.MLModel(os.path.join(MODELS_DIR, "SAM2LargeImageEncoderFLOAT16.mlpackage"))
    prompt_enc = ct.models.MLModel(os.path.join(MODELS_DIR, "SAM2LargePromptEncoderFLOAT16.mlpackage"))
    mask_dec = ct.models.MLModel(os.path.join(MODELS_DIR, "SAM2LargeMaskDecoderFLOAT16.mlpackage"))
    
    # SAM2 CoreML is exported for exactly 1 point
    points = np.array([[[512, 512]]], dtype=np.float32) # Shape: (1, 1, 2)
    labels = np.array([[1]], dtype=np.float32) # Shape: (1, 1)
    
    processed_count = {}
    total_processed = 0
    
    for item in all_images:
        class_name = "organic" if item[1] == "O" else item[1]
        if class_name not in processed_count:
            processed_count[class_name] = 0
            
        img_path = item[0]
        if not os.path.exists(img_path):
            continue
            
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
            
            # Check if the mask is actually the background (high border coverage)
            top = mask_orig[0, :]
            bottom = mask_orig[-1, :]
            left = mask_orig[:, 0]
            right = mask_orig[:, -1]
            border_sum = top.sum() + bottom.sum() + left.sum() + right.sum()
            border_len = (mask_orig.shape[0] + mask_orig.shape[1]) * 2
            coverage = border_sum / border_len
            
            if coverage > 0.40:
                # SAM2 segmented the background. Invert it to get the object.
                mask_orig = 1 - mask_orig
            
            is_good, bbox_info = check_mask_quality(mask_orig, class_name)
            if not is_good: continue
            
            xmin, ymin, xmax, ymax, largest_cc_mask = bbox_info
            
            # Sanity check: ensure the alpha channel foreground we are about to use matches the selected SAM2 foreground
            # largest_cc_mask should be a subset of mask_orig.
            # Check if there's any pixel in largest_cc_mask that is NOT in mask_orig
            sanity_diff = np.logical_and(largest_cc_mask == 1, mask_orig == 0).sum()
            if sanity_diff > 0:
                print(f"Sanity check failed on {img_path}: alpha has {sanity_diff} pixels not in original SAM2 foreground. Skipping.")
                continue
            
            rgba = np.zeros((orig_h, orig_w, 4), dtype=np.uint8)
            img_np = np.array(pil_img)
            rgba[:, :, :3] = img_np
            rgba[:, :, 3] = largest_cc_mask * 255
            
            cropped = rgba[ymin:ymax+1, xmin:xmax+1]
            
            class_dir = os.path.join(SPRITES_DIR, class_name)
            os.makedirs(class_dir, exist_ok=True)
            base_name = os.path.basename(img_path)
            out_path = os.path.join(class_dir, os.path.splitext(base_name)[0] + ".png")
            
            cv2.imwrite(out_path, cv2.cvtColor(cropped, cv2.COLOR_RGBA2BGRA))
            processed_count[class_name] += 1
            total_processed += 1
            
            if total_processed % 50 == 0:
                print(f"Processed {total_processed}/{len(all_images)}... (Time elapsed: {time.time() - start_time:.1f}s)")
                
        except Exception as e:
            print(f"Error on {img_path}: {e}")
            
    total_time = time.time() - start_time
    print(f"Finished generating sprites in {total_time:.1f}s.")
    print(f"Final pool counts: {processed_count}")

if __name__ == "__main__":
    main()
