import sys
import os
import random
import glob
import json
import yaml
import numpy as np
from PIL import Image, ImageDraw
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.dedup import find_duplicate_groups
from src.synthetic_detection import compose_belt_frame

YOLO_CLASSES = ["organic", "cardboard", "ewaste", "glass", "metal", "paper", "plastic"]
CLASS_TO_IDX = {c: i for i, c in enumerate(YOLO_CLASSES)}

def get_all_source_images():
    paths_and_labels = []

    yolo_dir = "data/yolo_data/cutouts"
    for c in YOLO_CLASSES:
        for p in glob.glob(os.path.join(yolo_dir, c, "*.png")):
            paths_and_labels.append((os.path.abspath(p), c))
            
    return paths_and_labels

def main():
    random.seed(42)
    np.random.seed(42)
    
    # Lives under data/ (not outputs/) alongside cascades_data and the
    # cutout source above -- this is the actual training dataset, not a
    # checkpoint/report artifact.
    out_dir = os.path.abspath("data/yolo_data/dataset")
    os.makedirs(os.path.join(out_dir, "images", "train"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "images", "val"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "labels", "train"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "labels", "val"), exist_ok=True)
    os.makedirs("data/yolo_data/samples", exist_ok=True)
    
    print("Gathering source images...")
    all_sources = get_all_source_images()
    print(f"Found {len(all_sources)} total source images.")
    
    path_to_label = {p: l for p, l in all_sources}
    all_paths = list(path_to_label.keys())
    
    print("Running duplicate detection to ensure leak-free splits...")
    groups, _ = find_duplicate_groups(all_paths)
    
    group_list = list(groups.values())
    random.shuffle(group_list)
    
    n_groups = len(group_list)
    n_train = int(n_groups * 0.85)
    
    train_pool = []
    val_pool = []
    
    for i, grp in enumerate(group_list):
        for p in grp:
            if i < n_train:
                train_pool.append((p, path_to_label[p]))
            else:
                val_pool.append((p, path_to_label[p]))
                
    print(f"Split source images: Train Pool={len(train_pool)}, Val Pool={len(val_pool)}")
    
    def generate_split(pool, split_name, num_frames, save_samples=0):
        print(f"\nGenerating {num_frames} {split_name} frames...")
        class_counts = {c: 0 for c in YOLO_CLASSES}
        frame_size = (640, 640)
        
        # Organize pool by class
        pool_by_class = {c: [] for c in YOLO_CLASSES}
        for path, label in pool:
            pool_by_class[label].append((path, label))
            
        # Filter out classes with no images in this split (shouldn't happen, but safe)
        available_classes = [c for c in YOLO_CLASSES if len(pool_by_class[c]) > 0]
        
        for i in tqdm(range(num_frames)):
            n_items = random.randint(2, 8)
            crops = []
            for _ in range(n_items):
                # Pick class uniformly
                chosen_class = random.choice(available_classes)
                # Pick random image from chosen class
                path, label = random.choice(pool_by_class[chosen_class])
                
                img = Image.open(path).convert("RGBA")
                crops.append((np.array(img), label))
                
            frame, gt_boxes = compose_belt_frame(
                crops, 
                frame_size=frame_size, 
                belt_color=(60, 60, 65), 
                min_scale=0.08, 
                max_scale=0.15, 
                allow_overlap=True
            )
            
            img_path = os.path.join(out_dir, "images", split_name, f"frame_{i:04d}.jpg")
            label_path = os.path.join(out_dir, "labels", split_name, f"frame_{i:04d}.txt")
            
            Image.fromarray(frame).save(img_path, quality=90)
            
            # Write YOLO labels
            with open(label_path, "w") as f:
                for label, box in gt_boxes:
                    class_counts[label] += 1
                    x1, y1, x2, y2 = box
                    # YOLO format: cls x_center y_center width height (normalized 0-1)
                    w = x2 - x1
                    h = y2 - y1
                    xc = x1 + w/2
                    yc = y1 + h/2
                    
                    norm_xc = xc / frame_size[0]
                    norm_yc = yc / frame_size[1]
                    norm_w = w / frame_size[0]
                    norm_h = h / frame_size[1]
                    
                    idx = CLASS_TO_IDX[label]
                    f.write(f"{idx} {norm_xc:.6f} {norm_yc:.6f} {norm_w:.6f} {norm_h:.6f}\n")
            
            # Save visual samples
            if i < save_samples:
                sample_img = Image.fromarray(frame).copy()
                draw = ImageDraw.Draw(sample_img)
                for label, box in gt_boxes:
                    draw.rectangle(box, outline="red", width=2)
                    draw.text((box[0], box[1]-10), label, fill="red")
                sample_img.save(f"data/yolo_data/samples/sample_{split_name}_{i}.jpg")
                
        return class_counts

    train_counts = generate_split(train_pool, "train", 2000)
    val_counts = generate_split(val_pool, "val", 400, save_samples=5)
    
    # Write data.yaml
    yaml_content = {
        "train": os.path.join(out_dir, "images", "train"),
        "val": os.path.join(out_dir, "images", "val"),
        "nc": len(YOLO_CLASSES),
        "names": YOLO_CLASSES
    }
    with open(os.path.join(out_dir, "data.yaml"), "w") as f:
        yaml.dump(yaml_content, f, default_flow_style=False)
        
    print("\nGeneration Complete.")
    print("Train Class Distribution:", train_counts)
    print("Val Class Distribution:", val_counts)

if __name__ == "__main__":
    main()
