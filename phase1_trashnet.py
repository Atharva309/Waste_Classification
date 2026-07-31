import os
import shutil
import glob
import subprocess
import zipfile

scratch_dir = "scratch/trashnet"
dest_base = "data/stage2_material"
classes = ["glass", "paper", "cardboard", "plastic", "metal"]

# Unzip
zip_path = os.path.join(scratch_dir, "data", "dataset-resized.zip")
extract_path = os.path.join(scratch_dir, "data")
if os.path.exists(zip_path):
    print("Unzipping dataset...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)

source_data = os.path.join(scratch_dir, "data", "dataset-resized")

copied = 0
for c in classes:
    src_class_dir = os.path.join(source_data, c)
    if not os.path.exists(src_class_dir):
        print(f"Warning: {src_class_dir} not found.")
        continue
        
    dest_class_dir = os.path.join(dest_base, c)
    os.makedirs(dest_class_dir, exist_ok=True)
    
    files = glob.glob(os.path.join(src_class_dir, "*.jpg"))
    for f in files:
        basename = os.path.basename(f)
        new_name = f"trashnet_{basename}"
        shutil.copy(f, os.path.join(dest_class_dir, new_name))
        copied += 1

print(f"Copied {copied} images from TrashNet.")
