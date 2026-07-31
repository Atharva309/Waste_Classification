import os
import shutil
import glob

base = "/Users/sachin/Documents/Claude-code/Waste_Classification-main/pytorch_pipeline"
inv_dir = os.path.join(base, "scripts", "investigations")
os.makedirs(inv_dir, exist_ok=True)

# 1. Archive tests and probes
to_archive = [
    "test_belt.py", "test_belt2.py", "test_belt3.py", "test_belt4.py", "test_belt5.py",
    "test_coreml.py", "test_inversion.py", "debug_mask.py", "probe_sam2.py", "probe_shapes.py",
    "print_layers.py", "preload_weights.py", "preview_sprites.py", "regenerate_inverted.py",
    "scan_inverted.py", "cleanup_sprites.py"
]

for f in to_archive:
    p = os.path.join(base, f)
    if not os.path.exists(p):
        p = os.path.join(base, "scripts", f)
    if os.path.exists(p):
        shutil.move(p, os.path.join(inv_dir, os.path.basename(p)))

# 2. Archive unreferenced scripts
for f in ["recover_masks.py", "refine_recovery.py", "annotate_sam.py"]:
    p = os.path.join(base, "scripts", f)
    if os.path.exists(p):
        shutil.move(p, os.path.join(inv_dir, os.path.basename(p)))

# 3. Delete deprecated scripts
for f in ["train_stage1.py", "train_stage2.py", "retrain_stage1_augmented.py", "retrain_stage2_early_stopping.py"]:
    p = os.path.join(base, "scripts", f)
    if os.path.exists(p):
        os.remove(p)

# 4. Delete logs and screenshots
for f in ["pipeline.log", "app.log", "screenshot.png"]:
    p = os.path.join(base, f)
    if os.path.exists(p):
        os.remove(p)
        
# 5. Move weights to outputs/
weights = ["FastSAM-s.pt", "yolov8n.pt"]
for w in weights:
    p = os.path.join(base, w)
    if os.path.exists(p):
        shutil.move(p, os.path.join(base, "outputs", w))
