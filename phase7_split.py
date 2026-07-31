import os
import json
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from src.splits import split_dataset_no_leakage

STAGE2_DIR = "data/stage2_material"
OUT_FILE = "outputs/stage2_split.json"

print("Running dedup and split for stage 2...")
res = split_dataset_no_leakage(STAGE2_DIR, train_frac=0.7, val_frac=0.15, test_frac=0.15)

# Save to outputs/stage2_split.json
out_data = {
    "train": res.train,
    "val": res.val,
    "test": res.test
}

with open(OUT_FILE, "w") as f:
    json.dump(out_data, f, indent=2)

print(f"Saved splits to {OUT_FILE}")
print(f"Train: {len(res.train)}, Val: {len(res.val)}, Test: {len(res.test)}")
