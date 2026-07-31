import os
import glob
import json
import shutil
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.dedup import find_duplicate_groups
from src.splits import split_dataset_no_leakage, list_images_by_class

DATA_SRC = "data/trash_classification_data-main"
STAGE2_DIR = "data/stage2_material"
STAGE1_TRAIN = "data/trash_classification_data-main/organictrashdetection/TRAIN"
STAGE1_TEST = "data/trash_classification_data-main/organictrashdetection/TEST"

def prep_stage2():
    print("Preparing stage 2 data (symlinks)...")
    classes = ["cardboard", "ewaste", "glass", "metal", "paper", "plastic"]
    os.makedirs(STAGE2_DIR, exist_ok=True)
    
    for c in classes:
        tgt_dir = os.path.join(STAGE2_DIR, c)
        if os.path.exists(tgt_dir):
            shutil.rmtree(tgt_dir)
        os.makedirs(tgt_dir)
        src_dir = os.path.join(DATA_SRC, "garbage_classification", c)
        
        if not os.path.exists(src_dir):
            print(f"Warning: {src_dir} does not exist.")
            continue
            
        for f in os.listdir(src_dir):
            if f.lower().endswith((".jpg", ".jpeg", ".png")):
                src_file = os.path.abspath(os.path.join(src_dir, f))
                tgt_file = os.path.join(tgt_dir, f)
                if not os.path.exists(tgt_file):
                    os.symlink(src_file, tgt_file)
                    
    print("Running dedup and split for stage 2...")
    res = split_dataset_no_leakage(STAGE2_DIR, train_frac=0.7, val_frac=0.15, test_frac=0.15)
    return {
        "dropped_duplicates": res.dropped_duplicates,
        "duplicate_groups_found": res.duplicate_groups_found,
        "train_size": len(res.train),
        "val_size": len(res.val),
        "test_size": len(res.test)
    }, res

def prep_stage1():
    print("Running dedup for stage 1...")
    train_paths = []
    for c in ["O", "R"]:
        train_paths.extend(glob.glob(os.path.join(STAGE1_TRAIN, c, "*.jpg")))
    
    test_paths = []
    for c in ["O", "R"]:
        test_paths.extend(glob.glob(os.path.join(STAGE1_TEST, c, "*.jpg")))
        
    all_paths = train_paths + test_paths
    groups, _ = find_duplicate_groups(all_paths)
    
    leak_found = False
    train_set = set(train_paths)
    test_set = set(test_paths)
    
    for gid, members in groups.items():
        if len(members) > 1:
            has_train = any(m in train_set for m in members)
            has_test = any(m in test_set for m in members)
            if has_train and has_test:
                leak_found = True
                break
                
    print(f"Stage 1 leakage found: {leak_found}")
    
    if leak_found:
        print("Merging and resplitting stage 1...")
        # Create a merged directory with symlinks for split_dataset_no_leakage
        merged_dir = "data/stage1_merged"
        if os.path.exists(merged_dir):
            shutil.rmtree(merged_dir)
        for c in ["O", "R"]:
            os.makedirs(os.path.join(merged_dir, c), exist_ok=True)
            for p in glob.glob(os.path.join(STAGE1_TRAIN, c, "*.jpg")) + glob.glob(os.path.join(STAGE1_TEST, c, "*.jpg")):
                os.symlink(os.path.abspath(p), os.path.join(merged_dir, c, os.path.basename(p)))
                
        res = split_dataset_no_leakage(merged_dir, train_frac=0.8, val_frac=0.1, test_frac=0.1)
        return {
            "leakage_found": True,
            "dropped_duplicates": res.dropped_duplicates,
            "duplicate_groups_found": res.duplicate_groups_found,
            "train_size": len(res.train),
            "val_size": len(res.val),
            "test_size": len(res.test)
        }, res
    else:
        # Keep original split. We still need a val set? Original has train/test. We can split test in half or keep it as test.
        # But for engine training we need a val loader. We will just use the test set as val and test.
        return {
            "leakage_found": False,
            "dropped_duplicates": 0,
            "duplicate_groups_found": sum(1 for g in groups.values() if len(g) > 1),
            "train_size": len(train_paths),
            "val_size": len(test_paths),
            "test_size": len(test_paths)
        }, None

if __name__ == "__main__":
    os.makedirs("outputs", exist_ok=True)
    stage2_info, stage2_res = prep_stage2()
    stage1_info, stage1_res = prep_stage1()
    
    report = {
        "stage2": stage2_info,
        "stage1": stage1_info
    }
    with open("outputs/split_info.json", "w") as f:
        f.write(json.dumps(report, indent=2))
        
    def save_split(res, path):
        if res is None:
            return
        with open(path, "w") as f:
            f.write(json.dumps({"train": res.train, "val": res.val, "test": res.test}, indent=2))
            
    save_split(stage2_res, "outputs/stage2_split.json")
    save_split(stage1_res, "outputs/stage1_split.json")
