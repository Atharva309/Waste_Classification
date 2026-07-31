"""
Recovery script -- rebuilds data/stage1_merged/{O,R}/ (a pure symlink farm)
after it was wrongly deleted during a cleanup pass (NOTES.md section 54).

IMPORTANT: this does NOT re-run split_dataset_no_leakage() the way
prep_data.py's prep_stage1() originally did. That matters: outputs/
stage1_split.json already exists and every downstream script (blocklist,
generate_yolo_cutouts_v2.py, generate_belt_dataset.py) already assumes its
exact current train/val/test assignment. Re-running the dedup+resplit
logic could shuffle which specific images land in which split, silently
invalidating everything already checked against the current split this
session. This script only recreates the physical symlinks at the paths
outputs/stage1_split.json already references -- it changes nothing about
which image is assigned to which split.

Safe to run multiple times (skips a symlink that already exists/resolves).

Run on your Mac, from pytorch_pipeline/:
    python3 scripts/restore_stage1_merged.py
"""
import os

STAGE1_TRAIN = "data/trash_classification_data-main/organictrashdetection/TRAIN"
STAGE1_TEST = "data/trash_classification_data-main/organictrashdetection/TEST"
MERGED_DIR = "data/stage1_merged"


def main():
    total_created, total_skipped, total_missing_source = 0, 0, 0

    for c in ["O", "R"]:
        out_dir = os.path.join(MERGED_DIR, c)
        os.makedirs(out_dir, exist_ok=True)

        sources = []
        for base in [STAGE1_TRAIN, STAGE1_TEST]:
            d = os.path.join(base, c)
            if not os.path.isdir(d):
                print(f"WARNING: source dir missing: {d}")
                continue
            for f in os.listdir(d):
                if f.lower().endswith((".jpg", ".jpeg", ".png")):
                    sources.append(os.path.join(d, f))

        for p in sources:
            link_path = os.path.join(out_dir, os.path.basename(p))
            if os.path.exists(link_path) or os.path.islink(link_path):
                total_skipped += 1
                continue
            src_abs = os.path.abspath(p)
            if not os.path.exists(src_abs):
                total_missing_source += 1
                print(f"  MISSING REAL SOURCE FILE (not recoverable): {src_abs}")
                continue
            os.symlink(src_abs, link_path)
            total_created += 1

        print(f"{c}: {len(sources)} source files found under TRAIN+TEST")

    print(f"\nDone. Created {total_created} symlinks, skipped {total_skipped} "
          f"already present, {total_missing_source} real source files were "
          f"missing (would be genuine data loss -- should be 0).")


if __name__ == "__main__":
    main()
