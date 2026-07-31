"""
Targeted fix-up: regenerate ONLY the 5 material cutout classes (cardboard,
glass, metal, paper, plastic) across train/val/test, using the corrected
trashnet_cutout (box_frac upper bound relaxed 0.96 -> 0.995, see NOTES.md
"Bug found mid-run"). Does NOT touch organic or ewaste -- both already
generated correctly in the previous run (organic uses forced_bbox_crop,
unaffected by this bug entirely; same for ewaste), and organic's train
split alone took ~76 minutes, not worth repeating.

What this does:
  1. Wipes and regenerates data/belt_composite_v1/stage2/{split}/{cls}/ for
     the 5 material classes only -- ewaste's files are left untouched.
  2. Rebuilds stage2_split.json: keeps ewaste's existing entries exactly as
     they were, replaces the 5 material classes' entries with the new
     (fixed-threshold) ones.
  3. Rebuilds stage1_split.json: keeps organic's existing entries exactly
     as they were (already correct, already on disk), replaces the
     "recyclable" entries with the freshly-rebuilt stage2 manifest
     (all 6 material classes pooled, relabeled) -- same reuse logic as
     generate_belt_dataset.py, just re-derived from the corrected data.

Run on your Mac, from a fresh terminal (the previous generate/train command
should be interrupted first -- Ctrl+C -- since it's still running on the
old, biased data):

    cd pytorch_pipeline
    python3 scripts/regenerate_materials_fixed.py
"""

import os
import sys
import json
import random
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from composite_preview import load_belt, _dedup_by_real_target  # noqa: E402
from generate_belt_dataset import (  # noqa: E402
    generate_material_class, verify_source_split_leak_free,
    MATERIAL_CUTOUT_CLASSES, SEED,
)

STAGE1_SPLIT_SRC = "outputs/stage1_split.json"
STAGE2_SPLIT_SRC = "outputs/stage2_split.json"
OUT_ROOT = "data/belt_composite_v1"
STAGE2_SPLIT_OUT = os.path.join(OUT_ROOT, "stage2_split.json")
STAGE1_SPLIT_OUT = os.path.join(OUT_ROOT, "stage1_split.json")


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def clear_class_dir(split_name, cls):
    d = os.path.join(OUT_ROOT, "stage2", split_name, cls)
    if os.path.isdir(d):
        for f in os.listdir(d):
            os.remove(os.path.join(d, f))
    else:
        os.makedirs(d, exist_ok=True)
    return d


def main():
    if not os.path.exists(STAGE2_SPLIT_OUT) or not os.path.exists(STAGE1_SPLIT_OUT):
        print(f"ERROR: {STAGE2_SPLIT_OUT} / {STAGE1_SPLIT_OUT} not found -- "
              f"run generate_belt_dataset.py (at least far enough to finish "
              f"organic/ewaste) before this script.")
        sys.exit(1)

    random.seed(SEED)

    with open(STAGE1_SPLIT_SRC) as f:
        s1 = json.load(f)
    with open(STAGE2_SPLIT_SRC) as f:
        s2 = json.load(f)

    verify_source_split_leak_free(s1, s2)

    with open(STAGE2_SPLIT_OUT) as f:
        old_stage2_manifest = json.load(f)

    # Preserve ewaste's existing entries, drop the 5 material classes' old
    # (biased) entries -- they'll be rebuilt fresh below.
    new_stage2_manifest = {"train": [], "val": [], "test": []}
    for split_name in ["train", "val", "test"]:
        new_stage2_manifest[split_name] = [
            entry for entry in old_stage2_manifest[split_name]
            if entry[1] == "ewaste"
        ]

    belt_im = load_belt()

    material_by_split = {}
    for split_name in ["train", "val", "test"]:
        by_class = {}
        for p, label in s2[split_name]:
            by_class.setdefault(label, []).append(p)
        material_by_split[split_name] = by_class

    stage2_summary = {}
    for split_name in ["train", "val", "test"]:
        for cls in MATERIAL_CUTOUT_CLASSES:
            out_dir = clear_class_dir(split_name, cls)
            paths = material_by_split[split_name].get(cls, [])
            written, failed = generate_material_class(
                cls, split_name, paths, belt_im, out_dir,
                new_stage2_manifest[split_name], cls
            )
            stage2_summary.setdefault(cls, {})[split_name] = written
            log(f"stage2/{split_name}/{cls}: wrote {written}, failed/skipped {failed} "
                f"(source images: {len(paths)})")

    with open(STAGE2_SPLIT_OUT, "w") as f:
        json.dump(new_stage2_manifest, f)
    log(f"Rewrote {STAGE2_SPLIT_OUT}")

    # Rebuild stage1's "recyclable" entries from the fresh stage2 manifest;
    # keep "organic" entries exactly as they already are on disk.
    with open(STAGE1_SPLIT_OUT) as f:
        old_stage1_manifest = json.load(f)

    new_stage1_manifest = {"train": [], "val": [], "test": []}
    for split_name in ["train", "val", "test"]:
        organic_entries = [
            entry for entry in old_stage1_manifest[split_name]
            if entry[1] == "organic"
        ]
        recyclable_entries = [
            [path, "recyclable"] for path, _ in new_stage2_manifest[split_name]
        ]
        new_stage1_manifest[split_name] = organic_entries + recyclable_entries
        log(f"stage1/{split_name}: kept {len(organic_entries)} organic entries, "
            f"rebuilt {len(recyclable_entries)} recyclable entries")

    with open(STAGE1_SPLIT_OUT, "w") as f:
        json.dump(new_stage1_manifest, f)
    log(f"Rewrote {STAGE1_SPLIT_OUT}")

    print("\n" + "=" * 60)
    print("STAGE 2 (material) summary -- regenerated classes only")
    total = 0
    for cls, per_split in stage2_summary.items():
        row_total = sum(per_split.values())
        total += row_total
        print(f"  {cls:10s}: train={per_split.get('train',0):6d}  "
              f"val={per_split.get('val',0):5d}  test={per_split.get('test',0):5d}  "
              f"total={row_total}")
    ewaste_total = sum(len(v) for v in [
        [e for e in new_stage2_manifest[s] if e[1] == "ewaste"] for s in ["train", "val", "test"]
    ])
    print(f"  ewaste (unchanged): {ewaste_total}")
    print(f"  STAGE 2 NEW TOTAL: {total + ewaste_total}")

    s1_total = sum(len(new_stage1_manifest[s]) for s in ["train", "val", "test"])
    print(f"  STAGE 1 NEW TOTAL (organic + rebuilt recyclable): {s1_total}")
    print("=" * 60)
    print("\nDone. Ready to run scripts/train_belt_cascade.py on the corrected data.")


if __name__ == "__main__":
    main()
