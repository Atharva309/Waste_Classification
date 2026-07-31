"""
Rebuild outputs/sprites/ from the TEST split ONLY, for every class, with
equal per-class counts.

Why this exists (see NOTES.md, "Sprite pool data leakage" section):
  - generate_sprites.py originally filtered sprites to the test split only.
  - phase6_sprites.py later regenerated the 5 TACO material classes directly
    from the raw, unsplit TACO annotations.json, ignoring the split entirely.
  - backfill_sprites.py topped up organic/ewaste (and re-supplemented TACO
    classes) by globbing every image in the raw, unsplit source folders
    (data/stage1_merged/O, data/stage2_material/<class>) to hit target
    counts, also ignoring the split.
  - Net result: the live webapp's test pool was 72-86% images the models
    were actually trained on, not held-out data. Every macro-F1 number
    produced against that pool was measuring memorization, not generalization.

This script fixes both problems in one pass:
  1. Leakage: every sprite is sampled ONLY from stage1_split.json["test"] /
     stage2_split.json["test"].
  2. Class imbalance: every class gets the same count (SPRITES_PER_CLASS),
     so get_random_sprite()'s uniform random.choice() over the whole pool
     naturally yields a near-even class distribution on the conveyor,
     instead of being ~56% plastic.

Run this on your Mac (NOT in a sandboxed/Linux shell -- the source images
under data/stage1_merged and data/stage2_material are symlinks that only
resolve on the machine that actually has the dataset):

    cd pytorch_pipeline
    python3 scripts/rebuild_sprites.py

It only touches outputs/sprites/. It does not touch any .pt checkpoint,
any training data, or outputs/unusable/.
"""

import os
import json
import random
import shutil

from PIL import Image

STAGE1_SPLIT = "outputs/stage1_split.json"
STAGE2_SPLIT = "outputs/stage2_split.json"
SPRITES_DIR = "outputs/sprites"

# Bottleneck class is metal, with 91 test images available (see NOTES.md).
# 90 is achievable for every class (organic has 957, the smallest material
# class -- metal -- has 91).
SPRITES_PER_CLASS = 90

SEED = 42


def load_test_items():
    """Returns {class_name: [source_path, ...]} using ONLY the test split."""
    with open(STAGE1_SPLIT) as f:
        s1 = json.load(f)
    with open(STAGE2_SPLIT) as f:
        s2 = json.load(f)

    items = {}

    organic_test = [p for p, label in s1["test"] if label == "O"]
    items["organic"] = organic_test

    by_class = {}
    for p, label in s2["test"]:
        by_class.setdefault(label, []).append(p)
    items.update(by_class)

    return items


def rebuild():
    random.seed(SEED)
    items = load_test_items()

    print("Test-split pool sizes (before sampling):")
    for cls, paths in items.items():
        print(f"  {cls}: {len(paths)} available")

    for cls, paths in items.items():
        out_dir = os.path.join(SPRITES_DIR, cls)

        if len(paths) < SPRITES_PER_CLASS:
            print(f"WARNING: {cls} only has {len(paths)} test images, "
                  f"using all of them (< {SPRITES_PER_CLASS} requested).")
            chosen = list(paths)
        else:
            chosen = random.sample(paths, SPRITES_PER_CLASS)

        # Wipe and recreate only this class's sprite dir.
        if os.path.isdir(out_dir):
            shutil.rmtree(out_dir)
        os.makedirs(out_dir, exist_ok=True)

        written = 0
        skipped = 0
        for src_path in chosen:
            stem = os.path.splitext(os.path.basename(src_path))[0]
            dst_path = os.path.join(out_dir, f"{stem}.png")
            try:
                with Image.open(src_path) as im:
                    im = im.convert("RGB")
                    im.save(dst_path, "PNG")
                written += 1
            except Exception as e:
                print(f"  skipping {src_path}: {e}")
                skipped += 1

        print(f"{cls}: wrote {written} sprites (test split only), "
              f"skipped {skipped}")

    print("\nDone. Restart the webapp to pick up the new sprite pool.")


if __name__ == "__main__":
    rebuild()
