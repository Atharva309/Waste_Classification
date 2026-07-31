"""
Backfill the plastic cutout pool back toward 1500 using genuinely NEW source
images only -- never currently in the live pool, and never confirmed bad by
manual audit -- without reintroducing already-removed contamination.

REWRITTEN (see NOTES.md section 45): the original version of this script
tried to replay generate_yolo_cutouts_v2.py's seed=42 shuffle to figure out
which source images were "already used" vs. "fresh," on the assumption that
paths[:1500] (pre-shuffle) would match what the original run actually
consumed. That assumption was checked directly and found FALSE: a live
cutout file's source image can appear as late as position 5372 of 5385 in a
fresh replay of that same shuffle. Most likely cause -- stage2_split.json's
stored ordering isn't stable across regenerations of that file (it's been
rebuilt at least once this session), so random.shuffle(seed=42) over a
differently-ordered input list produces a completely different permutation,
even though the underlying SET of source images is the same. Replaying the
shuffle is therefore not a reliable way to reconstruct "already used."

Fixed approach: don't try to reconstruct history via replay at all. Use set
difference against ground truth instead:
    candidates = (all current source images labeled "plastic")
                 - (basenames currently in the live cutout pool)
                 - (basenames in data/known_bad_source_images.json,
                    the persisted blocklist built from every audit round
                    this session -- see composite_preview.filter_known_bad)
This can't accidentally resurrect a currently-live image (redundant) or a
confirmed-bad one (blocklisted), regardless of any ordering quirks in the
split file. It also doesn't require ORIGINAL_TARGET/seed replay logic at
all -- one less way for the "already used" assumption to silently break
again if stage2_split.json changes again in the future.

Run on your Mac (or in a sandbox with cv2/numpy/PIL -- no torch needed):
    cd pytorch_pipeline
    python3 scripts/backfill_plastic_cutouts.py
"""
import os
import sys
import json
import random
import glob
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from composite_preview import (
    taco_cutout, trashnet_cutout, forced_bbox_crop, _dedup_by_real_target,
    load_known_bad_basenames,
)

STAGE2_SPLIT_PATH = "outputs/stage2_split.json"
OUT_DIR = "data/yolo_data/cutouts/plastic"
CANDIDATE_DIR = "outputs/plastic_backfill_candidates"
POINTER_PATH = os.path.join(CANDIDATE_DIR, "resume_pointer.json")
TARGET_POOL_SIZE = 1500
BUFFER = 1.3  # generate 30% extra so post-audit rejects still hit target
SHUFFLE_SEED = 42  # only affects candidate ORDER, not which set is eligible
TIME_BUDGET_SEC = 3600  # generous; meant for a real terminal, not the sandbox


def main():
    t0 = time.time()

    with open(STAGE2_SPLIT_PATH) as f:
        s2 = json.load(f)
    plastic_paths = [p for p, label in s2["train"] if label == "plastic"]
    paths = _dedup_by_real_target(plastic_paths)

    live_basenames = set(
        os.path.splitext(f)[0] for f in os.listdir(OUT_DIR) if f.endswith(".png")
    ) if os.path.isdir(OUT_DIR) else set()
    bad_basenames = load_known_bad_basenames()

    def basename_of(p):
        return os.path.splitext(os.path.basename(p))[0]

    candidates_pool = [
        p for p in paths
        if basename_of(p) not in live_basenames and basename_of(p) not in bad_basenames
    ]
    random.seed(SHUFFLE_SEED)
    random.shuffle(candidates_pool)

    needed = TARGET_POOL_SIZE - len(live_basenames)
    print(f"Live pool: {len(live_basenames)}")
    print(f"Blocklisted (confirmed bad, all-time): {len(bad_basenames)}")
    print(f"Total source images labeled plastic: {len(paths)}")
    print(f"Eligible fresh candidates (not live, not blocklisted): {len(candidates_pool)}")
    print(f"Need: {needed} to reach {TARGET_POOL_SIZE}")

    if needed <= 0:
        print("Already at or above target, nothing to do.")
        return

    os.makedirs(CANDIDATE_DIR, exist_ok=True)
    start_idx = 0
    manifest = []
    if os.path.exists(POINTER_PATH):
        with open(POINTER_PATH) as f:
            state = json.load(f)
        start_idx = state["next_idx"]
        manifest = state["manifest"]
        print(f"Resuming from index {start_idx}, {len(manifest)} already written")

    target_written = int(needed * BUFFER)
    written = len(manifest)
    cutout_count, fallback_count, failed = 0, 0, 0
    i = start_idx
    while i < len(candidates_pool) and written < target_written:
        if time.time() - t0 > TIME_BUDGET_SEC:
            print(f"Time budget hit, stopping at index {i}. Re-run this script to continue.")
            break
        p = candidates_pool[i]
        i += 1
        basename = basename_of(p)
        is_taco = os.path.basename(p).startswith("taco_")
        try:
            obj = taco_cutout(p) if is_taco else trashnet_cutout(p)
        except Exception:
            obj = None
        is_cutout = obj is not None
        if obj is None:
            try:
                obj = forced_bbox_crop(p)
            except Exception:
                obj = None
        if obj is None:
            failed += 1
            continue
        if is_cutout:
            cutout_count += 1
        else:
            fallback_count += 1
        out_path = os.path.join(CANDIDATE_DIR, f"{basename}.png")
        obj.save(out_path)
        manifest.append(basename + ".png")
        written += 1

    with open(POINTER_PATH, "w") as f:
        json.dump({"next_idx": i, "manifest": manifest}, f)

    print(f"\nThis run: scanned up to index {i}, {cutout_count} true cutouts, "
          f"{fallback_count} forced-crop fallback, {failed} failed.")
    print(f"Total written so far: {written} / target {target_written}")
    if written >= target_written:
        print(f"Target reached. Candidates in {CANDIDATE_DIR}/ ready for audit -- "
              f"NOT yet added to the live pool.")
    else:
        print("Target not yet reached -- re-run this script again to continue.")


if __name__ == "__main__":
    main()
