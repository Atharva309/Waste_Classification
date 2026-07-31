"""
Generate real object cutouts for YOLO's multi-object training frames using
THIS SESSION's own classical cutout pipeline (composite_preview.py's
taco_cutout / trashnet_cutout / forced_bbox_crop, with the sure-shot
fallback pattern) instead of FastSAM.

Why this replaces generate_yolo_cutouts.py: FastSAM's masks were already
documented (NOTES.md section 7) as inconsistent -- good on many images,
missing the object entirely on others, leaking outside the real boundary
on others -- and its per-image runtime was never precisely measured,
making it a real risk for an unattended overnight run. This session's own
classical cutout functions were validated extensively today building
data/cascades_data (leak-free, sure-shot fallback so nothing is ever
silently dropped, resolution-preserving) and are dramatically faster and
more predictable -- the full belt-composite dataset (46,124 images across
both cascade stages) took roughly 10-40 minutes total depending on disk
cache state.

Output contract matches generate_yolo_cutouts.py exactly, so
generate_yolo_dataset.py runs unchanged against this script's output:
    data/yolo_data/cutouts/<class>/<basename>.png
RGBA where a true cutout succeeds, RGB (opaque) where it falls back to a
forced bounding-box crop. Organic and ewaste always use the forced-crop
path -- same as data/cascades_data -- since they have no clean
background to chroma-key against (real, varied scenes: stock photos,
desks, market shots). This means organic/ewaste look like opaque "photo
card" rectangles in synthetic training frames rather than true silhouettes
-- an accepted tradeoff already made for the cascade's own training data,
not a new one introduced here. The GROUND TRUTH BOX itself is exactly
correct either way (it's the crop's real extent, regardless of whether the
crop is a true silhouette or a rectangle) -- only the visual realism of
the synthetic frame differs for these two classes.

Sourced from the ORIGINAL train-split source images
(outputs/stage1_split.json / outputs/stage2_split.json) -- the same source
of truth generate_yolo_cutouts.py already used and that
data/cascades_data was built from. Deliberately NOT sourced from:
  - data/cascades_data's own PNGs -- those already have a belt-texture
    background patch baked in around each object; scattering those into
    ANOTHER synthetic frame would paste a visibly mismatched little
    rectangle of belt texture around each object, not a clean cutout.
  - outputs/sprites_cutout_v1 -- that pool is built from the CASCADE's
    TEST split specifically so the live webapp's held-out evaluation stays
    meaningful. Training YOLO on those same images would silently
    invalidate any future live A/B between YOLO and the Cascade on the
    same served sprite pool.

Run on your Mac:

    cd pytorch_pipeline
    python3 scripts/generate_yolo_cutouts_v2.py
"""

import os
import sys
import json
import random
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from composite_preview import (  # noqa: E402
    taco_cutout, trashnet_cutout, forced_bbox_crop, _dedup_by_real_target,
    filter_known_bad,
)

STAGE1_SPLIT_PATH = "outputs/stage1_split.json"
STAGE2_SPLIT_PATH = "outputs/stage2_split.json"
# Lives under data/ (not outputs/) alongside cascades_data -- this is a
# real dataset, not a checkpoint/report artifact, per the same convention
# already used for the cascade's belt-composite data.
OUT_DIR = "data/yolo_data/cutouts"

FORCED_CROP_CLASSES = {"organic", "ewaste"}

TARGETS = {
    "cardboard": 1500, "ewaste": 1500, "glass": 1500, "metal": 1500,
    "paper": 1200, "plastic": 1500, "organic": 1500,
}

SEED = 42


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_train_paths():
    with open(STAGE1_SPLIT_PATH) as f:
        s1 = json.load(f)
    with open(STAGE2_SPLIT_PATH) as f:
        s2 = json.load(f)

    # Drop every source image confirmed by manual audit to be mislabeled
    # (NOTES.md sections 38-43) BEFORE dedup/shuffle -- otherwise a from-
    # scratch rebuild just re-draws the same contaminated images the audit
    # already removed.
    s1_train = filter_known_bad(s1["train"])
    s2_train = filter_known_bad(s2["train"])

    paths = {"organic": [p for p, label in s1_train if label == "O"]}
    material_train = {}
    for p, label in s2_train:
        material_train.setdefault(label, []).append(p)
    paths.update(material_train)
    return paths


def process_class(cls, paths, out_dir):
    paths = _dedup_by_real_target(paths)
    random.shuffle(paths)
    target = TARGETS.get(cls, 1500)

    # Plastic-specific: HARD EXCLUDE the generic Kaggle-sourced files (no
    # "taco_"/"trashnet_" prefix). NOTES.md section 51 -- the previous
    # version of this block ("priority ordering") still let the fill loop
    # fall back into this pool once TACO+TrashNet ran out, on the theory
    # that a blocklist of previously-found-bad images made the fallback
    # safe. It didn't: the blocklist (data/known_bad_source_images.json)
    # only covers images that happened to get selected and then manually
    # audited in a PAST run -- as of section 51 that's 415 of the 5,024
    # images in data/trash_classification_data-main/garbage_classification/
    # plastic/ (~8%). The other ~92% (4,618 images) has never been looked
    # at by anyone; a fresh rebuild drawing from it is a blind draw from a
    # pool already proven (by finding 425+ bad images in a small audited
    # subset of it) to have a real contamination rate. That's exactly what
    # put metal back in the live plastic pool after the section 41-49
    # cleanup + retrain. TACO and TrashNet are the only two sources that
    # have been FULLY re-reviewed end to end and come back 100% clean, so
    # they are now the ONLY allowed plastic source -- no fallback, no
    # exceptions, regardless of whether that means falling short of
    # `target`.
    if cls == "plastic":
        clean_only = [p for p in paths if os.path.basename(p).startswith(("taco_", "trashnet_"))]
        excluded_generic = len(paths) - len(clean_only)
        random.shuffle(clean_only)
        paths = clean_only
        log(f"  plastic source: {len(clean_only)} TACO/TrashNet (audited clean) only -- "
            f"{excluded_generic} generic Kaggle (unaudited) images excluded entirely, "
            f"no fallback (NOTES.md section 51)")
        if len(clean_only) < target:
            log(f"  WARNING: plastic clean-only pool ({len(clean_only)}) is below "
                f"target ({target}) -- will produce a SMALLER plastic set than other "
                f"classes rather than risk more contamination. Report this number "
                f"back, don't silently pad it.")

    written, cutout_count, fallback_count, failed = 0, 0, 0, 0
    processed = 0
    for p in paths:
        if written >= target:
            break
        processed += 1
        basename = os.path.splitext(os.path.basename(p))[0]

        if cls in FORCED_CROP_CLASSES:
            try:
                obj = forced_bbox_crop(p)
            except Exception as e:
                print(f"  hard failure (unreadable file) {p}: {e}")
                obj = None
            is_cutout = False
        else:
            is_taco = os.path.basename(p).startswith("taco_")
            try:
                obj = taco_cutout(p) if is_taco else trashnet_cutout(p)
            except Exception as e:
                print(f"  cutout error {p}: {e}")
                obj = None
            is_cutout = obj is not None
            if obj is None:
                try:
                    obj = forced_bbox_crop(p)
                except Exception as e:
                    print(f"  hard failure (unreadable file) {p}: {e}")
                    obj = None

        if obj is None:
            failed += 1
            continue

        if is_cutout:
            cutout_count += 1
        else:
            fallback_count += 1

        out_path = os.path.join(out_dir, f"{basename}.png")
        obj.save(out_path)
        written += 1

        if processed % 500 == 0:
            log(f"  ...{cls}: {processed} source images scanned, {written}/{target} written")

    log(f"  {cls}: {written} written ({cutout_count} true cutouts, {fallback_count} "
        f"fallback/forced crops), {failed} failed, target {target}, "
        f"{len(paths)} source images available")
    return written


def main():
    random.seed(SEED)
    os.makedirs(OUT_DIR, exist_ok=True)
    train_paths = load_train_paths()

    classes = ["organic", "cardboard", "ewaste", "glass", "metal", "paper", "plastic"]
    summary = {}
    for cls in classes:
        out_dir = os.path.join(OUT_DIR, cls)
        os.makedirs(out_dir, exist_ok=True)
        paths = train_paths.get(cls, [])
        log(f"Processing {cls} ({len(paths)} available train-split source images)...")
        summary[cls] = process_class(cls, paths, out_dir)

    print("\n--- Summary ---")
    for k, v in summary.items():
        print(f"{k}: {v}")
    print(f"Total: {sum(summary.values())}")


if __name__ == "__main__":
    main()
