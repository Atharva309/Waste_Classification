"""
Regenerate the LIVE webapp sprite pool so it's built from the exact same
content as the offline belt-composited test set (data/cascades_data's
test split) -- same source images, same cutout/crop logic, same
train/test boundary. Currently the live pool (outputs/sprites/, built by
rebuild_sprites.py + tight_crop_sprites.py) is full/tight-cropped photos,
not cutouts, and isn't guaranteed to trace to the same underlying test
images as the new offline test set.

IMPORTANT DIFFERENCE from generate_belt_dataset.py: this does NOT bake the
belt background into the sprite file. webapp/templates/index.html already
renders a moving belt background on the canvas and pastes each sprite PNG
on top with alpha blending (`ctx.drawImage(beltImg,...)` then
`ctx.drawImage(s.img,...)`), so baking a second, static belt patch into the
sprite would double-composite -- you'd see a rectangle of (slightly
mismatched, non-moving) belt texture sitting on top of the real moving
belt, which is exactly the "looks like a photo card" problem already fixed
once. Instead:
  - cardboard/glass/metal/paper/plastic: saved as TRUE transparent cutouts
    (RGBA, alpha = the object mask) -- the live canvas's own alpha-blended
    paste reproduces the same "object on belt" arrangement the model will
    have trained on, just done live instead of pre-baked.
  - organic/ewaste: forced bounding-box crop, opaque (matches how they're
    trained -- no segmentation attempted for these, real background kept).

Sourced ONLY from the test split (same test-split entries that feed
data/cascades_data/stage{1,2}_split.json's "test" list) -- so the live
webapp's ground-truth pool and the offline test-set numbers are measuring
the literal same held-out images, not just "the same kind of processing."

DOES NOT touch outputs/sprites/ (the currently-live pool) -- writes to
outputs/sprites_cutout_v1/ instead. Do not point webapp/app.py at this
folder until the new *_BELTCOMPOSITE.pt checkpoints are trained and loaded
-- the live model right now is still trained on full-background images, so
switching the live pool to cutouts before swapping checkpoints would feed
it out-of-domain input and tank live accuracy immediately (this is exactly
the domain mismatch this whole session's investigation was about).

Run on your Mac:

    cd pytorch_pipeline
    python3 scripts/generate_live_sprites_v2.py
"""

import os
import sys
import json
import shutil

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from composite_preview import taco_cutout, trashnet_cutout, forced_bbox_crop, filter_known_bad  # noqa: E402

STAGE1_SPLIT_SRC = "outputs/stage1_split.json"
STAGE2_SPLIT_SRC = "outputs/stage2_split.json"
OUT_ROOT = "outputs/sprites_cutout_v1"

MATERIAL_CUTOUT_CLASSES = ["cardboard", "glass", "metal", "paper", "plastic"]


def generate_material_sprites(cls, paths, out_dir):
    """Matches generate_belt_dataset.py's fallback behavior: if a true
    cutout can't be made, fall back to forced_bbox_crop (opaque, real
    background kept) rather than dropping the image -- the live pool
    should have the same cutout/fallback-crop mix as what the model was
    actually trained on, not a purer "cutouts only" pool that doesn't
    match training."""
    written, failed, cutout_count, fallback_count = 0, 0, 0, 0
    for i, p in enumerate(paths):
        is_taco = os.path.basename(p).startswith("taco_")
        try:
            obj = taco_cutout(p) if is_taco else trashnet_cutout(p)
        except Exception as e:
            print(f"  cutout error {p}: {e}")
            obj = None

        if obj is not None:
            cutout_count += 1
            obj.save(os.path.join(out_dir, f"{cls}_{i:04d}.png"))
            written += 1
            continue

        try:
            crop = forced_bbox_crop(p)
        except Exception as e:
            print(f"  hard failure (unreadable file) {p}: {e}")
            crop = None
        if crop is None:
            failed += 1
            continue
        fallback_count += 1
        crop.save(os.path.join(out_dir, f"{cls}_fallbackcrop_{i:04d}.png"))
        written += 1

    print(f"  {cls}: {cutout_count} true cutouts, {fallback_count} fallback crops")
    return written, failed


def generate_forced_crop_sprites(cls, paths, out_dir):
    written, failed = 0, 0
    for i, p in enumerate(paths):
        try:
            crop = forced_bbox_crop(p)
        except Exception as e:
            print(f"  skipping {p}: {e}")
            failed += 1
            continue
        crop.save(os.path.join(out_dir, f"{cls}_{i:04d}.png"))
        written += 1
    return written, failed


def main():
    with open(STAGE1_SPLIT_SRC) as f:
        s1 = json.load(f)
    with open(STAGE2_SPLIT_SRC) as f:
        s2 = json.load(f)

    # NOTES.md section 61: this script never applied the known-bad
    # blocklist OR the plastic hard-exclude at all -- it read straight
    # from outputs/stage2_split.json's raw test list. The live webapp's
    # "ground truth" plastic sprites (outputs/sprites_cutout_v1/plastic)
    # were generated Jul 26, before ANY of tonight's contamination fixes,
    # and were about to be tested live against the brand-new clean
    # checkpoint -- which would have graded the new model against the
    # exact same kind of mislabeled ground truth that made Stage 2's
    # first retrain look like a regression (section 57), just in the live
    # serving path instead of the offline test set.
    s1["test"] = filter_known_bad(s1["test"])
    s2["test"] = filter_known_bad(s2["test"])

    os.makedirs(OUT_ROOT, exist_ok=True)

    material_test = {}
    for p, label in s2["test"]:
        material_test.setdefault(label, []).append(p)

    # Same hard-exclude as generate_belt_dataset.py/generate_yolo_cutouts_v2.py
    # (sections 51/57): plastic ONLY from taco_/trashnet_-prefixed sources,
    # no generic-Kaggle fallback, for the same reason -- that source is the
    # one place contamination was ever found, and the blocklist alone only
    # covers what happened to get manually audited in a past run.
    if "plastic" in material_test:
        raw = material_test["plastic"]
        clean = [p for p in raw if os.path.basename(p).startswith(("taco_", "trashnet_"))]
        excluded = len(raw) - len(clean)
        material_test["plastic"] = clean
        print(f"plastic sprite source: {len(clean)} taco/trashnet (audited clean) only -- "
              f"{excluded} generic Kaggle (unaudited) excluded entirely (NOTES.md section 61)\n")

    print("Generating live sprite pool from the TEST split only "
          "(same images as data/cascades_data's test set)...\n")

    for cls in MATERIAL_CUTOUT_CLASSES:
        out_dir = os.path.join(OUT_ROOT, cls)
        # Wipe first, not just overwrite-by-index (NOTES.md section 58's
        # lesson applied here too): if the filtered/clean path list is
        # SHORTER than what a previous run wrote (true for plastic here,
        # 468 -> ~159), index-based filenames would leave old, higher-
        # index, still-contaminated sprites sitting alongside the new
        # ones -- and app.py loads this pool by listing the directory,
        # not from a manifest, so stale files would still get served live.
        if os.path.isdir(out_dir):
            shutil.rmtree(out_dir)
        os.makedirs(out_dir, exist_ok=True)
        paths = material_test.get(cls, [])
        written, failed = generate_material_sprites(cls, paths, out_dir)
        print(f"{cls}: {written} true cutouts written, {failed} skipped (source: {len(paths)})")

    out_dir = os.path.join(OUT_ROOT, "ewaste")
    if os.path.isdir(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    ewaste_paths = material_test.get("ewaste", [])
    written, failed = generate_forced_crop_sprites("ewaste", ewaste_paths, out_dir)
    print(f"ewaste: {written} forced-crop sprites written, {failed} skipped (source: {len(ewaste_paths)})")

    organic_paths = [p for p, label in s1["test"] if label == "O"]
    out_dir = os.path.join(OUT_ROOT, "organic")
    if os.path.isdir(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    written, failed = generate_forced_crop_sprites("organic", organic_paths, out_dir)
    print(f"organic: {written} forced-crop sprites written, {failed} skipped (source: {len(organic_paths)})")

    print(f"\nWrote to {OUT_ROOT}/ -- outputs/sprites/ (the live pool) is untouched.")
    print("Do not point webapp/app.py at this folder until the *_BELTCOMPOSITE "
          "checkpoints are trained and loaded -- see module docstring.")


if __name__ == "__main__":
    main()
