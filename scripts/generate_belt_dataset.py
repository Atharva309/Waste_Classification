"""
Full-scale belt-composited training set generator.

Extends the logic already validated in scripts/composite_preview.py (10
images/class preview, approved) to the ENTIRE usable dataset, split by
split, for both Stage 1 (organic vs recyclable) and Stage 2 (6 material
classes).

Per-class treatment (as directed):
  - cardboard / glass / metal / paper / plastic: real cutout.
      * TACO-sourced files (filename starts with "taco_") are already
        polygon-masked with the background flattened to a solid
        (128,128,128) gray -- chroma-keyed out precisely (see
        composite_preview.py's taco_cutout). Used TWICE per image in TRAIN
        only (different random placement each time), ONCE in val/test.
        EXCEPTION: plastic. NOTES.md already documents that mixing TACO
        into Stage 2 previously made plastic worse, specifically because
        TACO's own composition is ~64% plastic -- doubling plastic's TACO
        images would repeat that exact mistake. Plastic's TACO images are
        used ONCE even in train, same as everything else.
      * Non-TACO ("trashnet"/garbage_classification) files: plain white
        studio backgrounds -- border-color threshold cutout (see
        composite_preview.py's trashnet_cutout). Used once everywhere.
      * FALLBACK (see NOTES.md "sure-shot fallback"): if a true cutout
        can't be made (color-threshold segmentation fails), falls back to
        forced_bbox_crop (opaque, real background kept -- same treatment
        as organic/ewaste) instead of discarding the image. This was
        previously a hard failure that silently dropped 40-80% of source
        images per class; now every source image contributes at least one
        training example, either as a clean cutout or a plain crop.
      * Plastic additionally has its TRAIN split capped at
        PLASTIC_TRAIN_CAP (2800, matching roughly where cardboard/glass
        land) via random subsampling of the combined taco+trashnet
        candidate pool -- plastic's raw pool (6280) is 2-4x every other
        material class, and this cascade's live confusion matrix has
        consistently shown plastic as the dominant "attractor" class all
        session. val/test are NOT capped (only training-time imbalance is
        being addressed here, not the evaluation distribution).
  - organic, ewaste: no segmentation attempted -- confirmed by direct
    inspection to have real, varied backgrounds (stock photos, memes,
    product shots, desks -- no clean background to key on). Forced
    bounding-box crop only (opaque rectangle, real background kept inside
    the box), same detector as tight_crop_sprites.py with a center-crop
    fallback. Used once each, no duplication.
  - Stage 1's "recyclable" class: NOT generated from the original R source
    (`organictrashdetection`) at all. Instead reuses Stage 2's own
    composited images (all six material classes pooled, same train/val/
    test partition), relabeled "recyclable" -- no extra generation, no
    extra images on disk, just referenced again from stage1_split.json
    with a different label. This fixes a domain mismatch the original R
    source had: it came from an unrelated dataset with a different visual
    style than what Stage 2 actually classifies, so Stage 1 was learning
    "recyclable" as a different concept than what it hands off downstream.

CRITICAL, non-negotiable: every image is composited from its OWN split only
(train images only ever produce train output, same for val/test). No
crossover. This mirrors the exact leak-free discipline already enforced by
outputs/stage1_split.json / outputs/stage2_split.json -- generating this
dataset must not reintroduce the leakage this project already spent a
session fixing once.

Source images are also deduplicated by resolved file target before
sampling (see composite_preview.py's _dedup_by_real_target) -- cardboard in
particular has ~63% duplicate-content files in its raw listing.

Output:
  data/cascades_data/
    stage1/{train,val,test}/organic/*.png   (recyclable has no folder of its
      own -- its manifest entries point at the stage2/ files below, see
      "Stage 1's recyclable class" above)
    stage2/{train,val,test}/{cardboard,ewaste,glass,metal,paper,plastic}/*.png
    stage1_split.json   -- {"train": [[path,label], ...], "val": [...], "test": [...]}
    stage2_split.json   -- same format

Does not touch: original data/ folders, outputs/*.pt checkpoints, or the
live outputs/stage1_split.json / outputs/stage2_split.json.

Run on your Mac (source images are symlinks that only resolve there):

    cd pytorch_pipeline
    python3 scripts/generate_belt_dataset.py

This processes on the order of tens of thousands of images and will take a
while (expect somewhere in the ballpark of 20-40+ minutes depending on your
Mac) -- run it under caffeinate so the machine doesn't sleep partway
through (see the run command provided alongside this script).
"""

import os
import sys
import json
import random
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from composite_preview import (  # noqa: E402
    taco_cutout, trashnet_cutout, forced_bbox_crop,
    load_belt, make_composite, _dedup_by_real_target,
    filter_known_bad,
)

STAGE1_SPLIT_SRC = "outputs/stage1_split.json"
STAGE2_SPLIT_SRC = "outputs/stage2_split.json"
OUT_ROOT = "data/cascades_data"

TACO_MULTIPLIER_TRAIN = 2   # TACO images used twice in train, once in val/test
SEED = 42

MATERIAL_CUTOUT_CLASSES = ["cardboard", "glass", "metal", "paper", "plastic"]
FORCED_CROP_STAGE2_CLASSES = ["ewaste"]

# Plastic-specific overrides (see module docstring). PLASTIC_TRAIN_CAP
# still holds so plastic can't dominate the other 5 classes. The TACO
# multiplier is now the SAME as every other class (2x), reversed from the
# original 1x (NOTES.md sections 24/51/53). The original reasoning for 1x
# was to stop plastic getting doubly overrepresented on top of an already-
# large TACO+generic-Kaggle pool (TACO alone is naturally ~64% plastic by
# real-world litter composition, section 15). That concern no longer
# applies: as of section 51, plastic's source is HARD LIMITED to only
# TACO+real-TrashNet (the generic Kaggle pool that caused the original
# imbalance AND the contamination is excluded entirely now), so plastic
# went from having the largest raw pool of any class to the smallest
# (1,302 clean train images vs. 2,121-3,170 for the other material
# classes). Doubling TACO usage here is the same augmentation pattern
# already proven safe for cardboard/glass/metal/paper, applied to close
# that gap instead of a data-integrity shortcut -- each doubled use still
# gets a fresh random placement/rotation/scale via make_composite(), not a
# literal duplicate file.
PLASTIC_TACO_MULTIPLIER_TRAIN = 2
PLASTIC_TRAIN_CAP = 2800


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def make_class_dir(stage, split_name, cls):
    d = os.path.join(OUT_ROOT, stage, split_name, cls)
    os.makedirs(d, exist_ok=True)
    return d


def generate_material_class(cls, split_name, paths, belt_im, out_dir, manifest, label_name):
    paths = _dedup_by_real_target(paths)
    taco_paths = [p for p in paths if os.path.basename(p).startswith("taco_")]
    trashnet_paths = [p for p in paths if not os.path.basename(p).startswith("taco_")]

    if cls == "plastic":
        multiplier = PLASTIC_TACO_MULTIPLIER_TRAIN if split_name == "train" else 1
    else:
        multiplier = TACO_MULTIPLIER_TRAIN if split_name == "train" else 1

    if cls == "plastic":
        # HARD EXCLUDE generic Kaggle for plastic across EVERY split, not
        # just train. NOTES.md section 57: this was originally gated to
        # `split_name == "train"` only (section 51), which meant val/test
        # plastic ground truth still drew from the unaudited generic-Kaggle
        # pool -- 61% of val's and 66% of test's plastic images, checked
        # directly against outputs/stage2_split.json after the first
        # removedbadapples cascade run. A test set built from mislabeled
        # ground truth doesn't measure the model on harder-but-correct
        # data, it measures agreement with WRONG answers -- that first run
        # scored plastic recall 0.49 (down from 0.83 pre-cleanup) and every
        # other class's precision dropped too, and this contaminated-test-
        # set confound is the leading suspect, not a real regression from
        # the cleaner training data. Same fix as section 51, just applied
        # to every split now: only taco_ and real trashnet_-prefixed
        # sources (both fully re-reviewed, 100% clean) are used, no
        # fallback, regardless of split.
        real_trashnet = [p for p in trashnet_paths if os.path.basename(p).startswith("trashnet_")]
        excluded_generic = len(trashnet_paths) - len(real_trashnet)
        clean_only = [("taco", p) for p in taco_paths] + [("trashnet", p) for p in real_trashnet]
        random.shuffle(clean_only)
        if split_name == "train":
            # Only train gets the size cap -- val/test should use every
            # clean image available, not be artificially shrunk to match
            # train's cap (that cap exists purely to stop plastic
            # dominating the TRAINING signal, it has nothing to do with
            # eval).
            combined = clean_only[:PLASTIC_TRAIN_CAP]
        else:
            combined = clean_only
        n_taco = sum(1 for k, _ in combined if k == "taco")
        n_real_trashnet = sum(1 for k, _ in combined if k == "trashnet")
        log(f"  plastic ({split_name}) source mix: {n_taco} taco + {n_real_trashnet} real "
            f"trashnet (audited clean only) -- {excluded_generic} generic Kaggle (unaudited) "
            f"images excluded entirely, no fallback (NOTES.md section 51/57)")
        if split_name == "train" and len(clean_only) < PLASTIC_TRAIN_CAP:
            log(f"  WARNING: plastic clean-only pool ({len(clean_only)}) is below "
                f"PLASTIC_TRAIN_CAP ({PLASTIC_TRAIN_CAP}) -- will produce a SMALLER "
                f"plastic train set than the cap rather than risk more contamination. "
                f"Report this number back, don't silently pad it.")
        taco_paths = [p for kind, p in combined if kind == "taco"]
        trashnet_paths = [p for kind, p in combined if kind == "trashnet"]

    written = 0
    cutout_count = 0
    fallback_count = 0
    failed = 0
    idx = 0
    total_source = len(taco_paths) + len(trashnet_paths)
    processed = 0

    def _fallback(p):
        """No true cutout could be made -- fall back to the same forced
        bounding-box crop organic/ewaste already use (opaque, real
        background kept), rather than discarding the image. Guaranteed to
        return something: forced_bbox_crop has its own center-crop
        fallback and never returns None. This is what makes the pipeline a
        "sure shot" -- every source image contributes at least one
        training example, whether as a clean cutout or a plain crop."""
        try:
            return forced_bbox_crop(p)
        except Exception as e:
            print(f"  hard failure (unreadable file) {p}: {e}")
            return None

    for p in taco_paths:
        try:
            obj = taco_cutout(p)
        except Exception as e:
            print(f"  cutout error {p}: {e}")
            obj = None

        has_alpha = True
        if obj is None:
            obj = _fallback(p)
            has_alpha = False
            if obj is None:
                failed += 1
                processed += 1
                continue
            fallback_count += 1
        else:
            cutout_count += 1

        for _ in range(multiplier):
            comp = make_composite(belt_im, obj, has_alpha=has_alpha)
            suffix = "taco" if has_alpha else "taco_fallbackcrop"
            out_path = os.path.join(out_dir, f"{cls}_{suffix}_{idx:06d}.png")
            comp.save(out_path)
            manifest.append([out_path, label_name])
            idx += 1
            written += 1
        processed += 1
        if processed % 500 == 0:
            log(f"  ...{cls} ({split_name}): {processed}/{total_source} source images processed")

    for p in trashnet_paths:
        try:
            obj = trashnet_cutout(p)
        except Exception as e:
            print(f"  cutout error {p}: {e}")
            obj = None

        has_alpha = True
        if obj is None:
            obj = _fallback(p)
            has_alpha = False
            if obj is None:
                failed += 1
                processed += 1
                continue
            fallback_count += 1
        else:
            cutout_count += 1

        comp = make_composite(belt_im, obj, has_alpha=has_alpha)
        suffix = "trashnet" if has_alpha else "trashnet_fallbackcrop"
        out_path = os.path.join(out_dir, f"{cls}_{suffix}_{idx:06d}.png")
        comp.save(out_path)
        manifest.append([out_path, label_name])
        idx += 1
        written += 1
        processed += 1
        if processed % 500 == 0:
            log(f"  ...{cls} ({split_name}): {processed}/{total_source} source images processed")

    log(f"  {cls} ({split_name}): {cutout_count} true cutouts, {fallback_count} "
        f"fallback crops, {failed} hard failures")
    return written, failed


def generate_forced_crop_class(cls, split_name, paths, belt_im, out_dir, manifest, label_name):
    paths = _dedup_by_real_target(paths)
    written = 0
    failed = 0
    total_source = len(paths)
    for idx, p in enumerate(paths):
        try:
            crop = forced_bbox_crop(p)
        except Exception as e:
            print(f"  skipping {p}: {e}")
            failed += 1
            continue
        comp = make_composite(belt_im, crop, has_alpha=False)
        out_path = os.path.join(out_dir, f"{cls}_{idx:06d}.png")
        comp.save(out_path)
        manifest.append([out_path, label_name])
        written += 1
        if (idx + 1) % 500 == 0:
            log(f"  ...{cls} ({split_name}): {idx+1}/{total_source} source images processed")
    return written, failed


def _real_target(p):
    if os.path.islink(p):
        try:
            return os.path.realpath(p)
        except Exception:
            return p
    return p


def verify_source_split_leak_free(s1, s2):
    """This dataset's leak-freedom is entirely inherited from
    outputs/stage1_split.json / stage2_split.json -- generation maps each
    source split 1:1 to the matching output split with no cross-wiring
    (train sources only ever produce train output, etc.), so verifying the
    SOURCE split has no image whose real file target appears in more than
    one split is equivalent to verifying the generated dataset is leak-free
    too. Checked here, before spending 20-40+ minutes generating from it,
    rather than assumed."""
    problems = []
    for name, split_dict in [("stage1", s1), ("stage2", s2)]:
        target_to_splits = {}
        for split_name in ["train", "val", "test"]:
            for p, _label in split_dict[split_name]:
                t = _real_target(p)
                target_to_splits.setdefault(t, set()).add(split_name)
        crossing = {t: s for t, s in target_to_splits.items() if len(s) > 1}
        if crossing:
            problems.append((name, crossing))

    if problems:
        print("LEAK CHECK: FAILED")
        for name, crossing in problems:
            print(f"  {name}: {len(crossing)} images appear in more than one split!")
            for t, splits in list(crossing.items())[:5]:
                print(f"    {t} -> {splits}")
        print("Stopping before generating anything from a leaking split.")
        sys.exit(1)
    else:
        print("LEAK CHECK: PASSED -- no source image's real file target "
              "appears in more than one split, for either stage.")


def main():
    random.seed(SEED)

    with open(STAGE1_SPLIT_SRC) as f:
        s1 = json.load(f)
    with open(STAGE2_SPLIT_SRC) as f:
        s2 = json.load(f)

    # Drop every source image confirmed by manual audit to be mislabeled
    # (NOTES.md sections 38-43) BEFORE anything else -- same blocklist
    # generate_yolo_cutouts_v2.py uses, applied here too since both YOLO's
    # cutout pool and the cascade's belt-composite data draw from these same
    # two split files. Filtering (removing entries) can't introduce a leak,
    # so this runs before the leak check, not after.
    for split_name in ["train", "val", "test"]:
        s1[split_name] = filter_known_bad(s1[split_name])
        s2[split_name] = filter_known_bad(s2[split_name])

    verify_source_split_leak_free(s1, s2)

    belt_im = load_belt()
    os.makedirs(OUT_ROOT, exist_ok=True)

    # ---------------- Stage 2 ----------------
    stage2_manifest = {"train": [], "val": [], "test": []}
    stage2_summary = {}

    material_by_split = {}
    for split_name in ["train", "val", "test"]:
        by_class = {}
        for p, label in s2[split_name]:
            by_class.setdefault(label, []).append(p)
        material_by_split[split_name] = by_class

    for split_name in ["train", "val", "test"]:
        for cls in MATERIAL_CUTOUT_CLASSES:
            out_dir = make_class_dir("stage2", split_name, cls)
            paths = material_by_split[split_name].get(cls, [])
            written, failed = generate_material_class(
                cls, split_name, paths, belt_im, out_dir, stage2_manifest[split_name], cls
            )
            stage2_summary.setdefault(cls, {})[split_name] = written
            log(f"stage2/{split_name}/{cls}: wrote {written}, failed/skipped {failed} "
                f"(source images: {len(paths)})")

        for cls in FORCED_CROP_STAGE2_CLASSES:
            out_dir = make_class_dir("stage2", split_name, cls)
            paths = material_by_split[split_name].get(cls, [])
            written, failed = generate_forced_crop_class(
                cls, split_name, paths, belt_im, out_dir, stage2_manifest[split_name], cls
            )
            stage2_summary.setdefault(cls, {})[split_name] = written
            log(f"stage2/{split_name}/{cls}: wrote {written} (forced crop), failed/skipped {failed} "
                f"(source images: {len(paths)})")

    with open(os.path.join(OUT_ROOT, "stage2_split.json"), "w") as f:
        json.dump(stage2_manifest, f)
    log("Wrote data/cascades_data/stage2_split.json")

    # ---------------- Stage 1 ----------------
    stage1_manifest = {"train": [], "val": [], "test": []}
    stage1_summary = {}

    # organic: generated fresh from stage1_split.json's "O" entries (forced crop).
    for split_name in ["train", "val", "test"]:
        by_label = {"O": []}
        for p, label in s1[split_name]:
            if label == "O":
                by_label["O"].append(p)

        out_dir = make_class_dir("stage1", split_name, "organic")
        paths = by_label["O"]
        written, failed = generate_forced_crop_class(
            "organic", split_name, paths, belt_im, out_dir, stage1_manifest[split_name], "organic"
        )
        stage1_summary.setdefault("organic", {})[split_name] = written
        log(f"stage1/{split_name}/organic: wrote {written} (forced crop), failed/skipped {failed} "
            f"(source images: {len(paths)})")

    # recyclable: NOT generated from the original R source -- reuses Stage 2's
    # own composited images (all 6 material classes pooled), same split
    # partition, just relabeled. No new files written; same PNGs referenced
    # again from stage1_split.json with a different label.
    for split_name in ["train", "val", "test"]:
        recyclable_entries = [[path, "recyclable"] for path, _ in stage2_manifest[split_name]]
        stage1_manifest[split_name].extend(recyclable_entries)
        stage1_summary.setdefault("recyclable", {})[split_name] = len(recyclable_entries)
        log(f"stage1/{split_name}/recyclable: reused {len(recyclable_entries)} Stage 2 composites "
            f"(no new files written)")

    with open(os.path.join(OUT_ROOT, "stage1_split.json"), "w") as f:
        json.dump(stage1_manifest, f)
    log("Wrote data/cascades_data/stage1_split.json")

    # ---------------- summary ----------------
    print("\n" + "=" * 60)
    print("STAGE 2 (material) summary")
    total_s2 = 0
    for cls, per_split in stage2_summary.items():
        row_total = sum(per_split.values())
        total_s2 += row_total
        print(f"  {cls:10s}: train={per_split.get('train',0):6d}  "
              f"val={per_split.get('val',0):5d}  test={per_split.get('test',0):5d}  "
              f"total={row_total}")
    print(f"  STAGE 2 TOTAL: {total_s2}")

    print("\nSTAGE 1 (organic vs recyclable) summary")
    total_s1 = 0
    for cls, per_split in stage1_summary.items():
        row_total = sum(per_split.values())
        total_s1 += row_total
        print(f"  {cls:10s}: train={per_split.get('train',0):6d}  "
              f"val={per_split.get('val',0):5d}  test={per_split.get('test',0):5d}  "
              f"total={row_total}")
    print(f"  STAGE 1 TOTAL: {total_s1}")
    print(f"\nGRAND TOTAL (both stages): {total_s1 + total_s2}")
    print("=" * 60)


if __name__ == "__main__":
    main()
