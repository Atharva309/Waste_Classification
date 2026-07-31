"""
Prototype: composite cutout objects onto the real belt texture
(webapp/static/belt.jpg), sourced ONLY from the TRAIN split, so we can look
at samples before committing to a full retrain-on-belt-background pipeline.

See NOTES.md ("Belt-background retrain prototype") for the reasoning: the
model currently trains on plain-background photos, so a true transparent
cutout at test time is out-of-domain and hurts accuracy (already observed
live with the old SAM2 cutouts). The fix under test here is the other
direction -- make TRAINING images look like the belt, instead of making test
images look like training data.

Per-class source handling (as directed):
  - cardboard / glass / metal / paper / plastic: two source types exist in
    data/stage2_material/<class>/.
      * TACO-derived files (filename starts with "taco_") are ALREADY
        polygon-masked -- phase6_sprites.py cut them out using the real COCO
        segmentation polygon, then whatever produced the training-folder
        .jpg version flattened the transparent background to a solid
        (128,128,128) gray fill (confirmed by direct pixel sampling: >100k
        pixels exactly [128,128,128], compression noise elsewhere). That
        means a precise cutout falls right out of a simple chroma-key on
        that exact gray, no re-segmentation needed.
      * Non-TACO files are plain white-background product photos (the
        "trashnet"/garbage_classification set) -- cut out with the same
        border-color-threshold approach as tight_crop_sprites.py, extended
        from a bounding box to a full alpha mask.
    TACO images are sampled and used twice each (different random placement
    per use); trashnet images once each, to fill 10 composites/class.
  - organic / ewaste: backgrounds are real, varied scenes (wood desks,
    gardens, market photos -- see NOTES.md) with no clean background color
    to key on. No segmentation is attempted for these. Instead: a FORCED
    bounding-box crop (no alpha, keeps the image's own real background
    inside the box) -- same detector as tight_crop_sprites.py, but with a
    fallback to a fixed center-crop instead of "leave the image untouched"
    if the detector can't find a confident box. Pasted onto the belt as an
    opaque rectangle, not a cutout.

CRITICAL: only ever samples from split["train"] -- never test/val -- since
this is prototyping data for a future retrain, and mixing test images into
training data is exactly the leakage bug this session already fixed once
for the live sprite pool (see NOTES.md section 23).

Output: outputs/composite_preview/<class>/preview_NN.png -- 10 per class,
70 total. Preview only; does not touch the live sprite pool or any
checkpoint.

Run on your Mac (organic/ewaste/trashnet source images are symlinks that
only resolve there):

    cd pytorch_pipeline
    python3 scripts/composite_preview.py
"""

import os
import json
import random

import numpy as np
from PIL import Image
import cv2

STAGE1_SPLIT = "outputs/stage1_split.json"
STAGE2_SPLIT = "outputs/stage2_split.json"
BELT_IMG_PATH = "webapp/static/belt.jpg"
OUT_DIR = "outputs/composite_preview"

N_PER_CLASS = 10
CANVAS_FLOOR = 200    # minimum composite size -- purely to avoid a degenerate
                      # tiny output for an unusually small source crop; does
                      # NOT upsample the object itself (see make_composite)
MAX_OBJ_WIDTH = 480   # objects wider than this get downsized; comfortably
                      # above every real source crop width sampled this
                      # session (largest seen: 405x477), so this only clips
                      # rare outliers, not typical images
BORDER_FRAC = 0.05
COLOR_DIST_THRESHOLD = 32
TACO_GRAY = np.array([128, 128, 128], dtype=np.float32)
TACO_DIST_THRESHOLD = 18   # tighter, since the fill is an exact synthetic color
MARGIN_FRAC = 0.06

SEED = 123
random.seed(SEED)
np.random.seed(SEED)


# ---------- split loading ----------

KNOWN_BAD_SOURCE_IMAGES_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "known_bad_source_images.json",
)


def load_known_bad_basenames():
    """Basenames (no extension) of source images confirmed, by direct visual
    audit, to be mislabeled -- almost entirely metal cans/lids/hardware and a
    few glass/ceramic items mislabeled "plastic" in the merged Kaggle
    garbage_classification source data (NOTES.md sections 38-43). Diagnosed
    as concentrated in the "plastic" class specifically -- a 48-image random
    spot-check of metal/glass/cardboard/paper/ewaste's raw source folders
    found no equivalent contamination in any of them (NOTES.md section 44),
    so this blocklist is deliberately not applied as a blanket filter across
    all classes, just kept general enough to extend if that changes.

    Returns an empty set (not an error) if the file doesn't exist yet, so
    scripts that call this stay runnable before the file is first created."""
    if not os.path.exists(KNOWN_BAD_SOURCE_IMAGES_PATH):
        return set()
    with open(KNOWN_BAD_SOURCE_IMAGES_PATH) as f:
        return set(json.load(f))


def filter_known_bad(paths_and_labels):
    """Drop any (path, label) pair whose source basename is in the
    known-bad blocklist. Call this right after loading stage2_split.json
    (or stage1_split.json) and BEFORE dedup/shuffle, in every script that
    builds a training pool from it -- this is what stops a full from-
    scratch rebuild from silently re-selecting images already confirmed
    contaminated by manual audit."""
    bad = load_known_bad_basenames()
    if not bad:
        return paths_and_labels
    kept = [(p, label) for p, label in paths_and_labels
            if os.path.splitext(os.path.basename(p))[0] not in bad]
    dropped = len(paths_and_labels) - len(kept)
    if dropped:
        print(f"  [known-bad filter] dropped {dropped} blocklisted source images")
    return kept


def _dedup_by_real_target(paths):
    """The train split deliberately keeps every copy of a duplicate/near-
    duplicate cluster (see src/splits.py -- correct for training, since
    repeated signal there is harmless), but for sampling a *visually
    varied* preview set we want distinct underlying photos. Cardboard in
    particular has heavy duplication: 5714 of 9120 source files are
    re-uploaded copies of the same images (confirmed via symlink target).
    Resolve each path to its real target (or itself, if not a symlink) and
    keep only the first path seen per unique target."""
    seen_targets = set()
    deduped = []
    for p in paths:
        target = os.path.realpath(p) if os.path.islink(p) else p
        if target in seen_targets:
            continue
        seen_targets.add(target)
        deduped.append(p)
    return deduped


def load_train_pools():
    with open(STAGE1_SPLIT) as f:
        s1 = json.load(f)
    with open(STAGE2_SPLIT) as f:
        s2 = json.load(f)

    pools = {"organic": _dedup_by_real_target([p for p, label in s1["train"] if label == "O"])}

    material_train = {}
    for p, label in s2["train"]:
        material_train.setdefault(label, []).append(p)

    for cls, paths in material_train.items():
        paths = _dedup_by_real_target(paths)
        taco = [p for p in paths if os.path.basename(p).startswith("taco_")]
        trashnet = [p for p in paths if not os.path.basename(p).startswith("taco_")]
        pools[cls] = {"taco": taco, "trashnet": trashnet}

    return pools


# ---------- cutout / crop methods ----------

def estimate_border_color(arr):
    """Sample the full border ring (top/bottom rows, left/right columns).

    NOTE: a corner-only variant was tried and reverted (see NOTES.md
    "corner-based fix didn't hold up against real data") -- a clean
    synthetic test suggested corners would be more robust to objects
    touching the middle of an edge, but against the real dataset it made
    every class's yield WORSE (e.g. cardboard 1173 -> 1155, paper 502 ->
    374, worse than even the original unfixed version). Likely cause:
    real product photography has vignetting/shadows concentrated in
    corners specifically, and a much smaller sample (4 small corner
    patches vs the full perimeter) is more sensitive to that and to
    watermarks landing in-sample. Reverted to the full border ring, which
    empirically outperformed it. The relaxed box_frac threshold (0.995,
    see trashnet_cutout) is the fix that's actually working.
    """
    h, w = arr.shape[:2]
    bh = max(1, int(h * BORDER_FRAC))
    bw = max(1, int(w * BORDER_FRAC))
    border = np.concatenate([
        arr[:bh, :, :].reshape(-1, 3),
        arr[-bh:, :, :].reshape(-1, 3),
        arr[:, :bw, :].reshape(-1, 3),
        arr[:, -bw:, :].reshape(-1, 3),
    ], axis=0)
    return np.median(border, axis=0)


def clean_mask(mask):
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    return mask


def bbox_from_mask(mask, margin_frac=MARGIN_FRAC):
    ys, xs = np.where(mask > 0)
    if len(ys) == 0:
        return None
    h, w = mask.shape[:2]
    x1, x2 = xs.min(), xs.max()
    y1, y2 = ys.min(), ys.max()
    bw_, bh_ = (x2 - x1), (y2 - y1)
    mx, my = int(bw_ * margin_frac), int(bh_ * margin_frac)
    x1 = max(0, x1 - mx)
    y1 = max(0, y1 - my)
    x2 = min(w, x2 + mx)
    y2 = min(h, y2 + my)
    return x1, y1, x2, y2


def taco_cutout(path):
    """Chroma-key the flat (128,128,128) fill -> RGBA cutout."""
    im = Image.open(path).convert("RGB")
    arr = np.array(im)
    dist = np.linalg.norm(arr.astype(np.float32) - TACO_GRAY, axis=2)
    fg_mask = (dist > TACO_DIST_THRESHOLD).astype(np.uint8) * 255
    fg_mask = clean_mask(fg_mask)

    box = bbox_from_mask(fg_mask)
    if box is None:
        return None
    x1, y1, x2, y2 = box

    rgba = np.dstack([arr, fg_mask])
    crop = rgba[y1:y2, x1:x2]
    return Image.fromarray(crop, mode="RGBA")


def trashnet_cutout(path):
    """Border-color threshold -> RGBA cutout (white-background product photos).

    NOTE: box_frac upper bound was originally 0.96, intended to catch cases
    where the border-color estimate got poisoned (e.g. object touching the
    border) and the whole frame gets flagged as foreground. In practice,
    against the real garbage_classification dataset, this rejected 40-80%
    of otherwise-good images (see NOTES.md) -- many of these product photos
    legitimately fill nearly the whole frame with just a thin border, and
    0.96 was catching well-composed shots, not failures. Relaxed to 0.995 --
    only rejects near-total failures (mask covers essentially the entire
    frame with no margin at all), not legitimately tight compositions."""
    im = Image.open(path).convert("RGB")
    arr = np.array(im)
    bg_color = estimate_border_color(arr)
    dist = np.linalg.norm(arr.astype(np.float32) - bg_color.astype(np.float32), axis=2)
    fg_mask = (dist > COLOR_DIST_THRESHOLD).astype(np.uint8) * 255
    fg_mask = clean_mask(fg_mask)

    box = bbox_from_mask(fg_mask)
    if box is None:
        return None
    x1, y1, x2, y2 = box

    h, w = arr.shape[:2]
    box_frac = ((x2 - x1) * (y2 - y1)) / float(h * w)
    if box_frac > 0.995 or box_frac < 0.02:
        return None  # thresholding failed to isolate a subject cleanly

    rgba = np.dstack([arr, fg_mask])
    crop = rgba[y1:y2, x1:x2]
    return Image.fromarray(crop, mode="RGBA")


def forced_bbox_crop(path):
    """No alpha. Tight-ish rectangular crop, real background kept, with a
    fixed center-crop fallback so this never just returns the full image."""
    im = Image.open(path).convert("RGB")
    arr = np.array(im)
    bg_color = estimate_border_color(arr)
    dist = np.linalg.norm(arr.astype(np.float32) - bg_color.astype(np.float32), axis=2)
    fg_mask = (dist > COLOR_DIST_THRESHOLD).astype(np.uint8) * 255
    fg_mask = clean_mask(fg_mask)

    box = bbox_from_mask(fg_mask)
    h, w = arr.shape[:2]
    use_fallback = box is None
    if box is not None:
        x1, y1, x2, y2 = box
        box_frac = ((x2 - x1) * (y2 - y1)) / float(h * w)
        if box_frac > 0.94 or box_frac < 0.05:
            use_fallback = True

    if use_fallback:
        # fixed center crop: keep the central 75% of the frame
        cx1, cy1 = int(w * 0.125), int(h * 0.125)
        cx2, cy2 = int(w * 0.875), int(h * 0.875)
        crop = im.crop((cx1, cy1, cx2, cy2))
    else:
        crop = im.crop((x1, y1, x2, y2))

    return crop.convert("RGB")


# ---------- compositing ----------

def load_belt():
    return Image.open(BELT_IMG_PATH).convert("RGB")


def random_belt_patch(belt_im, size):
    w, h = belt_im.size
    size = min(size, w, h)  # never ask for a patch bigger than the source texture
    x = random.randint(0, max(0, w - size))
    y = random.randint(0, max(0, h - size))
    patch = belt_im.crop((x, y, x + size, y + size))
    if patch.size != (size, size):
        patch = patch.resize((size, size))
    return patch.copy()


def paste_scaled_rotated(canvas, obj_im, has_alpha):
    # NOTE: this used to resize obj_im DOWN to fit a small fixed canvas
    # (target_w = canvas.width * scale, with canvas fixed at 260px) --
    # meaning every object was capped to ~150-220px wide no matter its real
    # resolution (some source crops run 400px+), and then the training
    # pipeline resized AGAIN down to the network's 224px input. Two lossy
    # resizes stacked, throwing away real detail (fine metal scratches,
    # glass transparency, small text) before the network ever saw it. Now
    # the canvas itself is sized around the object's resolution (see
    # make_composite), so the object is never resized here at all except
    # for the MAX_OBJ_WIDTH safety cap on rare oversized outliers.
    if obj_im.width > MAX_OBJ_WIDTH:
        aspect = obj_im.height / obj_im.width
        obj_im = obj_im.resize((MAX_OBJ_WIDTH, max(1, int(MAX_OBJ_WIDTH * aspect))))

    # Always rotate in RGBA, even for opaque forced-crop objects (has_alpha
    # False). Rotating with expand=True grows the frame so the tilted
    # rectangle fits, and PIL fills the newly-exposed corners with the
    # image's background value -- for RGB that's solid black, which then
    # gets pasted straight onto the belt with no mask (the "black thingy"
    # around organic/ewaste, and around any material-class fallback crop).
    # Converting to RGBA first means those same expand-corners fill with
    # alpha=0 (transparent) instead, and pasting with the image's own alpha
    # as the mask makes them disappear into the belt exactly like true
    # cutouts already do.
    if obj_im.mode != "RGBA":
        obj_im = obj_im.convert("RGBA")

    angle = random.uniform(-15, 15)
    obj_im = obj_im.rotate(angle, expand=True, resample=Image.BICUBIC)

    max_x = max(1, canvas.width - obj_im.width)
    max_y = max(1, canvas.height - obj_im.height)
    px = random.randint(0, max_x)
    py = random.randint(0, max_y)

    canvas.paste(obj_im, (px, py), obj_im)
    return canvas


def make_composite(belt_im, obj_im, has_alpha):
    # Canvas size is derived from the OBJECT's own resolution, inverted from
    # the old fixed-canvas-then-shrink-object approach, so the object keeps
    # its native detail instead of being downsized to fit a small fixed
    # frame. `scale` is the same random occupied-fraction the object always
    # had (0.55-0.85 of frame width) -- just achieved by sizing the belt
    # patch to match the object, not by resizing the object to match a
    # fixed patch. Small source crops (some TACO crops are under 100px) end
    # up in a smaller composite occupying a correspondingly smaller
    # fraction of CANVAS_FLOOR rather than being upsampled to hit the
    # intended fraction -- upsampling a low-res crop adds blur, not real
    # detail, so it's better to let it be smaller than to fake it larger.
    scale = random.uniform(0.55, 0.85)
    effective_obj_w = min(obj_im.width, MAX_OBJ_WIDTH)
    canvas_size = max(CANVAS_FLOOR, int(effective_obj_w / scale))
    canvas = random_belt_patch(belt_im, canvas_size)
    canvas = paste_scaled_rotated(canvas, obj_im, has_alpha)
    return canvas


# ---------- main ----------

def process_material_class(cls, pool, belt_im, out_dir):
    taco_paths = pool.get("taco", [])
    trashnet_paths = pool.get("trashnet", [])

    n_taco_unique = min(3, len(taco_paths))
    n_trashnet = N_PER_CLASS - n_taco_unique * 2
    n_trashnet = max(0, min(n_trashnet, len(trashnet_paths)))

    chosen_taco = random.sample(taco_paths, n_taco_unique) if n_taco_unique else []
    chosen_trashnet = random.sample(trashnet_paths, n_trashnet) if n_trashnet else []

    written = 0
    idx = 0
    for p in chosen_taco:
        obj = taco_cutout(p)
        if obj is None:
            continue
        for _ in range(2):
            comp = make_composite(belt_im, obj, has_alpha=True)
            comp.save(os.path.join(out_dir, f"preview_{idx:02d}_taco.png"))
            idx += 1
            written += 1

    for p in chosen_trashnet:
        obj = trashnet_cutout(p)
        if obj is None:
            continue
        comp = make_composite(belt_im, obj, has_alpha=True)
        comp.save(os.path.join(out_dir, f"preview_{idx:02d}_trashnet.png"))
        idx += 1
        written += 1

    return written


def process_forced_crop_class(cls, paths, belt_im, out_dir):
    n = min(N_PER_CLASS, len(paths))
    chosen = random.sample(paths, n)
    written = 0
    for i, p in enumerate(chosen):
        crop = forced_bbox_crop(p)
        comp = make_composite(belt_im, crop, has_alpha=False)
        comp.save(os.path.join(out_dir, f"preview_{i:02d}_forcedcrop.png"))
        written += 1
    return written


def main():
    pools = load_train_pools()
    belt_im = load_belt()

    os.makedirs(OUT_DIR, exist_ok=True)

    material_classes = ["cardboard", "glass", "metal", "paper", "plastic"]
    for cls in material_classes:
        out_dir = os.path.join(OUT_DIR, cls)
        os.makedirs(out_dir, exist_ok=True)
        n = process_material_class(cls, pools[cls], belt_im, out_dir)
        n_taco_avail = len(pools[cls].get("taco", []))
        n_trashnet_avail = len(pools[cls].get("trashnet", []))
        print(f"{cls}: wrote {n} composites "
              f"(taco train available: {n_taco_avail}, trashnet train available: {n_trashnet_avail})")

    for cls in ["organic", "ewaste"]:
        out_dir = os.path.join(OUT_DIR, cls)
        os.makedirs(out_dir, exist_ok=True)
        paths = pools["organic"] if cls == "organic" else _dedup_by_real_target([
            p for p, label in json.load(open(STAGE2_SPLIT))["train"] if label == "ewaste"
        ])
        n = process_forced_crop_class(cls, paths, belt_im, out_dir)
        print(f"{cls}: wrote {n} composites (forced bbox crop, train available: {len(paths)})")

    print(f"\nDone. Review outputs/composite_preview/<class>/ before deciding "
          f"whether to scale this up into a full retrain.")


if __name__ == "__main__":
    main()
