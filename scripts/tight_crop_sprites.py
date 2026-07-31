"""
Tighten every sprite in outputs/sprites/ to a snug bounding box around the
object, instead of the full studio photo with lots of empty background
padding.

Why this exists (see NOTES.md): the belt UI renders the exact sprite file
that also gets classified (BeltLocalizer crops from the same rendered
frame), so "make it look like a floating object, not a photo card" and
"keep the accuracy gained from matching training-domain backgrounds" are in
tension if we go all the way to a transparent cutout (that was the old,
lower-accuracy SAM2 approach). This script is the middle path: trim the
excess plain background around the object, but keep a thin margin of real
background pixels around it, so the crop still looks like the same kind of
image the model trained on -- just without the big empty margins.

Method (classical CV, no new dependencies, no model weights):
  1. Sample the outer border ring of the image to estimate the background
     color (median, not a single corner pixel, so one weird pixel doesn't
     throw it off).
  2. Threshold every pixel by color distance from that background estimate
     -> binary foreground mask.
  3. Morphological close + open to remove small speckle/watermark noise.
  4. Bounding box of the surviving foreground pixels, expanded by a small
     margin, clipped to image bounds.
  5. Safety fallback: if the box covers almost the whole frame (>92%) or
     almost none of it (<3%), the image likely has a busy/non-uniform
     background (e.g. many organic photos -- wood tables, garden foliage,
     market scenes) where this heuristic doesn't apply cleanly. In that
     case the original image is left untouched rather than risk a bad crop.

Run this on your Mac (same reason as rebuild_sprites.py -- needs real pixel
access, not the sandboxed shell used for development):

    cd pytorch_pipeline
    python3 scripts/tight_crop_sprites.py

Operates in place on outputs/sprites/. Sprites are disposable/regenerable
(rebuild_sprites.py + this script reproduce them from source data), so this
does not follow the "never overwrite in place" rule that applies to .pt
checkpoints -- there is nothing here that can't be regenerated.
"""

import os
import numpy as np
from PIL import Image
import cv2

SPRITES_DIR = "outputs/sprites"

BORDER_FRAC = 0.06          # how much of the border ring to sample for bg color
COLOR_DIST_THRESHOLD = 32   # euclidean RGB distance to call a pixel "foreground"
MARGIN_FRAC = 0.08          # padding added around the tight bbox, relative to bbox size
MIN_BOX_FRAC = 0.03         # skip crop if box covers less than this fraction of the frame
MAX_BOX_FRAC = 0.92         # skip crop if box covers more than this fraction of the frame


def estimate_background_color(arr):
    h, w = arr.shape[:2]
    bh = max(1, int(h * BORDER_FRAC))
    bw = max(1, int(w * BORDER_FRAC))
    border_pixels = np.concatenate([
        arr[:bh, :, :].reshape(-1, 3),
        arr[-bh:, :, :].reshape(-1, 3),
        arr[:, :bw, :].reshape(-1, 3),
        arr[:, -bw:, :].reshape(-1, 3),
    ], axis=0)
    return np.median(border_pixels, axis=0)


def tight_bbox(arr):
    bg_color = estimate_background_color(arr)
    dist = np.linalg.norm(arr.astype(np.float32) - bg_color.astype(np.float32), axis=2)
    mask = (dist > COLOR_DIST_THRESHOLD).astype(np.uint8)

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    ys, xs = np.where(mask > 0)
    if len(ys) == 0:
        return None

    h, w = arr.shape[:2]
    x1, x2 = xs.min(), xs.max()
    y1, y2 = ys.min(), ys.max()

    box_frac = ((x2 - x1) * (y2 - y1)) / float(h * w)
    if box_frac < MIN_BOX_FRAC or box_frac > MAX_BOX_FRAC:
        return None

    bw_, bh_ = (x2 - x1), (y2 - y1)
    mx, my = int(bw_ * MARGIN_FRAC), int(bh_ * MARGIN_FRAC)
    x1 = max(0, x1 - mx)
    y1 = max(0, y1 - my)
    x2 = min(w, x2 + mx)
    y2 = min(h, y2 + my)
    return x1, y1, x2, y2


def process_class(cls):
    d = os.path.join(SPRITES_DIR, cls)
    if not os.path.isdir(d):
        return 0, 0

    cropped = 0
    skipped = 0
    for fname in os.listdir(d):
        if not fname.endswith(".png"):
            continue
        path = os.path.join(d, fname)
        try:
            im = Image.open(path).convert("RGB")
        except Exception as e:
            print(f"  {cls}/{fname}: could not open ({e}), skipping")
            skipped += 1
            continue

        arr = np.array(im)
        box = tight_bbox(arr)
        if box is None:
            skipped += 1
            continue

        x1, y1, x2, y2 = box
        cropped_im = im.crop((x1, y1, x2, y2))
        cropped_im.save(path, "PNG")
        cropped += 1

    return cropped, skipped


def main():
    classes = sorted(os.listdir(SPRITES_DIR)) if os.path.isdir(SPRITES_DIR) else []
    total_cropped = total_skipped = 0
    for cls in classes:
        c, s = process_class(cls)
        print(f"{cls}: tightened {c}, left as-is {s}")
        total_cropped += c
        total_skipped += s
    print(f"\nDone. {total_cropped} sprites tightened, {total_skipped} left "
          f"unchanged (likely non-uniform backgrounds, e.g. organic).")
    print("Restart the webapp to pick up the changes.")


if __name__ == "__main__":
    main()
