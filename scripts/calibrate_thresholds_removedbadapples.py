"""
Copy of calibrate_thresholds.py (which calibrated the BELTCOMPOSITE pair,
see NOTES.md section 30), re-pointed at the removedbadapples checkpoints
(NOTES.md section 62). STAGE1_CUTOFF_REMOVEDBADAPPLES/the "balanced"
0.40 threshold in webapp/app.py were both just reused from BELTCOMPOSITE
as an unvalidated approximation -- this project's own section 28/30
history already showed that exact shortcut doesn't transfer cleanly
(three live A/B runs, up to 7-10 points of macro-F1 recovered just from
recalibrating), and a live session on the removedbadapples pair (macro-F1
~58%, plastic recall 6.5%/13.2%, well below BELTCOMPOSITE's live ~74%)
is consistent with the same problem happening again, not a real model
regression. This script gets the real numbers instead of guessing again.

Original BELTCOMPOSITE docstring, for context on the method (still
accurate, just now applied to a different checkpoint pair):

Calibrate Stage 1's organic-cutoff and Stage 2's confidence threshold
specifically for the BELTCOMPOSITE checkpoints, instead of reusing the old
model's tuned constants (0.65 / 0.50) -- three live A/B runs today (see
NOTES.md section 28) showed those don't transfer cleanly: switching to
"High Recall" mode (0.30 threshold) alone recovered ~7-10 points of live
macro-F1, and plastic's F1 went from 24.4% to 54-60%, just by loosening
the Stage 2 gate. That's evidence the new checkpoints -- trained with
label smoothing and a much more visually mixed per-class training set
(true cutouts blended with opaque forced-crop fallbacks) -- produce
flatter, less-peaked softmax outputs than the old model the constants were
originally tuned around.

Method: run BOTH belt-composite checkpoints once over the VAL split
(never test -- calibrating against test would leak into the final
evaluation number), precompute every image's Stage 1 organic-probability
and Stage 2 top-class + top-probability, then grid-search over candidate
(stage1_cutoff, stage2_threshold) pairs, simulating the EXACT same
decision logic webapp/app.py's cascaded_classify_fn uses:

    if prob_organic >= stage1_cutoff: predict "organic"
    elif stage2_top_prob >= stage2_threshold: predict stage2_top_class
    else: predict "unknown"

...and scoring each candidate with this project's own metrics module
(src/metrics.py's confusion_matrix / per_class_precision_recall_f1 /
macro_and_weighted_f1 -- the same "unknown is never a true label" fix from
NOTES.md section 19), so the macro-F1 this script reports means the same
thing the live dashboard's macro-F1 means.

Stage 1 val images labeled "recyclable" in stage1_split.json are the exact
same files as stage2_split.json's val entries (recyclable is a relabeled
reuse of Stage 2's own data -- see NOTES.md section 24), so their true
7-way label is recovered by looking up the same path in stage2_split.json
rather than guessed or discarded.

Run on your Mac (needs the real venv with torch):

    cd pytorch_pipeline
    python3 scripts/calibrate_thresholds_removedbadapples.py
"""

import os
import sys
import json
import itertools

import torch
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.models import get_model_and_transforms
from src.metrics import per_class_precision_recall_f1, macro_and_weighted_f1

DATA_ROOT = "data/cascades_data"
STAGE1_SPLIT = os.path.join(DATA_ROOT, "stage1_split.json")
STAGE2_SPLIT = os.path.join(DATA_ROOT, "stage2_split.json")

STAGE1_CKPT = "outputs/stage1_mobilenet_removedbadapples.pt"
STAGE2_CKPT = "outputs/stage2_material_mobilenet_removedbadapples.pt"

STAGE2_CLASSES = ["cardboard", "ewaste", "glass", "metal", "paper", "plastic"]
ALL_CLASSES = ["organic", "cardboard", "ewaste", "glass", "metal", "paper", "plastic", "unknown"]
CLASS_TO_IDX = {c: i for i, c in enumerate(ALL_CLASSES)}

# Same values webapp/app.py's STAGE1_CUTOFF_REMOVEDBADAPPLES and the
# "balanced"-mode Stage 2 threshold override are ACTUALLY set to right now
# for this checkpoint pair (both just aliased from BELTCOMPOSITE, never
# calibrated -- see this file's docstring), included in the sweep so the
# report shows a direct "what's live right now vs. what it should be"
# comparison. NOT the original pre-BELTCOMPOSITE model's 0.65/0.50 -- that
# comparison isn't meaningful for this checkpoint pair.
CURRENT_STAGE1_CUTOFF = 0.50
CURRENT_STAGE2_THRESHOLD = 0.40

STAGE1_CUTOFF_CANDIDATES = sorted(set([round(x, 2) for x in np.arange(0.30, 0.91, 0.05)] + [CURRENT_STAGE1_CUTOFF]))
STAGE2_THRESHOLD_CANDIDATES = sorted(set([round(x, 2) for x in np.arange(0.10, 0.71, 0.05)] + [CURRENT_STAGE2_THRESHOLD]))

BATCH_SIZE = 32


def load_model(ckpt_path, num_classes, device):
    model, _, eval_tf = get_model_and_transforms("mobilenet_v3_large", num_classes, pretrained=False)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.to(device)
    model.eval()
    return model, eval_tf


@torch.no_grad()
def run_inference(model, transform, paths, device, batch_size=BATCH_SIZE):
    """Returns an (N, num_classes) numpy array of softmax probabilities, in
    the same order as `paths`."""
    all_probs = []
    batch_imgs = []
    for i, p in enumerate(paths):
        img = Image.open(p).convert("RGB")
        batch_imgs.append(transform(img))
        if len(batch_imgs) == batch_size or i == len(paths) - 1:
            x = torch.stack(batch_imgs).to(device)
            logits = model(x)
            all_probs.append(torch.softmax(logits, dim=1).cpu().numpy())
            batch_imgs = []
        if (i + 1) % 500 == 0:
            print(f"  ...{i+1}/{len(paths)} processed")
    return np.concatenate(all_probs, axis=0)


def main():
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    print(f"Device: {device}")

    with open(STAGE1_SPLIT) as f:
        s1 = json.load(f)
    with open(STAGE2_SPLIT) as f:
        s2 = json.load(f)

    stage2_val_label_by_path = {p: label for p, label in s2["val"]}

    all_paths, true_labels = [], []
    skipped = 0
    for p, label in s1["val"]:
        if label == "organic":
            true_labels.append("organic")
        else:
            material_label = stage2_val_label_by_path.get(p)
            if material_label is None:
                skipped += 1
                continue
            true_labels.append(material_label)
        all_paths.append(p)

    n_organic = sum(1 for l in true_labels if l == "organic")
    n_material = len(true_labels) - n_organic
    print(f"Calibration set: {len(all_paths)} val images "
          f"({n_organic} organic, {n_material} material, {skipped} skipped/unresolved)")

    print("Loading Stage 1 (removedbadapples)...")
    model1, transform1 = load_model(STAGE1_CKPT, 2, device)
    print("Loading Stage 2 (removedbadapples)...")
    model2, transform2 = load_model(STAGE2_CKPT, 6, device)

    print("Running Stage 1 inference over calibration set...")
    probs1 = run_inference(model1, transform1, all_paths, device)  # [:, 0] = organic prob

    print("Running Stage 2 inference over calibration set "
          "(run on every image -- cheap enough, and needed for every candidate stage1 cutoff)...")
    probs2 = run_inference(model2, transform2, all_paths, device)
    stage2_top_idx = probs2.argmax(axis=1)
    stage2_top_prob = probs2.max(axis=1)

    true_idx = np.array([CLASS_TO_IDX[l] for l in true_labels])

    print(f"\nSweeping {len(STAGE1_CUTOFF_CANDIDATES)} x {len(STAGE2_THRESHOLD_CANDIDATES)} "
          f"threshold combinations...")

    results = []
    for cutoff, threshold in itertools.product(STAGE1_CUTOFF_CANDIDATES, STAGE2_THRESHOLD_CANDIDATES):
        pred_idx = np.empty(len(all_paths), dtype=np.int64)
        is_organic_pred = probs1[:, 0] >= cutoff
        pred_idx[is_organic_pred] = CLASS_TO_IDX["organic"]

        not_organic = ~is_organic_pred
        confident = not_organic & (stage2_top_prob >= threshold)
        confident_indices = np.where(confident)[0]
        pred_idx[confident_indices] = [CLASS_TO_IDX[STAGE2_CLASSES[stage2_top_idx[i]]] for i in confident_indices]
        unknown_mask = not_organic & ~confident
        pred_idx[unknown_mask] = CLASS_TO_IDX["unknown"]

        m = per_class_precision_recall_f1(true_idx, pred_idx, len(ALL_CLASSES))
        f1_stats = macro_and_weighted_f1(m)
        results.append((cutoff, threshold, f1_stats["macro_f1"], f1_stats["weighted_f1"], int(unknown_mask.sum())))

    results.sort(key=lambda r: r[2], reverse=True)

    print("\nTop 10 (stage1_cutoff, stage2_threshold) combinations by macro-F1:")
    print(f"{'cutoff':>8} {'threshold':>10} {'macro_f1':>10} {'weighted_f1':>12} {'n_unknown':>10}")
    for cutoff, threshold, macro_f1, weighted_f1, n_unknown in results[:10]:
        print(f"{cutoff:>8.2f} {threshold:>10.2f} {macro_f1:>10.4f} {weighted_f1:>12.4f} {n_unknown:>10d}")

    best_cutoff, best_threshold, best_macro, best_weighted, best_unknown = results[0]
    print(f"\nRecommended: stage1_cutoff={best_cutoff}, stage2_threshold={best_threshold} "
          f"(macro-F1={best_macro:.4f}, weighted-F1={best_weighted:.4f}, "
          f"{best_unknown}/{len(all_paths)} routed to unknown)")

    current = next((r for r in results
                     if r[0] == CURRENT_STAGE1_CUTOFF and r[1] == CURRENT_STAGE2_THRESHOLD), None)
    if current:
        print(f"Current hardcoded values ({CURRENT_STAGE1_CUTOFF} / {CURRENT_STAGE2_THRESHOLD}): "
              f"macro-F1={current[2]:.4f}, weighted-F1={current[3]:.4f}, "
              f"{current[4]}/{len(all_paths)} routed to unknown")

    os.makedirs("outputs", exist_ok=True)
    with open("outputs/threshold_calibration_removedbadapples.json", "w") as f:
        json.dump({
            "recommended_stage1_cutoff": best_cutoff,
            "recommended_stage2_threshold": best_threshold,
            "recommended_macro_f1": best_macro,
            "recommended_weighted_f1": best_weighted,
            "current_stage1_cutoff": CURRENT_STAGE1_CUTOFF,
            "current_stage2_threshold": CURRENT_STAGE2_THRESHOLD,
            "current_macro_f1": current[2] if current else None,
            "current_weighted_f1": current[3] if current else None,
            "top_10": [
                {"stage1_cutoff": c, "stage2_threshold": t, "macro_f1": mf,
                 "weighted_f1": wf, "n_unknown": nu}
                for c, t, mf, wf, nu in results[:10]
            ],
            "n_calibration_images": len(all_paths),
        }, f, indent=2)
    print("\nWrote outputs/threshold_calibration_removedbadapples.json")


if __name__ == "__main__":
    main()
