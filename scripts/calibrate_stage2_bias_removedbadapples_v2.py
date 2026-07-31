"""
NOTES.md section 63 follow-up. removedbadapples_v2 is now the live
default (4-run avg macro-F1 69.9%, beating both the previous default and
the focal-loss variant decisively) -- but its own confusion matrices show
a real, systematic, NOT-random skew: cardboard's live precision is
consistently ~44-49% across all 3 confirmation runs (45/22, 50/24, 46/20
predicted-cardboard-vs-true-cardboard), meaning roughly half of
everything Stage 2 calls "cardboard" is actually something else --
ewaste, metal, paper, and plastic all leak into it repeatedly, every
run. Plastic specifically loses 4-8 of its own items to cardboard per
run, a meaningful chunk of its recall.

This is exactly the kind of bias a POST-HOC per-class logit correction
can fix without retraining: since cardboard is over-called and plastic
is under-called in a consistent, repeatable direction (not just this
session's noise), subtracting a fixed bias from cardboard's logit and/or
adding one to plastic's logit, before the argmax, should rebalance the
decision boundary directly -- the model's underlying features are fine
(that's why it's the best checkpoint tonight), the class SINK is what's
broken.

Method: run the already-trained removedbadapples_v2 Stage 2 checkpoint
(paired with the same old Stage 1 + 0.65 cutoff + 0.30 threshold the live
webapp currently uses) over the val split, get raw logits, then grid-
search (cardboard_bias, plastic_bias) pairs applied as
`logits + bias_vector` before softmax/argmax/threshold -- simulating the
EXACT same decision logic cascaded_classify_fn uses, scored with this
project's own src/metrics.py macro-F1 (same "unknown is never a true
label" convention as every other calibration this session). All other
classes stay unbiased (0) in this sweep -- cardboard/plastic are the two
classes with a proven, repeatable live skew; touching classes without
evidence of a problem risks trading one bias for another.

Run on your Mac:

    cd pytorch_pipeline
    python3 scripts/calibrate_stage2_bias_removedbadapples_v2.py
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

# Same pairing the live webapp uses for removedbadapples_v2 (NOTES.md
# section 62 follow-up): OLD Stage 1 (proven, not the leaky retrained
# one), augment-fixed Stage 2.
STAGE1_CKPT = "outputs/stage1_mobilenet.pt"
STAGE2_CKPT = "outputs/stage2_material_mobilenet_removedbadapples_v2.pt"
STAGE1_CUTOFF = 0.65
STAGE2_THRESHOLD = 0.30  # current live stopgap, held fixed while sweeping bias

STAGE2_CLASSES = ["cardboard", "ewaste", "glass", "metal", "paper", "plastic"]
ALL_CLASSES = ["organic", "cardboard", "ewaste", "glass", "metal", "paper", "plastic", "unknown"]
CLASS_TO_IDX = {c: i for i, c in enumerate(ALL_CLASSES)}
CB_IDX = STAGE2_CLASSES.index("cardboard")
PLASTIC_IDX = STAGE2_CLASSES.index("plastic")

# Logit-space bias candidates. +/-2.0 on a 6-way softmax is a large,
# decision-flipping shift -- range chosen to comfortably bracket where the
# effect saturates, not just a token nudge.
CARDBOARD_BIAS_CANDIDATES = [round(x, 2) for x in np.arange(-2.0, 0.01, 0.25)]  # 0 or negative only
PLASTIC_BIAS_CANDIDATES = [round(x, 2) for x in np.arange(0.0, 2.01, 0.25)]     # 0 or positive only

BATCH_SIZE = 32


def load_model(ckpt_path, num_classes, device, is_stage2=False):
    model, _, eval_tf = get_model_and_transforms("mobilenet_v3_large", num_classes, pretrained=False, is_stage2=is_stage2)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.to(device)
    model.eval()
    return model, eval_tf


@torch.no_grad()
def run_inference_logits(model, transform, paths, device, batch_size=BATCH_SIZE):
    all_logits = []
    batch = []
    for i, p in enumerate(paths):
        img = Image.open(p).convert("RGB")
        batch.append(transform(img))
        if len(batch) == batch_size or i == len(paths) - 1:
            x = torch.stack(batch).to(device)
            all_logits.append(model(x).cpu().numpy())
            batch = []
        if (i + 1) % 500 == 0:
            print(f"  ...{i+1}/{len(paths)} processed")
    return np.concatenate(all_logits, axis=0)


def softmax_np(x, axis=-1):
    e = np.exp(x - x.max(axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)


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

    print(f"Loading Stage 1 ({STAGE1_CKPT})...")
    model1, transform1 = load_model(STAGE1_CKPT, 2, device, is_stage2=False)
    print(f"Loading Stage 2 ({STAGE2_CKPT})...")
    model2, transform2 = load_model(STAGE2_CKPT, 6, device, is_stage2=True)

    print("Running Stage 1 inference...")
    logits1 = run_inference_logits(model1, transform1, all_paths, device)
    probs1 = softmax_np(logits1)

    print("Running Stage 2 inference (raw logits, bias applied per-candidate below)...")
    logits2 = run_inference_logits(model2, transform2, all_paths, device)

    true_idx = np.array([CLASS_TO_IDX[l] for l in true_labels])
    is_organic_pred = probs1[:, 0] >= STAGE1_CUTOFF
    not_organic = ~is_organic_pred

    print(f"\nSweeping {len(CARDBOARD_BIAS_CANDIDATES)} x {len(PLASTIC_BIAS_CANDIDATES)} "
          f"(cardboard_bias, plastic_bias) combinations, threshold fixed at {STAGE2_THRESHOLD}...")

    results = []
    for cb_bias, pl_bias in itertools.product(CARDBOARD_BIAS_CANDIDATES, PLASTIC_BIAS_CANDIDATES):
        bias_vec = np.zeros(6, dtype=np.float32)
        bias_vec[CB_IDX] = cb_bias
        bias_vec[PLASTIC_IDX] = pl_bias

        biased_logits2 = logits2 + bias_vec
        probs2 = softmax_np(biased_logits2)
        stage2_top_idx = probs2.argmax(axis=1)
        stage2_top_prob = probs2.max(axis=1)

        pred_idx = np.empty(len(all_paths), dtype=np.int64)
        pred_idx[is_organic_pred] = CLASS_TO_IDX["organic"]

        confident = not_organic & (stage2_top_prob >= STAGE2_THRESHOLD)
        confident_indices = np.where(confident)[0]
        pred_idx[confident_indices] = [CLASS_TO_IDX[STAGE2_CLASSES[stage2_top_idx[i]]] for i in confident_indices]
        unknown_mask = not_organic & ~confident
        pred_idx[unknown_mask] = CLASS_TO_IDX["unknown"]

        m = per_class_precision_recall_f1(true_idx, pred_idx, len(ALL_CLASSES))
        f1_stats = macro_and_weighted_f1(m)
        cardboard_precision = m["precision"][CLASS_TO_IDX["cardboard"]]
        plastic_recall = m["recall"][CLASS_TO_IDX["plastic"]]
        plastic_f1 = m["f1"][CLASS_TO_IDX["plastic"]]
        results.append((cb_bias, pl_bias, f1_stats["macro_f1"], f1_stats["weighted_f1"],
                         cardboard_precision, plastic_recall, plastic_f1))

    results.sort(key=lambda r: r[2], reverse=True)

    print("\nTop 15 (cardboard_bias, plastic_bias) combinations by macro-F1:")
    print(f"{'cb_bias':>8} {'pl_bias':>8} {'macro_f1':>10} {'weighted_f1':>12} "
          f"{'cb_prec':>8} {'plastic_R':>10} {'plastic_F1':>11}")
    for cb, pl, mf, wf, cbp, pr, pf in results[:15]:
        print(f"{cb:>8.2f} {pl:>8.2f} {mf:>10.4f} {wf:>12.4f} {cbp:>8.3f} {pr:>10.3f} {pf:>11.3f}")

    zero_bias = next((r for r in results if r[0] == 0.0 and r[1] == 0.0), None)
    if zero_bias:
        print(f"\nNo-bias baseline (0.0 / 0.0): macro-F1={zero_bias[2]:.4f}, "
              f"cardboard_precision={zero_bias[4]:.3f}, plastic_recall={zero_bias[5]:.3f}, "
              f"plastic_F1={zero_bias[6]:.3f}")

    best = results[0]
    print(f"\nRecommended: cardboard_bias={best[0]}, plastic_bias={best[1]} "
          f"(macro-F1={best[2]:.4f}, cardboard_precision={best[4]:.3f}, "
          f"plastic_recall={best[5]:.3f}, plastic_F1={best[6]:.3f})")

    os.makedirs("outputs", exist_ok=True)
    with open("outputs/stage2_bias_calibration_removedbadapples_v2.json", "w") as f:
        json.dump({
            "recommended_cardboard_bias": best[0],
            "recommended_plastic_bias": best[1],
            "recommended_macro_f1": best[2],
            "recommended_weighted_f1": best[3],
            "recommended_cardboard_precision": best[4],
            "recommended_plastic_recall": best[5],
            "recommended_plastic_f1": best[6],
            "no_bias_baseline_macro_f1": zero_bias[2] if zero_bias else None,
            "stage1_cutoff": STAGE1_CUTOFF,
            "stage2_threshold": STAGE2_THRESHOLD,
            "top_15": [
                {"cardboard_bias": cb, "plastic_bias": pl, "macro_f1": mf, "weighted_f1": wf,
                 "cardboard_precision": cbp, "plastic_recall": pr, "plastic_f1": pf}
                for cb, pl, mf, wf, cbp, pr, pf in results[:15]
            ],
            "n_calibration_images": len(all_paths),
        }, f, indent=2)
    print("\nWrote outputs/stage2_bias_calibration_removedbadapples_v2.json")


if __name__ == "__main__":
    main()
