"""
NOTES.md section 63 follow-up (round 2). The per-class logit bias
(cardboard=-0.75) fixed cardboard's precision (66.3% -> ~62-78% across
val/live) but did it by reallocating cardboard's suppressed probability
mass to whichever OTHER class had the next-highest logit -- live testing
showed this landed disproportionately on glass (plastic->glass leak rate
more than doubled, 7.9% -> 18.8%), dragging glass's own F1 down ~8 points
even though glass's logit was never touched. The user asked for a fix
that touches ONLY cardboard's numbers, not any other class's.

Different mechanism, not a bias: a cardboard-SPECIFIC confidence floor.
If Stage 2's argmax winner is cardboard but its probability doesn't clear
this floor, route straight to "unknown" instead of accepting cardboard --
crucially, NOT letting the second-place class win instead (that's what a
logit bias does, and that's what touched glass). Every image where
cardboard isn't the winner is completely untouched by this rule -- no
other class's predicted set changes AT ALL, so no other class's precision
or recall can change either. The only two numbers that move are
cardboard's own precision (up, weak-confidence cardboard calls get
filtered out) and cardboard's own recall (down slightly, since some
correct-but-low-confidence cardboard calls get filtered too -- there's no
way to tell correct from incorrect by confidence alone). "unknown" is
never a true label (NOTES.md section 19), so this doesn't distort
macro-F1 the way misrouting to a real class does.

Method: same val-split, same decision pipeline as every other calibration
this session (old Stage 1 @ 0.65 cutoff, Stage 2 threshold @ 0.30,
NO logit bias in this sweep -- testing the floor as a standalone
replacement, not stacked on top of the bias). Sweeps a single parameter,
the cardboard confidence floor, and prints the FULL per-class table for
top candidates specifically so the "no other class moves" claim is
verified with real numbers, not just asserted.

Run on your Mac:

    cd pytorch_pipeline
    python3 scripts/calibrate_cardboard_floor_removedbadapples_v2.py
"""
import os
import sys
import json

import torch
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.models import get_model_and_transforms
from src.metrics import per_class_precision_recall_f1, macro_and_weighted_f1

DATA_ROOT = "data/cascades_data"
STAGE1_SPLIT = os.path.join(DATA_ROOT, "stage1_split.json")
STAGE2_SPLIT = os.path.join(DATA_ROOT, "stage2_split.json")

STAGE1_CKPT = "outputs/stage1_mobilenet.pt"
STAGE2_CKPT = "outputs/stage2_material_mobilenet_removedbadapples_v2.pt"
STAGE1_CUTOFF = 0.65
STAGE2_THRESHOLD = 0.30  # global "unknown" threshold, held fixed

STAGE2_CLASSES = ["cardboard", "ewaste", "glass", "metal", "paper", "plastic"]
ALL_CLASSES = ["organic", "cardboard", "ewaste", "glass", "metal", "paper", "plastic", "unknown"]
CLASS_TO_IDX = {c: i for i, c in enumerate(ALL_CLASSES)}
CB_IDX = STAGE2_CLASSES.index("cardboard")

FLOOR_CANDIDATES = [round(x, 2) for x in np.arange(0.30, 0.91, 0.05)]

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


def print_full_table(m, macro_f1, weighted_f1, n_unknown, n_total, label):
    print(f"\n--- {label} ---")
    print(f"macro-F1={macro_f1:.4f} weighted-F1={weighted_f1:.4f} unknown={n_unknown}/{n_total}")
    for c in ["organic", "cardboard", "ewaste", "glass", "metal", "paper", "plastic"]:
        i = CLASS_TO_IDX[c]
        print(f"  {c.ljust(10)}: P={m['precision'][i]:.4f} R={m['recall'][i]:.4f} "
              f"F1={m['f1'][i]:.4f} support={int(m['support'][i])}")


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

    print(f"Calibration set: {len(all_paths)} val images ({skipped} skipped/unresolved)")

    print(f"Loading Stage 1 ({STAGE1_CKPT})...")
    model1, transform1 = load_model(STAGE1_CKPT, 2, device, is_stage2=False)
    print(f"Loading Stage 2 ({STAGE2_CKPT})...")
    model2, transform2 = load_model(STAGE2_CKPT, 6, device, is_stage2=True)

    print("Running Stage 1 inference...")
    logits1 = run_inference_logits(model1, transform1, all_paths, device)
    probs1 = softmax_np(logits1)

    print("Running Stage 2 inference...")
    logits2 = run_inference_logits(model2, transform2, all_paths, device)
    probs2_base = softmax_np(logits2)
    stage2_top_idx = probs2_base.argmax(axis=1)
    stage2_top_prob = probs2_base.max(axis=1)

    true_idx = np.array([CLASS_TO_IDX[l] for l in true_labels])
    is_organic_pred = probs1[:, 0] >= STAGE1_CUTOFF
    not_organic = ~is_organic_pred

    def evaluate(cardboard_floor):
        pred_idx = np.empty(len(all_paths), dtype=np.int64)
        pred_idx[is_organic_pred] = CLASS_TO_IDX["organic"]

        confident = not_organic & (stage2_top_prob >= STAGE2_THRESHOLD)
        confident_indices = np.where(confident)[0]
        for i in confident_indices:
            cls_i = stage2_top_idx[i]
            if cls_i == CB_IDX and stage2_top_prob[i] < cardboard_floor:
                pred_idx[i] = CLASS_TO_IDX["unknown"]
            else:
                pred_idx[i] = CLASS_TO_IDX[STAGE2_CLASSES[cls_i]]
        unknown_mask = not_organic & ~confident
        pred_idx[unknown_mask] = CLASS_TO_IDX["unknown"]

        m = per_class_precision_recall_f1(true_idx, pred_idx, len(ALL_CLASSES))
        f1_stats = macro_and_weighted_f1(m)
        n_unknown = int((pred_idx == CLASS_TO_IDX["unknown"]).sum())
        return m, f1_stats, n_unknown

    print("\nBaseline (no cardboard floor, floor == global threshold 0.30):")
    m0, f0, u0 = evaluate(STAGE2_THRESHOLD)
    print_full_table(m0, f0["macro_f1"], f0["weighted_f1"], u0, len(all_paths), "baseline")

    print(f"\nSweeping cardboard floor over {FLOOR_CANDIDATES}...")
    results = []
    for floor in FLOOR_CANDIDATES:
        m, f, u = evaluate(floor)
        cb_p = m["precision"][CLASS_TO_IDX["cardboard"]]
        cb_r = m["recall"][CLASS_TO_IDX["cardboard"]]
        # Verify the "no other class moves" claim directly, not just assert it.
        other_classes_unchanged = all(
            abs(m["precision"][CLASS_TO_IDX[c]] - m0["precision"][CLASS_TO_IDX[c]]) < 1e-9 and
            abs(m["recall"][CLASS_TO_IDX[c]] - m0["recall"][CLASS_TO_IDX[c]]) < 1e-9
            for c in ["organic", "ewaste", "glass", "metal", "paper", "plastic"]
        )
        results.append((floor, f["macro_f1"], cb_p, cb_r, u, other_classes_unchanged))

    print(f"\n{'floor':>7} {'macro_f1':>10} {'cb_precision':>13} {'cb_recall':>10} "
          f"{'n_unknown':>10} {'others_untouched':>18}")
    for floor, mf, cbp, cbr, u, untouched in results:
        print(f"{floor:>7.2f} {mf:>10.4f} {cbp:>13.3f} {cbr:>10.3f} {u:>10d} {str(untouched):>18}")

    # Pick the highest macro-F1 -- safe to do here (unlike the earlier
    # logit-bias sweep, section 63 round 1, where the top-macro-F1 pick
    # was misleading because macro-F1 hid a real regression in the class
    # being "fixed"): every candidate in THIS sweep is proven
    # (others_untouched=True, checked directly above) to leave every
    # other class byte-identical to baseline, so any macro-F1 movement can
    # ONLY be coming from cardboard's own precision/recall trade-off --
    # there's no way for this particular knob to hide a side effect
    # elsewhere the way the bias could.
    # NOTE: CLASS_TO_IDX["cardboard"] (index into the 8-class ALL_CLASSES
    # array m0["precision"] is built from), NOT CB_IDX (the 6-class
    # STAGE2_CLASSES index) -- an earlier version of this script mixed
    # the two up here and wrongly reported "no floor cleared the bar".
    cb_all_idx = CLASS_TO_IDX["cardboard"]
    best = max(results, key=lambda r: r[1])
    print(f"\nRecommended: cardboard_floor={best[0]} "
          f"(cardboard precision {m0['precision'][cb_all_idx]:.3f} -> {best[2]:.3f}, "
          f"recall {m0['recall'][cb_all_idx]:.3f} -> {best[3]:.3f}, "
          f"macro-F1={best[1]:.4f} vs baseline {f0['macro_f1']:.4f}, "
          f"other classes untouched={best[5]})")
    m_best, f_best, u_best = evaluate(best[0])
    print_full_table(m_best, f_best["macro_f1"], f_best["weighted_f1"], u_best, len(all_paths),
                      f"recommended (floor={best[0]})")

    os.makedirs("outputs", exist_ok=True)
    with open("outputs/cardboard_floor_calibration_removedbadapples_v2.json", "w") as f:
        json.dump({
            "recommended_cardboard_floor": best[0],
            "baseline_cardboard_precision": float(m0["precision"][cb_all_idx]),
            "recommended_cardboard_precision": best[2],
            "baseline_cardboard_recall": float(m0["recall"][cb_all_idx]),
            "recommended_cardboard_recall": best[3],
            "recommended_macro_f1": best[1],
            "baseline_macro_f1": f0["macro_f1"],
            "n_unknown": best[4],
            "other_classes_verified_untouched": best[5],
            "stage1_cutoff": STAGE1_CUTOFF,
            "stage2_threshold": STAGE2_THRESHOLD,
        }, f, indent=2)
    print("\nWrote outputs/cardboard_floor_calibration_removedbadapples_v2.json")


if __name__ == "__main__":
    main()
