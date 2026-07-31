"""
NOTES.md section 65. Same idea as
calibrate_cardboard_floor_removedbadapples_v2.py, applied to a second,
smaller sink found in the live runs AFTER the cardboard floor was fixed
and correctly re-layered: metal's own precision was mediocre and
consistent across all 3 confirmation runs (65.5%, 55.6%, 64.9%), and the
metal column in every one of those confusion matrices showed the same
two contributors every time -- plastic->metal leak was remarkably
consistent (4, 5, 4 across the three runs) and glass->metal was present
in all three too (2, 9, 5). organic/ewaste/cardboard/paper all had clean,
high precision in the same runs -- metal is a real, repeatable secondary
sink, not noise, just smaller in scale than cardboard's was.

Same mechanism, same reasoning: a metal-specific confidence floor. If
Stage 2's winner is metal but its probability doesn't clear the floor,
route to "unknown" instead of letting the runner-up win. Swept WITH the
cardboard floor already fixed at its calibrated value (0.65), not in
isolation -- the live system will have both active simultaneously, so
this measures the actual deployed configuration, not two independent
what-ifs that might interact.

IMPORTANT (NOTES.md section 64 follow-up, the lesson that cost a whole
live-test-and-revert cycle on the cardboard floor): this calibration
evaluates independent single images, matching the FINAL aggregated
per-object decision the live webapp makes after summing evidence across
an object's whole belt transit -- NOT each individual raw frame. Wire
this into templates/index.html's final-decision point (same place the
cardboard floor lives now), NOT into app.py's cascaded_classify_fn
(which runs per raw frame and would starve metal's running evidence
total the exact same way it broke cardboard the first time).

Run on your Mac:

    cd pytorch_pipeline
    python3 scripts/calibrate_metal_floor_removedbadapples_v2.py
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
STAGE2_THRESHOLD = 0.30      # global "unknown" threshold, held fixed
CARDBOARD_FLOOR = 0.65       # already calibrated (NOTES.md section 64), held fixed

STAGE2_CLASSES = ["cardboard", "ewaste", "glass", "metal", "paper", "plastic"]
ALL_CLASSES = ["organic", "cardboard", "ewaste", "glass", "metal", "paper", "plastic", "unknown"]
CLASS_TO_IDX = {c: i for i, c in enumerate(ALL_CLASSES)}
CB_IDX = STAGE2_CLASSES.index("cardboard")
METAL_IDX = STAGE2_CLASSES.index("metal")

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

    def evaluate(metal_floor):
        pred_idx = np.empty(len(all_paths), dtype=np.int64)
        pred_idx[is_organic_pred] = CLASS_TO_IDX["organic"]

        confident = not_organic & (stage2_top_prob >= STAGE2_THRESHOLD)
        confident_indices = np.where(confident)[0]
        for i in confident_indices:
            cls_i = stage2_top_idx[i]
            if cls_i == CB_IDX and stage2_top_prob[i] < CARDBOARD_FLOOR:
                pred_idx[i] = CLASS_TO_IDX["unknown"]
            elif cls_i == METAL_IDX and stage2_top_prob[i] < metal_floor:
                pred_idx[i] = CLASS_TO_IDX["unknown"]
            else:
                pred_idx[i] = CLASS_TO_IDX[STAGE2_CLASSES[cls_i]]
        unknown_mask = not_organic & ~confident
        pred_idx[unknown_mask] = CLASS_TO_IDX["unknown"]

        m = per_class_precision_recall_f1(true_idx, pred_idx, len(ALL_CLASSES))
        f1_stats = macro_and_weighted_f1(m)
        n_unknown = int((pred_idx == CLASS_TO_IDX["unknown"]).sum())
        return m, f1_stats, n_unknown

    print(f"\nBaseline (cardboard floor={CARDBOARD_FLOOR} active, metal floor == global threshold {STAGE2_THRESHOLD}):")
    m0, f0, u0 = evaluate(STAGE2_THRESHOLD)
    print_full_table(m0, f0["macro_f1"], f0["weighted_f1"], u0, len(all_paths), "baseline (cardboard fix only)")

    # NOTE: index into m["precision"]/m["recall"] using CLASS_TO_IDX
    # (the 8-class ALL_CLASSES array these are actually built over), NOT
    # METAL_IDX/CB_IDX (the 6-class STAGE2_CLASSES indices used only for
    # comparing against stage2_top_idx). An earlier version of this
    # script used METAL_IDX=3 here, which in ALL_CLASSES order is glass's
    # slot, not metal's -- silently printed glass's constant precision/
    # recall as if it were metal's for the entire sweep (0.571/0.818,
    # matching glass's real baseline numbers exactly). Same bug class as
    # the cardboard floor script's earlier CB_IDX mixup (NOTES.md
    # section 64) -- macro_f1 itself was never affected by this, only the
    # metal-specific display columns were wrong.
    metal_all_idx = CLASS_TO_IDX["metal"]

    print(f"\nSweeping metal floor over {FLOOR_CANDIDATES} (cardboard floor held at {CARDBOARD_FLOOR})...")
    results = []
    for floor in FLOOR_CANDIDATES:
        m, f, u = evaluate(floor)
        # Verify the "no other class moves" claim directly -- this time
        # excluding metal (being swept) AND cardboard (already floored,
        # unaffected by metal's floor since they're independent argmax
        # winners, but confirmed anyway rather than assumed).
        other_classes_unchanged = all(
            abs(m["precision"][CLASS_TO_IDX[c]] - m0["precision"][CLASS_TO_IDX[c]]) < 1e-9 and
            abs(m["recall"][CLASS_TO_IDX[c]] - m0["recall"][CLASS_TO_IDX[c]]) < 1e-9
            for c in ["organic", "cardboard", "ewaste", "glass", "paper", "plastic"]
        )
        results.append((floor, f["macro_f1"], m["precision"][metal_all_idx], m["recall"][metal_all_idx], u, other_classes_unchanged))

    print(f"\n{'floor':>7} {'macro_f1':>10} {'metal_precision':>16} {'metal_recall':>13} "
          f"{'n_unknown':>10} {'others_untouched':>18}")
    for floor, mf, mp, mr, u, untouched in results:
        print(f"{floor:>7.2f} {mf:>10.4f} {mp:>16.3f} {mr:>13.3f} {u:>10d} {str(untouched):>18}")

    best = max(results, key=lambda r: r[1])
    print(f"\nRecommended: metal_floor={best[0]} "
          f"(metal precision {m0['precision'][metal_all_idx]:.3f} -> {best[2]:.3f}, "
          f"recall {m0['recall'][metal_all_idx]:.3f} -> {best[3]:.3f}, "
          f"macro-F1={best[1]:.4f} vs baseline {f0['macro_f1']:.4f}, "
          f"other classes untouched={best[5]})")
    m_best, f_best, u_best = evaluate(best[0])
    print_full_table(m_best, f_best["macro_f1"], f_best["weighted_f1"], u_best, len(all_paths),
                      f"recommended (metal_floor={best[0]})")

    os.makedirs("outputs", exist_ok=True)
    with open("outputs/metal_floor_calibration_removedbadapples_v2.json", "w") as f:
        json.dump({
            "recommended_metal_floor": best[0],
            "cardboard_floor_held_at": CARDBOARD_FLOOR,
            "baseline_metal_precision": float(m0["precision"][metal_all_idx]),
            "recommended_metal_precision": best[2],
            "baseline_metal_recall": float(m0["recall"][metal_all_idx]),
            "recommended_metal_recall": best[3],
            "recommended_macro_f1": best[1],
            "baseline_macro_f1": f0["macro_f1"],
            "n_unknown": best[4],
            "other_classes_verified_untouched": best[5],
            "stage1_cutoff": STAGE1_CUTOFF,
            "stage2_threshold": STAGE2_THRESHOLD,
        }, f, indent=2)
    print("\nWrote outputs/metal_floor_calibration_removedbadapples_v2.json")


if __name__ == "__main__":
    main()
