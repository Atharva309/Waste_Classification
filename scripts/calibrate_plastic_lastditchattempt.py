"""
NOTES.md section 77. Live testing this morning showed plastic with a
completely different problem shape than cardboard/metal ever had:
consistently HIGH precision (71-87% across 3 live runs) and LOW recall
(33-37%). Cardboard/metal were "sinks" (low precision, high recall,
absorbing other classes) -- fixed with a confidence FLOOR that demotes
low-confidence winners to "unknown" (trades recall for precision).
Plastic needs the opposite: something that RECOVERS recall without
spending precision, since precision is already good.

Two mechanisms tested here, evaluated independently against the real
deployed config (old Stage 1 @ 0.65 cutoff + lastditchattempt Stage 2 +
cardboard floor @ 0.65 already active, matching production exactly):

1. Plastic "rescue" threshold: if plastic wins the argmax but its score
   falls between [rescue_threshold, STAGE2_THRESHOLD) -- i.e. currently
   routed to "unknown" -- accept it as plastic anyway. Mathematically
   isolated the same way the cardboard floor was proven isolated
   (NOTES.md section 64), just inverted: it only ever recovers items
   nobody else was going to claim (they were headed to "unknown", not to
   some other real class), so it cannot move any OTHER class's precision
   or recall. Verified directly below, not just asserted.

2. Plastic logit bias: adds a constant to plastic's raw logit before
   softmax, so it wins more borderline cases it's currently narrowly
   losing to metal/cardboard. NOT guaranteed isolated -- this is the same
   mechanism that hurt glass when tried on cardboard (section 63 round 1).
   Reported honestly with full per-class movement, not assumed safe.

Run on your Mac:

    cd pytorch_pipeline
    python3 scripts/calibrate_plastic_lastditchattempt.py
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
STAGE2_CKPT = "outputs/lastditchattempt.pt"
STAGE1_CUTOFF = 0.65
STAGE2_THRESHOLD = 0.30   # global "unknown" threshold, held fixed
CARDBOARD_FLOOR = 0.65    # already deployed, held fixed -- measuring the real config

STAGE2_CLASSES = ["cardboard", "ewaste", "glass", "metal", "paper", "plastic"]
ALL_CLASSES = ["organic", "cardboard", "ewaste", "glass", "metal", "paper", "plastic", "unknown"]
CLASS_TO_IDX = {c: i for i, c in enumerate(ALL_CLASSES)}
CB_IDX = STAGE2_CLASSES.index("cardboard")
PLASTIC_IDX = STAGE2_CLASSES.index("plastic")

RESCUE_CANDIDATES = [round(x, 2) for x in np.arange(0.30, 0.04, -0.02)]
BIAS_CANDIDATES = [round(x, 2) for x in np.arange(-0.25, 1.55, 0.15)]

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
    logits2_base = run_inference_logits(model2, transform2, all_paths, device)

    true_idx = np.array([CLASS_TO_IDX[l] for l in true_labels])
    is_organic_pred = probs1[:, 0] >= STAGE1_CUTOFF
    not_organic = ~is_organic_pred

    def evaluate(plastic_bias=0.0, plastic_rescue=STAGE2_THRESHOLD):
        logits2 = logits2_base.copy()
        logits2[:, PLASTIC_IDX] += plastic_bias
        probs2 = softmax_np(logits2)
        top_idx = probs2.argmax(axis=1)
        top_prob = probs2.max(axis=1)

        pred_idx = np.empty(len(all_paths), dtype=np.int64)
        pred_idx[is_organic_pred] = CLASS_TO_IDX["organic"]

        for i in np.where(not_organic)[0]:
            cls_i = top_idx[i]
            p = top_prob[i]
            if cls_i == CB_IDX and p < CARDBOARD_FLOOR:
                pred_idx[i] = CLASS_TO_IDX["unknown"]
            elif cls_i == PLASTIC_IDX:
                # Plastic's own effective bar: the lower rescue value
                # instead of the global threshold. Still routes to
                # unknown below THAT.
                pred_idx[i] = CLASS_TO_IDX["plastic"] if p >= plastic_rescue else CLASS_TO_IDX["unknown"]
            elif p >= STAGE2_THRESHOLD:
                pred_idx[i] = CLASS_TO_IDX[STAGE2_CLASSES[cls_i]]
            else:
                pred_idx[i] = CLASS_TO_IDX["unknown"]

        m = per_class_precision_recall_f1(true_idx, pred_idx, len(ALL_CLASSES))
        f1_stats = macro_and_weighted_f1(m)
        n_unknown = int((pred_idx == CLASS_TO_IDX["unknown"]).sum())
        return m, f1_stats, n_unknown

    print("\nBaseline (no plastic rescue, no plastic bias -- current deployed config):")
    m0, f0, u0 = evaluate(plastic_bias=0.0, plastic_rescue=STAGE2_THRESHOLD)
    print_full_table(m0, f0["macro_f1"], f0["weighted_f1"], u0, len(all_paths), "baseline")

    # --- Mechanism 1: plastic rescue threshold ---
    print(f"\n=== Sweeping plastic RESCUE threshold over {RESCUE_CANDIDATES} "
          f"(bias=0, provably isolated to plastic's own numbers) ===")
    rescue_results = []
    plastic_all_idx = CLASS_TO_IDX["plastic"]
    for rescue in RESCUE_CANDIDATES:
        m, f, u = evaluate(plastic_bias=0.0, plastic_rescue=rescue)
        pl_p, pl_r, pl_f1 = m["precision"][plastic_all_idx], m["recall"][plastic_all_idx], m["f1"][plastic_all_idx]
        others_unchanged = all(
            abs(m["precision"][CLASS_TO_IDX[c]] - m0["precision"][CLASS_TO_IDX[c]]) < 1e-9 and
            abs(m["recall"][CLASS_TO_IDX[c]] - m0["recall"][CLASS_TO_IDX[c]]) < 1e-9
            for c in ["organic", "cardboard", "ewaste", "glass", "metal", "paper"]
        )
        rescue_results.append((rescue, f["macro_f1"], pl_p, pl_r, pl_f1, u, others_unchanged))

    print(f"\n{'rescue':>7} {'macro_f1':>10} {'plastic_P':>10} {'plastic_R':>10} "
          f"{'plastic_F1':>11} {'n_unknown':>10} {'others_untouched':>18}")
    for rescue, mf, pp, pr, pf1, u, untouched in rescue_results:
        print(f"{rescue:>7.2f} {mf:>10.4f} {pp:>10.3f} {pr:>10.3f} {pf1:>11.3f} {u:>10d} {str(untouched):>18}")

    best_rescue = max(rescue_results, key=lambda r: r[4])  # best plastic F1, not macro-F1 -- this is a plastic-specific fix
    print(f"\nBest plastic F1 via rescue: rescue={best_rescue[0]} "
          f"(plastic P {m0['precision'][plastic_all_idx]:.3f} -> {best_rescue[2]:.3f}, "
          f"R {m0['recall'][plastic_all_idx]:.3f} -> {best_rescue[3]:.3f}, "
          f"F1 {m0['f1'][plastic_all_idx]:.3f} -> {best_rescue[4]:.3f}, "
          f"others_untouched={best_rescue[6]})")

    # --- Mechanism 2: plastic logit bias (NOT guaranteed isolated) ---
    print(f"\n=== Sweeping plastic BIAS over {BIAS_CANDIDATES} "
          f"(rescue=threshold i.e. off, full per-class movement reported honestly) ===")
    bias_results = []
    for bias in BIAS_CANDIDATES:
        m, f, u = evaluate(plastic_bias=bias, plastic_rescue=STAGE2_THRESHOLD)
        pl_p, pl_r, pl_f1 = m["precision"][plastic_all_idx], m["recall"][plastic_all_idx], m["f1"][plastic_all_idx]
        bias_results.append((bias, f["macro_f1"], pl_p, pl_r, pl_f1, u, m))

    print(f"\n{'bias':>7} {'macro_f1':>10} {'plastic_P':>10} {'plastic_R':>10} {'plastic_F1':>11} {'n_unknown':>10}")
    for bias, mf, pp, pr, pf1, u, _ in bias_results:
        print(f"{bias:>7.2f} {mf:>10.4f} {pp:>10.3f} {pr:>10.3f} {pf1:>11.3f} {u:>10d}")

    best_bias = max(bias_results, key=lambda r: r[4])
    print(f"\nBest plastic F1 via bias: bias={best_bias[0]} "
          f"(plastic P {m0['precision'][plastic_all_idx]:.3f} -> {best_bias[2]:.3f}, "
          f"R {m0['recall'][plastic_all_idx]:.3f} -> {best_bias[3]:.3f}, "
          f"F1 {m0['f1'][plastic_all_idx]:.3f} -> {best_bias[4]:.3f})")
    print("Full per-class table at that bias value (checking what it actually costs elsewhere):")
    print_full_table(best_bias[6], best_bias[1], macro_and_weighted_f1(best_bias[6])["weighted_f1"],
                      best_bias[5], len(all_paths), f"bias={best_bias[0]}")

    os.makedirs("outputs", exist_ok=True)
    with open("outputs/plastic_calibration_lastditchattempt.json", "w") as f:
        json.dump({
            "baseline_plastic_precision": float(m0["precision"][plastic_all_idx]),
            "baseline_plastic_recall": float(m0["recall"][plastic_all_idx]),
            "baseline_plastic_f1": float(m0["f1"][plastic_all_idx]),
            "baseline_macro_f1": f0["macro_f1"],
            "best_rescue_threshold": best_rescue[0],
            "best_rescue_plastic_f1": best_rescue[4],
            "best_rescue_others_untouched": best_rescue[6],
            "best_bias": best_bias[0],
            "best_bias_plastic_f1": best_bias[4],
            "best_bias_macro_f1": best_bias[1],
            "stage1_cutoff": STAGE1_CUTOFF,
            "stage2_threshold": STAGE2_THRESHOLD,
            "cardboard_floor": CARDBOARD_FLOOR,
        }, f, indent=2)
    print("\nWrote outputs/plastic_calibration_lastditchattempt.json")


if __name__ == "__main__":
    main()
