"""
NOTES.md section 62 follow-up -- the confusion matrix from the live
"its worse!" test pinpoints WHERE plastic recall is being lost, and it's
not primarily Stage 2's threshold:

  balanced mode:    true plastic (31) -> organic=9, unknown=10, plastic=2
  high recall mode: true plastic (38) -> organic=11, cardboard=6, unknown=4, plastic=5

Stage 1 (the organic-vs-recyclable gate) is misrouting roughly a third of
all live plastic straight to "organic" before it ever reaches Stage 2 --
that's the single biggest leak, bigger than the unknown-routing or any
Stage-2 material confusion. The user also reports the OLDER "focal" and
"tacotrashnet" checkpoint options score ~70 F1 live, well above
removedbadapples's 58%. Checking webapp/app.py: those older options don't
just use a different Stage 2 head -- they fall through to a DIFFERENT,
older Stage 1 model entirely (outputs/stage1_mobilenet.pt, the pre-
cleanup checkpoint) at cutoff 0.65, vs. removedbadapples's retrained
Stage 1 (outputs/stage1_mobilenet_removedbadapples.pt) at cutoff 0.50.
Two variables changed at once (which model + which cutoff), so which one
actually causes the leak is still unconfirmed.

This script isolates the cutoff variable using data we already have:
runs BOTH Stage 1 checkpoints (old default vs. new removedbadapples) over
every plastic-labeled val image, and for a sweep of cutoffs reports what
fraction of true plastic gets prob_organic >= cutoff (i.e. leaked to
"organic" before Stage 2 ever sees it). This is a much more direct,
class-specific diagnostic than the macro-F1 grid search in
calibrate_thresholds_removedbadapples.py, which averages over all 8
classes and can hide a leak this concentrated in one class.

Run on your Mac:

    cd pytorch_pipeline
    python3 scripts/diagnose_stage1_plastic_leak.py
"""
import os
import sys
import json

import torch
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.models import get_model_and_transforms

DATA_ROOT = "data/cascades_data"
STAGE1_SPLIT = os.path.join(DATA_ROOT, "stage1_split.json")
STAGE2_SPLIT = os.path.join(DATA_ROOT, "stage2_split.json")

CUTOFF_SWEEP = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]

MODELS = [
    ("OLD default (used by focal/tacotrashnet)", "outputs/stage1_mobilenet.pt", 0.65),
    ("NEW removedbadapples (current default)", "outputs/stage1_mobilenet_removedbadapples.pt", 0.50),
]


def load_model(ckpt_path, device):
    model, _, eval_tf = get_model_and_transforms("mobilenet_v3_large", 2, pretrained=False)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.to(device)
    model.eval()
    return model, eval_tf


@torch.no_grad()
def get_organic_probs(model, transform, paths, device, batch_size=32):
    probs = []
    batch = []
    for i, p in enumerate(paths):
        img = Image.open(p).convert("RGB")
        batch.append(transform(img))
        if len(batch) == batch_size or i == len(paths) - 1:
            x = torch.stack(batch).to(device)
            logits = model(x)
            probs.append(torch.softmax(logits, dim=1)[:, 0].cpu().numpy())
            batch = []
    return np.concatenate(probs, axis=0)


def main():
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    print(f"Device: {device}\n")

    with open(STAGE2_SPLIT) as f:
        s2 = json.load(f)

    plastic_val_paths = [p for p, label in s2["val"] if label == "plastic"]
    plastic_test_paths = [p for p, label in s2["test"] if label == "plastic"]
    print(f"True plastic images: {len(plastic_val_paths)} val, {len(plastic_test_paths)} test\n")

    all_plastic = plastic_val_paths + plastic_test_paths

    for name, ckpt_path, current_cutoff in MODELS:
        if not os.path.exists(ckpt_path):
            print(f"SKIP ({name}): {ckpt_path} not found\n")
            continue

        print("=" * 70)
        print(f"{name}\n  checkpoint: {ckpt_path}\n  currently live at cutoff: {current_cutoff}")
        print("=" * 70)

        model, transform = load_model(ckpt_path, device)
        probs_organic = get_organic_probs(model, transform, all_plastic, device)

        print(f"\nprob_organic stats over {len(all_plastic)} true-plastic images: "
              f"mean={probs_organic.mean():.3f}, median={np.median(probs_organic):.3f}, "
              f"max={probs_organic.max():.3f}")

        print(f"\n{'cutoff':>8} {'n_leaked_to_organic':>20} {'pct_leaked':>12}")
        for cutoff in CUTOFF_SWEEP:
            leaked = int((probs_organic >= cutoff).sum())
            pct = 100.0 * leaked / len(all_plastic)
            marker = "  <-- currently live" if abs(cutoff - current_cutoff) < 1e-9 else ""
            print(f"{cutoff:>8.2f} {leaked:>20d} {pct:>11.1f}%{marker}")
        print()

    print("If the NEW model's leak rate is high even at cutoff=0.65 (same cutoff")
    print("the OLD model uses successfully), the retrained Stage 1 model itself")
    print("is worse at telling plastic apart from organic -- a real model")
    print("regression, not a threshold problem, and would need Stage 1 retrained")
    print("or the old Stage 1 checkpoint kept in the removedbadapples pairing.")
    print("If the NEW model's leak rate at 0.65 drops close to the OLD model's")
    print("current leak rate, it's a threshold-only fix: raise")
    print("STAGE1_CUTOFF_REMOVEDBADAPPLES in webapp/app.py to 0.65 and retest live.")


if __name__ == "__main__":
    main()
