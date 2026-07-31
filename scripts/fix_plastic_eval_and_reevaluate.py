"""
NOTES.md section 57 -- targeted fix, no full retrain needed.

generate_belt_dataset.py's plastic hard-exclusion (section 51) was only
gated to split_name == "train" -- val/test plastic still drew from the
unaudited generic-Kaggle pool (61% of val, 66% of test, checked directly
against outputs/stage2_split.json). The first removedbadapples cascade
run (trained on clean data, evaluated against a still-contaminated test
set) scored plastic recall 0.49 (down from 0.83 pre-cleanup) and every
other class's precision dropped too -- consistent with the model being
correctly evaluated against WRONG ground-truth labels, not a real
regression from cleaner training data.

generate_belt_dataset.py's hard-exclude block has since been fixed to
apply to every split. This script does NOT re-run that full ~80-minute
training -- the already-trained checkpoint
(outputs/stage2_material_mobilenet_removedbadapples.pt) doesn't need to
change just because the TEST set was wrong. Instead:

  1. Regenerates ONLY data/cascades_data/stage2/{val,test}/plastic/ using
     the now-fixed (all-splits) hard-exclude logic, deleting the old
     (partially contaminated) files first.
  2. Patches data/cascades_data/stage2_split.json in place: replaces only
     the plastic entries in val/test, leaves train and every other class
     in every split completely untouched.
  3. Re-evaluates the ALREADY-TRAINED checkpoint against the corrected
     test set and prints the same metrics format train_belt_cascade.py
     does, so the corrected number is directly comparable.

Run on your Mac, from pytorch_pipeline/ (after train_belt_cascade.py has
already produced outputs/stage2_material_mobilenet_removedbadapples.pt):

    python3 scripts/fix_plastic_eval_and_reevaluate.py
"""
import os
import sys
import json
import shutil

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from composite_preview import load_belt, filter_known_bad
from generate_belt_dataset import generate_material_class, make_class_dir, OUT_ROOT

import torch
from torch.utils.data import DataLoader
from src.datasets import WasteImageDataset
from src.models import get_model_and_transforms
from src.engine import evaluate
from src.metrics import confusion_matrix

STAGE2_SPLIT_SRC = "outputs/stage2_split.json"
CASCADE_SPLIT = os.path.join(OUT_ROOT, "stage2_split.json")
CHECKPOINT = "outputs/stage2_material_mobilenet_removedbadapples.pt"
CLASSES = ["cardboard", "ewaste", "glass", "metal", "paper", "plastic"]


def regenerate_plastic_val_test():
    print("=" * 60)
    print("Step 1/2: regenerating stage2/{val,test}/plastic with the")
    print("now-fixed (all-splits) hard-exclude logic")
    print("=" * 60)

    with open(STAGE2_SPLIT_SRC) as f:
        s2 = json.load(f)
    for split_name in ["train", "val", "test"]:
        s2[split_name] = filter_known_bad(s2[split_name])

    with open(CASCADE_SPLIT) as f:
        cascade_manifest = json.load(f)

    belt_im = load_belt()

    for split_name in ["val", "test"]:
        out_dir = make_class_dir("stage2", split_name, "plastic")
        # Wipe the old (partially generic-Kaggle) files before rewriting --
        # filenames are index-based (plastic_taco_000123.png etc.), so a
        # stale leftover from the old, larger, contaminated pool could
        # otherwise survive alongside the new ones.
        if os.path.isdir(out_dir):
            shutil.rmtree(out_dir)
        os.makedirs(out_dir, exist_ok=True)

        paths = [p for p, label in s2[split_name] if label == "plastic"]
        new_entries = []
        written, failed = generate_material_class(
            "plastic", split_name, paths, belt_im, out_dir, new_entries, "plastic"
        )
        print(f"stage2/{split_name}/plastic: wrote {written}, failed/skipped {failed} "
              f"(source images: {len(paths)})")

        # Replace ONLY the plastic entries for this split in the manifest;
        # every other class, and every other split, is left byte-for-byte
        # untouched.
        kept = [[p, l] for p, l in cascade_manifest[split_name] if l != "plastic"]
        cascade_manifest[split_name] = kept + new_entries

    with open(CASCADE_SPLIT, "w") as f:
        json.dump(cascade_manifest, f)
    print(f"Patched {CASCADE_SPLIT} (val/test plastic entries only)")

    return cascade_manifest


def reevaluate(cascade_manifest):
    print()
    print("=" * 60)
    print("Step 2/2: re-evaluating the ALREADY-TRAINED checkpoint against")
    print("the corrected test set (no retraining)")
    print("=" * 60)

    if not os.path.exists(CHECKPOINT):
        print(f"ERROR: {CHECKPOINT} not found -- train_belt_cascade.py "
              f"must finish first.")
        sys.exit(1)

    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    class_to_idx = {c: i for i, c in enumerate(CLASSES)}
    num_classes = len(CLASSES)

    model, _, eval_tf = get_model_and_transforms(
        "mobilenet_v3_large", num_classes, pretrained=False, is_stage2=True
    )
    model.load_state_dict(torch.load(CHECKPOINT, map_location=device))
    model.to(device)
    model.eval()

    for split_name in ["val", "test"]:
        dataset = WasteImageDataset(cascade_manifest[split_name], class_to_idx, transform=eval_tf)
        loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)
        metrics = evaluate(model, loader, device, num_classes)

        print(f"\n--- Corrected {split_name.upper()} set (clean plastic ground truth) ---")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"Macro-F1: {metrics['macro_f1']:.4f}")
        print(f"Weighted-F1: {metrics['weighted_f1']:.4f}")
        print("Per-Class Metrics:")
        for i, c in enumerate(CLASSES):
            print(f"  {c.ljust(12)}: P={metrics['precision'][i]:.4f} "
                  f"R={metrics['recall'][i]:.4f} F1={metrics['f1'][i]:.4f}")

        if split_name == "test":
            cm = confusion_matrix(metrics["y_true"], metrics["y_pred"], num_classes)
            print("\nConfusion matrix (rows=true, cols=pred):", CLASSES)
            for i, row in enumerate(cm.tolist()):
                print(f"  {CLASSES[i].ljust(12)}: {row}")


def main():
    cascade_manifest = regenerate_plastic_val_test()
    reevaluate(cascade_manifest)


if __name__ == "__main__":
    main()
