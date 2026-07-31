"""
Train both cascade stages on the new belt-composited dataset
(data/cascades_data/, produced by scripts/generate_belt_dataset.py).

Adapted from scripts/retrain_cascade.py, with three deliberate differences:

  1. Reads data/cascades_data/stage{1,2}_split.json instead of the live
     outputs/stage{1,2}_split.json -- this is a full retrain on the new
     domain-matched data, not a quick inference-time comparison.
  2. No artificial compute budget. The original script capped itself at 9
     minutes total (5 for Stage 1) for a fast iteration loop; this dataset
     is roughly 5-6x larger and deserves a real training run. Uses epoch
     caps with early stopping (patience) instead of a wall-clock cutoff, so
     it stops when validation stops improving rather than mid-epoch.
  3. Saves to NEW checkpoint filenames -- never touches the live
     outputs/stage1_mobilenet.pt or outputs/stage2_material_mobilenet_*.pt.
     Per this project's standing rule, checkpoints are never overwritten in
     place.

Stage 2 class weighting: the new belt-composited dataset is still
imbalanced (plastic has the most source images of every material class),
but far less skewed than the raw live sprite pool was. A previous retrain
attempt (documented in NOTES.md) used naive linear inverse-frequency
weighting and over-corrected metal (the smallest class) into being
over-predicted. This run uses a square-root-dampened inverse-frequency
weight, capped at a max ratio, specifically to avoid repeating that
failure while still giving the smaller classes some help.

Run on your Mac after scripts/generate_belt_dataset.py has finished:

    cd pytorch_pipeline
    python3 scripts/train_belt_cascade.py
"""

import os
import sys
import json
import time
import copy
from collections import Counter

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.datasets import WasteImageDataset
from src.models import get_model_and_transforms, freeze_backbone, unfreeze_all
from src.engine import train_one_epoch, evaluate
from src.metrics import confusion_matrix

DATA_ROOT = "data/cascades_data"
STAGE1_SPLIT = os.path.join(DATA_ROOT, "stage1_split.json")
STAGE2_SPLIT = os.path.join(DATA_ROOT, "stage2_split.json")

STAGE1_OUT = "outputs/stage1_mobilenet_removedbadapples_v2.pt"
STAGE2_OUT = "outputs/stage2_material_mobilenet_removedbadapples_v2.pt"
REPORT_OUT = "outputs/training_report_removedbadapples_v2.json"
# _v2 (NOTES.md section 62 follow-up, augment_level="heavy" fix above) --
# NOT the same filenames as the currently-live removedbadapples checkpoints.
# Standing project rule: never overwrite a checkpoint in place. Keeps the
# current 0.8630-macro-F1 checkpoint around as a comparison baseline.

HEAD_EPOCHS_MAX = 25
FINETUNE_EPOCHS_MAX = 15
PATIENCE = 5  # stop a phase early if val macro-F1 hasn't improved in this many epochs
FINETUNE_IF_BELOW = 0.90
BATCH_SIZE = 32


def dampened_class_weights(labels, classes, dampen=0.5, max_ratio=3.0):
    """sqrt-dampened inverse-frequency weighting, capped at max_ratio between
    the largest and smallest weight -- deliberately gentler than plain
    inverse-frequency (see module docstring: a previous attempt at plain
    inverse-frequency weighting over-corrected the smallest class)."""
    counts = Counter(labels)
    total = len(labels)
    n = len(classes)
    raw = [ (total / (n * counts.get(c, 1))) ** dampen for c in classes ]
    lo, hi = min(raw), max(raw)
    if hi / max(lo, 1e-9) > max_ratio:
        # rescale so max/min == max_ratio, keeping the smallest weight at 1.0
        scale = (max_ratio - 1.0) / (hi - lo) if hi > lo else 0.0
        raw = [1.0 + (w - lo) * scale for w in raw]
    return raw


def run_stage(
    stage_name, split_path, classes, output_path,
    is_stage2, global_start,
):
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    print(f"\n{'='*60}\nStarting {stage_name} (Device: {device})\n{'='*60}")

    with open(split_path) as f:
        split_res = json.load(f)

    class_to_idx = {c: i for i, c in enumerate(classes)}
    num_classes = len(classes)

    train_labels = [item[1] for item in split_res["train"]]
    print(f"Train size: {len(split_res['train'])}, Val size: {len(split_res['val'])}, "
          f"Test size: {len(split_res['test'])}")
    print("Train class counts:", dict(Counter(train_labels)))

    if is_stage2:
        weights = dampened_class_weights(train_labels, classes, dampen=0.5, max_ratio=3.0)
    else:
        weights = dampened_class_weights(train_labels, classes, dampen=1.0, max_ratio=2.0)
    print("Class weights:", dict(zip(classes, [round(w, 3) for w in weights])))
    class_weights_tensor = torch.tensor(weights, dtype=torch.float32).to(device)

    smoothing = 0.1 if is_stage2 else 0.0
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights_tensor, label_smoothing=smoothing)

    print(f"Loading MobileNetV3-Large for {num_classes} classes...")
    # NOTES.md section 62 follow-up: FOCAL/TACOTRASHNET (the two checkpoints
    # that generalize well live, ~70 F1) both explicitly pass
    # augment_level="heavy" for Stage 2 (src/models.py: adds ColorJitter
    # hue=0.1 + RandomErasing(p=0.3) on top of the "gentle" preset). This
    # script never set augment_level at all, so it silently trained Stage 2
    # with "gentle" -- no hue jitter, no RandomErasing -- an omission, not a
    # deliberate choice (this file's own docstring lists three intentional
    # differences from retrain_cascade.py and augmentation isn't one of
    # them). RandomErasing specifically trains tolerance to partial
    # occlusion/cropping, which live belt crops have far more of than clean
    # product photos -- a plausible real contributor to the live vs. offline
    # gap this whole diagnosis has been chasing.
    model, train_tf, eval_tf = get_model_and_transforms(
        "mobilenet_v3_large", num_classes, pretrained=True, is_stage2=is_stage2,
        augment_level="heavy" if is_stage2 else "gentle"
    )

    train_dataset = WasteImageDataset(split_res["train"], class_to_idx, transform=train_tf)
    val_dataset = WasteImageDataset(split_res["val"], class_to_idx, transform=eval_tf)
    test_dataset = WasteImageDataset(split_res["test"], class_to_idx, transform=eval_tf)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model.to(device)

    # ---- Phase 1: head-only ----
    freeze_backbone(model)
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable, lr=1e-3)

    best_f1 = 0.0
    best_state = None
    best_epoch = -1
    epochs_since_improve = 0
    epoch = 0

    print("\n--- Phase 1: Head-Only Training ---")
    while epoch < HEAD_EPOCHS_MAX:
        epoch += 1
        t0 = time.time()
        loss = train_one_epoch(model, train_loader, optimizer, device, criterion=criterion)
        val_metrics = evaluate(model, val_loader, device, num_classes)
        f1, acc = val_metrics["macro_f1"], val_metrics["accuracy"]
        print(f"Epoch {epoch} [{time.time()-t0:.1f}s]: loss={loss:.4f} "
              f"val_macro_f1={f1:.4f} val_acc={acc:.4f} | total elapsed {time.time()-global_start:.0f}s")

        if f1 > best_f1:
            print(f"  -> new best ({best_f1:.4f} -> {f1:.4f})")
            best_f1 = f1
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            epochs_since_improve = 0
        else:
            epochs_since_improve += 1
            if epochs_since_improve >= PATIENCE:
                print(f"  -> no improvement in {PATIENCE} epochs, stopping Phase 1 early")
                break

    # ---- Phase 2: fine-tune (only if phase 1 wasn't already strong) ----
    if best_f1 < FINETUNE_IF_BELOW:
        print(f"\nPhase 1 best macro-F1 ({best_f1:.4f}) < {FINETUNE_IF_BELOW}. Starting Phase 2 fine-tuning.")
        if best_state is not None:
            model.load_state_dict(best_state)

        unfreeze_all(model, unfreeze_from_layer="features.14")
        trainable = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(trainable, lr=1e-5)

        epochs_since_improve = 0
        ft_epoch = 0
        while ft_epoch < FINETUNE_EPOCHS_MAX:
            ft_epoch += 1
            epoch += 1
            t0 = time.time()
            loss = train_one_epoch(model, train_loader, optimizer, device, criterion=criterion)
            val_metrics = evaluate(model, val_loader, device, num_classes)
            f1, acc = val_metrics["macro_f1"], val_metrics["accuracy"]
            print(f"Epoch {epoch} (FT) [{time.time()-t0:.1f}s]: loss={loss:.4f} "
                  f"val_macro_f1={f1:.4f} val_acc={acc:.4f} | total elapsed {time.time()-global_start:.0f}s")

            if f1 > best_f1:
                print(f"  -> new best ({best_f1:.4f} -> {f1:.4f})")
                best_f1 = f1
                best_state = copy.deepcopy(model.state_dict())
                best_epoch = epoch
                epochs_since_improve = 0
            else:
                epochs_since_improve += 1
                if epochs_since_improve >= PATIENCE:
                    print(f"  -> no improvement in {PATIENCE} epochs, stopping Phase 2 early")
                    break
    else:
        print(f"\nSkipping Phase 2: Phase 1 best macro-F1 ({best_f1:.4f}) >= {FINETUNE_IF_BELOW}.")

    print(f"\n{stage_name} training finished.")
    if best_state is not None:
        print(f"Restoring best model from epoch {best_epoch} (val_macro_f1={best_f1:.4f}).")
        model.load_state_dict(best_state)
    torch.save(model.state_dict(), output_path)
    print(f"Saved to {output_path} (new file -- live checkpoints untouched).")

    print(f"\n--- {stage_name} Final Test Set Evaluation ---")
    test_metrics = evaluate(model, test_loader, device, num_classes)
    print(f"Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"Macro-F1: {test_metrics['macro_f1']:.4f}")
    print(f"Weighted-F1: {test_metrics['weighted_f1']:.4f}")
    print("\nPer-Class Metrics:")
    for i, c in enumerate(classes):
        print(f"  {c.ljust(12)}: P={test_metrics['precision'][i]:.4f} "
              f"R={test_metrics['recall'][i]:.4f} F1={test_metrics['f1'][i]:.4f}")

    cm = confusion_matrix(test_metrics["y_true"], test_metrics["y_pred"], num_classes)

    return {
        "accuracy": test_metrics["accuracy"],
        "macro_f1": test_metrics["macro_f1"],
        "weighted_f1": test_metrics["weighted_f1"],
        "confusion_matrix": cm.tolist(),
        "classes": classes,
        "per_class": {
            classes[i]: {
                "precision": test_metrics["precision"][i],
                "recall": test_metrics["recall"][i],
                "f1": test_metrics["f1"][i],
            } for i in range(num_classes)
        },
        "best_epoch": best_epoch,
        "class_weights": dict(zip(classes, weights)),
    }


def main():
    if not os.path.exists(STAGE1_SPLIT) or not os.path.exists(STAGE2_SPLIT):
        print(f"ERROR: {STAGE1_SPLIT} / {STAGE2_SPLIT} not found. "
              f"Run scripts/generate_belt_dataset.py first.")
        sys.exit(1)

    global_start = time.time()

    stage1_metrics = run_stage(
        stage_name="Stage 1 (organic vs recyclable, belt-composited)",
        split_path=STAGE1_SPLIT,
        classes=["organic", "recyclable"],
        output_path=STAGE1_OUT,
        is_stage2=False,
        global_start=global_start,
    )

    stage2_metrics = run_stage(
        stage_name="Stage 2 (6 material classes, belt-composited)",
        split_path=STAGE2_SPLIT,
        classes=["cardboard", "ewaste", "glass", "metal", "paper", "plastic"],
        output_path=STAGE2_OUT,
        is_stage2=True,
        global_start=global_start,
    )

    total_time = time.time() - global_start
    print(f"\n{'='*60}\nAll training complete in {total_time/60:.1f} min\n{'='*60}")

    report = {
        "stage1_mobilenet_removedbadapples": stage1_metrics,
        "stage2_mobilenet_removedbadapples": stage2_metrics,
        "total_wall_clock_time_s": total_time,
        "data_root": DATA_ROOT,
    }
    with open(REPORT_OUT, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Wrote {REPORT_OUT}")


if __name__ == "__main__":
    main()
