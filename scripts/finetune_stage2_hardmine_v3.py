"""
Hard-example fine-tune of the current best Stage 2 checkpoint
(outputs/stage2_material_mobilenet_removedbadapples_v2.pt), NOT a
from-scratch retrain -- warm-starts from that checkpoint's own weights.

Why this is a different lever than "just train longer": the checkpoint
already converged on its training data (early stopping fired on a
macro-F1 plateau) -- more epochs on the SAME data distribution teaches it
nothing new. This instead reweights training TOWARD the specific images
it's already getting wrong or is unsure about (plastic<->cardboard,
plastic<->metal, glass<->metal -- the confusions found in tonight's live
confusion matrices), without introducing new data.

Two failure modes this project already hit once tonight, guarded against
explicitly:
  1. Naive reweighting over-corrects (NOTES.md section 15: plain inverse-
     frequency class weighting fixed metal's recall but tanked its
     precision by making the model over-predict metal broadly). Guard:
     hard examples are oversampled by INCLUSION (all of them go in every
     epoch), not by aggressive duplication multipliers, and are mixed
     with a comparable-sized random sample of ordinary ("easy") examples
     every epoch -- so the model keeps seeing the full distribution, not
     just its own mistakes.
  2. Overfitting to a small targeted subset very fast, especially with a
     warm-started (already-converged) model. Guard: near-frozen backbone
     (only unfreezes from the same features.14 layer the original
     Phase 2 fine-tune used, at the same 1e-5 LR), a hard epoch cap (10),
     early stopping on a real validation signal, and -- the important
     part -- validated and best-epoch-selected on the FULL val set's
     macro-F1 (all 6 classes), not hard-example accuracy specifically.
     If fine-tuning on the hard subset makes the FULL val set worse, that
     epoch will never be selected as best.

Hard example selection: runs the CURRENT best checkpoint (paired with the
same old Stage 1 the live webapp actually uses) over the full Stage 2
TRAIN split (not val -- val must stay held-out for honest early
stopping), and flags an image as "hard" if it's either (a) misclassified,
or (b) correctly classified but with top-class probability below
HARD_EXAMPLE_CONFIDENCE_THRESHOLD (borderline/uncertain calls -- standard
hard-example-mining practice, not just outright errors).

Saves to a NEW checkpoint name -- never overwrites the current best or
any other existing checkpoint.

Run on your Mac (safe to chain after train_evaluate_yolo.py or run
standalone):

    cd pytorch_pipeline
    python3 scripts/finetune_stage2_hardmine_v3.py
"""
import os
import sys
import json
import time
import copy
import random
from collections import Counter

import torch
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.datasets import WasteImageDataset
from src.models import get_model_and_transforms, unfreeze_all
from src.engine import train_one_epoch, evaluate
from src.metrics import confusion_matrix

DATA_ROOT = "data/cascades_data"
STAGE2_SPLIT = os.path.join(DATA_ROOT, "stage2_split.json")

STAGE2_CKPT_IN = "outputs/stage2_material_mobilenet_removedbadapples_v2.pt"
RUN_NAME = "lastditchattempt"
STAGE2_CKPT_OUT = f"outputs/{RUN_NAME}.pt"
REPORT_OUT = f"outputs/training_report_{RUN_NAME}.json"

STAGE2_CLASSES = ["cardboard", "ewaste", "glass", "metal", "paper", "plastic"]

# QUICK_TEST=1 env var shrinks this to a ~1-2 min smoke test (tiny hard-
# example scan subset, 2 epochs, patience effectively disabled) to verify
# the whole pipeline -- scan -> resample -> fine-tune -> restore-best ->
# test-eval -> report write -- runs cleanly before committing to the real
# overnight job. Untouched (QUICK_TEST unset/0) = the real planned run.
QUICK_TEST = os.environ.get("QUICK_TEST", "0") == "1"

HARD_EXAMPLE_CONFIDENCE_THRESHOLD = 0.60  # borderline calls, not just outright misses
EASY_EXAMPLE_RATIO = 1.5   # random easy examples per hard example, each epoch's resample
MAX_EPOCHS = 2 if QUICK_TEST else 10
PATIENCE = 2 if QUICK_TEST else 3
FINETUNE_LR = 1e-5         # same as the original Phase 2 fine-tune LR
UNFREEZE_FROM_LAYER = "features.14"
BATCH_SIZE = 32
HARD_SCAN_LIMIT = 200 if QUICK_TEST else None  # cap the train-set scan for the smoke test


def load_model(ckpt_path, num_classes, device, is_stage2=False):
    model, train_tf, eval_tf = get_model_and_transforms(
        "mobilenet_v3_large", num_classes, pretrained=False, is_stage2=is_stage2,
        augment_level="heavy" if is_stage2 else "gentle"
    )
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.to(device)
    return model, train_tf, eval_tf


@torch.no_grad()
def find_hard_examples(model, eval_tf, entries, class_to_idx, device, batch_size=BATCH_SIZE):
    """entries: list of [path, label]. Returns (hard, easy) as two lists of
    [path, label], using the EVAL transform (no augmentation) so confidence
    reflects the model's real calibration, not a randomly-augmented crop."""
    model.eval()
    hard, easy = [], []
    batch_imgs, batch_entries = [], []

    def flush():
        if not batch_imgs:
            return
        x = torch.stack(batch_imgs).to(device)
        logits = model(x)
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        for (path, label), p in zip(batch_entries, probs):
            true_idx = class_to_idx[label]
            top_idx = int(p.argmax())
            top_prob = float(p[top_idx])
            is_correct = (top_idx == true_idx)
            if (not is_correct) or top_prob < HARD_EXAMPLE_CONFIDENCE_THRESHOLD:
                hard.append([path, label])
            else:
                easy.append([path, label])

    for i, (path, label) in enumerate(entries):
        img = Image.open(path).convert("RGB")
        batch_imgs.append(eval_tf(img))
        batch_entries.append((path, label))
        if len(batch_imgs) == batch_size:
            flush()
            batch_imgs, batch_entries = [], []
        if (i + 1) % 1000 == 0:
            print(f"  ...{i+1}/{len(entries)} scanned for hard examples")
    flush()
    return hard, easy


def main():
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    print(f"Device: {device}")
    if QUICK_TEST:
        print("QUICK_TEST=1 -- running a shrunk smoke test, NOT the real fine-tune run.")

    if not os.path.exists(STAGE2_CKPT_IN):
        print(f"ERROR: {STAGE2_CKPT_IN} not found.")
        sys.exit(1)

    with open(STAGE2_SPLIT) as f:
        s2 = json.load(f)

    class_to_idx = {c: i for i, c in enumerate(STAGE2_CLASSES)}
    num_classes = len(STAGE2_CLASSES)

    print(f"Loading current best checkpoint ({STAGE2_CKPT_IN}) as the warm-start...")
    model, train_tf, eval_tf = load_model(STAGE2_CKPT_IN, num_classes, device, is_stage2=True)

    scan_pool = s2["train"][:HARD_SCAN_LIMIT] if HARD_SCAN_LIMIT else s2["train"]
    print(f"\nScanning {len(scan_pool)} train images for hard/borderline examples "
          f"(confidence threshold {HARD_EXAMPLE_CONFIDENCE_THRESHOLD})...")
    hard, easy = find_hard_examples(model, eval_tf, scan_pool, class_to_idx, device)
    print(f"\nFound {len(hard)} hard examples, {len(easy)} easy examples "
          f"({100*len(hard)/len(s2['train']):.1f}% of train flagged hard)")
    print("Hard example class breakdown:", dict(Counter(label for _, label in hard)))

    val_dataset = WasteImageDataset(s2["val"], class_to_idx, transform=eval_tf)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    print("\nBaseline (before fine-tuning) full val performance:")
    baseline_metrics = evaluate(model, val_loader, device, num_classes)
    print(f"  macro-F1={baseline_metrics['macro_f1']:.4f} acc={baseline_metrics['accuracy']:.4f}")

    # Near-frozen: only unfreeze from features.14 onward, same as the
    # original Phase 2 fine-tune -- this is a nudge, not a retrain.
    unfreeze_all(model, unfreeze_from_layer=UNFREEZE_FROM_LAYER)
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable, lr=FINETUNE_LR)
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.1)

    best_f1 = baseline_metrics["macro_f1"]
    best_state = copy.deepcopy(model.state_dict())
    best_epoch = 0
    epochs_since_improve = 0

    print(f"\nFine-tuning for up to {MAX_EPOCHS} epochs (patience={PATIENCE}, lr={FINETUNE_LR})...")
    print("Best-epoch selection uses FULL val macro-F1, not hard-example accuracy -- "
          "an epoch that overfits to the hard subset at the expense of the whole "
          "distribution will not be selected.")

    for epoch in range(1, MAX_EPOCHS + 1):
        # Fresh resample each epoch: all hard examples + a random draw of
        # easy examples (ratio-capped), so the model doesn't just memorize
        # a fixed hard set and keeps seeing the ordinary distribution too.
        n_easy = min(len(easy), int(len(hard) * EASY_EXAMPLE_RATIO))
        epoch_easy = random.sample(easy, n_easy) if n_easy < len(easy) else easy
        epoch_train_entries = hard + epoch_easy
        random.shuffle(epoch_train_entries)

        train_dataset = WasteImageDataset(epoch_train_entries, class_to_idx, transform=train_tf)
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

        t0 = time.time()
        loss = train_one_epoch(model, train_loader, optimizer, device, criterion=criterion)
        val_metrics = evaluate(model, val_loader, device, num_classes)
        f1, acc = val_metrics["macro_f1"], val_metrics["accuracy"]
        print(f"Epoch {epoch} [{time.time()-t0:.1f}s] (train size {len(epoch_train_entries)}, "
              f"{len(hard)} hard + {len(epoch_easy)} easy): loss={loss:.4f} "
              f"val_macro_f1={f1:.4f} val_acc={acc:.4f}")

        if f1 > best_f1:
            print(f"  -> new best full-val macro-F1 ({best_f1:.4f} -> {f1:.4f})")
            best_f1 = f1
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            epochs_since_improve = 0
        else:
            epochs_since_improve += 1
            if epochs_since_improve >= PATIENCE:
                print(f"  -> no improvement in {PATIENCE} epochs, stopping early")
                break

    print(f"\nRestoring best state (epoch {best_epoch}, val_macro_f1={best_f1:.4f}; "
          f"baseline was {baseline_metrics['macro_f1']:.4f}).")
    model.load_state_dict(best_state)

    if best_epoch == 0:
        print("\nWARNING: no epoch beat the baseline -- fine-tuning did not help. "
              "Saving the ORIGINAL checkpoint's weights unchanged (best_epoch=0 means "
              "best_state is still the pre-fine-tune baseline) so this is a documented "
              "no-op, not a silent regression.")

    torch.save(model.state_dict(), STAGE2_CKPT_OUT)
    print(f"Saved to {STAGE2_CKPT_OUT} (new file -- {STAGE2_CKPT_IN} untouched).")

    print("\n--- Final Test Set Evaluation ---")
    test_dataset = WasteImageDataset(s2["test"], class_to_idx, transform=eval_tf)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_metrics = evaluate(model, test_loader, device, num_classes)
    print(f"Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"Macro-F1: {test_metrics['macro_f1']:.4f}")
    print(f"Weighted-F1: {test_metrics['weighted_f1']:.4f}")
    print("Per-Class Metrics:")
    for i, c in enumerate(STAGE2_CLASSES):
        print(f"  {c.ljust(12)}: P={test_metrics['precision'][i]:.4f} "
              f"R={test_metrics['recall'][i]:.4f} F1={test_metrics['f1'][i]:.4f}")

    cm = confusion_matrix(test_metrics["y_true"], test_metrics["y_pred"], num_classes)
    print("\nConfusion matrix (rows=true, cols=pred):", STAGE2_CLASSES)
    for i, row in enumerate(cm.tolist()):
        print(f"  {STAGE2_CLASSES[i].ljust(12)}: {row}")

    report = {
        "baseline_checkpoint": STAGE2_CKPT_IN,
        "baseline_val_macro_f1": baseline_metrics["macro_f1"],
        "best_epoch": best_epoch,
        "best_val_macro_f1": best_f1,
        "n_hard_examples": len(hard),
        "n_easy_examples_pool": len(easy),
        "hard_example_confidence_threshold": HARD_EXAMPLE_CONFIDENCE_THRESHOLD,
        "test_accuracy": test_metrics["accuracy"],
        "test_macro_f1": test_metrics["macro_f1"],
        "test_weighted_f1": test_metrics["weighted_f1"],
        "test_confusion_matrix": cm.tolist(),
        "classes": STAGE2_CLASSES,
    }
    with open(REPORT_OUT, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nWrote {REPORT_OUT}")


if __name__ == "__main__":
    main()
