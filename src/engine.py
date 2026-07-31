"""
Training/eval loop for the PyTorch classifiers. Two-phase transfer learning
is driven from here via src/models.py's freeze_backbone/unfreeze_all:

  phase 1: freeze_backbone(model) -> train only the new head for a few
           epochs -> evaluate on val split
  phase 2: only if phase-1 val metrics are below a target, call
           unfreeze_all(model, unfreeze_from_layer=...) and continue
           training end-to-end at a lower LR

This is deliberately not "always fine-tune everything" -- with ~55k images
across up to 11 classes, full fine-tuning of a ResNet50 from the start
risks overfitting/catastrophic forgetting of ImageNet features faster than
head-only training does, and burns more compute for a real-time-constrained
downstream use case where we also care about not needing a giant GPU to
iterate.
"""
from __future__ import annotations
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .metrics import per_class_precision_recall_f1, macro_and_weighted_f1


def train_one_epoch(model, loader: DataLoader, optimizer, device, criterion=None):
    model.train()
    criterion = criterion or nn.CrossEntropyLoss()
    running_loss, n_batches = 0.0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        n_batches += 1
    return running_loss / max(n_batches, 1)


@torch.no_grad()
def evaluate(model, loader: DataLoader, device, num_classes: int):
    model.eval()
    all_preds, all_labels, all_scores = [], [], []
    for images, labels in loader:
        images = images.to(device)
        logits = model(images)
        probs = torch.softmax(logits, dim=1)
        preds = probs.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds.tolist())
        all_labels.extend(labels.numpy().tolist())
        all_scores.extend(probs.cpu().numpy().tolist())

    import numpy as np
    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    metrics = per_class_precision_recall_f1(y_true, y_pred, num_classes)
    metrics.update(macro_and_weighted_f1(metrics))
    metrics["accuracy"] = float((y_true == y_pred).mean())  # reported ALONGSIDE, never alone
    metrics["scores"] = np.array(all_scores)
    metrics["y_true"] = y_true
    metrics["y_pred"] = y_pred
    return metrics


def run_transfer_learning(
    model, train_loader, val_loader, device, num_classes,
    head_epochs=5, finetune_epochs=5, head_lr=1e-3, finetune_lr=1e-5,
    finetune_if_macro_f1_below=0.85, unfreeze_from_layer="layer4",
):
    """Orchestrates phase 1 (head-only) then conditionally phase 2 (fine-tune).
    Returns (model, history) where history logs val metrics per epoch/phase."""
    from .models import freeze_backbone, unfreeze_all

    history = {"phase1": [], "phase2": []}

    freeze_backbone(model)
    model.to(device)
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable, lr=head_lr)

    for epoch in range(head_epochs):
        t0 = time.time()
        loss = train_one_epoch(model, train_loader, optimizer, device)
        val_metrics = evaluate(model, val_loader, device, num_classes)
        history["phase1"].append({
            "epoch": epoch, "train_loss": loss,
            "val_macro_f1": val_metrics["macro_f1"], "val_accuracy": val_metrics["accuracy"],
            "time_s": time.time() - t0,
        })
        print(f"[phase1 head-only] epoch {epoch}: loss={loss:.4f} "
              f"val_macro_f1={val_metrics['macro_f1']:.4f} val_acc={val_metrics['accuracy']:.4f}")

    best_macro_f1 = history["phase1"][-1]["val_macro_f1"] if history["phase1"] else 0.0

    if best_macro_f1 < finetune_if_macro_f1_below:
        print(f"phase1 macro-F1={best_macro_f1:.4f} < target {finetune_if_macro_f1_below} "
              f"-> unfreezing from '{unfreeze_from_layer}' for fine-tuning")
        unfreeze_all(model, unfreeze_from_layer=unfreeze_from_layer)
        trainable = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(trainable, lr=finetune_lr)

        for epoch in range(finetune_epochs):
            t0 = time.time()
            loss = train_one_epoch(model, train_loader, optimizer, device)
            val_metrics = evaluate(model, val_loader, device, num_classes)
            history["phase2"].append({
                "epoch": epoch, "train_loss": loss,
                "val_macro_f1": val_metrics["macro_f1"], "val_accuracy": val_metrics["accuracy"],
                "time_s": time.time() - t0,
            })
            print(f"[phase2 fine-tune] epoch {epoch}: loss={loss:.4f} "
                  f"val_macro_f1={val_metrics['macro_f1']:.4f} val_acc={val_metrics['accuracy']:.4f}")
    else:
        print(f"phase1 macro-F1={best_macro_f1:.4f} >= target {finetune_if_macro_f1_below} "
              f"-> skipping fine-tuning, frozen backbone is sufficient")

    return model, history
