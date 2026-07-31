import sys
import os
import json
import time
import copy
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.datasets import WasteImageDataset
from src.models import get_model_and_transforms, freeze_backbone, unfreeze_all
from src.engine import train_one_epoch, evaluate
from src.metrics import confusion_matrix

TOTAL_BUDGET_S = 3600  # 1 hour

class FocalLoss(torch.nn.Module):
    def __init__(self, alpha=None, gamma=2.0, label_smoothing=0.0):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.ce = torch.nn.CrossEntropyLoss(weight=alpha, reduction='none', label_smoothing=label_smoothing)

    def forward(self, inputs, targets):
        ce_loss = self.ce(inputs, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()

def run_stage_training(
    stage_name, split_path, classes, global_start, time_limit_s, output_path,
    finetune_if_below=0.92, unfreeze_layer="features.14"
):
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    print(f"\n{'='*50}\nStarting {stage_name} Training (Device: {device})\n{'='*50}")
    
    with open(split_path, "r") as f:
        split_res = json.load(f)
        
    class_to_idx = {c: i for i, c in enumerate(classes)}
    num_classes = len(classes)
    
    is_stage2 = ("Stage 2" in stage_name)
    
    if is_stage2:
        new_train = []
        for item in split_res["train"]:
            fname = os.path.basename(item[0])
            label = item[1]
            if fname.startswith("trashnet_") or fname.startswith("taco_"):
                if label in ["metal", "ewaste"]:
                    new_train.extend([item] * 4)
                else:
                    new_train.append(item)
            else:
                new_train.append(item)
        split_res["train"] = new_train
        
        class_weights = [1.0] * num_classes
        if "metal" in class_to_idx:
            class_weights[class_to_idx["metal"]] = 1.75
    else:
        from collections import Counter
        train_labels = [item[1] for item in split_res["train"]]
        counts = Counter(train_labels)
        total_train = len(train_labels)
        class_weights = []
        for c in classes:
            w = total_train / (num_classes * counts.get(c, 1))
            class_weights.append(w)
            
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)
    smoothing = 0.1 if is_stage2 else 0.0
    if is_stage2:
        criterion = FocalLoss(alpha=class_weights_tensor, gamma=1.5, label_smoothing=smoothing)
    else:
        criterion = torch.nn.CrossEntropyLoss(weight=class_weights_tensor, label_smoothing=smoothing)
    
    print(f"Loading MobileNetV3-Large for {num_classes} classes...")
    model, train_tf, eval_tf = get_model_and_transforms("mobilenet_v3_large", num_classes, pretrained=True, is_stage2=is_stage2, augment_level="heavy" if is_stage2 else "gentle")
    
    train_dataset = WasteImageDataset(split_res["train"], class_to_idx, transform=train_tf)
    val_dataset = WasteImageDataset(split_res["val"], class_to_idx, transform=eval_tf)
    test_dataset = WasteImageDataset(split_res["test"], class_to_idx, transform=eval_tf)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)
    
    model.to(device)
    
    # Phase 1: Frozen backbone
    freeze_backbone(model)
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable, lr=1e-3)
    
    best_f1 = 0.0
    best_state = None
    best_epoch = -1
    epoch = 0
    
    print("\n--- Phase 1: Head-Only Training ---")
    while epoch < 30:
        epoch += 1
        elapsed = time.time() - global_start
        if elapsed > time_limit_s:
            print(f"Time limit reached ({elapsed:.1f}s > {time_limit_s}s) during Phase 1. Stopping.")
            break
            
        t0 = time.time()
        loss = train_one_epoch(model, train_loader, optimizer, device, criterion=criterion)
        val_metrics = evaluate(model, val_loader, device, num_classes)
        f1 = val_metrics["macro_f1"]
        acc = val_metrics["accuracy"]
        t_sec = time.time() - t0
        
        print(f"Epoch {epoch} [{t_sec:.1f}s]: loss={loss:.4f}, val_macro_f1={f1:.4f}, val_acc={acc:.4f} | Total Elapsed: {time.time()-global_start:.1f}s")
        
        if f1 > best_f1:
            print(f"  -> New best val_macro_f1! ({best_f1:.4f} -> {f1:.4f})")
            best_f1 = f1
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            
    # Phase 2: Fine-Tuning
    if best_f1 < finetune_if_below:
        elapsed = time.time() - global_start
        if elapsed < time_limit_s:
            print(f"\nPhase 1 best macro-F1 ({best_f1:.4f}) < {finetune_if_below}. Starting Phase 2 Fine-Tuning.")
            print(f"Unfreezing from layer: {unfreeze_layer}")
            
            # Load the best head-only state before finetuning
            if best_state is not None:
                model.load_state_dict(best_state)
                
            unfreeze_all(model, unfreeze_from_layer=unfreeze_layer)
            trainable = [p for p in model.parameters() if p.requires_grad]
            optimizer = torch.optim.Adam(trainable, lr=1e-5)
            
            while epoch < 30: # Continue epoch numbering
                epoch += 1
                elapsed = time.time() - global_start
                if elapsed > time_limit_s:
                    print(f"Time limit reached ({elapsed:.1f}s > {time_limit_s}s) during Phase 2. Stopping.")
                    break
                    
                t0 = time.time()
                loss = train_one_epoch(model, train_loader, optimizer, device, criterion=criterion)
                val_metrics = evaluate(model, val_loader, device, num_classes)
                f1 = val_metrics["macro_f1"]
                acc = val_metrics["accuracy"]
                t_sec = time.time() - t0
                
                print(f"Epoch {epoch} (FT) [{t_sec:.1f}s]: loss={loss:.4f}, val_macro_f1={f1:.4f}, val_acc={acc:.4f} | Total Elapsed: {time.time()-global_start:.1f}s")
                
                if f1 > best_f1:
                    print(f"  -> New best val_macro_f1! ({best_f1:.4f} -> {f1:.4f})")
                    best_f1 = f1
                    best_state = copy.deepcopy(model.state_dict())
                    best_epoch = epoch
        else:
            print(f"\nSkipping Phase 2 Fine-Tuning: Time limit reached ({elapsed:.1f}s >= {time_limit_s}s).")
    else:
        print(f"\nSkipping Phase 2 Fine-Tuning: Phase 1 best macro-F1 ({best_f1:.4f}) >= {finetune_if_below}.")
            
    print(f"\n{stage_name} Training finished.")
    if best_state is not None:
        print(f"Restoring best model from Epoch {best_epoch} (val_macro_f1={best_f1:.4f}).")
        model.load_state_dict(best_state)
        torch.save(model.state_dict(), output_path)
        print(f"Saved best model to {output_path}.")
    else:
        print("WARNING: No best state found. Saving current state.")
        torch.save(model.state_dict(), output_path)
        
    print(f"\n--- {stage_name} Final Test Set Evaluation ---")
    test_metrics = evaluate(model, test_loader, device, num_classes)
    print(f"Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"Macro-F1: {test_metrics['macro_f1']:.4f}")
    print(f"Weighted-F1: {test_metrics['weighted_f1']:.4f}")
    
    print("\nPer-Class Metrics:")
    for i, c in enumerate(classes):
        p = test_metrics["precision"][i]
        r = test_metrics["recall"][i]
        f = test_metrics["f1"][i]
        print(f"  {c.ljust(12)}: P={p:.4f}, R={r:.4f}, F1={f:.4f}")
        
    cm = confusion_matrix(test_metrics["y_true"], test_metrics["y_pred"], num_classes)
    
    return {
        "accuracy": test_metrics['accuracy'],
        "macro_f1": test_metrics['macro_f1'],
        "weighted_f1": test_metrics['weighted_f1'],
        "confusion_matrix": cm.tolist(),
        "per_class": {
            classes[i]: {
                "precision": test_metrics["precision"][i],
                "recall": test_metrics["recall"][i],
                "f1": test_metrics["f1"][i]
            } for i in range(num_classes)
        },
        "best_epoch": best_epoch
    }

def main():
    global_start = time.time()
    
    global_start = time.time()

    
    # Stage 2
    stage2_metrics = run_stage_training(
        stage_name="Stage 2 (6 Material Classes)",
        split_path="outputs/stage2_split.json",
        classes=["cardboard", "ewaste", "glass", "metal", "paper", "plastic"],
        global_start=global_start,
        time_limit_s=TOTAL_BUDGET_S,  # Full 900s for both combined
        output_path="outputs/stage2_material_mobilenet_FOCAL.pt",
        finetune_if_below=0.92,
        unfreeze_layer="features.14"
    )
    
    total_time = time.time() - global_start
    print(f"\n{'='*50}\nAll Training Complete in {total_time:.1f}s!\n{'='*50}")
    
    # Update training_report.json
    report_path = "outputs/training_report.json"
    if os.path.exists(report_path):
        with open(report_path, "r") as f:
            report = json.load(f)
            
        # Don't rotate since this is a separate run type
        report["stage2_mobilenet_FOCAL"] = stage2_metrics
        report["total_wall_clock_time_s"] = total_time
        
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)
        print(f"Updated {report_path}")

if __name__ == "__main__":
    main()
