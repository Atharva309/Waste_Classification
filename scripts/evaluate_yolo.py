import os
import json
import torch
import sys
import numpy as np
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ultralytics import YOLO
from src.metrics import compute_map
from src.latency import benchmark_latency
from src.iou import iou_matrix

def main():
    model_path = "outputs/yolo_detector.pt"
    if not os.path.exists(model_path):
        print(f"Error: {model_path} not found.")
        return

    model = YOLO(model_path)
    val_images_dir = "outputs/yolo_dataset/images/val"
    val_labels_dir = "outputs/yolo_dataset/labels/val"

    classes = ["organic", "cardboard", "ewaste", "glass", "metal", "paper", "plastic"]
    
    per_image_predictions = []
    per_image_ground_truths = []

    print("Running inference on validation set for custom metrics computation...")
    import glob
    image_paths = sorted(glob.glob(os.path.join(val_images_dir, "*.jpg")))

    # For confusion matrix / P / R / F1 (at a specific conf threshold, e.g. 0.5)
    CONF_THRESH = 0.5
    IOU_THRESH = 0.5
    
    y_true = []
    y_pred = []
    
    for img_path in image_paths:
        base_name = os.path.basename(img_path).replace(".jpg", ".txt")
        label_path = os.path.join(val_labels_dir, base_name)
        
        gts = {c: [] for c in range(len(classes))}
        gt_boxes_flat = []
        gt_classes_flat = []
        
        if os.path.exists(label_path):
            with open(label_path, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        c = int(parts[0])
                        cx, cy, w, h = map(float, parts[1:])
                        x1 = max(0.0, cx - w / 2)
                        y1 = max(0.0, cy - h / 2)
                        x2 = min(1.0, cx + w / 2)
                        y2 = min(1.0, cy + h / 2)
                        
                        # scale to 640x640 since YOLO output is absolute pixels
                        x1 *= 640
                        y1 *= 640
                        x2 *= 640
                        y2 *= 640
                        
                        gts[c].append([x1, y1, x2, y2])
                        gt_boxes_flat.append([x1, y1, x2, y2])
                        gt_classes_flat.append(c)
                        
        for c in range(len(classes)):
            gts[c] = np.array(gts[c], dtype=np.float64)
            if gts[c].ndim == 1:
                gts[c] = gts[c].reshape(0, 4)
        
        per_image_ground_truths.append(gts)

        # Run inference
        results = model(img_path, verbose=False, conf=0.01)[0]
        
        preds = {c: ([], []) for c in range(len(classes))}
        pred_boxes_conf = []
        pred_classes_conf = []
        
        if len(results.boxes) > 0:
            boxes = results.boxes.xyxy.cpu().numpy()
            scores = results.boxes.conf.cpu().numpy()
            cls_ids = results.boxes.cls.cpu().numpy().astype(int)
            
            for b, s, c in zip(boxes, scores, cls_ids):
                preds[c][0].append(b)
                preds[c][1].append(s)
                if s >= CONF_THRESH:
                    pred_boxes_conf.append(b)
                    pred_classes_conf.append(c)
                
        for c in range(len(classes)):
            if len(preds[c][0]) > 0:
                preds[c] = (np.array(preds[c][0], dtype=np.float64), np.array(preds[c][1], dtype=np.float64))
            else:
                preds[c] = (np.zeros((0, 4), dtype=np.float64), np.zeros((0,), dtype=np.float64))
                
        per_image_predictions.append(preds)
        
        # Now match for Confusion Matrix
        pred_boxes_conf = np.array(pred_boxes_conf, dtype=np.float64)
        pred_classes_conf = np.array(pred_classes_conf, dtype=int)
        
        if len(pred_boxes_conf) > 0:
            scores_conf = np.array([scores[i] for i, s in enumerate(scores) if s >= CONF_THRESH])
            order = np.argsort(-scores_conf)
            pred_boxes_conf = pred_boxes_conf[order]
            pred_classes_conf = pred_classes_conf[order]
        else:
            pred_boxes_conf = np.zeros((0, 4))
            pred_classes_conf = np.zeros(0, dtype=int)
            
        gt_boxes_flat = np.array(gt_boxes_flat, dtype=np.float64)
        gt_classes_flat = np.array(gt_classes_flat, dtype=int)
        
        claimed = np.zeros(len(gt_boxes_flat), dtype=bool)
        
        if len(pred_boxes_conf) > 0 and len(gt_boxes_flat) > 0:
            ious = iou_matrix(pred_boxes_conf, gt_boxes_flat)
            for i, p_cls in enumerate(pred_classes_conf):
                valid_ious = ious[i].copy()
                valid_ious[claimed] = -1
                
                best_j = np.argmax(valid_ious)
                if valid_ious[best_j] >= IOU_THRESH:
                    y_true.append(gt_classes_flat[best_j])
                    y_pred.append(p_cls)
                    claimed[best_j] = True
                else:
                    y_true.append(len(classes))
                    y_pred.append(p_cls)
        elif len(pred_boxes_conf) > 0:
            for p_cls in pred_classes_conf:
                y_true.append(len(classes))
                y_pred.append(p_cls)
                
        for j, t_cls in enumerate(gt_classes_flat):
            if not claimed[j]:
                y_true.append(t_cls)
                y_pred.append(len(classes))


    print("Computing mAP...")
    map_results = compute_map(per_image_predictions, per_image_ground_truths, list(range(len(classes))), iou_threshold=0.5)

    ap_per_class = {classes[c]: float(ap) for c, ap in map_results["AP_per_class"].items()}

    from src.metrics import confusion_matrix as custom_cm
    cm = custom_cm(y_true, y_pred, len(classes) + 1)
    
    tp = np.diag(cm)[:-1].astype(float)
    fp = cm.sum(axis=0)[:-1] - tp
    fn = cm.sum(axis=1)[:-1] - tp
    
    precision = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) > 0)
    recall = np.divide(tp, tp + fn, out=np.zeros_like(tp), where=(tp + fn) > 0)
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros_like(tp), where=(precision + recall) > 0)

    print("Running Ultralytics val...")
    metrics = model.val(data="outputs/yolo_dataset/data.yaml", split="val", verbose=False)
    
    print("Benchmarking latency...")
    dummy_frame = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)

    def predict_fn(x):
        model(x, verbose=False)
        
    def make_input_fn():
        return dummy_frame
        
    latency_stats = benchmark_latency(predict_fn, make_input_fn, n_warmup=10, n_iters=100)

    try:
        model_cpu = YOLO(model_path)
        model_cpu.to('cpu')
        
        def predict_fn_cpu(x):
            model_cpu(x, verbose=False)
            
        latency_cpu = benchmark_latency(predict_fn_cpu, make_input_fn, n_warmup=10, n_iters=100)
    except Exception as e:
        print("Failed to benchmark CPU:", e)
        latency_cpu = None

    yolo_eval = {
        "ultralytics_map50": float(metrics.box.map50),
        "ultralytics_map50_95": float(metrics.box.map),
        "custom_map50": float(map_results["mAP"]),
        "custom_precision": precision.tolist(),
        "custom_recall": recall.tolist(),
        "custom_f1": f1.tolist(),
        "custom_confusion_matrix": cm.tolist(),
        "custom_ap_per_class": ap_per_class,
        "latency_device": latency_stats.as_dict()
    }
    
    if latency_cpu:
        yolo_eval["latency_cpu"] = latency_cpu.as_dict()
        
    report_path = "outputs/training_report.json"
    if os.path.exists(report_path):
        with open(report_path, "r") as f:
            report = json.load(f)
            
        report["yolo_evaluation"] = yolo_eval
        
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)
        print("Updated outputs/training_report.json")
    else:
        print("Report not found, skipping update.")

if __name__ == "__main__":
    main()
