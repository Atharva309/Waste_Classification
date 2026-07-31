"""
Evaluation metrics for both the classification heads (organic/inorganic,
7-class, 11-class) and the synthetic detection task (bounding boxes on
belt frames).

Design decisions, stated explicitly because they matter in an interview:

1. Plain accuracy is never reported alone. Every class-imbalanced setting
   here (waste categories are not evenly represented, and "background /
   no object" massively outnumbers real detections on a belt) gets
   per-class precision, recall, F1, and a confusion matrix.

2. mAP is computed the way object detection literature computes it: for
   each class, sort predictions by confidence, greedily match each
   prediction to the highest-IoU unmatched ground-truth box of the same
   class (IoU >= iou_threshold counts as a match, from src/iou.py),
   build a precision-recall curve over that ranked list, and take the
   area under it (AP). mAP is the mean of AP across classes. This only
   makes sense for the synthetic-detection task (src/synthetic_detection.py)
   since the original classification dataset has no bounding boxes --
   mAP on a plain classification task is not a standard or meaningful
   metric, so for the classifiers we report P/R/F1/confusion matrix
   instead, not mAP.

3. PR curves, not ROC curves, are used to visualize threshold trade-offs
   for the minority/positive classes -- see `PR_VS_ROC_JUSTIFICATION`
   at the bottom of this file for the reasoning, verified against the
   actual class distribution before being implemented.
"""
from __future__ import annotations
import numpy as np
from .iou import iou_matrix


def _trapz(y, x):
    """np.trapezoid on NumPy>=2.0, falls back to np.trapz on older NumPy."""
    fn = getattr(np, "trapezoid", None) or np.trapz
    return fn(y, x)

# ---------------------------------------------------------------------------
# Classification metrics
# ---------------------------------------------------------------------------

def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> np.ndarray:
    """Rows = true class, cols = predicted class. No sklearn dependency
    required (kept dependency-free deliberately; sklearn's version is used
    elsewhere for cross-checking, see tests)."""
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm


def per_class_precision_recall_f1(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int):
    """Returns dict of arrays: precision, recall, f1, support (each length num_classes)."""
    cm = confusion_matrix(y_true, y_pred, num_classes)
    tp = np.diag(cm).astype(np.float64)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    support = cm.sum(axis=1)

    precision = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) > 0)
    recall = np.divide(tp, tp + fn, out=np.zeros_like(tp), where=(tp + fn) > 0)
    f1 = np.divide(2 * precision * recall, precision + recall,
                    out=np.zeros_like(tp), where=(precision + recall) > 0)

    return {
        "precision": precision, "recall": recall, "f1": f1,
        "support": support, "confusion_matrix": cm,
    }


def macro_and_weighted_f1(metrics_dict) -> dict:
    """Macro-F1 is averaged only over classes that actually occur in the
    ground truth (support > 0). A class with zero true instances (e.g. the
    "unknown" routing bucket, which is never a real ground-truth label --
    only ever a prediction) has undefined recall, not zero recall; forcing
    it to 0 and including it in the average would silently drag every
    macro-F1 down by a fixed amount regardless of model quality, purely
    because that class structurally can never have a true positive. This
    was a real bug, not a style choice -- caught by comparing the reported
    number against a hand-computed sum/7 vs sum/8 and finding the dashboard
    matched sum/8 exactly on multiple live runs. Weighted-F1 already
    handled this correctly on its own, since multiplying by support=0
    naturally zeroes out a class's contribution without needing this fix."""
    f1, support = metrics_dict["f1"], metrics_dict["support"]
    present = support > 0
    macro = float(np.mean(f1[present])) if present.any() else 0.0
    weighted = float(np.sum(f1 * support) / max(support.sum(), 1))
    return {"macro_f1": macro, "weighted_f1": weighted}


def binary_pr_curve(y_true: np.ndarray, scores: np.ndarray, num_thresholds: int = 200):
    """Precision-recall curve for a binary/one-vs-rest problem, computed
    from scratch by sweeping thresholds over the score range (this is what
    sklearn.metrics.precision_recall_curve does internally, made explicit
    since 'implement from scratch' is part of the brief for this project).

    Returns (precisions, recalls, thresholds), plus average precision (AP)
    = area under the PR curve via the trapezoidal rule on recall-sorted points.
    """
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores, dtype=np.float64)
    thresholds = np.linspace(scores.min(), scores.max(), num_thresholds)[::-1]

    precisions, recalls = [], []
    P = max(y_true.sum(), 1)
    for t in thresholds:
        pred = (scores >= t).astype(int)
        tp = np.sum((pred == 1) & (y_true == 1))
        fp = np.sum((pred == 1) & (y_true == 0))
        precisions.append(tp / (tp + fp) if (tp + fp) > 0 else 1.0)
        recalls.append(tp / P)

    precisions, recalls = np.array(precisions), np.array(recalls)
    order = np.argsort(recalls)
    ap = float(_trapz(precisions[order], recalls[order]))
    return precisions, recalls, thresholds, ap


# ---------------------------------------------------------------------------
# Detection metrics (mAP via IoU matching) -- for the synthetic belt-frame task
# ---------------------------------------------------------------------------

def match_detections_to_gt(pred_boxes, pred_scores, gt_boxes, iou_threshold=0.5):
    """Greedy matching of predictions (already sorted by confidence, desc)
    to ground-truth boxes of the SAME class within one image.

    Returns a boolean array `tp` of length len(pred_boxes): True if that
    prediction is a true positive (matched to a not-yet-claimed GT box with
    IoU >= iou_threshold), False if it's a false positive (duplicate or
    no matching GT).
    """
    n_pred = len(pred_boxes)
    n_gt = len(gt_boxes)
    tp = np.zeros(n_pred, dtype=bool)
    if n_gt == 0 or n_pred == 0:
        return tp

    ious = iou_matrix(np.asarray(pred_boxes), np.asarray(gt_boxes))  # (n_pred, n_gt)
    claimed = np.zeros(n_gt, dtype=bool)

    order = np.argsort(-np.asarray(pred_scores))
    for i in order:
        row = ious[i].copy()
        row[claimed] = -1.0
        best_j = np.argmax(row)
        if row[best_j] >= iou_threshold:
            tp[i] = True
            claimed[best_j] = True
    return tp


def average_precision_for_class(all_pred_scores, all_tp_flags, num_gt):
    """Standard AP: sort all predictions (across the whole eval set) for one
    class by confidence descending, compute cumulative precision/recall,
    integrate area under the PR curve (trapezoidal, no 11-point interpolation
    -- the modern/COCO-style continuous method)."""
    order = np.argsort(-np.asarray(all_pred_scores))
    tp = np.asarray(all_tp_flags)[order].astype(np.float64)
    fp = 1.0 - tp

    cum_tp = np.cumsum(tp)
    cum_fp = np.cumsum(fp)

    recalls = cum_tp / max(num_gt, 1)
    precisions = cum_tp / np.maximum(cum_tp + cum_fp, 1e-12)

    # make precision monotonically non-increasing when read right-to-left
    # (standard AP smoothing so a lucky later high-precision point doesn't
    # get ignored -- prevents zig-zag from deflating the integral)
    precisions = np.maximum.accumulate(precisions[::-1])[::-1]

    recalls = np.concatenate([[0.0], recalls])
    precisions = np.concatenate([[precisions[0] if len(precisions) else 1.0], precisions])
    ap = float(_trapz(precisions, recalls))
    return ap, precisions, recalls


def compute_map(per_image_predictions, per_image_ground_truths, class_ids, iou_threshold=0.5):
    """
    per_image_predictions: list over images of dict {class_id: (boxes, scores)}
    per_image_ground_truths: list over images of dict {class_id: boxes}
    class_ids: iterable of class ids to evaluate

    Returns {"mAP": float, "AP_per_class": {class_id: ap}}
    """
    ap_per_class = {}
    for c in class_ids:
        scores_all, tp_all = [], []
        num_gt = 0
        for preds, gts in zip(per_image_predictions, per_image_ground_truths):
            p_boxes, p_scores = preds.get(c, (np.zeros((0, 4)), np.zeros((0,))))
            g_boxes = gts.get(c, np.zeros((0, 4)))
            num_gt += len(g_boxes)
            if len(p_boxes) == 0:
                continue
            tp_flags = match_detections_to_gt(p_boxes, p_scores, g_boxes, iou_threshold)
            scores_all.extend(list(p_scores))
            tp_all.extend(list(tp_flags))

        if num_gt == 0:
            continue
        ap, _, _ = average_precision_for_class(scores_all, tp_all, num_gt)
        ap_per_class[c] = ap

    mAP = float(np.mean(list(ap_per_class.values()))) if ap_per_class else 0.0
    return {"mAP": mAP, "AP_per_class": ap_per_class}


PR_VS_ROC_JUSTIFICATION = """
Why PR curves, not ROC curves, are the primary evaluation view here:

ROC curves plot TPR (recall) vs FPR = FP / (FP + TN). The "N" (negative)
population for a crop/weed detector, or for a per-image class in a
belt-frame full of background, is enormous relative to the "P" (positive)
population -- most of any camera frame is not-weed, and in the classifier
setting most classes have far fewer examples than others (medical/e-waste
vs. plastic/paper, for instance). When TN is huge, FPR = FP/(FP+TN) stays
near zero even as the absolute number of false positives grows into the
thousands, because it's being divided by a denominator dominated by TN.
That makes ROC curves look uniformly good ("hug the top-left corner")
even for a detector that is, in absolute terms, throwing a lot of false
positives -- exactly the failure mode a sprayer can't afford (each FP is
an herbicide hit on a crop plant).

Precision = TP / (TP + FP) does not have TN in it at all, so it directly
exposes how often a "weed" call is wrong, independent of how large the
background/negative class is. A PR curve therefore stays informative
under the class imbalance this project actually has (both in the 7/11-way
waste classes and in the belt-frame background-vs-object setting), while
an ROC curve would visually flatter the model. This is the standard
justification (Davis & Goadrich, 2006, "The Relationship Between
Precision-Recall and ROC Curves") and it is why COCO/PASCAL VOC-style
detection mAP is itself built on precision-recall, not ROC.

Concretely in this repo: `binary_pr_curve` and `average_precision_for_class`
are used for the reported curves and AP/mAP numbers; ROC/AUC is not used
as a headline metric anywhere, though computing it as a side-by-side
comparison plot to make this argument visually is a reasonable thing to
include in the write-up.
"""
