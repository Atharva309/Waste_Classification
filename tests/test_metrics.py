import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.metrics import (confusion_matrix, per_class_precision_recall_f1,
                          binary_pr_curve, match_detections_to_gt,
                          average_precision_for_class, compute_map)

def test_confusion_matrix_matches_sklearn():
    from sklearn.metrics import confusion_matrix as skcm
    rng = np.random.RandomState(1)
    y_true = rng.randint(0, 4, size=200)
    y_pred = rng.randint(0, 4, size=200)
    mine = confusion_matrix(y_true, y_pred, 4)
    theirs = skcm(y_true, y_pred, labels=[0,1,2,3])
    assert np.array_equal(mine, theirs)

def test_precision_recall_f1_matches_sklearn():
    from sklearn.metrics import precision_score, recall_score, f1_score
    rng = np.random.RandomState(2)
    y_true = rng.randint(0, 3, size=300)
    y_pred = rng.randint(0, 3, size=300)
    m = per_class_precision_recall_f1(y_true, y_pred, 3)
    p_sk = precision_score(y_true, y_pred, average=None, labels=[0,1,2], zero_division=0)
    r_sk = recall_score(y_true, y_pred, average=None, labels=[0,1,2], zero_division=0)
    f_sk = f1_score(y_true, y_pred, average=None, labels=[0,1,2], zero_division=0)
    assert np.allclose(m["precision"], p_sk, atol=1e-9)
    assert np.allclose(m["recall"], r_sk, atol=1e-9)
    assert np.allclose(m["f1"], f_sk, atol=1e-9)

def test_perfect_classifier_ap_is_one():
    # detections perfectly match GT with high scores, no FPs
    pred_boxes = np.array([[0,0,10,10],[20,20,30,30]], dtype=float)
    gt_boxes = pred_boxes.copy()
    scores = np.array([0.99, 0.95])
    tp = match_detections_to_gt(pred_boxes, scores, gt_boxes, iou_threshold=0.5)
    assert tp.all()
    ap, _, _ = average_precision_for_class(scores, tp, num_gt=2)
    assert abs(ap - 1.0) < 1e-6

def test_duplicate_prediction_is_false_positive():
    gt_boxes = np.array([[0,0,10,10]], dtype=float)
    pred_boxes = np.array([[0,0,10,10],[0.5,0.5,10.5,10.5]], dtype=float)  # 2 preds, 1 gt
    scores = np.array([0.9, 0.8])
    tp = match_detections_to_gt(pred_boxes, scores, gt_boxes, iou_threshold=0.5)
    assert tp.tolist() == [True, False]  # only the higher-scoring one claims the GT

def test_compute_map_simple_two_class():
    preds = [{
        0: (np.array([[0,0,10,10]]), np.array([0.9])),
        1: (np.array([[20,20,30,30]]), np.array([0.8])),
    }]
    gts = [{
        0: np.array([[0,0,10,10]]),
        1: np.array([[20,20,30,30]]),
    }]
    result = compute_map(preds, gts, class_ids=[0,1], iou_threshold=0.5)
    assert abs(result["mAP"] - 1.0) < 1e-6

def test_pr_curve_monotonic_recall():
    rng = np.random.RandomState(3)
    y_true = (rng.rand(500) < 0.1).astype(int)  # imbalanced, like real classes here
    scores = y_true * rng.rand(500) * 0.5 + rng.rand(500) * 0.5
    p, r, t, ap = binary_pr_curve(y_true, scores)
    assert 0.0 <= ap <= 1.0
    assert r[0] <= r[-1]  # recall increases as threshold sweeps down

if __name__ == "__main__":
    fns = [v for k, v in list(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"OK: {fn.__name__}")
    print(f"\n{len(fns)} metrics tests passed")
