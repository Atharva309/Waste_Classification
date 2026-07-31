import sys, os
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.inference import ThresholdedDetector, PRESET_THRESHOLDS

def test_mode_switch_changes_threshold():
    d = ThresholdedDetector(predict_scores_fn=lambda x: x)
    d.set_mode("high_precision")
    assert d.confidence_threshold == PRESET_THRESHOLDS["high_precision"]
    d.set_mode("high_recall")
    assert d.confidence_threshold == PRESET_THRESHOLDS["high_recall"]

def test_predict_filters_classification_dict():
    d = ThresholdedDetector(predict_scores_fn=lambda x: {"weed": 0.9, "crop": 0.2}, confidence_threshold=0.5)
    result = d.predict(None)
    assert result == {"weed": 0.9}

def test_predict_filters_detection_triple():
    boxes = [(0,0,10,10), (20,20,30,30)]
    scores = [0.9, 0.3]
    classes = ["weed", "weed"]
    d = ThresholdedDetector(predict_scores_fn=lambda x: (boxes, scores, classes), confidence_threshold=0.5)
    kept_boxes, kept_scores, kept_classes = d.predict(None)
    assert kept_boxes == [(0,0,10,10)]
    assert kept_scores == [0.9]

def test_high_recall_keeps_more_than_high_precision():
    boxes = [(0,0,1,1)] * 5
    scores = [0.35, 0.55, 0.65, 0.80, 0.90]
    classes = ["c"] * 5
    d = ThresholdedDetector(predict_scores_fn=lambda x: (boxes, scores, classes))
    d.set_mode("high_recall")
    n_recall = len(d.predict(None)[0])
    d.set_mode("high_precision")
    n_precision = len(d.predict(None)[0])
    assert n_recall > n_precision

def test_threshold_for_target_precision_achievable():
    rng = np.random.RandomState(0)
    y_true = np.array([1]*50 + [0]*50)
    # positives get higher scores on average -> precision improves as threshold rises
    scores = np.concatenate([rng.uniform(0.5, 1.0, 50), rng.uniform(0.0, 0.6, 50)])
    d = ThresholdedDetector(predict_scores_fn=lambda x: x)
    t = d.threshold_for_target_precision(y_true, scores, target_precision=0.9)
    assert t is not None
    # applying that threshold should actually achieve >= 0.9 precision
    pred = (scores >= t).astype(int)
    tp = np.sum((pred==1) & (y_true==1)); fp = np.sum((pred==1) & (y_true==0))
    precision = tp / max(tp+fp, 1)
    assert precision >= 0.9 - 1e-6, precision

def test_threshold_for_target_precision_unachievable_returns_none():
    y_true = np.array([1,0,1,0,1,0])
    scores = np.array([0.5,0.5,0.5,0.5,0.5,0.5])  # no separation -> can't hit 0.99 precision
    d = ThresholdedDetector(predict_scores_fn=lambda x: x)
    t = d.threshold_for_target_precision(y_true, scores, target_precision=0.99)
    assert t is None

if __name__ == "__main__":
    fns = [v for k, v in list(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"OK: {fn.__name__}")
    print(f"\n{len(fns)} inference/threshold tests passed")
