"""
Tunable confidence-threshold wrapper.

Real deployment framing: on a sprayer, a false positive (spraying herbicide
on a crop plant misclassified as weed) and a false negative (missing an
actual weed) have very different costs -- crop damage/yield loss vs. one
weed surviving to the next pass. Rather than bake in a single hardcoded
threshold, the detector exposes a `confidence_threshold` that can be tuned
per deployment context:

  - "high_precision" mode: raise the threshold, only act on confident calls
    -> fewer false positives, more missed detections. Appropriate when
    crop damage is the dominant cost (e.g. high-value crop, early growth
    stage where crop and weed look similar).
  - "high_recall" mode: lower the threshold, act on more calls -> more
    false positives (over-spraying, e.g. spraying borderline cases), fewer
    missed weeds. Appropriate when the weed itself is the dominant cost
    (e.g. weeds that go to seed quickly and are cheap to over-spray for).

This is the same knob whether the model is a per-crop classifier's softmax
score or a detector's objectness*class score -- both cases just need a
scalar "confidence" fed through the PR curve from src/metrics.py to pick a
threshold, rather than guessing.
"""
from __future__ import annotations
from dataclasses import dataclass


PRESET_THRESHOLDS = {
    "high_recall": 0.30,     # act on weaker signals -- fewer misses, more false alarms
    "balanced": 0.50,
    "high_precision": 0.75,  # only act when confident -- fewer false alarms, more misses
}


@dataclass
class ThresholdedDetector:
    predict_scores_fn: callable  # (input) -> dict[class_id, score] or (boxes, scores, classes)
    confidence_threshold: float = 0.5

    def set_mode(self, mode: str):
        if mode not in PRESET_THRESHOLDS:
            raise ValueError(f"mode must be one of {list(PRESET_THRESHOLDS)}, got {mode}")
        self.confidence_threshold = PRESET_THRESHOLDS[mode]
        return self

    def set_threshold(self, value: float):
        if not (0.0 <= value <= 1.0):
            raise ValueError("threshold must be in [0, 1]")
        self.confidence_threshold = value
        return self

    def predict(self, x):
        """Runs predict_scores_fn and filters to only calls at/above the
        current threshold. Works for either a classification-style
        {class_id: score} dict or a detection-style (boxes, scores, classes)
        triple -- shape-detects at call time."""
        raw = self.predict_scores_fn(x)

        if isinstance(raw, dict):
            return {c: s for c, s in raw.items() if s >= self.confidence_threshold}

        boxes, scores, classes = raw
        keep = [i for i, s in enumerate(scores) if s >= self.confidence_threshold]
        return (
            [boxes[i] for i in keep],
            [scores[i] for i in keep],
            [classes[i] for i in keep],
        )

    def threshold_for_target_precision(self, y_true, scores, target_precision: float):
        """Given a labeled validation set (scores from this detector, ground-
        truth labels), pick the lowest threshold that achieves at least
        target_precision -- i.e. derive the operating point from data rather
        than guessing the preset constants above. Uses binary_pr_curve from
        src/metrics.py."""
        from .metrics import binary_pr_curve
        precisions, recalls, thresholds, _ = binary_pr_curve(y_true, scores)
        candidates = [(t, p, r) for t, p, r in zip(thresholds, precisions, recalls) if p >= target_precision]
        if not candidates:
            return None  # target precision not achievable at any threshold on this data
        # among thresholds hitting the precision target, pick the one with best recall
        best = max(candidates, key=lambda trp: trp[2])
        return best[0]
