"""
Non-Maximum Suppression, implemented from scratch (no torchvision.ops.nms,
no cv2.dnn.NMSBoxes). Two variants:

  - nms(...)       : classic "hard" NMS -- boxes with IoU > threshold vs. a
                     kept box are dropped entirely.
  - soft_nms(...)  : Soft-NMS (Bodla et al., 2017) -- overlapping boxes have
                     their score decayed (linearly or with a Gaussian) instead
                     of being deleted outright. Useful when objects genuinely
                     overlap (e.g. weeds occluding crop leaves), where hard
                     NMS can wrongly kill a real second detection.

Both operate on a single class at a time (typical detection pipelines run
NMS per-class); box format is (x1, y1, x2, y2).
"""
from __future__ import annotations
import numpy as np
from .iou import iou_matrix


def nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float = 0.5) -> np.ndarray:
    """Classic greedy NMS.

    Args:
        boxes: (N, 4) array of (x1,y1,x2,y2)
        scores: (N,) confidence scores
        iou_threshold: boxes overlapping a kept box by more than this are suppressed

    Returns:
        indices (into the original arrays) of kept boxes, sorted by descending score.
    """
    boxes = np.asarray(boxes, dtype=np.float64)
    scores = np.asarray(scores, dtype=np.float64)
    n = len(scores)
    if n == 0:
        return np.array([], dtype=np.int64)

    order = np.argsort(-scores)
    keep = []
    suppressed = np.zeros(n, dtype=bool)

    for idx_pos in range(n):
        i = order[idx_pos]
        if suppressed[i]:
            continue
        keep.append(i)
        if idx_pos == n - 1:
            break
        remaining = order[idx_pos + 1:]
        remaining = remaining[~suppressed[remaining]]
        if len(remaining) == 0:
            continue
        ious = iou_matrix(boxes[i:i + 1], boxes[remaining])[0]  # (len(remaining),)
        to_suppress = remaining[ious > iou_threshold]
        suppressed[to_suppress] = True

    return np.array(keep, dtype=np.int64)


def soft_nms(
    boxes: np.ndarray,
    scores: np.ndarray,
    method: str = "gaussian",
    iou_threshold: float = 0.3,
    sigma: float = 0.5,
    score_threshold: float = 0.001,
) -> tuple[np.ndarray, np.ndarray]:
    """Soft-NMS. Instead of deleting overlapping boxes, decays their score.

    method="linear":   score *= (1 - iou)              if iou > iou_threshold, else unchanged
    method="gaussian":  score *= exp(-(iou**2) / sigma)  applied to all boxes (smooth decay)

    Returns:
        (kept_indices, decayed_scores_for_kept_indices), both sorted by
        descending decayed score. A box is dropped once its decayed score
        falls below score_threshold.
    """
    if method not in ("linear", "gaussian"):
        raise ValueError("method must be 'linear' or 'gaussian'")

    boxes = np.asarray(boxes, dtype=np.float64).copy()
    scores = np.asarray(scores, dtype=np.float64).copy()
    n = len(scores)
    if n == 0:
        return np.array([], dtype=np.int64), np.array([])

    indices = np.arange(n)
    kept_idx = []
    kept_score = []

    active = np.ones(n, dtype=bool)

    for _ in range(n):
        remaining = indices[active]
        if len(remaining) == 0:
            break
        # pick current max-score box among active
        local_best = remaining[np.argmax(scores[remaining])]
        if scores[local_best] < score_threshold:
            break
        kept_idx.append(local_best)
        kept_score.append(scores[local_best])
        active[local_best] = False

        others = indices[active]
        if len(others) == 0:
            break
        ious = iou_matrix(boxes[local_best:local_best + 1], boxes[others])[0]

        if method == "linear":
            decay = np.where(ious > iou_threshold, 1 - ious, 1.0)
        else:  # gaussian
            decay = np.exp(-(ious ** 2) / sigma)

        scores[others] = scores[others] * decay
        # deactivate boxes that decayed below threshold so they're never picked
        active[others] = active[others] & (scores[others] >= score_threshold)

    return np.array(kept_idx, dtype=np.int64), np.array(kept_score)
