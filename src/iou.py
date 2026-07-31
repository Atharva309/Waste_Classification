"""
Intersection-over-Union (IoU) implemented from scratch.

Box format everywhere in this module: (x1, y1, x2, y2), absolute pixel
coordinates, x2 > x1 and y2 > y1 (i.e. corner format, not cx/cy/w/h).

Two entry points:
  - iou_single(box_a, box_b): scalar IoU for one pair of boxes.
  - iou_matrix(boxes_a, boxes_b): vectorized IoU for every (a, b) pair,
    returns an (N, M) matrix. This is the one NMS/mAP code should use --
    looping iou_single over N*M pairs in Python is fine for a demo but
    is the wrong way to do this at scale, so we show both.
"""
from __future__ import annotations
import numpy as np


def _area(x1, y1, x2, y2):
    w = np.maximum(0.0, x2 - x1)
    h = np.maximum(0.0, y2 - y1)
    return w * h


def iou_single(box_a, box_b) -> float:
    """IoU of two boxes, each (x1, y1, x2, y2). Returns 0.0 for degenerate
    or non-overlapping boxes (never divides by zero)."""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter_area

    if union <= 0.0:
        return 0.0
    return float(inter_area / union)


def iom_single(box_a, box_b) -> float:
    """Intersection-over-min-area of two boxes, each (x1, y1, x2, y2).

    Unlike IoU, this isn't penalized by a size mismatch between the two
    boxes -- it asks "what fraction of the SMALLER box is covered by the
    overlap", not "what fraction of their combined footprint". Two boxes
    around the same physical object but regressed slightly differently in
    size (e.g. by two different YOLO class heads on the same anchor) can
    have a much lower IoU than a human would expect from looking at them,
    because a modest size difference shrinks IoU's union-based denominator
    a lot; IoM stays high as long as one box is mostly contained in the
    overlap region. This is what cross-class same-object dedup should be
    checking, not plain IoU (see webapp/app.py step 3b and NOTES.md
    section 49). Returns 0.0 for degenerate or non-overlapping boxes
    (never divides by zero)."""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    min_area = min(area_a, area_b)

    if min_area <= 0.0:
        return 0.0
    return float(inter_area / min_area)


def iou_matrix(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """Vectorized IoU between every box in boxes_a (N,4) and boxes_b (M,4).

    Returns an (N, M) float array. Uses broadcasting instead of nested
    Python loops -- this is the version training/eval code should call
    when N and M are in the hundreds or thousands (e.g. NMS candidates
    vs. ground truth per image).
    """
    boxes_a = np.asarray(boxes_a, dtype=np.float64)
    boxes_b = np.asarray(boxes_b, dtype=np.float64)
    if boxes_a.ndim == 1:
        boxes_a = boxes_a[None, :]
    if boxes_b.ndim == 1:
        boxes_b = boxes_b[None, :]

    N, M = boxes_a.shape[0], boxes_b.shape[0]
    if N == 0 or M == 0:
        return np.zeros((N, M), dtype=np.float64)

    a = boxes_a[:, None, :]   # (N, 1, 4)
    b = boxes_b[None, :, :]   # (1, M, 4)

    inter_x1 = np.maximum(a[..., 0], b[..., 0])
    inter_y1 = np.maximum(a[..., 1], b[..., 1])
    inter_x2 = np.minimum(a[..., 2], b[..., 2])
    inter_y2 = np.minimum(a[..., 3], b[..., 3])

    inter_w = np.clip(inter_x2 - inter_x1, 0, None)
    inter_h = np.clip(inter_y2 - inter_y1, 0, None)
    inter_area = inter_w * inter_h  # (N, M)

    area_a = _area(a[..., 0], a[..., 1], a[..., 2], a[..., 3])  # (N,1)
    area_b = _area(b[..., 0], b[..., 1], b[..., 2], b[..., 3])  # (1,M)
    union = area_a + area_b - inter_area

    iou = np.where(union > 0, inter_area / np.where(union > 0, union, 1), 0.0)
    return iou
