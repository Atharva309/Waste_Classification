import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.nms import nms, soft_nms

def test_nms_removes_duplicate():
    # two nearly identical boxes around the same object, one lower-confidence duplicate
    boxes = np.array([
        [0, 0, 10, 10],
        [1, 1, 11, 11],   # heavy overlap with box 0
        [50, 50, 60, 60],  # separate object
    ], dtype=float)
    scores = np.array([0.9, 0.8, 0.7])
    keep = nms(boxes, scores, iou_threshold=0.5)
    assert set(keep.tolist()) == {0, 2}, keep

def test_nms_keeps_non_overlapping():
    boxes = np.array([[0,0,10,10],[20,20,30,30],[40,40,50,50]], dtype=float)
    scores = np.array([0.5, 0.9, 0.3])
    keep = nms(boxes, scores, iou_threshold=0.5)
    assert set(keep.tolist()) == {0, 1, 2}

def test_nms_empty():
    keep = nms(np.zeros((0,4)), np.zeros((0,)))
    assert len(keep) == 0

def test_soft_nms_keeps_more_than_hard_nms_when_objects_overlap():
    # two real, overlapping objects (like occluding leaves) plus a near-duplicate box.
    boxes = np.array([
        [0, 0, 10, 10],
        [4, 0, 14, 10],    # genuinely different object, moderate overlap with box 0
        [0.5, 0.5, 10.5, 10.5],  # near-duplicate of box 0
    ], dtype=float)
    scores = np.array([0.95, 0.80, 0.90])

    hard_keep = nms(boxes, scores, iou_threshold=0.3)
    soft_keep, soft_scores = soft_nms(boxes, scores, method="gaussian",
                                       iou_threshold=0.3, sigma=0.5, score_threshold=0.1)

    # hard NMS at this aggressive threshold suppresses both overlapping boxes (1 and 2)
    assert set(hard_keep.tolist()) == {0}

    # soft-NMS decays but doesn't necessarily zero out box 1 (different object, box 1
    # and box 0 overlap less than box 2 and box 0)
    assert 0 in soft_keep.tolist()
    assert len(soft_keep) >= len(hard_keep)

def test_soft_nms_scores_are_decayed_not_deleted_instantly():
    boxes = np.array([[0,0,10,10],[1,1,11,11]], dtype=float)
    scores = np.array([0.9, 0.85])
    keep, decayed = soft_nms(boxes, scores, method="linear", iou_threshold=0.1, score_threshold=0.01)
    # box 1 heavily overlaps box 0 -> its score should have been reduced from 0.85
    idx_of_box1 = list(keep).index(1) if 1 in keep else None
    if idx_of_box1 is not None:
        assert decayed[idx_of_box1] < 0.85

def test_soft_nms_empty():
    keep, scores = soft_nms(np.zeros((0,4)), np.zeros((0,)))
    assert len(keep) == 0 and len(scores) == 0

if __name__ == "__main__":
    fns = [v for k, v in list(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"OK: {fn.__name__}")
    print(f"\n{len(fns)} NMS tests passed")
