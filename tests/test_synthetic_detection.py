import sys, os
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.synthetic_detection import compose_belt_frame, propose_regions_via_background_subtraction, make_belt_background
from src.iou import iou_single

def _crop(color, size=(40,40)):
    arr = np.ones((size[0], size[1], 3), dtype=np.uint8) * np.array(color, dtype=np.uint8)
    return arr

def test_belt_background_shape_and_color():
    bg = make_belt_background(100, 80, color=(60,60,65), noise_std=0, seed=0)
    assert bg.shape == (80, 100, 3)
    assert np.allclose(bg[0,0], [60,60,65], atol=1)

def test_compose_frame_no_overlap_boxes_are_disjoint():
    crops = [(_crop((255,0,0)), "plastic"), (_crop((0,255,0)), "paper"), (_crop((0,0,255)), "metal")]
    frame, gt = compose_belt_frame(crops, frame_size=(300,300), allow_overlap=False, seed=1)
    assert frame.shape == (300, 300, 3)
    assert len(gt) <= 3
    boxes = [g[1] for g in gt]
    for i in range(len(boxes)):
        for j in range(i+1, len(boxes)):
            assert iou_single(boxes[i], boxes[j]) == 0.0, "allow_overlap=False violated"

def test_compose_frame_gt_boxes_actually_contain_the_pasted_color():
    crops = [(_crop((200,10,10)), "plastic")]
    frame, gt = compose_belt_frame(crops, frame_size=(200,200), belt_color=(60,60,65), seed=2)
    assert len(gt) == 1
    cls, (x1,y1,x2,y2) = gt[0]
    center = frame[(y1+y2)//2, (x1+x2)//2]
    assert center[0] > 150  # should be reddish, matching the pasted crop, not belt-gray

def test_background_subtraction_finds_pasted_object():
    crops = [(_crop((255,255,255), size=(60,60)), "cardboard")]  # bright vs dark belt
    frame, gt = compose_belt_frame(crops, frame_size=(300,300), belt_color=(40,40,40), seed=3)
    proposals = propose_regions_via_background_subtraction(frame, belt_color=(40,40,40),
                                                             color_distance_threshold=25, min_area=50)
    assert len(proposals) >= 1
    # the GT box and at least one proposal should overlap substantially
    gt_box = gt[0][1]
    best_iou = max(iou_single(gt_box, p) for p in proposals)
    assert best_iou > 0.5, f"best proposal IoU with GT too low: {best_iou}"

def test_background_subtraction_empty_frame_no_proposals():
    frame = make_belt_background(200, 200, color=(60,60,65), noise_std=1, seed=4)
    proposals = propose_regions_via_background_subtraction(frame, belt_color=(60,60,65),
                                                             color_distance_threshold=25, min_area=50)
    assert len(proposals) == 0

if __name__ == "__main__":
    fns = [v for k, v in list(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"OK: {fn.__name__}")
    print(f"\n{len(fns)} synthetic-detection tests passed")
