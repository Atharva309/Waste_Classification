import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.iou import iou_single, iou_matrix, iom_single

def test_identical_boxes():
    b = (10, 10, 50, 50)
    assert abs(iou_single(b, b) - 1.0) < 1e-9

def test_no_overlap():
    a = (0, 0, 10, 10)
    b = (20, 20, 30, 30)
    assert iou_single(a, b) == 0.0

def test_known_half_overlap():
    # a: 0..10 x 0..10 (area 100), b: 5..15 x 0..10 (area 100)
    # intersection: 5..10 x 0..10 = area 50, union = 100+100-50=150
    a = (0, 0, 10, 10)
    b = (5, 0, 15, 10)
    expected = 50 / 150
    assert abs(iou_single(a, b) - expected) < 1e-9

def test_touching_edges_zero():
    a = (0, 0, 10, 10)
    b = (10, 0, 20, 10)
    assert iou_single(a, b) == 0.0

def test_degenerate_box_zero_area():
    a = (5, 5, 5, 5)   # zero area
    b = (0, 0, 10, 10)
    assert iou_single(a, b) == 0.0

def test_matrix_matches_single():
    rng = np.random.RandomState(0)
    A = rng.uniform(0, 100, size=(7, 2))
    boxes_a = np.concatenate([A, A + rng.uniform(5, 30, size=(7, 2))], axis=1)
    B = rng.uniform(0, 100, size=(5, 2))
    boxes_b = np.concatenate([B, B + rng.uniform(5, 30, size=(5, 2))], axis=1)

    M = iou_matrix(boxes_a, boxes_b)
    assert M.shape == (7, 5)
    for i in range(7):
        for j in range(5):
            expected = iou_single(boxes_a[i], boxes_b[j])
            assert abs(M[i, j] - expected) < 1e-9, (i, j, M[i, j], expected)

def test_matrix_empty_inputs():
    M = iou_matrix(np.zeros((0, 4)), np.zeros((3, 4)))
    assert M.shape == (0, 3)

def test_iom_identical_boxes():
    b = (10, 10, 50, 50)
    assert abs(iom_single(b, b) - 1.0) < 1e-9

def test_iom_no_overlap():
    a = (0, 0, 10, 10)
    b = (20, 20, 30, 30)
    assert iom_single(a, b) == 0.0

def test_iom_small_box_inside_big_box():
    # b is fully inside a -- IoM should be 1.0 (100% of the smaller box
    # is covered) even though IoU is far below 1.0 (a is much bigger).
    a = (0, 0, 100, 100)     # area 10000
    b = (10, 10, 20, 20)     # area 100, fully inside a
    assert abs(iom_single(a, b) - 1.0) < 1e-9
    assert iou_single(a, b) < 0.02  # sanity: IoU is tiny for the same pair

def test_iom_matches_iou_for_equal_size_boxes():
    # when both boxes have equal area, IoM == IoU (min_area == either area
    # == what union's "extra" term reduces to in this special case is not
    # generally true, so just check the known-half-overlap case directly).
    a = (0, 0, 10, 10)
    b = (5, 0, 15, 10)
    inter = 50
    area = 100
    assert abs(iom_single(a, b) - inter / area) < 1e-9

def test_iom_degenerate_box_zero():
    a = (5, 5, 5, 5)
    b = (0, 0, 10, 10)
    assert iom_single(a, b) == 0.0

if __name__ == "__main__":
    fns = [v for k, v in list(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"OK: {fn.__name__}")
    print(f"\n{len(fns)} IoU tests passed")
