import sys, os, tempfile, shutil
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from PIL import Image
from src.dedup import average_hash, hamming_distance, find_duplicate_groups

def _save(path, arr):
    Image.fromarray(arr.astype(np.uint8)).save(path)

def test_identical_images_hash_equal():
    tmp = tempfile.mkdtemp()
    try:
        rng = np.random.RandomState(0)
        arr = rng.randint(0, 255, size=(64, 64, 3))
        p1 = os.path.join(tmp, "a.png"); p2 = os.path.join(tmp, "b.png")
        _save(p1, arr); _save(p2, arr)
        assert average_hash(p1) == average_hash(p2)
    finally:
        shutil.rmtree(tmp)

def test_near_duplicate_jpeg_recompress_close_hash():
    tmp = tempfile.mkdtemp()
    try:
        rng = np.random.RandomState(1)
        arr = rng.randint(0, 255, size=(128, 128, 3))
        p1 = os.path.join(tmp, "orig.png")
        p2 = os.path.join(tmp, "recompressed.jpg")
        _save(p1, arr)
        Image.fromarray(arr.astype(np.uint8)).save(p2, quality=70)  # lossy recompress
        d = hamming_distance(average_hash(p1), average_hash(p2))
        assert d <= 8, f"expected near-duplicate hash distance, got {d}"
    finally:
        shutil.rmtree(tmp)

def test_different_images_hash_far_apart():
    # flat black/white images are a known aHash degenerate case (every pixel
    # equals the mean -> hash is all-zero for both), so use structured,
    # visually distinct random images instead -- the realistic case for
    # actually-different waste photos.
    tmp = tempfile.mkdtemp()
    try:
        rng1 = np.random.RandomState(10)
        rng2 = np.random.RandomState(99)
        img1 = rng1.randint(0, 255, size=(64, 64, 3))
        img2 = rng2.randint(0, 255, size=(64, 64, 3))
        p1 = os.path.join(tmp, "img1.png"); p2 = os.path.join(tmp, "img2.png")
        _save(p1, img1); _save(p2, img2)
        d = hamming_distance(average_hash(p1), average_hash(p2))
        assert d > 8, f"expected clearly different images to hash far apart, got {d}"
    finally:
        shutil.rmtree(tmp)

def test_find_duplicate_groups_clusters_correctly():
    tmp = tempfile.mkdtemp()
    try:
        rng = np.random.RandomState(3)
        img_a = rng.randint(0, 255, size=(64, 64, 3))
        img_b = rng.randint(0, 255, size=(64, 64, 3))
        paths = []
        for i in range(3):
            p = os.path.join(tmp, f"a_{i}.png")
            noisy = np.clip(img_a + rng.randint(-3, 3, size=img_a.shape), 0, 255)
            _save(p, noisy)
            paths.append(p)
        for i in range(2):
            p = os.path.join(tmp, f"b_{i}.png")
            noisy = np.clip(img_b + rng.randint(-3, 3, size=img_b.shape), 0, 255)
            _save(p, noisy)
            paths.append(p)

        groups, path_to_group = find_duplicate_groups(paths, max_distance=5)
        # the 3 "a" images should share a group id, the 2 "b" images should share a different one
        a_groups = {path_to_group[p] for p in paths if "a_" in os.path.basename(p)}
        b_groups = {path_to_group[p] for p in paths if "b_" in os.path.basename(p)}
        assert len(a_groups) == 1, a_groups
        assert len(b_groups) == 1, b_groups
        assert a_groups != b_groups
    finally:
        shutil.rmtree(tmp)

if __name__ == "__main__":
    fns = [v for k, v in list(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"OK: {fn.__name__}")
    print(f"\n{len(fns)} dedup tests passed")
