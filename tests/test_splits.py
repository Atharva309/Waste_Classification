import sys, os, tempfile, shutil
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from PIL import Image
from src.splits import split_dataset_no_leakage, list_images_by_class
from src.dedup import average_hash, hamming_distance

def _make_dataset(tmp, n_groups_per_class=20, dup_per_group=3):
    classes = ["plastic", "paper", "metal"]
    rng = np.random.RandomState(42)
    for c in classes:
        os.makedirs(os.path.join(tmp, c), exist_ok=True)
        for g in range(n_groups_per_class):
            base = rng.randint(0, 255, size=(48, 48, 3))
            for d in range(dup_per_group):
                noisy = np.clip(base + rng.randint(-2, 2, size=base.shape), 0, 255).astype(np.uint8)
                Image.fromarray(noisy).save(os.path.join(tmp, c, f"{c}_{g}_{d}.png"))
    return classes

def test_no_near_duplicate_leakage_across_splits():
    tmp = tempfile.mkdtemp()
    try:
        _make_dataset(tmp)
        result = split_dataset_no_leakage(tmp, train_frac=0.6, val_frac=0.2, test_frac=0.2,
                                           dedup_max_distance=5, seed=0)

        train_paths = [p for p, _ in result.train]
        val_paths = [p for p, _ in result.val]
        test_paths = [p for p, _ in result.test]

        assert len(train_paths) > 0 and len(val_paths) > 0 and len(test_paths) > 0
        assert result.duplicate_groups_found >= 3 * 20 * 0  # sanity, groups exist
        assert result.duplicate_groups_found > 0

        # brute-force cross-check: no image in val/test should be a near-duplicate
        # (hamming distance <= 5) of any image in train
        train_hashes = [average_hash(p) for p in train_paths]
        for split_name, paths in [("val", val_paths), ("test", test_paths)]:
            for p in paths:
                h = average_hash(p)
                for th in train_hashes:
                    assert hamming_distance(h, th) > 5, f"{split_name} image {p} leaks into train"
        print(f"  groups found: {result.duplicate_groups_found}, "
              f"train={len(train_paths)} val={len(val_paths)} test={len(test_paths)}, "
              f"dropped_dup_copies_in_eval={result.dropped_duplicates}")
    finally:
        shutil.rmtree(tmp)

def test_split_fractions_roughly_respected():
    tmp = tempfile.mkdtemp()
    try:
        _make_dataset(tmp, n_groups_per_class=30, dup_per_group=1)  # no dups this time
        result = split_dataset_no_leakage(tmp, train_frac=0.7, val_frac=0.15, test_frac=0.15, seed=1)
        total = len(result.train) + len(result.val) + len(result.test)
        train_frac = len(result.train) / total
        assert 0.55 < train_frac < 0.85, train_frac
    finally:
        shutil.rmtree(tmp)

if __name__ == "__main__":
    fns = [v for k, v in list(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"OK: {fn.__name__}")
    print(f"\n{len(fns)} split tests passed")
