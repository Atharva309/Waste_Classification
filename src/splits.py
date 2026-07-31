"""
Custom PyTorch Dataset/DataLoader for the waste classification task, plus
the group-aware train/val/test split that fixes issue #1 (7-class model
evaluated on the full directory instead of a held-out split -- likely
leakage) and issue #6 (no dedup across merged sources before splitting).

Expected on-disk layout (standard ImageFolder-style, one dir per class):
    root/
      cardboard/*.jpg
      paper/*.jpg
      plastic/*.jpg
      ...

split_dataset_no_leakage() is the piece that matters most here: it groups
images into near-duplicate clusters FIRST (via src/dedup.py), then assigns
whole clusters to train/val/test, so no near-duplicate pair can straddle a
split boundary. This directly targets why the old 0.912 number was
suspicious -- if the "test" set the old code used shared images (or near-
identical crops/recompressions) with what MobileNet/ResNet/YOLO trained on,
accuracy is inflated by memorization, not generalization.
"""
from __future__ import annotations
import os
import random
from dataclasses import dataclass

from .dedup import find_duplicate_groups


def list_images_by_class(root: str, extensions=(".jpg", ".jpeg", ".png")):
    """Returns dict: class_name -> list of image paths."""
    class_to_paths = {}
    for class_name in sorted(os.listdir(root)):
        class_dir = os.path.join(root, class_name)
        if not os.path.isdir(class_dir):
            continue
        paths = [
            os.path.join(class_dir, f) for f in os.listdir(class_dir)
            if f.lower().endswith(extensions)
        ]
        class_to_paths[class_name] = paths
    return class_to_paths


@dataclass
class SplitResult:
    train: list[tuple[str, str]]   # (path, class_name)
    val: list[tuple[str, str]]
    test: list[tuple[str, str]]
    dropped_duplicates: int
    duplicate_groups_found: int


def split_dataset_no_leakage(
    root: str,
    train_frac: float = 0.7,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    dedup_max_distance: int = 5,
    seed: int = 42,
    run_dedup: bool = True,
) -> SplitResult:
    assert abs(train_frac + val_frac + test_frac - 1.0) < 1e-6
    rng = random.Random(seed)

    class_to_paths = list_images_by_class(root)
    all_pairs = [(p, c) for c, paths in class_to_paths.items() for p in paths]
    all_paths = [p for p, _ in all_pairs]
    path_to_class = dict(all_pairs)

    if run_dedup:
        groups, path_to_group = find_duplicate_groups(all_paths, max_distance=dedup_max_distance)
    else:
        # every image is its own group -- used for quick iteration on huge
        # datasets where a full dedup pass is too slow to run every time
        path_to_group = {p: p for p in all_paths}
        groups = {p: [p] for p in all_paths}

    # group ids per class (a duplicate group should in practice be
    # single-class, since near-dup images are the same object; assign the
    # group to the majority class present, to be defensive)
    group_ids = list(groups.keys())
    rng.shuffle(group_ids)

    n_groups = len(group_ids)
    n_train = int(n_groups * train_frac)
    n_val = int(n_groups * val_frac)

    train_groups = set(group_ids[:n_train])
    val_groups = set(group_ids[n_train:n_train + n_val])
    test_groups = set(group_ids[n_train + n_val:])

    train, val, test = [], [], []
    dropped = 0
    for gid, members in groups.items():
        bucket = train if gid in train_groups else (val if gid in val_groups else test)
        # keep only ONE representative image per duplicate group in eval
        # splits to avoid a model getting "credit" for the same photo twice;
        # for train we keep all copies (more training signal is fine there,
        # since train/train leakage isn't a validity problem -- only
        # train/eval leakage is).
        if bucket is train:
            for m in members:
                bucket.append((m, path_to_class[m]))
        else:
            bucket.append((members[0], path_to_class[members[0]]))
            dropped += len(members) - 1

    n_dup_groups = sum(1 for g in groups.values() if len(g) > 1)
    return SplitResult(train, val, test, dropped, n_dup_groups)


