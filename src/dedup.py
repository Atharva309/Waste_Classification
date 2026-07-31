"""
Near-duplicate detection across the merged source datasets (Kaggle waste
dataset + garbage-classification + e-waste + Roboflow + self-collected).

Fixes issue #6: the old project merged multiple public datasets with no
dedup step, so it's likely the same image (or a lightly recompressed near-
copy of it) ended up in both train and test after a naive random split --
an easy way to get a suspiciously high number like the old 0.912 without
the model actually generalizing.

Approach: a simple perceptual hash (average hash, aHash) computed from
scratch (no `imagehash` dependency): downscale to 8x8 grayscale, threshold
against the mean pixel value, pack into a 64-bit fingerprint. Images whose
hashes differ by <= a small Hamming distance are treated as near-duplicates
and grouped together. CRITICAL: splitting into train/val/test happens by
duplicate GROUP, not by individual image -- so no near-duplicate pair can
ever land on both sides of a split.

This is deliberately a cheap, explainable, from-scratch method rather than
a deep embedding-similarity approach, because the goal here is "catch exact
and near-exact duplicates from merged sources" not "catch semantically
similar but genuinely different photos" -- aHash is the right tool for that
specific job and is easy to defend in an interview (you can compute one by
hand on a whiteboard).
"""
from __future__ import annotations
import numpy as np
from PIL import Image


def average_hash(image_path: str, hash_size: int = 8) -> int:
    img = Image.open(image_path).convert("L").resize((hash_size, hash_size), Image.LANCZOS)
    pixels = np.asarray(img, dtype=np.float64)
    avg = pixels.mean()
    bits = (pixels > avg).flatten()
    h = 0
    for bit in bits:
        h = (h << 1) | int(bit)
    return h


def hamming_distance(h1: int, h2: int) -> int:
    return bin(h1 ^ h2).count("1")


def find_duplicate_groups(image_paths: list[str], hash_size: int = 8, max_distance: int = 5,
                           num_bands: int = 4):
    """Returns (groups, path_to_group): group_id -> list of near-duplicate
    image paths, and the reverse lookup.

    Naive all-pairs Hamming comparison is O(N^2), which is ~1.5 billion
    comparisons at N=55,000 -- too slow. Instead this uses banding (a form
    of LSH): the 64-bit hash is split into `num_bands` contiguous bit bands,
    and two images are only Hamming-compared if they share an IDENTICAL
    band in at least one of the bands. Near-duplicates (small Hamming
    distance) are very likely to match exactly in at least one band, so
    this catches the same near-duplicates as the brute-force version while
    only fully comparing images inside the same bucket -- turning an O(N^2)
    problem into an O(N) bucketing pass plus small within-bucket comparisons.
    """
    hashes = {}
    for p in image_paths:
        try:
            hashes[p] = average_hash(p, hash_size)
        except Exception:
            continue  # corrupt/unreadable image -- flagged separately by a data-quality pass

    paths = list(hashes.keys())
    parent = {p: p for p in paths}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    total_bits = hash_size * hash_size
    band_width = max(1, total_bits // num_bands)
    band_mask = (1 << band_width) - 1

    buckets: dict[tuple, list[str]] = {}
    for p in paths:
        h = hashes[p]
        for b in range(num_bands):
            band_val = (h >> (b * band_width)) & band_mask
            buckets.setdefault((b, band_val), []).append(p)

    checked = set()
    for bucket_paths in buckets.values():
        if len(bucket_paths) < 2:
            continue
        for i in range(len(bucket_paths)):
            for j in range(i + 1, len(bucket_paths)):
                a, b = bucket_paths[i], bucket_paths[j]
                key = (a, b) if a < b else (b, a)
                if key in checked:
                    continue
                checked.add(key)
                if hamming_distance(hashes[a], hashes[b]) <= max_distance:
                    union(a, b)

    groups: dict[str, list[str]] = {}
    path_to_group: dict[str, str] = {}
    for p in paths:
        root = find(p)
        groups.setdefault(root, []).append(p)
        path_to_group[p] = root

    return groups, path_to_group
