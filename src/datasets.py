"""
Dataset/DataLoader wrapper around the leak-free split logic in src/splits.py
(kept in a separate module so the split/dedup logic can be unit tested
without a torch dependency).
"""
from __future__ import annotations
from PIL import Image
from torch.utils.data import Dataset

from .splits import split_dataset_no_leakage, list_images_by_class, SplitResult  # re-exported


class WasteImageDataset(Dataset):
    """Standard classification Dataset: (path, class_name) pairs -> (tensor, label_idx)."""

    def __init__(self, samples: list[tuple[str, str]], class_to_idx: dict[str, int], transform=None):
        self.samples = samples
        self.class_to_idx = class_to_idx
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, class_name = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        label = self.class_to_idx[class_name]
        return img, label

    @staticmethod
    def build_class_to_idx(class_names: list[str]) -> dict[str, int]:
        return {c: i for i, c in enumerate(sorted(set(class_names)))}
