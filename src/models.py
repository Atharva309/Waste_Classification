"""
PyTorch model definitions via torchvision, replacing the old Keras
MobileNet/ResNet implementation.

Fixes issue #2 from the old project: preprocessing now uses each backbone's
own ImageNet transform (mean/std normalization) via
`torchvision.models.<Weights>.transforms()`, not a blanket `rescale=1./255`.
ResNet50 and MobileNetV3 do NOT expect [0,1]-scaled input; they expect
ImageNet-normalized input, and get it here automatically and correctly.

Fixes issue #7/#8: architecture is explicitly CNN-based (ResNet50 /
MobileNetV3-Large from torchvision), loaded with ImageNet weights, and
transfer learning is staged: freeze backbone -> train head only -> optionally
unfreeze and fine-tune end-to-end if the frozen-backbone result is not good
enough. This mirrors what you'd actually do at a company with limited
labeled data (crop/weed images) and a lot of compute pressure.
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torchvision
from torchvision.models import resnet50, ResNet50_Weights, mobilenet_v3_large, MobileNet_V3_Large_Weights


SUPPORTED_ARCHS = ("resnet50", "mobilenet_v3_large")


def get_model_and_transforms(arch: str, num_classes: int, pretrained: bool = True, is_stage2: bool = False, augment_level: str = "gentle"):
    """Returns (model, train_transform, eval_transform).

    The transforms come directly from torchvision's packaged weights metadata
    -- this is the correct, version-matched preprocessing for the specific
    pretrained checkpoint (resize, center crop, ImageNet mean/std). No manual
    rescale=1/255 anywhere.
    """
    if arch not in SUPPORTED_ARCHS:
        raise ValueError(f"arch must be one of {SUPPORTED_ARCHS}, got {arch}")

    if arch == "resnet50":
        weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        model = resnet50(weights=weights)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
        head_module_names = ["fc"]
    else:  # mobilenet_v3_large
        weights = MobileNet_V3_Large_Weights.IMAGENET1K_V2 if pretrained else None
        model = mobilenet_v3_large(weights=weights)
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)
        head_module_names = ["classifier"]

    eval_transform = weights.transforms() if weights is not None else _fallback_transform()
    # Light augmentation on top of the same normalization for training.
    train_transform = _build_train_transform(eval_transform, is_stage2=is_stage2, augment_level=augment_level)

    model._head_module_names = head_module_names  # stashed for freeze/unfreeze helpers
    return model, train_transform, eval_transform


def _fallback_transform():
    from torchvision import transforms as T
    return T.Compose([
        T.Resize(256), T.CenterCrop(224), T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def _build_train_transform(eval_transform, is_stage2=False, augment_level="gentle"):
    from torchvision import transforms as T
    # Reuse the eval transform's resize/crop/normalize stats, just add
    # flips/jitter for training. Extracting resize size defensively.
    resize_size = 224
    for t in getattr(eval_transform, "transforms", []):
        if hasattr(t, "size"):
            size = t.size
            resize_size = size[0] if isinstance(size, (list, tuple)) else size
            break
            
    if is_stage2:
        if augment_level == "heavy":
            return T.Compose([
                T.RandomResizedCrop(resize_size, scale=(0.7, 1.0)),
                T.RandomHorizontalFlip(p=0.5),
                T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.4, hue=0.1),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                T.RandomErasing(p=0.3, scale=(0.02, 0.15))
            ])
        else:
            return T.Compose([
                T.RandomResizedCrop(resize_size, scale=(0.7, 1.0)),
                T.RandomHorizontalFlip(p=0.5),
                T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
    else:
        return T.Compose([
            T.RandomResizedCrop(resize_size, scale=(0.7, 1.0)),
            T.RandomHorizontalFlip(p=0.5),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])


def freeze_backbone(model: nn.Module):
    """Phase 1 of transfer learning: freeze everything except the head."""
    head_names = getattr(model, "_head_module_names", ["fc"])
    for name, param in model.named_parameters():
        param.requires_grad = any(name.startswith(h) for h in head_names)
    return model


def unfreeze_all(model: nn.Module, unfreeze_from_layer: str | None = None):
    """Phase 2 (only if phase-1 metrics are insufficient): unfreeze for full
    fine-tuning. `unfreeze_from_layer` lets you unfreeze only the last N
    blocks instead of the whole network, which is usually the better move
    with a dataset this size (avoids catastrophic forgetting of ImageNet
    features on ~55k images)."""
    if unfreeze_from_layer is None:
        for param in model.parameters():
            param.requires_grad = True
        return model

    unfreezing = False
    for name, param in model.named_parameters():
        if unfreeze_from_layer in name:
            unfreezing = True
        param.requires_grad = unfreezing or param.requires_grad
    return model


def trainable_param_count(model: nn.Module) -> tuple[int, int]:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total
