"""
Conveyor-belt real-time-feed simulation: items ("trash") cross a moving
belt one frame at a time, the pipeline detects + classifies + deduplicates
them, and routes each accepted detection into the correct output bin
(segregation) -- the actual end-to-end demo, not just a static single-image
classifier call.

Pipeline per frame:
  1. propose_regions_via_background_subtraction (src/synthetic_detection.py)
     -- classical CV region proposals against the known belt background.
  2. classify_fn(crop) for each proposal -> (class_name, confidence). This
     is a pluggable callable: swap in the trained PyTorch classifier
     (src/models.py) once available, or a placeholder for architecture
     testing (this repo ships both, see scripts/demo_conveyor.py).
  3. ThresholdedDetector (src/inference.py) filters by confidence mode
     (high_recall / balanced / high_precision).
  4. Per-class NMS or Soft-NMS (src/nms.py) removes duplicate proposals on
     the same physical object.
  5. Segregation: each surviving detection is routed into a bin keyed by
     predicted class -- the "detector correctly identifying trash and
     segregating it" behavior. Bin routing + per-frame latency are logged.

This mirrors the belt/moving-camera framing of See & Spray: sequential
frames, a real-time per-frame latency budget, and an actual downstream
action taken per detection (route to a bin here; trigger a nozzle there).
"""
from __future__ import annotations
import time
from dataclasses import dataclass, field

import numpy as np
from PIL import Image, ImageDraw

from .synthetic_detection import BeltLocalizer
from .nms import nms, soft_nms
from .iou import iou_matrix


@dataclass
class FrameResult:
    frame_index: int
    detections: list  # list of (class_name, box, score)
    n_proposals: int
    latency_ms: float


@dataclass
class BeltRunSummary:
    frame_results: list = field(default_factory=list)
    bins: dict = field(default_factory=dict)  # class_name -> count segregated

    def total_segregated(self):
        return sum(self.bins.values())


class ConveyorBeltSimulator:
    def __init__(
        self,
        classify_fn,               # crop_array (H,W,3 uint8) -> (class_name, confidence)
        belt_color=(60, 60, 65),
        nms_method: str = "soft_gaussian",  # "hard" | "soft_linear" | "soft_gaussian"
        iou_threshold: float = 0.4,
        confidence_mode: str = "balanced",
    ):
        self.classify_fn = classify_fn
        self.belt_color = belt_color
        self.nms_method = nms_method
        self.iou_threshold = iou_threshold
        self.belt_localizer = BeltLocalizer()

        from .inference import ThresholdedDetector, PRESET_THRESHOLDS
        self.confidence_threshold = PRESET_THRESHOLDS[confidence_mode]
        self.confidence_mode = confidence_mode

    def set_confidence_mode(self, mode: str):
        from .inference import PRESET_THRESHOLDS
        self.confidence_threshold = PRESET_THRESHOLDS[mode]
        self.confidence_mode = mode

    def _dedup_per_class(self, boxes_by_class):
        """Applies hard NMS or Soft-NMS independently per predicted class."""
        final = {}
        for cls, (boxes, scores) in boxes_by_class.items():
            boxes = np.array(boxes, dtype=float)
            scores = np.array(scores, dtype=float)
            if len(boxes) == 0:
                continue
            if self.nms_method == "hard":
                keep = nms(boxes, scores, iou_threshold=self.iou_threshold)
                final[cls] = (boxes[keep], scores[keep])
            else:
                method = "linear" if self.nms_method == "soft_linear" else "gaussian"
                keep, decayed = soft_nms(boxes, scores, method=method,
                                          iou_threshold=self.iou_threshold, score_threshold=0.05)
                final[cls] = (boxes[keep], decayed)
        return final

    def process_frame(self, frame: np.ndarray, frame_index: int = 0) -> FrameResult:
        t0 = time.perf_counter()

        proposals, _ = self.belt_localizer.update(frame)

        boxes_by_class: dict[str, tuple[list, list]] = {}
        for tid, box in proposals:
            x1, y1, x2, y2 = box
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            cls, score = self.classify_fn(crop)
            if score < self.confidence_threshold:
                continue
            boxes_by_class.setdefault(cls, ([], []))
            boxes_by_class[cls][0].append(box)
            boxes_by_class[cls][1].append(score)

        deduped = self._dedup_per_class(boxes_by_class)

        detections = []
        for cls, (boxes, scores) in deduped.items():
            for b, s in zip(boxes, scores):
                detections.append((cls, tuple(b), float(s)))

        latency_ms = (time.perf_counter() - t0) * 1000.0
        return FrameResult(frame_index, detections, len(proposals), latency_ms)

    def run(self, frames: list[np.ndarray]) -> BeltRunSummary:
        summary = BeltRunSummary()
        for i, frame in enumerate(frames):
            result = self.process_frame(frame, frame_index=i)
            summary.frame_results.append(result)
            for cls, box, score in result.detections:
                summary.bins[cls] = summary.bins.get(cls, 0) + 1
        return summary

    def latency_stats(self, summary: BeltRunSummary):
        from .latency import _percentile
        times = sorted(r.latency_ms for r in summary.frame_results)
        if not times:
            return {}
        return {
            "mean_ms": float(np.mean(times)), "p50_ms": _percentile(times, 50),
            "p95_ms": _percentile(times, 95), "max_ms": float(np.max(times)),
            "throughput_fps": 1000.0 / np.mean(times),
        }


def draw_detections(frame: np.ndarray, detections, class_colors: dict | None = None) -> Image.Image:
    img = Image.fromarray(frame).convert("RGB")
    draw = ImageDraw.Draw(img)
    default_colors = ["#e63946", "#2a9d8f", "#f4a261", "#264653", "#e9c46a",
                       "#8ab17d", "#457b9d", "#ff70a6", "#606c38", "#6d597a", "#219ebc"]
    class_colors = class_colors or {}
    for i, (cls, box, score) in enumerate(detections):
        color = class_colors.get(cls, default_colors[hash(cls) % len(default_colors)])
        x1, y1, x2, y2 = box
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        draw.text((x1 + 2, max(0, y1 - 12)), f"{cls} {score:.2f}", fill=color)
    return img


def save_belt_gif(frames: list[np.ndarray], all_detections: list[list], out_path: str, duration_ms=600):
    annotated = [draw_detections(f, d) for f, d in zip(frames, all_detections)]
    if not annotated:
        return
    annotated[0].save(out_path, save_all=True, append_images=annotated[1:],
                       duration=duration_ms, loop=0)
