"""
Turns the classification dataset into a genuine (small, synthetic)
LOCALIZATION task, which is what makes IoU / NMS / Soft-NMS / mAP actually
meaningful here rather than decorative.

The original dataset (Kaggle waste + garbage-classification + e-waste +
Roboflow + self-collected) is whole-image classification: one label per
image, no bounding boxes. See & Spray-style problems are localization
problems (find and classify each plant in a frame), so a classification-only
pipeline understates what the role needs. Rather than claim bounding-box
detection on data that doesn't have boxes, this module is explicit about
what's real and what's synthetic:

  1. compose_belt_frame(): pastes several class-labeled crops onto a
     uniform "belt" background at known random positions/scales -> we get
     a frame WITH ground-truth boxes, honestly labeled as synthetic
     composition, not a real multi-object photograph.

  2. propose_regions_via_background_subtraction(): a classical-CV region
     proposal step that works BECAUSE the belt background is a known,
     roughly uniform color -- foreground pixels are found by color distance
     from the background, then connected-component boxes are extracted.
     This is a legitimate, explainable technique for a real fixed-camera
     conveyor belt (not a hand-wave): background subtraction against a
     static/near-static background is standard in industrial vision, and
     is the reason a "conveyor belt" framing was chosen for this demo
     rather than pretending to do general-scene object detection with a
     classifier that was never trained for it.

  3. The CNN classifier (src/models.py) then classifies each proposed
     crop, and NMS/Soft-NMS (src/nms.py) deduplicates overlapping
     proposals per class, and mAP (src/metrics.py) is computed against the
     known synthetic ground truth via IoU (src/iou.py).

This composition is the honest way to demo detection-style evaluation
without bounding-box-labeled training data, and it is a close proxy for the
belt-segregation scenario the project is actually simulating.
"""
from __future__ import annotations
import numpy as np
from PIL import Image
from scipy import ndimage as ndi  # optional convenience; see note below


def make_belt_background(width: int, height: int, color=(60, 60, 65), noise_std: float = 3.0,
                          seed: int | None = None) -> np.ndarray:
    rng = np.random.RandomState(seed)
    base = np.ones((height, width, 3), dtype=np.float64) * np.array(color)
    noise = rng.normal(0, noise_std, size=base.shape)
    return np.clip(base + noise, 0, 255).astype(np.uint8)


def compose_belt_frame(
    crops: list[tuple[np.ndarray, str]],  # (RGB array, class_name)
    frame_size: tuple[int, int] = (640, 480),
    belt_color=(60, 60, 65),
    min_scale: float = 0.15,
    max_scale: float = 0.35,
    allow_overlap: bool = False,
    seed: int | None = None,
):
    """Pastes each crop onto a belt-colored background at a random position
    and scale. Returns (frame_array, list of (class_name, (x1,y1,x2,y2))).

    allow_overlap=False rejects placements that IoU-overlap an already-
    placed box (retries a few times then gives up on that crop) -- keeps
    ground truth boxes unambiguous, matching how items would realistically
    be spaced on a belt.
    """
    from .iou import iou_single

    rng = np.random.RandomState(seed)
    w, h = frame_size
    frame = make_belt_background(w, h, belt_color, seed=seed).astype(np.float64)
    gt = []

    for crop_arr, class_name in crops:
        ch, cw = crop_arr.shape[:2]
        for attempt in range(20):
            scale = rng.uniform(min_scale, max_scale)
            new_w = max(8, int(w * scale))
            new_h = max(8, int(new_w * ch / cw))
            if new_w >= w or new_h >= h:
                continue
            x1 = rng.randint(0, w - new_w)
            y1 = rng.randint(0, h - new_h)
            box = (x1, y1, x1 + new_w, y1 + new_h)

            if not allow_overlap and any(iou_single(box, g[1]) > 0.0 for g in gt):
                continue

            resized = np.array(Image.fromarray(crop_arr).resize((new_w, new_h), Image.LANCZOS))
            
            if resized.shape[-1] == 4:
                alpha = (resized[:, :, 3] / 255.0)[..., np.newaxis]
                rgb = resized[:, :, :3]
                bg = frame[y1:y1 + new_h, x1:x1 + new_w]
                frame[y1:y1 + new_h, x1:x1 + new_w] = alpha * rgb + (1 - alpha) * bg
            else:
                frame[y1:y1 + new_h, x1:x1 + new_w] = resized
                
            gt.append((class_name, box))
            break
        # if all attempts failed, this crop is skipped for this frame (belt too crowded)

    return frame.astype(np.uint8), gt


import cv2

class BeltLocalizer:
    """
    Adaptive background subtraction and lightweight centroid tracking for the conveyor belt.
    Uses MOG2 to adapt to lighting drift, applies morphological filters to remove noise,
    and tracks the single largest object across frames.
    """
    def __init__(self, min_area=200, min_dim=32, pad=12, dist_threshold=150.0):
        self.min_area = min_area
        self.min_dim = min_dim
        self.pad = pad
        self.dist_threshold = dist_threshold
        
        # MOG2 adapts over time. history=100 is enough for a fast belt.
        # varThreshold=25, detectShadows=False
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=25, detectShadows=False)
        
        self.next_id = 1
        self.tracks = {}  # id -> {'centroid': (cx, cy), 'last_seen': frame_idx, 'crops': []}
        self.frame_idx = 0

    def update(self, frame: np.ndarray):
        """
        Updates the background model and returns (proposals, fg_mask).
        proposals is a list of tuples: (track_id, (x1, y1, x2, y2))
        fg_mask is the morphological-filtered foreground mask (for debugging).
        """
        self.frame_idx += 1
        
        # 1. Adaptive Background Subtraction
        fg_mask = self.bg_subtractor.apply(frame, learningRate=-1)
        
        # 2. Morphological filtering (open to remove speckles, close to fill holes)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
        
        # 3. Find contours
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Purge stale tracks (not seen for > 15 frames)
        stale_ids = [tid for tid, track in self.tracks.items() if self.frame_idx - track['last_seen'] > 15]
        for tid in stale_ids:
            del self.tracks[tid]
            
        if not contours:
            return [], fg_mask
            
        # 4. Filter and cap contours
        h_frame, w_frame = frame.shape[:2]
        valid_contours = []
        for c in contours:
            area = cv2.contourArea(c)
            if self.min_area <= area <= (h_frame * w_frame * 0.8):
                valid_contours.append((c, area))
        
        # Top 6 by area
        valid_contours.sort(key=lambda x: x[1], reverse=True)
        valid_contours = valid_contours[:6]
        
        if not valid_contours:
            return [], fg_mask
            
        # 5. Greedy centroid matching
        active_tids = list(self.tracks.keys())
        centroids_det = []
        for c, area in valid_contours:
            x, y, w, h = cv2.boundingRect(c)
            cx, cy = x + w / 2.0, y + h / 2.0
            centroids_det.append((cx, cy, c))
            
        distances = []
        for i, (cx, cy, c) in enumerate(centroids_det):
            for j, tid in enumerate(active_tids):
                track = self.tracks[tid]
                tcx, tcy = track['centroid']
                dist = np.hypot(cx - tcx, cy - tcy)
                distances.append((dist, i, j))
                
        distances.sort(key=lambda x: x[0])
        
        matched_dets = set()
        matched_tracks = set()
        matches = []
        
        for dist, i, j in distances:
            if dist > self.dist_threshold:
                continue
            if i in matched_dets or j in matched_tracks:
                continue
            matched_dets.add(i)
            matched_tracks.add(j)
            matches.append((i, active_tids[j]))
            
        # 6. Apply matches, create new tracks, apply padding
        proposals = []
        for i, (cx, cy, c) in enumerate(centroids_det):
            if i in matched_dets:
                tid = next(t_id for t_idx, t_id in matches if t_idx == i)
                self.tracks[tid]['centroid'] = (cx, cy)
                self.tracks[tid]['last_seen'] = self.frame_idx
            else:
                tid = self.next_id
                self.next_id += 1
                self.tracks[tid] = {'centroid': (cx, cy), 'last_seen': self.frame_idx, 'crops': []}
                
            x, y, w, h = cv2.boundingRect(c)
            new_w = max(w + 2 * self.pad, self.min_dim)
            new_h = max(h + 2 * self.pad, self.min_dim)
            
            x1 = int(max(0, cx - new_w / 2.0))
            y1 = int(max(0, cy - new_h / 2.0))
            x2 = int(min(w_frame, cx + new_w / 2.0))
            y2 = int(min(h_frame, cy + new_h / 2.0))
            
            crop = frame[y1:y2, x1:x2].copy()
            if crop.size > 0:
                self.tracks[tid]['crops'].append(crop)
                
            proposals.append((tid, (x1, y1, x2, y2)))
            
        return proposals, fg_mask
