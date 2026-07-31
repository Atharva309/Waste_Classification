---
title: Trashcam App
emoji: ♻️
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: "4.36.0"
app_file: app.py
pinned: false
---

# Waste Classification -- PyTorch Rebuild

A real-time waste-sorting computer vision project built around a simulated
conveyor belt: objects move across a live camera feed, get localized,
classified into one of six recyclable material classes (or "organic"), and
segregated into bins -- with per-class precision/recall/F1, mAP, and latency
tracked the whole time. Originally a rebuild of an earlier Keras project
(see the repo root for that legacy version); this PyTorch rebuild fixes its
methodology problems (see "Fixes to the 8 known issues" below) and has since
grown into a full working system with two independent detection
architectures, live A/B testing, and a from-scratch calibration layer.

## Two architectures, one live webapp

- **Cascade classifier** (`detector_model=cascade`): classical background
  subtraction (`BeltLocalizer`) proposes object regions, then a two-stage
  MobileNetV3 pipeline classifies each one -- Stage 1 is a binary
  organic-vs-recyclable gate, and only if Stage 1 says "not organic" does
  Stage 2 run its 6-class material classifier (cardboard, ewaste, glass,
  metal, paper, plastic). This gating mirrors classic cascade classifiers
  (Viola-Jones face detection, MTCNN): a cheap filter handles the easy
  majority case, so the expensive specialized model only runs on what
  actually needs it.
- **YOLO detector** (`detector_model=yolo`): a single YOLOv8n forward pass
  predicts box + class together. Faster and handles overlapping objects
  better (98%+ of deliberately-overlapping test items separated into
  distinct detections, vs. ~45% for background subtraction), at some
  accuracy cost since its classification head is a lightweight side-output
  of a detection-focused network.
- **Hybrid / "Frankenstein"** (`detector_model=frankenstein`, the
  recommended default): takes the best of both -- YOLO's learned,
  overlap-tolerant box proposals for localization, then discards YOLO's own
  classification and feeds every box through the cascade's proven, heavily
  calibrated Stage 2 classifier instead. This is the best-performing
  configuration found (~74.7% average live macro-F1 across final testing,
  vs. ~71.5% for the cascade alone).

All three are live-selectable in the webapp (`webapp/app.py` +
`webapp/templates/index.html`) with tunable confidence presets
(high recall / balanced / high precision) and a choice of NMS strategy
(hard, soft-linear, soft-Gaussian).

## Evaluation results

**Stage 2 material classifier (`lastditchattempt`, the deployed default)**
-- offline test-set confusion matrix, 88.2% accuracy / 88.4% macro-F1:

![Cascade Stage 2 confusion matrix](docs/images/cascade_stage2_confusion_matrix.png)

**YOLOv8n detector (`yolofirstactual`, the deployed default)** -- offline
evaluation and single-frame inference latency:

![YOLO evaluation summary](docs/images/yolo_eval_summary.png)

*(These are regenerated from the saved evaluation JSON in `outputs/`, not
Ultralytics' own training-run plots -- those lived under `runs/`, which was
cleaned up as training-run scratch space. Re-run
`scripts/train_evaluate_yolo.py` if you want the original Ultralytics
plots back.)*

**Live, end-to-end results** (the numbers that actually matter -- full
belt simulation, multi-frame evidence accumulation, real-time):

| Configuration | Avg. live macro-F1 |
|---|---|
| Cascade alone (`lastditchattempt` + cardboard floor) | ~71.5% |
| **Hybrid / Frankenstein** (YOLO localization + cascade classification) | **~74.7%** |

Known limitation: plastic recall is the weakest spot in both
configurations, root-caused to genuine TACO source-image quality (tiny,
barely-visible plastic litter fragments) rather than a fixable pipeline or
calibration bug -- confirmed by directly inspecting the failing live crops.
See `NOTES.md` for the full investigation and every other design decision,
bug, and dead end from building this.

## Fixes to the 8 known issues in the original Keras project

1. **7-class eval on the full directory (leakage).** `src/splits.py`'s
   `split_dataset_no_leakage()` does a proper train/val/test split.
2. **`rescale=1./255` for ResNet.** `src/models.py` uses each torchvision
   backbone's own packaged `weights.transforms()` (ImageNet mean/std
   normalization), not manual [0,1] rescaling.
3. **Plain accuracy reported throughout.** `src/engine.evaluate()` always
   returns accuracy *alongside* per-class precision/recall/F1 and a
   confusion matrix.
4. **No confusion matrix / per-class P/R/F1.** `src/metrics.py` --
   cross-checked against sklearn in `tests/test_metrics.py`.
5. **7-class vs. 11-class training-time confound.** Flagged as an
   experiment-design issue; `src/engine.run_transfer_learning()`'s
   phase-1/phase-2 epoch counts are explicit config, not implicit magic
   numbers.
6. **No dedup across merged sources.** `src/dedup.py` (perceptual hash +
   LSH banding) runs before splitting; verified in `tests/test_splits.py`.
7. **No systematic latency measurement.** `src/latency.py`: warmup
   iterations excluded, full percentile report (p50/p90/p95/p99/max),
   batch=1 (the realistic belt scenario).
8. **Entirely Keras/TensorFlow.** This whole repo -- `src/models.py`,
   `src/engine.py`, `src/datasets.py`.

## Layout

```
src/
  iou.py                    # IoU single + vectorized
  nms.py                    # NMS + Soft-NMS from scratch
  metrics.py                 # confusion matrix, P/R/F1, PR curves, mAP
  dedup.py                    # perceptual hash + LSH near-dup grouping
  splits.py                    # leak-free group-aware train/val/test split
  datasets.py                   # torch Dataset wrapping splits.py
  models.py                      # ResNet50/MobileNetV3 + transfer-learning helpers
  engine.py                       # train/eval loop, transfer-learning orchestration
  latency.py                       # warmup + percentile latency benchmarking
  inference.py                      # tunable confidence threshold presets
  synthetic_detection.py             # BeltLocalizer + crop compositing
webapp/
  app.py                               # Flask backend: cascade/YOLO/hybrid inference
  templates/index.html                  # live belt simulation + dashboard UI
scripts/                                  # training, calibration, dataset-generation scripts
tests/                                     # automated tests, run via tests/run_all.py
outputs/                                    # deployed checkpoints + evaluation reports
NOTES.md                                     # full project history: decisions, bugs, dead ends
```

Run the automated tests with:
```
python3 tests/run_all.py
```

## Running the webapp

```
pip install -r requirements.txt
python3 webapp/app.py
```

Then open the local URL it prints. Pick a model from the "Model" dropdown
(Hybrid is the recommended default), start the belt, and watch live
per-class metrics build up on the Session Metrics Dashboard tab.

## Data

The datasets this project was trained on (`data/cascades_data`,
`data/yolo_data`) are not included in this repository -- they were large
image sets used only during training/fine-tuning and removed once the
final checkpoints in `outputs/` were trained. See `NOTES.md`'s Data
Pipeline section for the sources used (TACO, TrashNet, and merged
Kaggle/Roboflow garbage-classification sets) and how they were composited
into training data.
