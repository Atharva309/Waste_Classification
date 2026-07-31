# Script Redundancy Mapping

| Script | Relationship / Note |
|-------------------|----------------------------------------------|
| `scripts/train_stage1.py` | Superseded by `scripts/retrain_cascade.py` (combined Stage 1 + 2 retraining). See `scripts/retrain_cascade.py` for the unified training pipeline. |
| `scripts/train_stage2.py` | Superseded by `scripts/retrain_cascade.py` (combined Stage 1 + 2 retraining). See `scripts/retrain_cascade.py` for the unified training pipeline. |
| `scripts/generate_yolo_cutouts.py` | Stage 1: produces alpha cutouts from source images. |
| `scripts/generate_yolo_dataset.py` | Stage 2: composites cutouts into labeled training frames (reads from `outputs/yolo_cutouts`). |

Each superseded script now contains a comment at the top pointing to the newer implementation.
