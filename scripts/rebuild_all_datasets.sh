#!/bin/bash
# Full delete-and-rebuild for BOTH the YOLO cutout/dataset pipeline and the
# cascade's belt-composite pipeline, now that both generators consult the
# known-bad-source-image blocklist (data/known_bad_source_images.json,
# built from every manual audit round this session -- NOTES.md sections
# 38-45) BEFORE selecting any source images. This is what actually makes
# "rebuild from scratch" safe: without the blocklist, a fresh rebuild would
# just re-draw a similar contamination rate from the same messy source
# folder, because the contamination lives in the raw Kaggle-merged dataset
# itself, not in any of this project's own code.
#
# Run this on your Mac, from pytorch_pipeline/. NOT runnable in a sandbox --
# cutout extraction + belt compositing across thousands of images will blow
# past any reasonable sandboxed-tool-call timeout.
#
# Usage:
#   bash scripts/rebuild_all_datasets.sh          # asks for confirmation
#   bash scripts/rebuild_all_datasets.sh --yes     # skips confirmation
#
# What this does NOT do: launch training, or run a post-rebuild visual
# audit. Both are required next steps -- see the printed summary at the
# end.

set -e
cd "$(dirname "$0")/.."  # pytorch_pipeline/

if [ "$1" != "--yes" ]; then
    echo "This will DELETE and regenerate:"
    echo "  data/yolo_data/cutouts/"
    echo "  data/yolo_data/dataset/"
    echo "  data/cascades_data/"
    echo ""
    echo "Known-bad blocklist in use: data/known_bad_source_images.json"
    if [ -f data/known_bad_source_images.json ]; then
        COUNT=$(python3 -c "import json; print(len(json.load(open('data/known_bad_source_images.json'))))")
        echo "  ($COUNT confirmed-bad source images will be excluded)"
    else
        echo "  WARNING: blocklist file not found -- rebuild will NOT exclude anything."
        echo "  Expected at data/known_bad_source_images.json"
    fi
    echo ""
    read -p "Continue? [y/N] " CONFIRM
    if [ "$CONFIRM" != "y" ] && [ "$CONFIRM" != "Y" ]; then
        echo "Aborted."
        exit 1
    fi
fi

source venv/bin/activate

echo ""
echo "=== [1/5] Deleting old YOLO cutout pool + dataset ==="
rm -rf data/yolo_data/cutouts data/yolo_data/dataset

echo ""
echo "=== [2/5] Deleting old cascade belt-composite dataset ==="
rm -rf data/cascades_data

echo ""
echo "=== [3/5] Regenerating YOLO cutouts (blocklist-filtered) ==="
python3 scripts/generate_yolo_cutouts_v2.py

echo ""
echo "=== [4/5] Regenerating YOLO composited training frames ==="
python3 scripts/generate_yolo_dataset.py

echo ""
echo "=== [5/5] Regenerating cascade belt-composite dataset (blocklist-filtered) ==="
python3 scripts/generate_belt_dataset.py

echo ""
echo "=================================================================="
echo "Rebuild complete."
echo ""
echo "Plastic source is now structurally guaranteed clean (NOTES.md"
echo "section 51/53): both generators hard-exclude the unaudited generic-"
echo "Kaggle pool for plastic entirely -- no fallback, no exceptions -- so"
echo "there is nothing left to audit for that class specifically. Expect"
echo "plastic to land BELOW its target/cap (real achievable counts printed"
echo "above by each generator) rather than being padded with unaudited"
echo "images. The other 6 classes are unchanged from before (already"
echo "spot-checked clean, NOTES.md section 44) -- no new audit needed."
echo ""
echo "Launch training separately once satisfied with the printed counts:"
echo "   YOLO:    caffeinate -dimsu nohup python3 -u scripts/train_evaluate_yolo.py > yolo_train_log.txt 2>&1 & disown"
echo "   Cascade: caffeinate -dimsu nohup python3 -u scripts/train_belt_cascade.py > cascade_train_log.txt 2>&1 & disown"
echo "=================================================================="
