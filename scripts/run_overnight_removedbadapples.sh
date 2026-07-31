#!/bin/bash
# Unattended overnight pipeline: rebuild both datasets (blocklist-filtered,
# same sizes/targets as before -- NOTES.md sections 44-46), then train YOLO,
# then train the cascade. Meant to run via caffeinate+nohup so it survives
# the terminal closing and the Mac trying to sleep (see NOTES.md section 13
# for why both are needed together, neither alone is sufficient).
#
# Old checkpoints are never touched:
#   - outputs/yolo_detector.pt                          (untouched)
#   - outputs/stage1_mobilenet.pt                        (untouched)
#   - outputs/stage2_material_mobilenet_*.pt              (untouched)
#   - outputs/stage{1,2}_mobilenet_BELTCOMPOSITE.pt        (untouched)
# New checkpoints land at:
#   - outputs/yolo_detector_removedbadapples.pt
#   - outputs/stage1_mobilenet_removedbadapples.pt
#   - outputs/stage2_material_mobilenet_removedbadapples.pt
#
# Run this (from pytorch_pipeline/):
#   caffeinate -dimsu nohup bash scripts/run_overnight_removedbadapples.sh > overnight_log.txt 2>&1 &
#   disown
#
# Then check progress any time with:
#   tail -f overnight_log.txt
#
# WATCH THE FIRST FEW MINUTES before actually going to sleep -- this script
# cannot be fully tested end-to-end ahead of time (no GPU/torch available in
# the environment used to write it), so the first real signal that
# everything is wired correctly is: dataset rebuild finishing cleanly, then
# YOLO's "Epoch 1/500" line appearing with real loss numbers. If either of
# those doesn't happen within ~15-20 minutes, something is wrong and it's
# not worth leaving running unattended.

set -u
cd "$(dirname "$0")/.."  # pytorch_pipeline/
mkdir -p logs

STAGE1_STAMP() { date "+%Y-%m-%d %H:%M:%S"; }

echo "[$(STAGE1_STAMP)] ===== STAGE 1/3: Rebuilding datasets ====="
source venv/bin/activate
bash scripts/rebuild_all_datasets.sh --yes 2>&1 | tee logs/1_rebuild.log
REBUILD_STATUS=${PIPESTATUS[0]}

if [ "$REBUILD_STATUS" -ne 0 ]; then
    echo "[$(STAGE1_STAMP)] REBUILD FAILED (exit $REBUILD_STATUS). Stopping -- not attempting training on a possibly-broken dataset."
    echo "See logs/1_rebuild.log for details."
    exit 1
fi
echo "[$(STAGE1_STAMP)] Rebuild finished OK."

echo ""
echo "[$(STAGE1_STAMP)] ===== STAGE 2/3: Training YOLO (removedbadapples) ====="
python3 -u scripts/train_evaluate_yolo.py 2>&1 | tee logs/2_yolo_train.log
YOLO_STATUS=${PIPESTATUS[0]}
if [ "$YOLO_STATUS" -ne 0 ]; then
    echo "[$(STAGE1_STAMP)] YOLO TRAINING FAILED (exit $YOLO_STATUS). See logs/2_yolo_train.log."
    echo "Continuing to cascade training anyway -- the two are independent."
else
    echo "[$(STAGE1_STAMP)] YOLO training finished OK. New weights: outputs/yolo_detector_removedbadapples.pt"
fi

echo ""
echo "[$(STAGE1_STAMP)] ===== STAGE 3/3: Training cascade (removedbadapples) ====="
python3 -u scripts/train_belt_cascade.py 2>&1 | tee logs/3_cascade_train.log
CASCADE_STATUS=${PIPESTATUS[0]}
if [ "$CASCADE_STATUS" -ne 0 ]; then
    echo "[$(STAGE1_STAMP)] CASCADE TRAINING FAILED (exit $CASCADE_STATUS). See logs/3_cascade_train.log."
else
    echo "[$(STAGE1_STAMP)] Cascade training finished OK. New weights: outputs/stage1_mobilenet_removedbadapples.pt, outputs/stage2_material_mobilenet_removedbadapples.pt"
fi

echo ""
echo "===== SUMMARY ====="
echo "Rebuild:  $([ "$REBUILD_STATUS" -eq 0 ] && echo OK || echo FAILED)"
echo "YOLO:     $([ "$YOLO_STATUS" -eq 0 ] && echo OK || echo FAILED)"
echo "Cascade:  $([ "$CASCADE_STATUS" -eq 0 ] && echo OK || echo FAILED)"
echo "Logs in pytorch_pipeline/logs/"
echo "[$(STAGE1_STAMP)] Done."
