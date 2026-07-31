import os
import sys
import json
import random
import time
import io
import base64
import torch
import numpy as np
import statistics
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flask import Flask, request, jsonify, render_template

from src.models import get_model_and_transforms
from src.synthetic_detection import BeltLocalizer
from src.nms import nms, soft_nms
from src.iou import iou_single, iom_single
from src.inference import PRESET_THRESHOLDS
from src.metrics import confusion_matrix, per_class_precision_recall_f1, macro_and_weighted_f1, compute_map
from src.latency import _percentile

from ultralytics import YOLO

app = Flask(__name__)

# Global State
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
# Kept model set (cleaned up -- see NOTES.md "Checkpoint / File Glossary"):
# Stage 1 is always stage1_mobilenet.pt (the old, proven organic gate --
# every surviving Stage 2 checkpoint pairs with it). Stage 2 offers two
# checkpoints for live A/B: lastditchattempt (default) and its direct
# parent removedbadapples_v2 (previous default, kept for comparison). YOLO
# is yolofirstactual only (the fully-converged, best-performing retrain,
# used both standalone and as Frankenstein's localizer). Every other
# checkpoint that used to be loaded here (FOCAL, TACOTRASHNET,
# BELTCOMPOSITE, faultyvaltest, removedbadapples, removedbadapples_v2_focal,
# the original/pre-cleanup YOLO checkpoints) was a superseded or one-off
# comparison artifact -- deleted from disk and removed from the UI.
model_stage1 = None
model_stage2_removedbadapples_v2 = None
model_stage2_lastditchattempt = None
model_yolo_yolofirstactual = None
transform_stage1 = None
transform_stage2_mb = None
belt_localizer = None

test_pool = []
history = []
scan_history = []
latency_history = []
_last_scan_combo = (None, None)

STAGE2_CLASSES = ["cardboard", "ewaste", "glass", "metal", "paper", "plastic"]
ALL_CLASSES = ["organic", "cardboard", "ewaste", "glass", "metal", "paper", "plastic", "unknown"]

def init_app():
    global model_stage1, model_stage2_removedbadapples_v2
    global model_stage2_lastditchattempt, model_yolo_yolofirstactual
    global transform_stage1, transform_stage2_mb, test_pool
    global belt_localizer

    print(f"Initializing models on {device}...")
    belt_localizer = BeltLocalizer()

    # Load Stage 1 -- the old, pre-cleanup organic/recyclable gate. Every
    # surviving Stage 2 checkpoint below pairs with this one; it was proven
    # live to not leak plastic into "organic" the way its retrained
    # replacements did (see NOTES.md).
    s1_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "outputs", "stage1_mobilenet.pt"))
    m1, _, t1 = get_model_and_transforms("mobilenet_v3_large", 2, pretrained=False)
    m1.load_state_dict(torch.load(s1_path, map_location=device))
    m1.to(device)
    m1.eval()
    model_stage1 = m1
    transform_stage1 = t1
    print(f"Loaded Stage 1 from {s1_path} (mtime: {time.ctime(os.path.getmtime(s1_path))})")

    # Load Stage 2 -- "removedbadapples_v2" (previous default, kept as a live
    # A/B reference; augment-fixed retrain on the belt-composited,
    # contamination-cleaned dataset. Live avg macro-F1 ~69.9%.)
    s2_rba2_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "outputs", "stage2_material_mobilenet_removedbadapples_v2.pt"))
    m2_rba2, _, t2_mb = get_model_and_transforms("mobilenet_v3_large", 6, pretrained=False)
    m2_rba2.load_state_dict(torch.load(s2_rba2_path, map_location=device))
    m2_rba2.to(device)
    m2_rba2.eval()
    model_stage2_removedbadapples_v2 = m2_rba2
    transform_stage2_mb = t2_mb
    print(f"Loaded Stage 2 (removedbadapples_v2, previous default) from {s2_rba2_path} (mtime: {time.ctime(os.path.getmtime(s2_rba2_path))})")

    # Load Stage 2 -- "lastditchattempt" (DEFAULT: hard-example fine-tune,
    # warm-started from removedbadapples_v2's own weights -- see
    # scripts/finetune_stage2_hardmine_v3.py. Live avg macro-F1 ~71.5%,
    # tighter run-to-run consistency than its parent. Uses the same Stage 1
    # as removedbadapples_v2 -- the fine-tune never touched Stage 1.)
    s2_lda_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "outputs", "lastditchattempt.pt"))
    m2_lda, _, _ = get_model_and_transforms("mobilenet_v3_large", 6, pretrained=False)
    m2_lda.load_state_dict(torch.load(s2_lda_path, map_location=device))
    m2_lda.to(device)
    m2_lda.eval()
    model_stage2_lastditchattempt = m2_lda
    print(f"Loaded Stage 2 (lastditchattempt, DEFAULT) from {s2_lda_path} (mtime: {time.ctime(os.path.getmtime(s2_lda_path))})")

    # Load YOLO Detector -- "yolofirstactual" (the only YOLO checkpoint kept:
    # fully-converged full retrain, 271 epochs. Ultralytics mAP@0.5 0.852,
    # mAP@0.5:0.95 0.810. Used both standalone (1-Stage YOLOv8n mode) and as
    # Frankenstein's localizer.)
    yolo_path_yolofirstactual = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "outputs", "yolofirstactual.pt"))
    model_yolo_yolofirstactual = YOLO(yolo_path_yolofirstactual)
    if device.type != "cpu":
        model_yolo_yolofirstactual.to(device)
    print(f"Loaded YOLO Detector (yolofirstactual, DEFAULT) from {yolo_path_yolofirstactual} "
          f"(mtime: {time.ctime(os.path.getmtime(yolo_path_yolofirstactual))})")

    # Load Test Pool

    
    print("Loading test sprites...")
    test_pool.clear()
    # Known class sprites -- sourced from sprites_cutout_v1 (true cutouts/
    # forced-crops, test-split-only, same source images as the
    # belt-composited cascade training data -- see NOTES.md "Data Pipeline").
    # This matches the domain both surviving Stage 2 checkpoints
    # (lastditchattempt, removedbadapples_v2) were trained on -- see
    # cascaded_classify_fn. Regenerate via scripts/rebuild_sprites.py if this
    # pool ever needs to be rebuilt.
    #
    # IMPORTANT: sprites_cutout_v1 was generated with one file per test-split
    # image, per class -- and the classes are wildly different sizes
    # (organic 957, plastic 468, metal only 91), because it just carried over
    # each class's natural test-split size instead of being equalized. Since
    # get_random_sprite() below does a flat random.choice() over this whole
    # list, an unequalized pool means organic gets drawn ~46% of the time and
    # metal ~4% -- this is the EXACT SAME class-imbalance bug section 23
    # already found and fixed for the old outputs/sprites pool (fixed there
    # by equalizing per-class counts). Fixing it the same way here: bucket by
    # class subfolder, then randomly subsample every class down to the
    # smallest class's count before flattening into test_pool.
    sprites_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "outputs", "sprites_cutout_v1"))
    sprites_archive = sprites_dir + ".tar.gz"
    if not os.path.exists(sprites_dir) and os.path.exists(sprites_archive):
        import tarfile
        print(f"Extracting {sprites_archive} to bypass HF API rate limits...")
        with tarfile.open(sprites_archive, "r:gz") as tar:
            def is_within_directory(directory, target):
                abs_directory = os.path.abspath(directory)
                abs_target = os.path.abspath(target)
                prefix = os.path.commonprefix([abs_directory, abs_target])
                return prefix == abs_directory
            
            def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
                for member in tar.getmembers():
                    member_path = os.path.join(path, member.name)
                    if not is_within_directory(path, member_path):
                        raise Exception("Attempted Path Traversal in Tar File")
                tar.extractall(path, members, numeric_owner=numeric_owner) 
                
            safe_extract(tar, os.path.dirname(sprites_dir))

    if os.path.exists(sprites_dir):
        by_class = {}
        for cls in sorted(os.listdir(sprites_dir)):
            cls_dir = os.path.join(sprites_dir, cls)
            if not os.path.isdir(cls_dir):
                continue
            by_class[cls] = [os.path.join(cls_dir, f) for f in os.listdir(cls_dir) if f.endswith(".png")]
        if by_class:
            min_count = min(len(v) for v in by_class.values())
            print(f"Equalizing sprite pool per class to {min_count} "
                  f"(raw counts: {dict((c, len(v)) for c, v in by_class.items())})")
            for cls, files in by_class.items():
                test_pool.extend(random.sample(files, min_count) if len(files) > min_count else files)
    # Unusable (unknown) images
    unusable_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "outputs", "unusable"))
    if os.path.exists(unusable_dir):
        for root, _, files in os.walk(unusable_dir):
            for f in files:
                if f.endswith(".png"):
                    test_pool.append(os.path.join(root, f))
    print(f"Loaded {len(test_pool)} test sprites into the pool.")
    print(f"Loaded {len(test_pool)} test sprites into the pool.")

# Stage 1 organic-cutoff. Both surviving Stage 2 checkpoints (lastditchattempt,
# removedbadapples_v2) pair with the same old, proven Stage 1 model at this
# cutoff -- see cascaded_classify_fn.
STAGE1_CUTOFF_DEFAULT = 0.65

# REPLACED (NOTES.md section 63, round 2): the per-class logit bias
# (cardboard=-0.75) fixed cardboard's precision, but live testing showed
# it did so by reallocating cardboard's suppressed probability mass to
# whichever class had the next-highest logit -- this landed
# disproportionately on glass (plastic->glass leak rate more than
# doubled), dragging glass's own F1 down ~8 points even though glass's
# logit was never touched. User asked for a fix that touches ONLY
# cardboard's numbers. Replaced with CARDBOARD_CONFIDENCE_FLOOR below --
# a stricter acceptance bar for cardboard specifically, routing to
# "unknown" (not the runner-up class) when it isn't met. This is
# mathematically guaranteed to leave every other class's predicted set
# unchanged, since it only ever intervenes on images where cardboard was
# already the argmax winner (verified directly in
# calibrate_cardboard_floor_removedbadapples_v2.py's output, not just
# asserted). STAGE2_BIAS_REMOVEDBADAPPLES_V2 kept at all-zeros (disabled,
# not deleted -- can be reinstated if the floor alone isn't enough).
STAGE2_BIAS_REMOVEDBADAPPLES_V2 = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]).to(device)
# Calibrated by scripts/calibrate_cardboard_floor_removedbadapples_v2.py
# (NOTES.md section 64): val sweep peaked at macro-F1 0.7850 (vs. 0.7738
# baseline) at floor=0.65 -- cardboard precision 0.663 -> 0.896, recall
# 0.879 -> 0.780, F1 0.756 -> 0.834. Verified mathematically, not just
# claimed: the macro-F1 delta (+0.0112) exactly equals cardboard's own F1
# delta divided by 7 classes (+0.0112), proving 100% of the macro-F1 gain
# came from cardboard alone -- every other class's precision/recall was
# independently confirmed byte-identical to baseline at every floor value
# tested.
#
# NOT applied here (NOTES.md section 64 follow-up) -- cascaded_classify_fn
# runs once per raw frame, and the live system aggregates many frames'
# votes per tracked object before deciding a final class (see
# templates/index.html). Applying the floor per-frame starved cardboard's
# running evidence total across an object's transit, which let OTHER
# classes win votes they wouldn't have otherwise -- live testing showed
# organic/glass/metal/paper/plastic all moved, not just cardboard. The
# calibration script's "single image" methodology actually matches the
# FINAL aggregated decision, not each raw frame -- so 0.65 is applied
# client-side instead, to the one decision per object, where it belongs.
# Kept here as the source-of-truth value / for any future backend use,
# not because cascaded_classify_fn reads it anymore.
CARDBOARD_CONFIDENCE_FLOOR_REMOVEDBADAPPLES_V2 = 0.65

# NOTES.md section 65: same mechanism, second sink (metal absorbing
# plastic and glass, consistent across all 3 live confirmation runs of
# the corrected cardboard floor). Calibrated by
# scripts/calibrate_metal_floor_removedbadapples_v2.py WITH the cardboard
# floor already held at 0.65 (both are active simultaneously live): val
# sweep peaked at macro-F1 0.7900 (vs. 0.7850 cardboard-only baseline) at
# floor=0.60 -- metal precision 0.731 -> 0.899, recall 0.806 -> 0.724,
# F1 0.767 -> 0.802. Every other class (including cardboard) confirmed
# byte-identical to baseline at every floor tested. Also NOT applied here
# for the same reason as cardboard's floor above -- applied client-side,
# to the final aggregated decision, not per raw frame.
METAL_CONFIDENCE_FLOOR_REMOVEDBADAPPLES_V2 = 0.60

# NOTES.md section 79: plastic's live problem is the mirror image of
# cardboard/metal's -- high precision (71-87% across live runs), low
# recall (25-51%), with metal and cardboard specifically shown (via the
# section 77 plastic-diag logs) to be absorbing plastic's items, which
# also drags metal/cardboard's OWN precision down in the process. A
# confidence floor can't help here -- it only ever trades recall away, and
# recall is exactly what's missing. This is a targeted swap instead: when
# Stage 2's winner is specifically metal or cardboard (not any other
# class) but plastic's own probability is close behind, prefer plastic.
# Scoped narrowly on purpose -- glass/paper/ewaste/organic wins are never
# touched, and metal/cardboard wins by a wide, genuine margin over plastic
# are never touched either, only the specific close three-way calls this
# was built for. UNLIKE the cardboard/metal floors, this could NOT be
# pre-validated with an offline calibration script -- section 77 already
# showed the offline val set doesn't reproduce plastic's live failure mode
# (offline recall 80.7%, live recall 25-51%), so there's no representative
# offline distribution to sweep this margin against. Starting value is a
# reasoned guess, not a validated one -- must be live-tested and tuned
# directly, same way the resolution/padding/conf fixes were.
PLASTIC_RESCUE_MARGIN_LASTDITCHATTEMPT = 0.20
# Bumped 0.10 -> 0.20 after live testing: at 0.10 the effect was within
# noise (plastic recall 46.2%/30.4%/48.3%, no clearer than pre-swap runs)
# -- too narrow a margin for plastic to actually win many close calls.
# Still a live-tuned guess, not an offline-validated value (section 79 already
# established why no representative offline distribution exists to sweep
# this against) -- if 0.20 overcorrects (metal/cardboard recall dropping,
# plastic precision cratering), it's a one-line number to walk back down.


@torch.no_grad()
def cascaded_classify_fn(crop_arr: np.ndarray, stage2_checkpoint="lastditchattempt"):
    # stage2_checkpoint selects the Stage 2 material classifier only -- both
    # surviving options (lastditchattempt, removedbadapples_v2) pair with the
    # same Stage 1 organic gate (model_stage1, cutoff 0.65): lastditchattempt
    # is a warm-started fine-tune of removedbadapples_v2 and never touched
    # Stage 1 at all, and removedbadapples_v2's own retrained Stage 1 was
    # live-tested and found to leak plastic into "organic" -- the old, proven
    # Stage 1 model beat it decisively for both checkpoints. See NOTES.md.
    img = Image.fromarray(crop_arr).convert("RGB")
    stage1_model = model_stage1
    stage1_cutoff = STAGE1_CUTOFF_DEFAULT

    # Stage 1
    x1 = transform_stage1(img).unsqueeze(0).to(device)
    logits1 = stage1_model(x1)
    probs1 = torch.softmax(logits1, dim=1)[0]

    if probs1[0] > stage1_cutoff:
        # Stage 1 is a binary organic-vs-not gate -- no per-material
        # breakdown exists for an organic verdict, so top3 is just the one
        # entry. NOTES.md section 80: added so the bin history UI can show
        # the next two runner-up predictions, not just the winner.
        return "organic", float(probs1[0]), [["organic", float(probs1[0])]]

    # Stage 2 -- lastditchattempt (default) or its parent removedbadapples_v2
    # (previous default, kept for live A/B).
    if stage2_checkpoint == "removedbadapples_v2":
        stage2_model = model_stage2_removedbadapples_v2
    else:
        stage2_model = model_stage2_lastditchattempt
    x2 = transform_stage2_mb(img).unsqueeze(0).to(device)
    logits2 = stage2_model(x2)

    if stage2_checkpoint == "removedbadapples_v2":
        # STAGE2_BIAS_REMOVEDBADAPPLES_V2 is currently all-zeros (disabled,
        # see the constant's comment above) -- kept as a no-op add so it
        # can be reinstated without restructuring this function.
        logits2 = logits2 + STAGE2_BIAS_REMOVEDBADAPPLES_V2

    probs2 = torch.softmax(logits2, dim=1)[0]
    idx2 = int(probs2.argmax())
    # NOTES.md section 64 follow-up: the cardboard confidence floor was
    # originally applied HERE, per raw frame -- WRONG layer. The webapp
    # doesn't classify an object once; it scans it every frame during its
    # belt transit and accumulates confidence per class
    # (index.html's class_evidence.sum), picking whichever class has the
    # highest ACCUMULATED total as the final verdict. Demoting individual
    # cardboard frames to "unknown" here starved cardboard's running
    # evidence total across the transit, which let other classes win the
    # aggregated vote that wouldn't have otherwise -- live testing showed
    # organic/glass/metal/paper/plastic all moved (not just cardboard),
    # contradicting the offline single-image proof. The floor is now
    # applied client-side, to the FINAL aggregated decision only (see
    # templates/index.html's loop() function, where s.best_scan is
    # finalized) -- matching what the calibration script actually measured
    # (independent single images, not per-frame votes feeding an
    # aggregator).

    # NOTES.md section 79: plastic rescue swap. Unlike the cardboard/metal
    # floors, this belongs PER-FRAME, not at the final aggregated decision
    # -- it's not a quality gate on an already-decided verdict (which needs
    # to see the aggregated confidence to avoid starving evidence, per
    # section 64's lesson), it's a per-frame competing-class decision, the
    # same layer idx2/argmax already operates at. It only changes which
    # class's evidence bucket THIS frame's score gets added to -- the
    # existing, already-tuned evidence-accumulation system in loop()
    # handles the rest unmodified.
    if stage2_checkpoint == "lastditchattempt" and STAGE2_CLASSES[idx2] in ("metal", "cardboard"):
        plastic_idx = STAGE2_CLASSES.index("plastic")
        if probs2[plastic_idx] >= probs2[idx2] - PLASTIC_RESCUE_MARGIN_LASTDITCHATTEMPT:
            idx2 = plastic_idx

    # NOTES.md section 80: top3 = the full Stage 2 distribution's top 3
    # classes by probability, computed from probs2 BEFORE any of the above
    # overrides matter for display purposes -- deliberately reflects what
    # Stage 2 itself actually thought, not just the final (possibly
    # rescued/floored/thresholded) verdict, so the bin history UI can show
    # genuine runner-up context even for items where the winner got
    # overridden downstream.
    top3_idx = torch.argsort(probs2, descending=True)[:3].tolist()
    top3 = [[STAGE2_CLASSES[i], float(probs2[i])] for i in top3_idx]

    return STAGE2_CLASSES[idx2], float(probs2[idx2]), top3

def calculate_iou(box1, box2):
    x_left = max(box1[0], box2[0])
    y_top = max(box1[1], box2[1])
    x_right = min(box1[2], box2[2])
    y_bottom = min(box1[3], box2[3])
    if x_right < x_left or y_bottom < y_top:
        return 0.0
    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    return intersection_area / float(box1_area + box2_area - intersection_area)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/get_random_sprite", methods=["GET"])
def get_random_sprite():
    if not test_pool:
        return jsonify({"error": "No test pool"}), 404
        
    path = random.choice(test_pool)
    abs_path = os.path.abspath(path)
    
    # Extract true_label from the directory structure outputs/sprites/<label>/<file>
    label = os.path.basename(os.path.dirname(abs_path))
    
    try:
        with open(abs_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
            return jsonify({"image": f"data:image/png;base64,{b64}", "true_label": label})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/reset", methods=["POST"])
def reset_bins():
    global history, scan_history, latency_history, belt_localizer
    history = []
    scan_history = []
    latency_history = []
    belt_localizer = BeltLocalizer()
    return jsonify({cls: 0 for cls in ALL_CLASSES})

@app.route("/history/<bin_name>")
def get_history(bin_name):
    items = [h for h in history if h["pred_class"] == bin_name]
    
    tp = sum(1 for h in history if h["pred_class"] == bin_name and h["true_label"] == bin_name)
    fp = sum(1 for h in history if h["pred_class"] == bin_name and h["true_label"] != bin_name)
    fn = sum(1 for h in history if h["true_label"] == bin_name and h["pred_class"] != bin_name)
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    
    return jsonify({
        "items": items,
        "stats": {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": precision,
            "recall": recall
        }
    })

@app.route("/dashboard_stats")
def get_dashboard_stats():
    c2i = {c: i for i, c in enumerate(ALL_CLASSES)}
    y_true = [c2i.get(h["true_label"], 7) for h in history]
    y_pred = [c2i.get(h["pred_class"], 7) for h in history]
    
    cm = confusion_matrix(y_true, y_pred, 8)
    metrics = per_class_precision_recall_f1(y_true, y_pred, 8)
    macro_weighted = macro_and_weighted_f1(metrics)
    
    per_image_preds = [s[0] for s in scan_history]
    per_image_gts = [s[1] for s in scan_history]
    class_ids = STAGE2_CLASSES + ["organic"]
    map_result = compute_map(per_image_preds, per_image_gts, class_ids, iou_threshold=0.5)
    
    lat = {}
    if latency_history:
        sorted_lat = sorted(latency_history)
        mean_lat = statistics.mean(sorted_lat)
        lat = {
            "mean": mean_lat,
            "p50": _percentile(sorted_lat, 50),
            "p95": _percentile(sorted_lat, 95),
            "p99": _percentile(sorted_lat, 99),
            "throughput": 1000.0 / mean_lat if mean_lat > 0 else 0
        }
    else:
        lat = {"mean": 0, "p50": 0, "p95": 0, "p99": 0, "throughput": 0}
        
    tp_sum = int(np.diag(cm).sum())
    fp_sum = int(cm.sum(axis=0).sum() - tp_sum)
    fn_sum = int(cm.sum(axis=1).sum() - tp_sum)

    return jsonify({
        "confusion_matrix": cm.tolist(),
        "classes": ALL_CLASSES,
        "per_class": {
            "precision": metrics["precision"].tolist(),
            "recall": metrics["recall"].tolist(),
            "f1": metrics["f1"].tolist(),
            "support": metrics["support"].tolist()
        },
        "macro_f1": macro_weighted["macro_f1"],
        "weighted_f1": macro_weighted["weighted_f1"],
        "mAP": map_result["mAP"],
        "AP_per_class": map_result["AP_per_class"],
        "latency": lat,
        "counts": {
            "total": len(history),
            "tp": tp_sum,
            "fp": fp_sum,
            "fn": fn_sum
        }
    })

@app.route("/scan", methods=["POST"])
def scan():
    data = request.json or {}
    image_b64 = data.get("image", "")
    confidence_mode = data.get("confidence_mode", "balanced")
    nms_method = data.get("nms_method", "soft_gaussian")
    detector_model = data.get("detector_model")
    if detector_model not in ["cascade", "frankenstein", "yolo"]:
        detector_model = "frankenstein"
    stage2_checkpoint = data.get("stage2_checkpoint", "lastditchattempt")
    # "lastditchattempt" (NEW DEFAULT): hard-example fine-tune of
    # removedbadapples_v2 (see scripts/finetune_stage2_hardmine_v3.py),
    # confirmed over 3 live runs AFTER the cardboard confidence floor was
    # correctly extended to it (its first live test, without the floor,
    # looked like a regression -- macro-F1 66.8%/59.6% -- purely because the
    # floor wasn't wired up yet, not a real problem with the fine-tune
    # itself). With the floor applied: macro-F1 71.9%/71.3%/71.4% (avg
    # 71.5%), consistently and clearly ahead of removedbadapples_v2's own
    # 69.9% average, and notably tighter/more stable across runs. Cardboard
    # precision landed at 81.3%/82.1%/86.4% -- genuinely fixed, not a fluke.
    # removedbadapples_v2 stays loaded/selectable as the previous default.
    completeness_filter = data.get("completeness_filter", False)
    sprites_gt = data.get("sprites_gt", []) # list of {"id": "...", "box": [x1,y1,x2,y2], "label": "..."}

    # Debug: print to the server console ONLY when the active combo actually
    # changes (not per-frame -- /scan fires many times/sec while an object
    # is in the zone, so an unconditional print would flood the terminal).
    # Added after the user asked "are you sure this is lastditchattempt
    # classifying?" for the second time -- rather than just asserting yes,
    # this gives a real, authoritative answer straight from what the server
    # actually received and used for this request, not what the UI was
    # assumed to be sending.
    global _last_scan_combo
    _combo = (detector_model, stage2_checkpoint)
    if _combo != _last_scan_combo:
        print(f"[/scan] ACTIVE COMBO CHANGED -> detector_model={detector_model} "
              f"stage2_checkpoint={stage2_checkpoint}")
        _last_scan_combo = _combo

    threshold = PRESET_THRESHOLDS.get(confidence_mode, 0.5)
    # Calibrated overrides for "balanced" mode, per Stage 2 checkpoint --
    # see NOTES.md.
    if stage2_checkpoint == "removedbadapples_v2" and confidence_mode == "balanced":
        threshold = 0.30
    elif stage2_checkpoint == "lastditchattempt" and confidence_mode == "balanced":
        # lastditchattempt is a near-frozen (features.14+ only), 2-epoch,
        # 1e-5 LR fine-tune of removedbadapples_v2 -- its softmax shape
        # should be close to its parent's, not a fresh flat-softmax risk
        # like a from-scratch retrain would be. Inheriting removedbadapples_v2's
        # 0.30 stopgap as the starting point rather than the raw 0.50
        # default -- still NOT a value calibrated for this exact checkpoint,
        # a real sweep (scripts/calibrate_thresholds_removedbadapples.py
        # re-pointed at lastditchattempt.pt) would be needed to confirm it.
        threshold = 0.30

    if image_b64.startswith("data:image"):
        image_b64 = image_b64.split(",")[1]
        
    img_bytes = base64.b64decode(image_b64)
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    frame = np.array(img)
    
    t0 = time.time()
    
    boxes_by_class = {}
    
    if detector_model == "cascade":
        # 1. Background Subtraction
        proposals, _ = belt_localizer.update(frame)
        frame_h, frame_w = frame.shape[:2]

        # 2. Classify and Threshold
        for tid, box in proposals:
            x1, y1, x2, y2 = box
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            # Completeness filter: BeltLocalizer clamps boxes to the frame
            # boundary (max(0,...)/min(w_frame,...)), so a box touching the
            # edge means the object is actually partially off-screen --
            # entering or exiting the scan zone, not fully visible. These are
            # exactly the low-quality frames that confidence was implicitly
            # filtering out already (see NOTES.md section 20); this makes
            # that filtering explicit instead of implicit.
            edge_margin = 2
            touches_edge = (
                x1 <= edge_margin or y1 <= edge_margin or
                x2 >= frame_w - edge_margin or y2 >= frame_h - edge_margin
            )
            if completeness_filter and touches_edge:
                continue

            pred_cls, score, top3 = cascaded_classify_fn(crop, stage2_checkpoint=stage2_checkpoint)

            # Route to "unknown" if below threshold (Cascade guarantees object presence)
            if score < threshold:
                pred_cls = "unknown"

            boxes_by_class.setdefault(pred_cls, ([], [], []))
            boxes_by_class[pred_cls][0].append(box)
            boxes_by_class[pred_cls][1].append(score)
            boxes_by_class[pred_cls][2].append(top3)

    elif detector_model == "frankenstein":
        # YOLO for box proposals ONLY -- its own predicted class and
        # confidence are discarded entirely, every box gets cropped and
        # classified through the SAME cascaded_classify_fn (Stage 1 + Stage
        # 2) already proven/calibrated for the cascade path above, reusing
        # its threshold + cardboard floor rather than inventing a second
        # calibration. Rationale (tonight's live testing): YOLO's own
        # boxes/IoU are solid, but even after fixing the canvas-resolution
        # bug its own classification lagged cascade's badly (macro-F1 ~53%
        # vs cascade's ~71%) -- consistent with YOLOv8n's classification
        # head being a lightweight side-output of a detection-focused
        # backbone with zero calibration work, vs. cascade's dedicated,
        # heavily-tuned MobileNetV3 classifier. Also a bet that YOLO's
        # LEARNED box proposals (trained with allow_overlap=True) separate
        # touching/overlapping objects better than BeltLocalizer's
        # background-subtraction, which has no explicit overlap handling.
        active_yolo = model_yolo_yolofirstactual
        # conf=0.25/iou=0.5 (Ultralytics' own standard operating defaults),
        # NOT the pure-YOLO path's conf=0.01/iou=0.99 below -- those
        # permissive values exist specifically to feed the app's own
        # downstream cross-class dedup with lots of raw candidates, needed
        # there because picking the right CLASS among overlapping same-
        # location boxes matters. Here YOLO's class output is unused, so we
        # just want one clean box per real object -- its own standard
        # operating point is tuned to produce exactly that.
        # NOTES.md section 75: imgsz=640 forces YOLO's OWN internal
        # inference to run at its trained resolution regardless of how
        # large the input frame actually is (now 1200x1200 -- see the
        # scaleFactor change in templates/index.html, bumped to match
        # cascade's crop-quality resolution). Ultralytics automatically
        # rescales returned box coordinates back to this frame's actual
        # size, so localization accuracy is unaffected -- but the crop
        # taken below now comes from the full 1200x1200 frame instead of a
        # 640x640 one, giving Stage 2's classifier a native-resolution crop
        # close to what cascade mode itself provides, instead of one
        # requiring a 2.7-5x upsample.
        # NOTES.md section 78: conf dropped from 0.25 to 0.10. Frankenstein
        # discards YOLO's own class AND confidence entirely -- the real
        # quality gate is Stage 2's calibrated threshold/floor downstream --
        # so there's no reason to also demand YOLO be confident before we
        # even look at a box. Motivated by the plastic-diag logs (section
        # 77) showing most objects only accumulated 1-4 total evidence
        # frames across their whole transit under Frankenstein, vs.
        # BeltLocalizer's near-continuous background-subtraction tracking --
        # the resulting high run-to-run variance (macro-F1 62.4-78.5% across
        # 6 runs) traces back to too few detection opportunities per object,
        # not classifier noise. Lowering conf gives YOLO more chances per
        # frame to propose "something is here," even at low confidence,
        # closer to how consistently BeltLocalizer re-catches the same blob.
        res = active_yolo(frame, conf=0.10, iou=0.5, imgsz=640, verbose=False, device="cpu")[0]
        frame_h, frame_w = frame.shape[:2]
        for box in res.boxes:
            rx1, ry1, rx2, ry2 = [float(v) for v in box.xyxy[0].cpu().numpy()]

            # NOTES.md section 76: match BeltLocalizer's own crop
            # convention exactly (src/synthetic_detection.py's update(),
            # pad=12/min_dim=32) -- YOLO's raw box is trained to regress
            # the sprite's exact tight rectangle (compose_belt_frame's
            # ground truth has no padding baked in), but BeltLocalizer
            # expands its tight contour box by 12px on every side and
            # enforces a 32px minimum before cascade's classifier ever
            # sees it. Stage 2's calibration (threshold, cardboard floor)
            # was tuned end-to-end against BeltLocalizer's padded crops in
            # every live test this session -- applying the identical
            # transform here so Frankenstein feeds Stage 2 a crop shaped
            # the same way, not just a similarly-sized one.
            bx, by, bw, bh = rx1, ry1, rx2 - rx1, ry2 - ry1
            bcx, bcy = bx + bw / 2.0, by + bh / 2.0
            BELT_PAD, BELT_MIN_DIM = 12, 32
            new_w = max(bw + 2 * BELT_PAD, BELT_MIN_DIM)
            new_h = max(bh + 2 * BELT_PAD, BELT_MIN_DIM)
            x1 = int(max(0, bcx - new_w / 2.0))
            y1 = int(max(0, bcy - new_h / 2.0))
            x2 = int(min(frame_w, bcx + new_w / 2.0))
            y2 = int(min(frame_h, bcy + new_h / 2.0))

            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            edge_margin = 2
            touches_edge = (
                x1 <= edge_margin or y1 <= edge_margin or
                x2 >= frame_w - edge_margin or y2 >= frame_h - edge_margin
            )
            if completeness_filter and touches_edge:
                continue

            pred_cls, score, top3 = cascaded_classify_fn(crop, stage2_checkpoint=stage2_checkpoint)

            if score < threshold:
                pred_cls = "unknown"

            boxes_by_class.setdefault(pred_cls, ([], [], []))
            boxes_by_class[pred_cls][0].append([x1, y1, x2, y2])
            boxes_by_class[pred_cls][1].append(score)
            boxes_by_class[pred_cls][2].append(top3)

    else:
        # 1. YOLO Single-Stage
        active_yolo = model_yolo_yolofirstactual
        res = active_yolo(frame, conf=0.01, iou=0.99, verbose=False, device="cpu")[0]
        yolo_classes = ["organic", "cardboard", "ewaste", "glass", "metal", "paper", "plastic"]
        for box in res.boxes:
            cls_id = int(box.cls[0].item())
            pred_cls = yolo_classes[cls_id]
            score = float(box.conf[0].item())
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            
            if score < threshold:
                # YOLO doesn't guarantee object presence; drop low-confidence duplicate anchors
                continue

            # NOTES.md section 80: YOLO's raw output is one class per box,
            # not a full distribution -- no real "2nd/3rd place" exists here
            # the way it does for cascade's softmax. Single-entry top3 so
            # the downstream data shape stays consistent; the bin history
            # UI just won't have runner-ups to show for pure-YOLO items.
            top3 = [[pred_cls, score]]

            boxes_by_class.setdefault(pred_cls, ([], [], []))
            boxes_by_class[pred_cls][0].append([x1, y1, x2, y2])
            boxes_by_class[pred_cls][1].append(score)
            boxes_by_class[pred_cls][2].append(top3)

    # 3. NMS per class
    deduped = {}
    for cls, (boxes, scores, top3s) in boxes_by_class.items():
        boxes_arr = np.array(boxes, dtype=float)
        scores_arr = np.array(scores, dtype=float)
        if len(boxes_arr) == 0:
            continue

        if nms_method == "hard":
            keep = nms(boxes_arr, scores_arr, iou_threshold=0.4)
            deduped[cls] = (boxes_arr[keep], scores_arr[keep], [top3s[i] for i in keep])
        else:
            method = "linear" if nms_method == "soft_linear" else "gaussian"
            keep, decayed = soft_nms(boxes_arr, scores_arr, method=method, iou_threshold=0.4, score_threshold=0.05)
            deduped[cls] = (boxes_arr[keep], decayed, [top3s[i] for i in keep])

    # 3b. Cross-class dedup -- YOLO only. The per-class NMS above only
    # suppresses duplicate boxes WITHIN the same class. The cascade path
    # can't produce cross-class duplicates for one object (one background-
    # subtraction proposal -> one softmax classification -> exactly one
    # label), but YOLO's raw per-class detection heads can and do emit
    # multiple candidate boxes for the SAME physical object under
    # DIFFERENT classes (e.g. a "plastic" box and an "organic" box both
    # covering the same location), especially at the permissive conf=0.01
    # used above. Live testing showed exactly this signature -- plastic
    # recall high but precision low, and cardboard/metal/paper repeatedly
    # misattributed to plastic in the confusion matrix -- consistent with
    # spurious same-location "plastic" boxes never getting suppressed
    # because they were never compared against the true class's box.
    # Greedily keep only the highest-confidence detection among any group
    # of heavily-overlapping cross-class boxes.
    #
    # NOTES.md section 49: this fix was already deployed (dataset cleanup +
    # retrain, sections 41-48) and the live signature came back byte-for-
    # byte identical (plastic precision 31.1%/recall 66.7%, overall mAP
    # 29.1% -- matching the ORIGINAL pre-fix numbers in section 35 almost
    # exactly), even though the retrained model's own offline mAP@0.5
    # *improved* (0.821 vs 0.816). That combination -- offline better,
    # live unchanged -- means the bad training data was never the cause of
    # THIS symptom; the dedup step itself was under-triggering. Root cause:
    # it compared boxes with plain IoU, which is sensitive to size, not
    # just position. YOLO's per-class heads regress a same-object box
    # differently in SIZE across classes (not just class label), so a
    # same-location "plastic" phantom box and the true class's box can
    # have IoU as low as ~0.3 even though one is basically sitting inside
    # the other -- below the 0.5 IoU cutoff this step used, so the
    # duplicate survived. Switched to IoM (intersection over the SMALLER
    # box's area, src/iou.py:iom_single) which isn't penalized by a size
    # mismatch -- a box fully contained in another scores IoM 1.0 no
    # matter how much bigger the other one is -- and is the right test for
    # "these two boxes are almost certainly the same object."
    if detector_model != "cascade" and deduped:
        all_dets = [
            [cls, np.asarray(b, dtype=float), float(s), t3]
            for cls, (boxes, scores, top3s) in deduped.items()
            for b, s, t3 in zip(boxes, scores, top3s)
        ]
        all_dets.sort(key=lambda d: -d[2])

        kept = []
        for cls, box, score, t3 in all_dets:
            if any(iom_single(box, kbox) >= 0.6 for _, kbox, _, _ in kept):
                continue
            kept.append((cls, box, score, t3))

        deduped = {}
        for cls, box, score, t3 in kept:
            deduped.setdefault(cls, ([], [], []))
            deduped[cls][0].append(box)
            deduped[cls][1].append(score)
            deduped[cls][2].append(t3)
        deduped = {c: (np.array(b), np.array(s), t3s) for c, (b, s, t3s) in deduped.items()}

    # 4. Process Detections
    detections = []

    for cls, (boxes, scores, top3s) in deduped.items():
        for b, s, t3 in zip(boxes, scores, top3s):
            box_tuple = tuple(int(x) for x in b)
            detections.append((cls, box_tuple, float(s), t3))
            
    latency_ms = (time.time() - t0) * 1000.0
    latency_history.append(latency_ms)
    
    return jsonify({
        "latency_ms": latency_ms,
        "detections": detections
    })

@app.route("/finalize_item", methods=["POST"])
def finalize_item():
    data = request.json or {}
    pred_class = data.get("pred_class")
    confidence = data.get("confidence", 0.0)
    true_label = data.get("true_label", "unknown")
    crop_b64 = data.get("crop_b64", "")
    box = data.get("box", [0, 0, 0, 0])
    gt_box = data.get("gt_box", [0, 0, 0, 0])
    top3 = data.get("top3", [])  # NOTES.md section 80: runner-up predictions for the bin history UI

    if pred_class != "missed":
        history.append({
            "crop": crop_b64,
            "pred_class": pred_class,
            "confidence": float(confidence),
            "true_label": true_label,
            "top3": top3
        })
        
        pred_by_class = {pred_class: (np.array([box], dtype=float), np.array([confidence], dtype=float))}
        gt_by_class = {true_label: np.array([gt_box], dtype=float)}
        scan_history.append((pred_by_class, gt_by_class))
    else:
        # Missed detection (true FN -- no box matched this ground-truth
        # sprite at all). Previously this branch only fed scan_history
        # (used for Overall mAP), never `history` -- which is what "Total
        # Processed", the per-class table, and the confusion matrix are
        # all built from. That meant a stricter confidence threshold
        # (e.g. "High Precision") really was cutting more detections, but
        # every one of those cut items just silently vanished from the
        # dashboard instead of showing up as a miss -- explaining why FP
        # always exactly equalled FN in every session (history only ever
        # held genuine class-vs-class confusions, never a true "nothing
        # detected" case) and why the "unknown" row/column was always
        # stuck at 0 support despite existing in the schema for exactly
        # this. Recording it as pred_class="unknown" now (matching the
        # frontend's own visual treatment of a missed item, index.html:
        # `s.predicted_class = 'unknown'`) so raising the threshold's
        # real effect -- more true misses -- is actually visible instead
        # of just nudging Overall mAP by a couple points unexplained.
        history.append({
            "crop": crop_b64,
            "pred_class": "unknown",
            "confidence": float(confidence),
            "true_label": true_label
        })
        pred_by_class = {}
        gt_by_class = {true_label: np.array([gt_box], dtype=float)}
        scan_history.append((pred_by_class, gt_by_class))
        
    counts = {c: sum(1 for h in history if h["pred_class"] == c) for c in ALL_CLASSES}
    return jsonify({"counts": counts})

if __name__ == "__main__":
    init_app()
    app.run(host="0.0.0.0", port=7860, debug=False)
