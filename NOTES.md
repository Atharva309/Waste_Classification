# Waste Classification Project — Condensed Notes

Condensed from a ~5,900-line chronological journal (82 sections) into a
topic-organized reference. Full blow-by-blow narration, exact live-run
numbers, and repeated debugging back-and-forth have been cut; final/
representative numbers and the reasoning behind decisions are kept.

**Working method used throughout the project, worth remembering:** don't
trust a prose summary or an offline benchmark — verify against the real
artifact (actual crop images, actual code, actual file counts) before
believing something is fixed, and never trust a single live run (this
project got burned by single-run "wins" repeatedly; 3+ runs became the
standard before calling anything real).

---

## 1. Project Overview

Rebuild of an earlier Keras/TensorFlow project that had 8 real methodology
problems (test-set leakage, wrong ResNet preprocessing, accuracy-only
reporting, no confusion matrix, confounded comparisons, no dedup across
merged datasets, no real latency benchmarking, no PyTorch). This rebuild
(`pytorch_pipeline/`) fixes all 8 verifiably: from-scratch IoU/NMS/Soft-NMS
(unit tested), a metrics module cross-checked against sklearn, perceptual-
hash+LSH dedup before every split, correct ImageNet preprocessing, and
percentile-based latency benchmarking. `tests/run_all.py` covers this.

**Two architectures were built and evaluated, not one**, to make a real
engineering tradeoff demonstrable rather than picking one classifier
arbitrarily:

- **Version B — two-stage cascade.** Stage 1: MobileNetV3-Large binary
  organic-vs-recyclable gate. Stage 2 (only runs if Stage 1 says
  not-organic): MobileNetV3 or ResNet50, 6-class material classifier
  (cardboard / ewaste / glass / metal / paper / plastic). Mirrors R-CNN-
  style separation of localization and classification — each stage
  specializes, can be improved independently.
- **Version A — single-stage YOLOv8n detector.** One forward pass predicts
  box + class. Trades some accuracy for speed that doesn't degrade as
  object density in a frame grows — the real justification (measured, not
  asserted): in overlap tests YOLO separated 98%+ of deliberately-
  overlapping items into distinct detections, vs. classical background
  subtraction's ~45% merge rate.

**Deliberately not built:** a literal crop/weed detector copying Blue
River's product. Kept the trash/conveyor-belt framing (same core CV
problem — real-time multi-object detection under a latency budget on a
moving scene) so the project stands on its own in an interview rather than
mirroring a target company's own product back at them.

A live Flask webapp simulates an animated conveyor belt, polls continuously
(not triggered by privileged simulation state — this was an early bug,
fixed), and supports both architectures plus a later third hybrid mode
("Frankenstein") via a model dropdown.

---

## 2. Data Pipeline (historical record — raw source data has since been deleted)

**Sources used:**
- `organictrashdetection` — Stage 1's organic/recyclable binary data.
- `garbage_classification` (a merged Kaggle/Roboflow/self-collected style
  pool) — Stage 2's 6 material classes.
- **TACO** (official litter-photo dataset, remapped to project classes) —
  download was flaky (stalled on a dead Flickr link, partial ~820/1500),
  used as a supplement. TACO's own class composition is naturally ~64%
  plastic (real litter mostly is), which caused real problems later (see
  Dead Ends).
- **TrashNet** (5 of 6 classes mapped cleanly) — small, clean, curated;
  consistently the least-contaminated sub-source.

**Object annotation (for YOLO training boxes):** source photos are
folder-labeled only, no boxes. FastSAM (lightweight YOLOv8-based
segmentation) was used to extract bounding boxes at scale — its masks were
inconsistent but its **boxes were reliably tight even when masks were bad**,
so FastSAM boxes became the sole annotation source for YOLO training.
SAM2 (a real ViT-based segmenter) was evaluated as an alternative and gave
better masks, but was rejected for cutout compositing because chroma-key
cutouts broke on transparent glass (green screen shows through) and
reflective metal (chroma spill) — a compositing problem, not a
segmentation one.

**Synthetic training data generation:**
- **YOLO frames**: real object cutouts/crops composited onto synthetic
  640×640 belt frames with known ground-truth boxes and deliberate overlap
  (`compose_belt_frame`), via real RGBA alpha blending (an early bug pasted
  whole rectangular photos including background — fixed, see Bugs).
- **Cascade retrain data** (`data/cascades_data`, formerly
  `belt_composite_v1`): objects composited onto the actual belt texture
  background (not plain studio photos) so the model's training domain
  matches what the live camera actually sees. Used a "sure-shot fallback"
  design: attempt a real alpha cutout (TACO chroma-key or TrashNet
  border-threshold); if that fails, fall back to an aggressive opaque
  bounding-box crop rather than dropping the image — guarantees every
  source image contributes a training example.
- Classes with no clean background to segment (organic, ewaste, and Stage
  1's "recyclable") always use a forced bounding-box crop.
- Objects are composited at **native resolution** (canvas sized around the
  object, not the object shrunk to fit a fixed small canvas) to avoid
  compounding two lossy resizes before the network's own 224×224 resize.

**Known, real data-quality issues found by direct inspection (not
metrics):** organic's source data contains non-waste images (stock photos,
a meme, a painting, a landscape) — a plausible major contributor to Stage
1's weak organic recall. Cardboard's source folder has ~63% literal
duplicate re-uploads (harmless for train, handled correctly by
cluster-aware leak-free splitting). Most seriously: **the generic
Kaggle-style "plastic" folder had real label-noise contamination** — metal
cans, tins, foil, and hardware mislabeled as plastic, roughly 12-17% of
that sub-source (TACO and TrashNet's plastic images were confirmed clean).
This was audited and partially fixed via a persisted blocklist
(`data/known_bad_source_images.json`) and eventually a **hard exclusion**
of the generic-Kaggle plastic pool entirely, keeping only TACO/TrashNet
plastic sources (see Dead Ends and Bugs).

---

## 3. Architecture & Key Design Decisions

- **Two-stage cascade vs. single classifier.** See Overview — a real
  accuracy/simplicity-vs-speed/density tradeoff, not an arbitrary choice.
- **Trash/belt framing instead of a literal crop/weed clone** of a target
  company's product — same underlying CV problem, distinct project.
- **Per-frame multi-frame evidence accumulation on the live belt, instead
  of trusting a single frame or a flat average/vote.** An object crossing
  the scan zone gets scanned many times. The system sums per-class
  confidence across all frames matched to that object's track and picks
  the class with the highest **total**, not the single highest-confidence
  frame and not a flat average. This was arrived at after two failed
  alternatives: pure single-frame-max (works okay but wastes signal) and
  flat-weighted temporal aggregation (averaging softmax / majority voting)
  — both gave every frame equal weight, so an object's own blurry/partial
  entry-exit frames could outvote its few genuinely good frames, and both
  were tried extensively and consistently made live results worse (see
  Dead Ends). Confidence-summed evidence avoids this: a class that's
  repeatedly detected with decent confidence accumulates a large total,
  while one outlier spike can't dominate unless it's dramatically more
  confident than everything else combined.
- **Never overwrite a checkpoint file in place; always give retrains new
  filenames.** Adopted after a real incident: an early retrain script
  overwrote the live checkpoint with no backup, and the pre-experiment
  91.4%/90.9% Stage 2 checkpoint was permanently lost. Every checkpoint
  since is additive — old ones stay loaded/selectable for A/B, never
  deleted without explicit confirmation.
- **"Unknown" is never a valid ground-truth label**, only a routing bucket
  a prediction can land in. A metrics bug (see Bugs) averaged it into
  macro-F1 as an 8th class with permanently-zero F1, silently understating
  every macro-F1 the project ever reported until fixed.
- **Where a fix belongs matters, and this recurred as a real, learned
  principle**: confidence floors that decide whether an *already-decided*
  final verdict should be trusted (cardboard floor, metal floor) must live
  at the **final aggregated per-object decision point** (client-side, after
  evidence accumulation) — applying them per-frame instead starves the
  evidence total for frames that were actually correct and silently shifts
  outcomes for unrelated classes (a real bug, found and fixed twice, see
  Bugs/Dead Ends). Competing-class swaps that change which class's bucket
  a frame's score gets added to (the plastic-vs-metal/cardboard rescue)
  belong at the **per-frame layer**, since they operate exactly where the
  argmax decision is made and the existing aggregation logic handles
  combining frames afterward unmodified.
- **Cross-class detection dedup uses IoM (intersection-over-minimum), not
  IoU**, for YOLO's raw output. YOLO's per-class heads can emit
  differently-**sized** boxes for the same physical object; plain IoU
  (divides by union) can stay well under threshold even when one box is
  basically nested inside the other, letting duplicates survive plain NMS.
- **Frankenstein hybrid mode**: use YOLO purely for localization (its
  learned, overlap-tolerant box proposals) and discard its classification
  entirely, feeding every box through the cascade's proven, calibrated
  Stage 2 classifier instead. Built after establishing that YOLO's boxes
  were solid but its lightweight built-in classification head was
  genuinely weaker than the dedicated, heavily-calibrated cascade
  classifier (a real architectural difference — YOLOv8n's classification
  is a side-output of a ~3M-parameter detection-focused network, receiving
  no calibration, vs. cascade's dedicated MobileNetV3 with a full night of
  tuning). Required matching Frankenstein's crop resolution and padding
  convention to what the cascade's own localizer (`BeltLocalizer`)
  produces before it actually worked (see Bugs).

---

## 4. Major Bugs Found and Fixed

1. **Stale/worse Stage 2 checkpoint loaded by default in the webapp.** The
   live app was loading a checkpoint scoring 82.2%/78.2% (accuracy/macro-F1)
   offline while a better checkpoint (FOCAL, 87.6%/85.9%) was already
   trained and sitting unused on disk. Swapped the default; live macro-F1
   rose 48.2%→53.6% immediately.
2. **YOLO canvas resolution mismatch (400 vs 640) causing blur.** The
   webapp's live scan canvas rendered YOLO's input at 400×400 while the
   model was trained at 640×640 frames; Ultralytics silently upsampled
   ~1.6x internally, blurring fine detail a heavily-converged model relied
   on. Root-caused by ruling out a class-index mismatch first, then a
   direct paused-frame comparison against ground truth. Fixed by rendering
   natively at 640 (scaleFactor 1.6) instead of relying on the blurry
   internal upsample.
3. **A second, Frankenstein-specific crop-resolution mismatch.** After
   fix #2, Frankenstein's crops (fed to the cascade's Stage 2 classifier)
   were still much smaller than cascade-alone's own crops, needing nearly
   2x the upsampling to reach the classifier's 224×224 input. Fixed by
   bumping Frankenstein's frame-capture scaleFactor to match cascade's (3x)
   while explicitly forcing YOLO's own internal inference to run at its
   trained 640 resolution (`imgsz=640`) regardless of input frame size —
   Ultralytics decouples the two, so localization accuracy wasn't traded
   for crop quality.
4. **Crop-padding-convention mismatch between YOLO boxes and
   BeltLocalizer's padded boxes.** Cascade's own localizer pads every
   detected box (`pad=12`, `min_dim=32`) before cropping; YOLO's boxes are
   trained to be tight with no margin. Since Stage 2's calibration was all
   tuned against BeltLocalizer's padded crops, Frankenstein was feeding it
   differently-shaped input. Fixed by applying the identical pad/min-dim
   transform to YOLO's boxes in Frankenstein mode before cropping —
   closed the remaining live gap to full parity with cascade-alone.
5. **A confidence floor scoped correctly by checkpoint name but not
   extended to a newly-added checkpoint.** The cardboard confidence floor
   (fixing a class-sink problem) was gated to one specific Stage 2
   checkpoint; when a new, genuinely-improved checkpoint
   (`lastditchattempt`) was added as a selectable option, its first live
   tests looked like a regression purely because the floor wasn't wired up
   for it yet. Extending the same floor check to the new checkpoint
   revealed the fine-tune was actually a real win.
6. **`console.log`'s object-collapsing hid diagnostic data.** A debug log
   meant to show per-class evidence breakdown printed a raw JS object,
   which Chrome's console collapses into a clickable, non-copyable
   placeholder — effectively invisible for copy/paste diagnosis. Fixed by
   building the breakdown into a plain-text log string instead.

---

## 5. Dead Ends / Fallbacks Tried and Abandoned

- **Per-frame confidence floors applied inside the classify function
  (before aggregation) instead of at the final decision.** Demoting
  individual cardboard/metal *frames* to "unknown" starved that class's
  running evidence total over an object's whole transit, letting other
  classes silently win objects that should have landed on cardboard/metal
  — even though the offline calibration "proved" other classes were
  mathematically untouched (true only at the single-frame level it was
  measured at). Moved to the client-side final-decision point instead,
  where it actually worked as intended.
- **Metal confidence floor.** Worked exactly as designed offline and for
  its own precision (avg 78.6% live), but recall collapsed further than
  predicted and metal's own F1 went down overall — a net loss on its own
  target class, unlike cardboard's floor (a genuine worthwhile tradeoff).
  Reverted.
- **Per-class logit bias** (suppressing cardboard's raw logit to fix its
  over-triggering). Worked mechanically, but since suppressing one class's
  score just reassigns the argmax winner to whichever other class had the
  next-highest logit, it measurably dragged down glass's precision as a
  side effect — an entangled fix. Replaced by the confidence-floor
  approach, which only ever intervenes on frames where the target class
  already won, leaving every other class's predictions untouched.
- **An offline plastic-calibration sweep whose baseline didn't reproduce
  the live symptom.** Live plastic showed high precision / low recall;
  the offline val-set baseline showed the *opposite* shape (good recall,
  moderate precision) — meaning the sweep was answering a different
  question than the one live testing was asking. Its recommendations were
  correctly distrusted rather than applied.
- **Temporal aggregation via flat-averaged softmax or majority voting**
  across an object's frames — tried in six different forms (recalibrated
  thresholds, no marginal band, majority vote, etc.), and live testing
  disagreed with the offline benchmark's claimed improvement every single
  time, sometimes wiping out a whole class's recall to 0%. Root cause:
  equal-weighting every frame lets a plurality of an object's own blurry/
  partial frames outvote its few genuinely good ones. Removed from the
  codebase entirely; later replaced by confidence-*summed* evidence (a
  different, non-equal-weighting mechanism — see Design Decisions).
- **Mixing TACO+TrashNet into Stage 2 training to fix metal's confusion
  with plastic.** Made things worse: metal's F1 dropped further because
  TACO's own composition is naturally ~64% plastic, and naive oversampling
  amplified an already plastic-heavy addition instead of correcting
  metal's deficit.
- **Naive inverse-frequency class weighting for metal.** Over-corrected —
  metal's precision collapsed as the model started calling other classes
  metal instead.
- **Watershed segmentation** for splitting overlapping belt objects.
  Solves a problem this belt doesn't actually have (items are placed to
  never touch) while being bad at the one it does have — it fragmented
  single objects into up to 9 disjoint pieces by finding false internal
  "peaks" (labels, highlights, shadows). Replaced entirely by
  `BeltLocalizer` (adaptive background subtraction + centroid tracking).
- **SAM2 for cutout/mask generation.** Better mask quality than FastSAM,
  but broke chroma-key compositing on transparent (glass) and reflective
  (metal) materials. Kept FastSAM's boxes only.
- **Corner-based background-color estimation** for classical cutout
  segmentation (sampling 4 corners instead of the full border ring, to
  avoid contamination when an object touches an edge). Passed a clean
  synthetic test but made every class's real yield *worse* — real product
  photography has vignetting/shadows concentrated in corners, so the
  "more robust" idea was noisier in practice. Reverted to the full border
  ring.
- **A frame-completeness filter** (skip classifying any box touching the
  frame edge, on the theory it's a partially-visible object). Consistently
  hurt macro-F1 by ~12 points in both checkpoint conditions tested —
  cutting too deep into thin-support classes' frame counts. Dropped
  (kept in code, defaulted off).
- **Deciding to stop retraining and pursue only post-hoc/inference-side
  fixes.** After a belt-domain-matched retrain didn't clearly beat the old
  pipeline live despite every individual fix being correctly diagnosed,
  and after diminishing returns on manual data-contamination cleanup, the
  explicit late-project call was "no training again" — everything from
  that point on (threshold calibration, confidence floors, rescue swaps,
  the Frankenstein hybrid) was built without retraining any model weights.
- **Plastic's low recall diagnosed as a genuine source-data-quality
  ceiling, not a fixable pipeline bug.** Direct inspection of the actual
  live crops for failing plastic items showed many were TACO litter photos
  where the plastic is a tiny, barely-visible fragment in a busy outdoor
  scene — hard to identify even for a human. No amount of threshold/bias
  calibration can fix a crop that doesn't contain enough real signal;
  flagged as needing a data audit/filter and likely a retrain, explicitly
  left undone this session.

---

## 6. Calibration / Rescue Mechanisms (final, deployed)

- **Cardboard confidence floor (0.65).** If Stage 2's final winning class
  for an object is cardboard but its confidence doesn't clear 0.65, route
  to "unknown" instead. Lives at the client-side final-decision point.
  Fixes a real, repeatable cardboard over-triggering ("sink") problem;
  genuinely improves cardboard precision at a small, accepted cost to a
  couple of other classes (glass mainly) that absorb the redirected
  traffic.
- **Plastic rescue swap (margin 0.20).** Per-frame: if Stage 2's argmax
  winner is specifically metal or cardboard, but plastic's own probability
  is within the margin of the winner's, swap the winner to plastic before
  it's added to the evidence total. Scoped narrowly (only metal/cardboard
  wins, only close calls) so it doesn't touch genuine metal/cardboard wins
  by a wide margin. Lives at the per-frame layer, since it's a
  competing-class decision at the same point the raw argmax happens, not a
  quality gate on an already-finalized verdict.
- **Unknown-bin second-guess rescue (min confidence 0.15).** If an
  object's final verdict landed on "unknown" (via a confidence floor or
  threshold routing), check its 2nd-place prediction; if that clears 0.15,
  promote it to the real final answer instead. Since "unknown" is never a
  valid ground-truth label, there's no scenario where this can be worse
  than leaving the item unknown. Lives at the client-side final-decision
  point, same layer as the confidence floors.
- **General principle behind where each fix lives** (this recurred as a
  real, learned rule, not just organizational convenience): floors/gates
  that judge an *already-decided* verdict belong at the final aggregated
  decision point; swaps that change which class a *raw per-frame score*
  gets attributed to belong at the per-frame layer, before aggregation.
  Mixing these up (applying a floor per-frame) was a real, twice-diagnosed
  bug.
- **Threshold calibration methodology**, used repeatedly: grid-search
  `(stage1_cutoff, stage2_threshold)` (or a bias/floor value) over the
  **validation split only** (never test), scored using this project's own
  macro-F1 (which correctly excludes "unknown"), using the exact same
  decision logic the live app uses. Every offline-recommended value was
  live-tested before being trusted — offline gains repeatedly failed to
  transfer 1:1 to live behavior (sometimes not at all), so this was never
  treated as sufficient on its own.

---

## 7. Final Results / Current State

**Live-confirmed final numbers** (averaged across the final rounds of live
testing, each 3+ runs):

- **Cascade alone** (Stage 2 = `lastditchattempt`, Stage 1 = the old,
  proven pre-cleanup checkpoint, cardboard floor 0.65): **~71.5% average
  live macro-F1**, tight run-to-run consistency.
- **Frankenstein hybrid** (YOLO `yolofirstactual` for localization at
  `conf=0.10`, matched to the cascade's crop resolution/padding
  convention, cascade's `lastditchattempt` for classification, plastic
  rescue margin 0.20, unknown-bin rescue): **~74.7% average live macro-F1**
  across the final 3 runs, including one larger 500-item run at **74.4%**
  — the best-performing configuration found this project.

**Deployed defaults:** `lastditchattempt.pt` (Stage 2 material classifier),
the old, pre-cleanup `stage1_mobilenet.pt` (Stage 1 organic gate — proven
to not leak plastic to "organic," unlike its retrained replacement),
`yolofirstactual.pt` (YOLO detector, used both standalone and as
Frankenstein's localizer), and Frankenstein mode as the best-tested option
(cascade-alone remains fully viable and simpler, ~3 points behind).

**Known unresolved issue: plastic recall.** Root-caused to genuine TACO
source-image data quality (tiny, barely-visible plastic litter fragments),
not a pipeline, calibration, or model-architecture bug — confirmed by
directly inspecting the actual failing live crops. A real fix needs a data
audit/filter pass on the plastic sprite/training pool and likely a
retrain; both were explicitly deferred, since the project's late-stage
decision was to stop retraining and exhaust post-hoc fixes first.

**Other standing open items, not resolved this project:** a full audit of
the remaining ~3,668 never-reviewed generic-Kaggle plastic source images
(currently hard-excluded rather than audited); a possible future
hard-example fine-tune targeted at cardboard/paper; YOLO's cross-class
duplicate handling and multi-object localizer edge cases (objects that
visually touch with zero gap still merge into one detection — an inherent
limitation of connected-components detection, not fixed).

---

## 8. Checkpoint / File Glossary

Only checkpoints whose purpose is clearly established in the notes:

- **`lastditchattempt.pt`** (Stage 2) — warm-started hard-example fine-tune
  from `removedbadapples_v2`, focused on its weak/borderline predictions
  without a full retrain. Final deployed default; a real, confirmed
  improvement over its parent.
- **`removedbadapples_v2.pt`** (Stage 1+2 pair) — cascade retrained on the
  belt-composited dataset after plastic-contamination cleanup, with the
  `augment_level="heavy"` fix applied to Stage 2 (matching FOCAL/
  TACOTRASHNET's recipe) and plain weighted CrossEntropyLoss (beat a
  focal-loss variant, `removedbadapples_v2_focal`, decisively in live A/B).
  Superseded by `lastditchattempt` for Stage 2.
- **`removedbadapples.pt`** / **`..._FAULTYVALTEST.pt`** — earlier
  "contamination cleanup" retrain attempts. The FAULTYVALTEST version's
  own val/test evaluation set was itself still ~61-66% unaudited/
  contaminated plastic ground truth, making its offline "regression" look
  worse than the model actually was; renamed to flag it as an invalid
  comparison rather than deleted.
- **`stage1_mobilenet_BELTCOMPOSITE.pt`** / **`stage2_material_mobilenet_
  BELTCOMPOSITE.pt`** — first belt-domain-matched retrain (cutouts
  composited onto belt texture instead of plain studio photos). Strong
  offline, but didn't clearly beat the old pipeline live until later
  fixes (augmentation, thresholds) were layered on for later runs.
- **`stage2_material_mobilenet_FOCAL.pt`** — early Stage 2 retrain on the
  original full-background `garbage_classification`+TACO/TrashNet data
  using focal loss. Historically one of the two strongest live performers
  (~70 F1) before the belt-composite work began.
- **`stage2_material_mobilenet_TACOTRASHNET.pt`** — same data/recipe as
  FOCAL but plain cross-entropy loss; essentially tied with FOCAL,
  confirming loss function wasn't the differentiator. Was the official
  default for a period.
- **`yolofirstactual.pt`** — the final, fully-converged YOLO retrain (271
  epochs, contaminated-plastic hard-excluded, `close_mosaic=10`,
  `batch=32`). Best offline YOLO result, and after resolution/padding
  fixes, the localizer used in the best-performing Frankenstein
  configuration.
- **`yolo_detector.pt`** / **`yolo_detector_removedbadapples.pt`** — earlier
  YOLO checkpoints (original, and post-cleanup-but-pre-`yolofirstactual`).
  Kept for reference/fallback, not deployed.
- **`stage1_mobilenet.pt`** — the original, pre-cleanup Stage 1
  organic/inorganic gate. Proven not to leak plastic into "organic" the
  way its later retrained replacement did; ended up paired with newer
  Stage 2 checkpoints in the final deployed configuration instead of its
  own retrained successor.

---

## 9. Interview-Prep Notes

**Why a two-stage cascade instead of one classifier?** Same idea as a
classic face-detection cascade (Viola-Jones, MTCNN): a cheap binary gate
filters out the easy majority case first, so the more expensive, more
specialized full classifier only ever runs on the harder subset — each
stage can be tuned and improved independently instead of one model trying
to do both jobs at once.

**Is this kind of post-hoc calibration tuning realistic in real-world ML,
or a hack?** Yes — it maps directly onto standard, real techniques:
**reject-option classification** (the confidence floors, which route
low-confidence verdicts to a human-review/unknown bucket instead of
forcing a guess), **per-class threshold calibration** (the grid-searched
stage1_cutoff/stage2_threshold pairs, tuned per checkpoint rather than
reused), and **N-best rescoring** (the top-3 prediction tracking and the
unknown-bin second-guess rescue, which uses the runner-up prediction when
the top choice is untrustworthy).
