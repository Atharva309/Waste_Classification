import os
import json
import torch
import glob
import numpy as np
import sys
from PIL import Image

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ultralytics import YOLO
from src.metrics import compute_map
from src.latency import benchmark_latency

def compute_iou(box1, box2):
    ixmin = max(box1[0], box2[0])
    iymin = max(box1[1], box2[1])
    ixmax = min(box1[2], box2[2])
    iymax = min(box1[3], box2[3])
    iw = max(ixmax - ixmin, 0.)
    ih = max(iymax - iymin, 0.)
    inters = iw * ih
    if inters <= 0:
        return 0.0
    uni = ((box1[2]-box1[0])*(box1[3]-box1[1]) +
           (box2[2]-box2[0])*(box2[3]-box2[1]) - inters)
    return inters / uni if uni > 0 else 0.0

RUN_NAME = "yolofirstactual"

# QUICK_TEST=1 env var shrinks this to a ~1-2 min smoke test (2 epochs, tiny
# time cap, latency bench and custom-eval loop trimmed to a handful of
# images) so the full pipeline -- train -> save -> custom eval -> latency
# bench -> report write -- can be verified end-to-end before committing to
# an unattended overnight run. Untouched (QUICK_TEST unset/0) = the real
# planned run: epochs=500/patience=100/time=8, full val set.
QUICK_TEST = os.environ.get("QUICK_TEST", "0") == "1"


def main():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Training on device: {device}")
    if QUICK_TEST:
        print("QUICK_TEST=1 -- running a shrunk smoke test, NOT the real training run.")

    # 1. Train YOLOv8n
    # Round 2 overnight run (NOTES.md section 36-37): the first real run
    # completed cleanly (66 epochs in 2.003 hours) but was stopped by the
    # `time=2` wall-clock cap, not by convergence -- patience=100 was
    # nowhere close to firing at 66 epochs. Live testing afterward showed
    # a real plastic over-prediction bias consistent with an undertrained,
    # unsharpened decision boundary, not a data quantity/quality problem
    # (checked: plastic's precision/recall signature -- high recall, low
    # precision -- is the signature of over-triggering, not the low/low
    # signature a genuinely under-represented class would show; plastic's
    # cutout quality also isn't the worst of the 7 classes). So this round
    # removes the artificial ceiling and lets patience=100 actually govern
    # when training stops, per explicit user instruction to "let it run
    # freely" for a full night. `time=8` is kept as a generous safety-net
    # only (matches the project's own prior successful-run convention,
    # section 13, where an 8-hour cap was set and the run converged well
    # under it on its own) -- not expected to be hit, just insurance
    # against an unexpected non-convergence eating the whole night.
    model = YOLO("yolov8n.pt")

    # name="yolo_train_removedbadapples_v2" (not "yolo_train_removedbadapples"):
    # this is a SECOND run, on top of the already-deployed removedbadapples
    # checkpoint (outputs/yolo_detector_removedbadapples.pt, currently the
    # live webapp default). Reusing the old name with exist_ok=True would
    # have overwritten that checkpoint's own run directory AND the deployed
    # .pt file itself via the cp command below -- caught and renamed to _v2
    # before running, per the project's own "never overwrite a working
    # checkpoint in place" rule (and the explicit "dont delete old weights"
    # instruction). This run trains on the hard-excluded plastic dataset
    # (NOTES.md section 51/53, stronger than the blocklist-only fix the
    # first removedbadapples run had) and will be paired with the section
    # 49/50 webapp fixes (IoM cross-class dedup, sum-of-confidence
    # aggregation) that also postdate the first run and have never been
    # live-tested together with a retrained model yet.
    #
    # Two hyperparameter changes from the first removedbadapples run, both
    # standard Ultralytics best practice (NOTES.md section 56):
    #   - close_mosaic=10: mosaic augmentation (4-image blend) stays on for
    #     most of training -- already Ultralytics' default behavior, not
    #     newly enabled here -- but now explicitly turns off for the final
    #     10 epochs so the model fine-tunes box regression against real,
    #     undistorted single-image layouts rather than mosaic'd composites,
    #     instead of relying on the library default silently doing this.
    #   - batch=32 (was 16): more stable batchnorm statistics, still a
    #     conservative fixed value rather than Ultralytics' batch=-1
    #     auto-sizing -- MPS (Apple Silicon) autobatch support is not as
    #     mature/reliable as CUDA's, and this is an unattended overnight
    #     run where a batch-size-related crash costs the whole night, so
    #     a modest fixed increase was chosen over an untested automatic one.
    # Deliberately NOT changed: mixup (stays at Ultralytics' default, off).
    # The source material for this round suggested mosaic+mixup together,
    # but mixup blends two DIFFERENT images' pixels and labels together --
    # given tonight's whole story is plastic's decision boundary already
    # being too permissive/bleeding into other classes, adding an
    # augmentation that literally blends plastic and non-plastic pixels
    # together is a real risk of making that specific problem worse, and
    # it would be a second untested new variable stacked on this run
    # alongside close_mosaic/batch/the dataset fix/the two webapp fixes.
    # Left for a follow-up round if this run's own result suggests it's
    # still needed, not bundled in blind.
    results = model.train(
        data="data/yolo_data/dataset/data.yaml",
        epochs=2 if QUICK_TEST else 500,
        patience=100,
        time=0.03 if QUICK_TEST else 8,
        device=device,
        cache="ram",
        degrees=15,
        close_mosaic=0 if QUICK_TEST else 10,  # close_mosaic=10 on a 2-epoch run would fire immediately; skip for the smoke test
        project="outputs",
        name=RUN_NAME,
        exist_ok=True,
        imgsz=640,
        batch=32,
        verbose=False
    )

    # 2. Evaluate with Ultralytics metrics
    # CORRECTED BACK (smoke test, real run): despite project="outputs",
    # name=<run name> above, this Ultralytics version's actual save_dir
    # (confirmed directly from a live log on an earlier run: "save_dir=
    # .../runs/detect/outputs/yolo_train" and "Results saved to .../runs/
    # detect/outputs/yolo_train") DOES still nest under runs/detect/. An
    # earlier edit assumed project= fully overrides that prefix -- it
    # doesn't, in this version -- and that assumption was wrong and broke
    # a previously-correct line. Verified by actually running the script
    # end-to-end, not by reasoning about the API from memory. The path
    # below just substitutes this run's own name= value into that same
    # confirmed runs/detect/outputs/<name>/ nesting.
    model_trained = YOLO(f"runs/detect/outputs/{RUN_NAME}/weights/best.pt")
    val_results = model_trained.val(data="data/yolo_data/dataset/data.yaml", verbose=False)
    map50 = val_results.box.map50
    map50_95 = val_results.box.map

    print(f"Ultralytics mAP@0.5: {map50}")
    print(f"Ultralytics mAP@0.5:0.95: {map50_95}")

    # Save the best weights under a NEW filename -- outputs/yolo_detector.pt
    # (the live, previously-deployed checkpoint) is deliberately left
    # untouched, same "never overwrite in place" rule the cascade side
    # already followed.
    os.system(f"cp runs/detect/outputs/{RUN_NAME}/weights/best.pt outputs/{RUN_NAME}.pt")

    # 3. Custom Evaluation (mAP and Overlap Separation)
    model = YOLO(f"outputs/{RUN_NAME}.pt")
    model.to(device)

    val_images = glob.glob("data/yolo_data/dataset/images/val/*.jpg")
    val_images.sort()
    if QUICK_TEST:
        val_images = val_images[:15]  # full custom-eval loop on 400 images isn't needed to prove the pipeline works

    classes = ["organic", "cardboard", "ewaste", "glass", "metal", "paper", "plastic"]
    
    per_image_preds = []
    per_image_gts = []
    
    overlap_frames = 0
    correctly_separated_frames = 0
    
    for img_path in val_images:
        label_path = img_path.replace("images", "labels").replace(".jpg", ".txt")
        
        gt_boxes = {}
        all_gt_boxes = []
        
        img = Image.open(img_path)
        width, height = img.size
        
        if os.path.exists(label_path):
            with open(label_path, "r") as f:
                lines = f.readlines()
                for line in lines:
                    parts = line.strip().split()
                    if not parts:
                        continue
                    cls_id = int(parts[0])
                    cls_name = classes[cls_id]
                    x_c, y_c, w, h = map(float, parts[1:])
                    
                    xmin = (x_c - w/2) * width
                    xmax = (x_c + w/2) * width
                    ymin = (y_c - h/2) * height
                    ymax = (y_c + h/2) * height
                    box = [xmin, ymin, xmax, ymax]
                    
                    gt_boxes.setdefault(cls_name, []).append(box)
                    all_gt_boxes.append(box)
                    
        gt_by_class = {c: np.array(b) for c, b in gt_boxes.items()}
        per_image_gts.append(gt_by_class)
        
        has_overlap = False
        for i in range(len(all_gt_boxes)):
            for j in range(i+1, len(all_gt_boxes)):
                if compute_iou(all_gt_boxes[i], all_gt_boxes[j]) > 0.0:
                    has_overlap = True
                    break
            if has_overlap:
                break
                
        if has_overlap:
            overlap_frames += 1
            
        # conf=0.001 (near-zero) is deliberate, not a typo: model()'s default
        # conf=0.25 is an operating threshold for real deployment, not for
        # AP computation. AP needs the full precision-recall sweep across
        # every confidence level (this is what Ultralytics' own internal
        # .val() does, and why its mAP50 above came out to 0.259 while this
        # loop's compute_map first came out to 0.0014 on the same model --
        # found by comparing the two numbers on a real smoke-test run and
        # noticing the ~180x gap couldn't be explained by "1 epoch is
        # weak" alone). Gating to conf=0.25 here silently threw away nearly
        # every detection before compute_map ever saw it.
        res = model(img_path, verbose=False, conf=0.001)[0]
        
        pred_by_class = {}
        all_det_boxes = []
        for box in res.boxes:
            cls_id = int(box.cls[0].item())
            cls_name = classes[cls_id]
            conf = float(box.conf[0].item())
            xyxy = box.xyxy[0].cpu().numpy()
            
            pred_by_class.setdefault(cls_name, ([], []))
            pred_by_class[cls_name][0].append(xyxy)
            pred_by_class[cls_name][1].append(conf)
            all_det_boxes.append(xyxy)
            
        pred_by_class = {c: (np.array(b), np.array(s)) for c, (b, s) in pred_by_class.items()}
        per_image_preds.append(pred_by_class)
        
        if has_overlap:
            matched_gt = set()
            for b_idx, gt_b in enumerate(all_gt_boxes):
                for det_b in all_det_boxes:
                    if compute_iou(gt_b, det_b) > 0.3:
                        matched_gt.add(b_idx)
                        break
            if len(matched_gt) == len(all_gt_boxes):
                correctly_separated_frames += 1
                
    map_result = compute_map(per_image_preds, per_image_gts, class_ids=classes, iou_threshold=0.5)
    
    print(f"Custom mAP@0.5: {map_result['mAP']}")
    print(f"Overlap frames separated: {correctly_separated_frames}/{overlap_frames}")
    
    # 4. Latency Benchmark
    print("Benchmarking latency on Device...")
    dummy_input = np.random.randint(0, 255, (400, 400, 3), dtype=np.uint8)
    
    def predict_fn_dev(x):
        return model(x, verbose=False)
        
    def make_input_fn():
        return dummy_input
        
    dev_stats = benchmark_latency(predict_fn_dev, make_input_fn,
                                   n_warmup=2 if QUICK_TEST else 10,
                                   n_iters=5 if QUICK_TEST else 50)

    print("Benchmarking latency on CPU...")
    model_cpu = YOLO(f"outputs/{RUN_NAME}.pt")
    model_cpu.to("cpu")

    def predict_fn_cpu(x):
        return model_cpu(x, verbose=False)

    cpu_stats = benchmark_latency(predict_fn_cpu, make_input_fn,
                                   n_warmup=2 if QUICK_TEST else 10,
                                   n_iters=5 if QUICK_TEST else 50)
    
    # 5. Update Report -- adds a NEW key to the same shared report file;
    # the existing "yolo_detector" key (previous run's results) is left
    # untouched since we only assign into a new key below.
    report_path = "outputs/training_report.json"
    with open(report_path, "r") as f:
        report = json.load(f)

    report_key = f"{RUN_NAME}_quicktest" if QUICK_TEST else RUN_NAME
    report[report_key] = {
        "ultralytics_map50": map50,
        "ultralytics_map50_95": map50_95,
        "custom_map50": map_result["mAP"],
        "overlap_frames": overlap_frames,
        "correctly_separated_frames": correctly_separated_frames,
        "separation_rate": correctly_separated_frames / overlap_frames if overlap_frames > 0 else 0.0,
        "latency_device": dev_stats.as_dict(),
        "latency_cpu": cpu_stats.as_dict()
    }

    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"Done! Saved outputs/{RUN_NAME}.pt, report key '{report_key}' in outputs/training_report.json")

if __name__ == "__main__":
    main()
