"""Runs every hand-rolled test module without needing pytest installed.
NOTE: test_engine_smoke.py (if present) requires torch/torchvision and is
skipped automatically if they aren't importable -- everything else here
(iou, nms, metrics, dedup, splits, latency, inference, synthetic_detection)
has zero torch dependency and always runs."""
import sys, os, importlib, traceback

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
TEST_DIR = os.path.dirname(__file__)

modules = sorted(f[:-3] for f in os.listdir(TEST_DIR) if f.startswith("test_") and f.endswith(".py"))

total_pass, total_fail = 0, 0
for mod_name in modules:
    try:
        mod = importlib.import_module(mod_name)
    except ImportError as e:
        print(f"SKIP {mod_name} (missing dependency: {e})")
        continue

    fns = [v for k, v in vars(mod).items() if k.startswith("test_") and callable(v)]
    for fn in fns:
        try:
            fn()
            total_pass += 1
        except Exception:
            total_fail += 1
            print(f"FAIL {mod_name}.{fn.__name__}")
            traceback.print_exc()

print(f"\n{total_pass} passed, {total_fail} failed")
sys.exit(1 if total_fail else 0)
