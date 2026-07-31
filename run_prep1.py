import sys
import json
import os
sys.path.append(".")
from scripts.prep_data import prep_stage1

info, res = prep_stage1()
print("Info:", info)

if res is not None:
    with open("outputs/stage1_split.json", "w") as f:
        json.dump({"train": res.train, "val": res.val, "test": res.test}, f, indent=2)
else:
    print("Warning: res is None, stage1_split.json not updated from res, will rely on original logic")
