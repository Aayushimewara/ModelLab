"""Build the task data (data/task/*.jsonl) and the pretraining text (data/pretrain/*.txt).

    python scripts/build_data.py

Reads the Python standard library of the running interpreter. Different Python versions give
different data, so the committed task files are the ones the reported results use.
The pretraining text is large (tens of MB) and is not committed, only rebuilt here.
"""
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from modellab import data  # noqa: E402

splits = data.build_task()
data.write_task(splits, ROOT / "data" / "task")
held = {e["sig"] for split in ("val", "test") for e in splits[split]}
train, valid, info = data.build_pretrain_text(held)
# final leak guard: drop val/test examples whose signature appears in a `def` line of the pretraining text
import re
removed = {}
for split in ("val", "test"):
    keep = []
    for e in splits[split]:
        if re.search(r"def\s+\w+\s*" + re.escape("(" + e["sig"] + ")"), train):
            removed[split] = removed.get(split, 0) + 1
        else:
            keep.append(e)
    splits[split] = keep
data.write_task(splits, ROOT / "data" / "task")
info["removed_for_def_leak"] = removed
pre = ROOT / "data" / "pretrain"
pre.mkdir(parents=True, exist_ok=True)
(pre / "train.txt").write_text(train)
(pre / "valid.txt").write_text(valid)
summary = {"python": sys.version.split()[0], "pretrain": info,
           "task": {k: {"examples": len(v), "modules": len({e["module"] for e in v}),
                        "complex_default_share": round(sum(e["complex"] for e in v) / len(v), 3),
                        "n_params": dict(sorted(Counter(e["n_params"] for e in v).items()))} for k, v in splits.items()}}
(ROOT / "data" / "summary.json").write_text(json.dumps(summary, indent=1))
print(json.dumps(summary, indent=1))
