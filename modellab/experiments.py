"""The experiment plan and runner.

    python -m modellab.experiments baselines     # zero / few shot, no training
    python -m modellab.experiments tune          # learning rate per method, scored on the VAL questions only
    python -m modellab.experiments main          # full vs LoRA vs QLoRA, 3 seeds, scored on TEST
    python -m modellab.experiments ablate        # rank, target layers, training set size
    python -m modellab.experiments all

Every run is a separate process that writes results/runs/<run_id>.json. Finished runs are skipped,
so an interrupted experiment resumes where it stopped.
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "results" / "runs"
TUNING = ROOT / "results" / "tuning.json"

LR_GRID = {"full": [3e-5, 1e-4, 3e-4, 1e-3, 3e-3], "lora": [1e-3, 3e-3, 1e-2], "qlora": [1e-3, 3e-3, 1e-2]}
MAIN_SEEDS = [0, 1, 2]
ABLATION_SEEDS = [0, 1]
RANKS = [1, 2, 4, 16, 32]
DATA_SIZES = [100, 300, 1000]


def rid(stage: str, **kw) -> str:
    parts = [stage] + [f"{k}{v}" for k, v in kw.items()]
    return "-".join(str(p) for p in parts).replace(".", "p").replace("e-0", "e-")


def baseline_specs() -> list[dict]:
    out = [{"run_id": "baseline-zeroshot", "method": "none", "shots": 0}]
    for k in (1, 3):
        out.append({"run_id": f"baseline-{k}shot-random", "method": "none", "shots": k, "shot_mode": "random"})
    out.append({"run_id": "baseline-3shot-retrieved", "method": "none", "shots": 3, "shot_mode": "retrieved"})
    return out


def tune_specs() -> list[dict]:
    out = []
    for method, lrs in LR_GRID.items():
        for lr in lrs:
            out.append({"run_id": rid("tune", m=method, lr=lr), "method": method, "rank": 8, "lr": lr, "seed": 0,
                        "eval_test": False})
    return out


def chosen_lrs() -> dict:
    if not TUNING.is_file():
        raise SystemExit("run `python -m modellab.experiments tune` first (it picks the learning rates)")
    return json.loads(TUNING.read_text())["chosen_lr"]


def main_specs() -> list[dict]:
    lr = chosen_lrs()
    out = []
    for seed in MAIN_SEEDS:
        for method in ("full", "lora", "qlora"):
            spec = {"run_id": rid("main", m=method, s=seed), "method": method, "rank": 8, "lr": lr[method], "seed": seed}
            if seed == 0:
                spec.update(save_adapter=method != "full", bench=False)
            out.append(spec)
    return out


def ablation_specs() -> list[dict]:
    lr = chosen_lrs()
    out = []
    for seed in ABLATION_SEEDS:
        for r in RANKS:
            out.append({"run_id": rid("rank", r=r, s=seed), "method": "lora", "rank": r, "lr": lr["lora"], "seed": seed})
        for t in ("attn", "mlp"):
            out.append({"run_id": rid("targets", t=t, s=seed), "method": "lora", "rank": 8, "targets": t, "lr": lr["lora"], "seed": seed})
        for n in DATA_SIZES:
            for method in ("full", "lora"):
                out.append({"run_id": rid("data", m=method, n=n, s=seed), "method": method, "rank": 8, "n_train": n,
                            "lr": lr[method], "seed": seed})
    return out


STAGES = {"baselines": baseline_specs, "tune": tune_specs, "main": main_specs, "ablate": ablation_specs}


def run_specs(specs: list[dict]) -> None:
    RUNS.mkdir(parents=True, exist_ok=True)
    todo = [s for s in specs if not (RUNS / f"{s['run_id']}.json").is_file()]
    print(f"{len(specs)} runs planned, {len(specs) - len(todo)} already done, {len(todo)} to run", flush=True)
    for i, spec in enumerate(todo, 1):
        out = RUNS / f"{spec['run_id']}.json"
        t = time.time()
        proc = subprocess.run([sys.executable, "-m", "modellab.finetune", "--spec", json.dumps(spec), "--out", str(out)],
                              cwd=ROOT, capture_output=True, text=True)
        tail = [l for l in proc.stdout.strip().splitlines() if l][-1:] or [""]
        if proc.returncode != 0 or not out.is_file():
            print(f"[{i}/{len(todo)}] FAILED {spec['run_id']}\n{proc.stderr[-1500:]}", flush=True)
            continue
        print(f"[{i}/{len(todo)}] {time.time() - t:5.0f}s  {tail[0]}", flush=True)


def select_learning_rates() -> dict:
    """Best learning rate per method by VAL exact match (ties broken by VAL field F1, then the smaller rate)."""
    table: dict = {}
    for method, lrs in LR_GRID.items():
        rows = []
        for lr in lrs:
            f = RUNS / f"{rid('tune', m=method, lr=lr)}.json"
            if not f.is_file():
                continue
            r = json.loads(f.read_text())
            rows.append({"lr": lr, "val_em": r["val"]["em"], "val_f1": r["val"]["field_f1"], "final_train_loss": r["training"]["final_train_loss"]})
        table[method] = rows
    chosen = {m: sorted(rows, key=lambda x: (-x["val_em"], -x["val_f1"], x["lr"]))[0]["lr"] for m, rows in table.items() if rows}
    TUNING.write_text(json.dumps({"chosen_lr": chosen, "grid": table, "selected_on": "validation modules only"}, indent=1))
    return chosen


def main() -> None:
    stage = sys.argv[1] if len(sys.argv) > 1 else "all"
    order = ["baselines", "tune", "main", "ablate"] if stage == "all" else [stage]
    for name in order:
        if name not in STAGES:
            raise SystemExit(f"unknown stage {name}. Options: {list(STAGES) + ['all']}")
        if name in ("main", "ablate") and not TUNING.is_file():
            pass  # chosen_lrs() will raise a clear error
        print(f"== stage: {name}", flush=True)
        run_specs(STAGES[name]())
        if name == "tune":
            print("chosen learning rates:", select_learning_rates(), flush=True)


if __name__ == "__main__":
    main()
