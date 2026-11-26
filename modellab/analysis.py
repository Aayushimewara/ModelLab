"""Turn results/runs/*.json into results/results.json: tables, confidence intervals, paired comparisons.

Two kinds of uncertainty are reported and they answer different questions:
  * test-set uncertainty: a bootstrap over the test questions (what if I had drawn different test modules?)
  * training uncertainty: the spread across random seeds (what if I trained again?)
With only 2 or 3 seeds the second is rough, so both are shown.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from .data import load_task, naive_split_baseline, parse_baseline

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "results" / "runs"


def load_runs(runs_dir: Path = RUNS) -> dict[str, dict]:
    return {p.stem: json.loads(p.read_text()) for p in sorted(runs_dir.glob("*.json"))}


def bootstrap_mean(values: np.ndarray, n_boot: int = 4000, seed: int = 0) -> tuple[float, float, float]:
    """values: one number per test question. Returns mean and 95% interval from resampling questions."""
    v = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(v), size=(n_boot, len(v)))
    m = v[idx].mean(axis=1)
    return float(v.mean()), float(np.quantile(m, 0.025)), float(np.quantile(m, 0.975))


def paired_diff(a: np.ndarray, b: np.ndarray, n_boot: int = 4000, seed: int = 0) -> tuple[float, float, float]:
    return bootstrap_mean(np.asarray(a, float) - np.asarray(b, float), n_boot, seed)


def seed_avg(runs: list[dict], split: str = "test") -> np.ndarray:
    """Per question exact match, averaged over the seeds of one configuration."""
    return np.mean([r[split]["per_example_em"] for r in runs], axis=0)


def _mean(runs: list[dict], getter) -> float | None:
    vals = [getter(r) for r in runs]
    vals = [v for v in vals if v is not None]
    return float(np.mean(vals)) if vals else None


def summarize_group(runs: list[dict], split: str = "test") -> dict:
    ems = [r[split]["em"] for r in runs]
    m, lo, hi = bootstrap_mean(seed_avg(runs, split))
    out = {"n_seeds": len(runs), "em_by_seed": [round(e, 4) for e in ems], "em_mean": round(m, 4),
           "em_seed_std": round(float(np.std(ems, ddof=1)), 4) if len(ems) > 1 else None,
           "em_ci_questions": [round(lo, 4), round(hi, 4)],
           "field_f1": round(_mean(runs, lambda r: r[split]["field_f1"]), 4),
           "valid_rate": round(_mean(runs, lambda r: r[split]["valid_rate"]), 4)}
    for k in ("trainable_params", "total_params", "weight_bytes", "optimizer_state_bytes"):
        out[k] = int(runs[0][k])
    out["trainable_fraction"] = round(runs[0]["trainable_fraction"], 5)
    out["general_loss_before"] = round(_mean(runs, lambda r: r["general_loss_before"]), 4)
    out["general_loss_after"] = round(_mean(runs, lambda r: r["general_loss_after"]), 4)
    out["forgetting_nats"] = round(_mean(runs, lambda r: r["forgetting_nats"]), 4)
    if "training" in runs[0]:
        out["train_seconds"] = round(_mean(runs, lambda r: r["training"]["train_seconds"]), 1)
        out["tokens_per_second"] = round(_mean(runs, lambda r: r["training"]["tokens_per_second"]), 0)
        out["final_train_loss"] = round(_mean(runs, lambda r: r["training"]["final_train_loss"]), 4)
        out["steps"] = runs[0]["training"]["steps"]
    out["peak_rss_mb"] = round(_mean(runs, lambda r: r["peak_rss_mb_after_training"]), 0)
    br: dict = defaultdict(list)
    for r in runs:
        for k, v in r[split]["breakdown"].items():
            if v["em"] is not None:
                br[k].append((v["em"], v["n"]))
    out["breakdown"] = {k: {"em": round(float(np.mean([e for e, _ in v])), 4), "n": v[0][1]} for k, v in br.items()}
    return out


def group_by(runs: dict[str, dict], prefix: str, key) -> dict:
    groups: dict = defaultdict(list)
    for rid, r in runs.items():
        if rid.startswith(prefix):
            groups[key(r["spec"])].append(r)
    return groups


def aggregate(runs_dir: Path = RUNS) -> dict:
    runs = load_runs(runs_dir)
    task = load_task(ROOT / "data" / "task")
    out: dict = {"n_runs": len(runs)}

    # ---- baselines that need no model training
    test = task["test"]
    out["heuristics"] = {
        "reference_parser_em": float(np.mean([parse_baseline(e["sig"]) == e["target"] for e in test])),
        "naive_comma_split_em": float(np.mean([naive_split_baseline(e["sig"]) == e["target"] for e in test])),
    }
    out["baselines"] = {rid.replace("baseline-", ""): {"test_em": round(r["test"]["em"], 4), "test_field_f1": round(r["test"]["field_f1"], 4),
                                                       "valid_rate": round(r["test"]["valid_rate"], 4)}
                        for rid, r in runs.items() if rid.startswith("baseline-")}

    tuning = ROOT / "results" / "tuning.json"
    if tuning.is_file():
        out["tuning"] = json.loads(tuning.read_text())

    # ---- main comparison
    main = group_by(runs, "main-", lambda s: s["method"])
    out["main"] = {m: summarize_group(rs) for m, rs in main.items()}
    out["paired"] = {}
    names = {"lora vs full": ("lora", "full"), "qlora vs lora": ("qlora", "lora"), "qlora vs full": ("qlora", "full")}
    for label, (a, b) in names.items():
        if a in main and b in main:
            d, lo, hi = paired_diff(seed_avg(main[a]), seed_avg(main[b]))
            seed_d = [x["test"]["em"] - y["test"]["em"] for x, y in zip(sorted(main[a], key=lambda r: r["spec"]["seed"]), sorted(main[b], key=lambda r: r["spec"]["seed"]))]
            out["paired"][label] = {"diff_em": round(d, 4), "ci_questions": [round(lo, 4), round(hi, 4)],
                                    "diff_by_seed": [round(x, 4) for x in seed_d]}
    # serving speed comes from the standalone benchmark (python -m modellab.bench), not from the per-run numbers
    bench = ROOT / "results" / "inference_bench.json"
    out["inference"] = json.loads(bench.read_text()) if bench.is_file() else {}
    out["predictions_example"] = {}

    # ---- ablations
    rank = group_by(runs, "rank-", lambda s: s["rank"])
    if "lora" in main:   # r=8 comes from the main runs, seeds 0 and 1 only so the seeds match
        rank[8] = [r for r in main["lora"] if r["spec"]["seed"] in (0, 1)]
    out["rank"] = {str(k): summarize_group(v) for k, v in sorted(rank.items())}
    targets = group_by(runs, "targets-", lambda s: s["targets"])
    if "lora" in main:
        targets["all"] = [r for r in main["lora"] if r["spec"]["seed"] in (0, 1)]
    out["targets"] = {k: summarize_group(v) for k, v in targets.items()}
    data: dict = defaultdict(dict)
    for rid, r in runs.items():
        if rid.startswith("data-"):
            data[r["spec"]["method"]].setdefault(r["spec"]["n_train"], []).append(r)
    for m in ("full", "lora"):
        if m in main:
            data[m][len(task["train"])] = [r for r in main[m] if r["spec"]["seed"] in (0, 1)]
    out["data_size"] = {m: {str(n): summarize_group(v) for n, v in sorted(d.items())} for m, d in data.items()}
    return out


if __name__ == "__main__":
    res = aggregate()
    (ROOT / "results" / "results.json").write_text(json.dumps(res, indent=1))
    print("wrote results/results.json from", res["n_runs"], "runs")
