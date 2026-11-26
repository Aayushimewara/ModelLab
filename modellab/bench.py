"""Serving speed of the four ways to deploy these models, measured together on a quiet machine.

    python -m modellab.bench

Forms measured (greedy decoding, exactly 48 new tokens, bfloat16 autocast, all parameters frozen):
  full model        a plain model. A fully fine-tuned model has the same architecture and so the same speed
                    as the base weights used here (speed does not depend on weight values).
  LoRA, separate    base model plus adapter kept apart: every adapted layer does extra small matrix products
  LoRA, merged      adapter folded into the base weights (merge_and_unload): same cost as a plain model
  QLoRA             4-bit base plus adapter. Runs bitsandbytes' CPU path

Rounds are interleaved (form A, B, C, D, A, B, ...) so a slow stretch of the machine hits every form alike.
Parameters are frozen for every form because bfloat16 autocast keeps a cache of the cast weights only for
parameters that require gradients. A model that still has trainable parameters would otherwise look faster for
a reason that has nothing to do with LoRA. (The per-run numbers recorded during training were affected by
exactly this and are not used.)
"""
from __future__ import annotations

import json
import os
import platform
import time
from pathlib import Path

import numpy as np
import torch
from peft import PeftModel

from .data import load_task, prompt_text
from .finetune import RunSpec, load_base
from .tokenizer import load_tokenizer

ROOT = Path(__file__).resolve().parent.parent
FORMS = {"full": "Full model", "lora_separate": "LoRA, adapter kept separate", "lora_merged": "LoRA, adapter merged",
         "qlora": "QLoRA, 4-bit base + adapter"}


def build_forms(base: str, lora_adapter: str, qlora_adapter: str) -> dict:
    spec_l = RunSpec(run_id="bench", method="lora", base=base)
    spec_q = RunSpec(run_id="bench", method="qlora", base=base)
    forms = {
        "full": load_base(spec_l),
        "lora_separate": PeftModel.from_pretrained(load_base(spec_l), lora_adapter),
        "lora_merged": PeftModel.from_pretrained(load_base(spec_l), lora_adapter).merge_and_unload(),
        "qlora": PeftModel.from_pretrained(load_base(spec_q), qlora_adapter),
    }
    for model in forms.values():
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)
    return forms


@torch.no_grad()
def _time_generate(model, enc, n_new: int, pad_id: int) -> float:
    t = time.perf_counter()
    with torch.autocast("cpu", dtype=torch.bfloat16):
        model.generate(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"], max_new_tokens=n_new,
                       min_new_tokens=n_new, do_sample=False, pad_token_id=pad_id)
    return time.perf_counter() - t


def run(base: str | None = None, rounds: int = 5, n_new: int = 48, batch_sizes=(1, 16), out: str | None = None) -> dict:
    base = base or str(ROOT / "models" / "base")
    torch.set_num_threads(os.cpu_count() or 2)
    tok = load_tokenizer(Path(base) / "tokenizer")
    prompts = [prompt_text(e["sig"]) for e in load_task(ROOT / "data" / "task")["test"][:max(batch_sizes)]]
    forms = build_forms(base, str(ROOT / "results" / "adapters" / "main-mlora-s0"),
                        str(ROOT / "results" / "adapters" / "main-mqlora-s0"))
    result = {"n_new_tokens": n_new, "rounds": rounds, "cpu_count": os.cpu_count(), "machine": platform.platform(),
              "torch": torch.__version__, "forms": {}}
    for bs in batch_sizes:
        enc = tok(prompts[:bs], return_tensors="pt", padding=True, add_special_tokens=False)
        times = {k: [] for k in forms}
        for r in range(rounds + 1):                       # round 0 is a warm up and is dropped
            for k, model in forms.items():
                dt = _time_generate(model, enc, n_new, tok.pad_token_id)
                if r > 0:
                    times[k].append(dt)
        for k, ts in times.items():
            tps = [bs * n_new / t for t in ts]
            result["forms"].setdefault(k, {})[f"batch{bs}"] = {
                "tokens_per_second_median": round(float(np.median(tps)), 1),
                "tokens_per_second_min": round(min(tps), 1), "tokens_per_second_max": round(max(tps), 1)}
    path = Path(out) if out else ROOT / "results" / "inference_bench.json"
    path.write_text(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    res = run()
    for k, v in res["forms"].items():
        print(f"{FORMS[k]:32s} batch1 {v['batch1']['tokens_per_second_median']:7.1f}   batch16 {v['batch16']['tokens_per_second_median']:7.1f}")
