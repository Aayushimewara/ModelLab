"""Fine-tune the base model on the signature task. One run per process, so memory numbers are clean.

    python -m modellab.finetune --spec '{"run_id": "demo", "method": "lora", "rank": 8, "lr": 1e-3}' --out results/runs/demo.json

Methods
    none    no training. Used for zero shot and few shot baselines.
    full    every weight is trained.
    lora    base frozen in fp32, low rank adapters (PEFT) on the chosen linear layers.
    qlora   base frozen and stored in 4 bit NF4 with double quantization (bitsandbytes), bf16 compute, LoRA adapters (PEFT).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

try:
    import resource
except ImportError:
    resource = None
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch

from .data import load_task
from .evaluate import evaluate_model, general_loss, pick_shots
from .model import ATTN_TARGETS, LINEAR_TARGETS, MLP_TARGETS
from .tokenize_task import batches, collate, encode_all
from .tokenizer import load_tokenizer

ROOT = Path(__file__).resolve().parent.parent
TARGET_SETS = {"all": LINEAR_TARGETS, "attn": ATTN_TARGETS, "mlp": MLP_TARGETS}


@dataclass
class RunSpec:
    run_id: str
    method: str = "lora"              # none | full | lora | qlora
    rank: int = 8
    alpha: float | None = None        # defaults to 2 * rank
    dropout: float = 0.05
    targets: str = "all"              # all | attn | mlp
    lr: float = 1e-3
    epochs: int = 4
    min_steps: int = 150
    bs: int = 16
    n_train: int = 0                  # 0 means the whole training set
    seed: int = 0
    wd: float = 0.0
    double_quant: bool = True
    eval_val: bool = True
    eval_test: bool = True
    save_adapter: bool = False
    bench: bool = False
    shots: int = 0                    # only for method "none"
    shot_mode: str = "random"
    base: str = str(ROOT / "models" / "base")
    data: str = str(ROOT / "data" / "task")
    pretrain_data: str = str(ROOT / "data" / "pretrain")
    extra: dict = field(default_factory=dict)


# ------------------------------------------------------------------ model construction
def load_base(spec: RunSpec):
    from transformers import AutoModelForCausalLM
    if spec.method == "qlora":
        from transformers import BitsAndBytesConfig
        q = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=spec.double_quant,
                               bnb_4bit_compute_dtype=torch.bfloat16)
        return AutoModelForCausalLM.from_pretrained(spec.base, quantization_config=q, device_map="cpu", dtype=torch.float32)
    return AutoModelForCausalLM.from_pretrained(spec.base, dtype=torch.float32)


def logical_param_counts(model) -> dict:
    """Parameter counts that treat a packed 4 bit weight as the two weights it stores. (PEFT's own count
    reports packed bytes as parameters, which understates the size of a quantized model.)"""
    total, trainable, base_bytes = 0, 0, 0
    seen = set()
    for p in model.parameters():
        if id(p) in seen:
            continue
        seen.add(id(p))
        n = p.numel() * (2 if p.dtype == torch.uint8 else 1)
        total += n
        if p.requires_grad:
            trainable += n
        base_bytes += p.numel() * p.element_size()
    for b in model.buffers():
        base_bytes += b.numel() * b.element_size()
    return {"total_params": total, "trainable_params": trainable, "weight_bytes": base_bytes}


def apply_method(model, spec: RunSpec):
    if spec.method in ("none", "full"):
        return model
    from peft import LoraConfig, get_peft_model
    torch.manual_seed(spec.seed)
    cfg = LoraConfig(r=spec.rank, lora_alpha=spec.alpha if spec.alpha is not None else 2 * spec.rank,
                     lora_dropout=spec.dropout, target_modules=list(TARGET_SETS[spec.targets]), bias="none",
                     task_type="CAUSAL_LM")
    return get_peft_model(model, cfg)


def lr_at(step: int, total: int, peak: float, warmup_frac: float = 0.06, floor: float = 0.1) -> float:
    warm = max(1, int(total * warmup_frac))
    if step < warm:
        return peak * (step + 1) / warm
    p = (step - warm) / max(1, total - warm)
    return peak * (floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * min(1.0, p))))


def assert_backward_ready(model) -> None:
    """Refuse to train on 4 bit layers that were already switched to bitsandbytes' inference-only CPU format.
    Training on them runs without error but the gradients that reach the adapters are wrong (see run())."""
    for name, mod in model.named_modules():
        qs = getattr(getattr(mod, "weight", None), "quant_state", None)
        if qs is not None and getattr(qs, "packing_format_for_cpu", False):
            raise RuntimeError(f"{name} is in bitsandbytes' inference-only packed format; gradients through it would be wrong. "
                               "Do not run an eval-mode, no-grad forward on a 4 bit model you plan to train.")


def train(model, tok, examples: list[dict], spec: RunSpec) -> dict:
    assert_backward_ready(model)
    data = encode_all(tok, examples)
    steps = max(spec.min_steps, math.ceil(len(data) / spec.bs) * spec.epochs)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=spec.lr, weight_decay=spec.wd, betas=(0.9, 0.999))
    model.train()
    losses, tokens = [], 0
    t0 = time.time()
    for step, batch in enumerate(batches(data, spec.bs, steps, spec.seed)):
        for g in opt.param_groups:
            g["lr"] = lr_at(step, steps, spec.lr)
        b = collate(batch, tok.pad_token_id)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            loss = model(**b).loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        losses.append(loss.item())
        tokens += int(b["attention_mask"].sum())
    secs = time.time() - t0
    return {"steps": steps, "train_seconds": round(secs, 2), "train_tokens": tokens, "tokens_per_second": round(tokens / secs, 1),
            "seconds_per_step": round(secs / steps, 4), "final_train_loss": round(float(np.mean(losses[-20:])), 4),
            "loss_curve": [round(float(np.mean(losses[i:i + 10])), 4) for i in range(0, len(losses), 10)]}


def peak_rss_mb() -> float:
    if resource is not None:
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return round(rss / (1024 * 1024 if sys.platform == "darwin" else 1024), 1)

    if sys.platform == "win32":
        import ctypes

        class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
            _fields_ = [
                ("cb", ctypes.c_ulong),
                ("PageFaultCount", ctypes.c_ulong),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        counters = PROCESS_MEMORY_COUNTERS()
        counters.cb = ctypes.sizeof(counters)
        if ctypes.windll.psapi.GetProcessMemoryInfo(ctypes.windll.kernel32.GetCurrentProcess(),
                                                   ctypes.byref(counters), counters.cb):
            return round(counters.PeakWorkingSetSize / (1024 * 1024), 1)

    return 0.0


# ------------------------------------------------------------------ inference speed
@torch.no_grad()
def bench_inference(model, tok, prompts: list[str], n_new: int = 48, repeats: int = 3, batch_sizes=(1, 16)) -> dict:
    """Tokens per second for greedy decoding of exactly `n_new` new tokens (eos disabled so every run does equal work)."""
    out = {}
    model.eval()
    for p in model.parameters():   # bf16 autocast caches weight casts only for parameters that require grad, which would flatter a model that still has them
        p.requires_grad_(False)
    for bs in batch_sizes:
        enc = tok(prompts[:bs], return_tensors="pt", padding=True, add_special_tokens=False)
        times = []
        for r in range(repeats + 1):
            t = time.time()
            with torch.autocast("cpu", dtype=torch.bfloat16):
                model.generate(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"], max_new_tokens=n_new,
                               min_new_tokens=n_new, do_sample=False, pad_token_id=tok.pad_token_id)
            if r > 0:
                times.append(time.time() - t)
        out[f"batch{bs}_tokens_per_second"] = round(bs * n_new / float(np.median(times)), 1)
    return out


# ------------------------------------------------------------------ the run
def run(spec: RunSpec) -> dict:
    torch.set_num_threads(os.cpu_count() or 2)
    torch.manual_seed(spec.seed)
    np.random.seed(spec.seed)
    tok = load_tokenizer(Path(spec.base) / "tokenizer")
    task = load_task(spec.data)
    train_all = task["train"]
    if spec.n_train and spec.n_train < len(train_all):
        rng = np.random.default_rng(1000 + spec.seed)
        train_ex = [train_all[i] for i in rng.choice(len(train_all), size=spec.n_train, replace=False)]
    else:
        train_ex = train_all

    model = load_base(spec)
    valid_ids = np.load(Path(spec.pretrain_data) / "valid.tokens.npy") if (Path(spec.pretrain_data) / "valid.tokens.npy").is_file() \
        else np.asarray(tok(Path(spec.pretrain_data, "valid.txt").read_text(), add_special_tokens=False)["input_ids"], dtype=np.uint16)

    res: dict = {"spec": asdict(spec), "n_train": len(train_ex)}
    if spec.method == "qlora":
        # On CPUs with AVX512-BF16, bitsandbytes converts a 4 bit layer to an inference-only packed format the
        # first time it runs in eval mode on an input that has no gradient. That format has no backward pass,
        # so one such forward silently stops gradients from flowing through the frozen base for the rest of the
        # run. Measure the quantized base on a throwaway copy so the training copy is never touched.
        probe = load_base(spec)
        res["general_loss_before"] = round(general_loss(probe, valid_ids), 4)   # the quantized base
        del probe
    else:
        res["general_loss_before"] = round(general_loss(model, valid_ids), 4)

    model = apply_method(model, spec)
    res.update(logical_param_counts(model))
    res["trainable_fraction"] = round(res["trainable_params"] / res["total_params"], 5)
    res["optimizer_state_bytes"] = res["trainable_params"] * 8     # AdamW keeps two fp32 values per trainable weight

    if spec.method != "none":
        res["training"] = train(model, tok, train_ex, spec)
    res["peak_rss_mb_after_training"] = peak_rss_mb()
    if spec.save_adapter and spec.method in ("lora", "qlora"):
        out = ROOT / "results" / "adapters" / spec.run_id
        out.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(out)

    if spec.eval_val:
        shots = pick_shots(train_all, task["val"], spec.shots, spec.shot_mode, spec.seed) if spec.method == "none" else None
        res["val"] = evaluate_model(model, tok, task["val"], shots)
    if spec.eval_test:
        shots = pick_shots(train_all, task["test"], spec.shots, spec.shot_mode, spec.seed) if spec.method == "none" else None
        res["test"] = evaluate_model(model, tok, task["test"], shots)
    res["general_loss_after"] = round(general_loss(model, valid_ids), 4)
    res["forgetting_nats"] = round(res["general_loss_after"] - res["general_loss_before"], 4)

    if spec.bench:
        from .data import prompt_text
        prompts = [prompt_text(e["sig"]) for e in task["test"][:16]]
        res["inference"] = {"adapter_unmerged" if spec.method in ("lora", "qlora") else spec.method: bench_inference(model, tok, prompts)}
        if spec.method == "lora":
            merged = model.merge_and_unload()
            res["inference"]["lora_merged"] = bench_inference(merged, tok, prompts)
            model = merged

    res["peak_rss_mb"] = peak_rss_mb()
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True, help="JSON object with RunSpec fields")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    spec = RunSpec(**json.loads(args.spec))
    t = time.time()
    res = run(spec)
    res["wall_seconds"] = round(time.time() - t, 1)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res))
    summary = {k: res.get(k) for k in ("trainable_params", "general_loss_before", "general_loss_after", "peak_rss_mb")}
    for s in ("val", "test"):
        if s in res:
            summary[f"{s}_em"] = round(res[s]["em"], 4)
    print(spec.run_id, summary, flush=True)


if __name__ == "__main__":
    main()
