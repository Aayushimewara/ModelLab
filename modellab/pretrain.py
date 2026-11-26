"""Pretrain the small base model from scratch on Python source text.

    python -m modellab.pretrain --minutes 30

Resumable: a checkpoint is written every `--ckpt_every` steps and picked up on restart.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch

from .model import count_params, new_model, small_llama_config
from .tokenizer import load_tokenizer, train_tokenizer

ROOT = Path(__file__).resolve().parent.parent


def encode_file(tok, path: Path, cache: Path) -> np.ndarray:
    if cache.is_file():
        return np.load(cache)
    text = path.read_text()
    ids: list[int] = []
    chunk = 2_000_000
    for i in range(0, len(text), chunk):
        ids.extend(tok(text[i:i + chunk], add_special_tokens=False)["input_ids"])
    arr = np.asarray(ids, dtype=np.uint16)
    np.save(cache, arr)
    return arr


def batch_from(arr: np.ndarray, rng: np.random.Generator, bs: int, T: int) -> tuple[torch.Tensor, torch.Tensor]:
    starts = rng.integers(0, len(arr) - T - 1, size=bs)
    x = np.stack([arr[s:s + T] for s in starts]).astype(np.int64)
    return torch.from_numpy(x), torch.from_numpy(x.copy())


@torch.no_grad()
def eval_loss(model, arr: np.ndarray, T: int, n_windows: int = 160, bs: int = 16, seed: int = 123) -> float:
    model.eval()
    rng = np.random.default_rng(seed)
    losses = []
    for _ in range(n_windows // bs):
        x, y = batch_from(arr, rng, bs, T)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            losses.append(model(input_ids=x, labels=y).loss.item())
    model.train()
    return float(np.mean(losses))


def lr_at(step: int, total: int, peak: float, warmup: int, floor: float = 0.1) -> float:
    if step < warmup:
        return peak * (step + 1) / warmup
    p = (step - warmup) / max(1, total - warmup)
    return peak * (floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * min(1.0, p))))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=float, default=30.0, help="time budget, sets the number of steps")
    ap.add_argument("--steps", type=int, default=0, help="fixed number of steps (overrides --minutes)")
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--seq", type=int, default=256)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--wd", type=float, default=0.1)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--vocab", type=int, default=4096)
    ap.add_argument("--ckpt_every", type=int, default=150)
    ap.add_argument("--out", default=str(ROOT / "models" / "base"))
    ap.add_argument("--data", default=str(ROOT / "data" / "pretrain"))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.set_num_threads(os.cpu_count() or 2)
    out, data = Path(args.out), Path(args.data)
    out.mkdir(parents=True, exist_ok=True)
    tok_dir = out / "tokenizer"
    if not (tok_dir / "tokenizer.json").is_file():
        print("training tokenizer", flush=True)
        train_tokenizer([(data / "train.txt").read_text()], tok_dir, args.vocab)
    tok = load_tokenizer(tok_dir)
    train_ids = encode_file(tok, data / "train.txt", data / "train.tokens.npy")
    valid_ids = encode_file(tok, data / "valid.txt", data / "valid.tokens.npy")
    print(f"tokens: train {len(train_ids):,}  valid {len(valid_ids):,}  vocab {len(tok):,}", flush=True)

    cfg = small_llama_config(vocab_size=len(tok), max_len=args.seq)
    model = new_model(cfg, args.seed)
    decay = [p for n, p in model.named_parameters() if p.ndim >= 2]
    no_decay = [p for n, p in model.named_parameters() if p.ndim < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": args.wd}, {"params": no_decay, "weight_decay": 0.0}],
                            lr=args.lr, betas=(0.9, 0.95))
    print(f"parameters: {count_params(model)/1e6:.2f}M", flush=True)

    ckpt = out / "pretrain_ckpt.pt"
    step, total, log, elapsed_before = 0, args.steps, [], 0.0
    rng = np.random.default_rng(args.seed)
    if ckpt.is_file():
        st = torch.load(ckpt, weights_only=False)
        model.load_state_dict(st["model"]); opt.load_state_dict(st["opt"])
        step, total, log, elapsed_before = st["step"], st["total"], st["log"], st["elapsed"]
        rng = np.random.default_rng(args.seed + step)
        print(f"resumed at step {step}/{total}", flush=True)

    model.train()
    t_start = time.time()
    log_path = out / "pretrain_log.jsonl"
    while True:
        if total and step >= total:
            break
        if not total and step == 8:  # measure speed, then fix the number of steps for the time budget
            sec_per_step = (time.time() - t_start) / 8
            total = max(200, int(args.minutes * 60 / sec_per_step))
            print(f"{sec_per_step:.2f}s/step -> {total} steps for {args.minutes} minutes", flush=True)
        lr = lr_at(step, total or 10**9, args.lr, args.warmup)
        for g in opt.param_groups:
            g["lr"] = lr
        x, y = batch_from(train_ids, rng, args.bs, args.seq)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            loss = model(input_ids=x, labels=y).loss
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); opt.zero_grad(set_to_none=True)
        step += 1
        if step % 25 == 0 or step == total:
            rec = {"step": step, "loss": round(loss.item(), 4), "lr": round(lr, 6), "grad_norm": round(float(gn), 3),
                   "elapsed_s": round(elapsed_before + time.time() - t_start, 1)}
            if step % 150 == 0 or step == total:
                rec["valid_loss"] = round(eval_loss(model, valid_ids, args.seq), 4)
            log.append(rec)
            with log_path.open("a") as f:
                f.write(json.dumps(rec) + "\n")
            print(rec, flush=True)
        if step % args.ckpt_every == 0 and total:
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "step": step, "total": total, "log": log,
                        "elapsed": elapsed_before + time.time() - t_start}, ckpt)

    final_valid = eval_loss(model, valid_ids, args.seq, n_windows=320)
    model.save_pretrained(out)
    meta = {"parameters": count_params(model), "steps": step, "tokens_seen": step * args.bs * args.seq,
            "train_tokens_available": int(len(train_ids)), "final_valid_loss_nats": round(final_valid, 4),
            "final_valid_perplexity": round(math.exp(final_valid), 3), "wall_seconds": round(elapsed_before + time.time() - t_start, 1),
            "config": cfg.to_dict(), "args": vars(args)}
    (out / "pretrain_meta.json").write_text(json.dumps(meta, indent=1, default=str))
    print("saved", out, "valid loss", round(final_valid, 4), flush=True)


if __name__ == "__main__":
    main()
