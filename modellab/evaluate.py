"""Decoding and scoring."""
from __future__ import annotations

import math
import re
import time

import numpy as np
import torch

from .data import completion_text, few_shot_prompt, prompt_text

LABEL_RE = re.compile(r"^req: (.*) \| opt: (.*) \| var: (.*)$")


def parse_label(text: str) -> set[tuple[str, str]] | None:
    """(category, name) pairs in a label string, or None if the string is not in the expected format."""
    m = LABEL_RE.match(text.strip())
    if not m:
        return None
    pairs = set()
    for cat, body in zip(("req", "opt", "var"), m.groups()):
        body = body.strip()
        if body in ("", "-"):
            continue
        for name in body.split(", "):
            pairs.add((cat, name.strip()))
    return pairs


def score_one(pred: str, gold: str) -> dict:
    gp, pp = parse_label(gold), parse_label(pred)
    valid = pp is not None
    pp = pp or set()
    tp = len(gp & pp)
    return {"em": float(pred.strip() == gold.strip()), "valid": float(valid), "tp": tp, "n_pred": len(pp), "n_gold": len(gp)}


def aggregate(scores: list[dict]) -> dict:
    tp = sum(s["tp"] for s in scores)
    npred = sum(s["n_pred"] for s in scores)
    ngold = sum(s["n_gold"] for s in scores)
    p = tp / npred if npred else 0.0
    r = tp / ngold if ngold else 0.0
    return {"em": float(np.mean([s["em"] for s in scores])), "valid_rate": float(np.mean([s["valid"] for s in scores])),
            "field_precision": p, "field_recall": r, "field_f1": 2 * p * r / (p + r) if p + r else 0.0, "n": len(scores)}


def breakdown(examples: list[dict], per_example_em: list[float]) -> dict:
    """Exact match split by number of parameters and by whether any default value is complex."""
    out: dict = {}
    buckets = {"1-2 params": lambda e: e["n_params"] <= 2, "3-4 params": lambda e: 3 <= e["n_params"] <= 4,
               "5+ params": lambda e: e["n_params"] >= 5, "simple defaults": lambda e: not e["complex"],
               "complex defaults": lambda e: e["complex"]}
    for name, pred in buckets.items():
        idx = [i for i, e in enumerate(examples) if pred(e)]
        out[name] = {"n": len(idx), "em": float(np.mean([per_example_em[i] for i in idx])) if idx else None}
    return out


@torch.no_grad()
def generate(model, tok, prompts: list[str], max_new_tokens: int = 80, batch_size: int = 32) -> list[str]:
    """Greedy decoding. Returns the first line of each completion. Prompts are sorted by length to limit padding."""
    model.eval()
    order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))
    outs: list[str] = [""] * len(prompts)
    for s in range(0, len(order), batch_size):
        idx = order[s:s + batch_size]
        enc = tok([prompts[i] for i in idx], return_tensors="pt", padding=True, add_special_tokens=False)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            gen = model.generate(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"], max_new_tokens=max_new_tokens,
                                 do_sample=False, eos_token_id=tok.eos_token_id, pad_token_id=tok.pad_token_id)
        new = gen[:, enc["input_ids"].shape[1]:]
        for i, row in zip(idx, new):
            outs[i] = tok.decode(row, skip_special_tokens=True).split("\n")[0].strip()
    return outs


def evaluate_model(model, tok, examples: list[dict], shots: list[list[dict]] | None = None, batch_size: int = 32) -> dict:
    """Exact match and field level scores. `shots[i]` is the list of in-context examples for example i (few-shot)."""
    prompts = [few_shot_prompt(shots[i], e["sig"]) if shots else prompt_text(e["sig"]) for i, e in enumerate(examples)]
    t = time.time()
    preds = generate(model, tok, prompts, batch_size=batch_size)
    secs = time.time() - t
    scores = [score_one(p, e["target"]) for p, e in zip(preds, examples)]
    agg = aggregate(scores)
    per_em = [s["em"] for s in scores]
    return {**agg, "per_example_em": per_em, "breakdown": breakdown(examples, per_em), "predictions": preds, "decode_seconds": round(secs, 2)}


@torch.no_grad()
def general_loss(model, token_ids: np.ndarray, T: int = 256, n_windows: int = 160, bs: int = 16, seed: int = 123) -> float:
    """Average next token loss (nats) on held out Python source. Used to measure forgetting."""
    from .pretrain import batch_from
    model.eval()
    rng = np.random.default_rng(seed)
    losses = []
    for _ in range(n_windows // bs):
        x, y = batch_from(token_ids, rng, bs, T)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            losses.append(model(input_ids=x, labels=y).loss.item())
    return float(np.mean(losses))


def perplexity(loss_nats: float) -> float:
    return math.exp(loss_nats)


def pick_shots(train: list[dict], examples: list[dict], k: int, mode: str = "random", seed: int = 0) -> list[list[dict]]:
    """In-context examples for every query. 'random': the same k random training examples for everyone.
    'retrieved': the k training examples most similar to the query (character n-gram TF-IDF cosine)."""
    if k == 0:
        return [[] for _ in examples]
    if mode == "random":
        rng = np.random.default_rng(seed)
        chosen = [train[i] for i in rng.choice(len(train), size=k, replace=False)]
        return [chosen for _ in examples]
    from sklearn.feature_extraction.text import TfidfVectorizer
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), sublinear_tf=True)
    X = vec.fit_transform([t["sig"] for t in train])
    Q = vec.transform([e["sig"] for e in examples])
    sims = (Q @ X.T).toarray()
    return [[train[j] for j in np.argsort(-sims[i])[:k][::-1]] for i in range(len(examples))]
