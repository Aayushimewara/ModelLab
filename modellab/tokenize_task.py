"""Turn task examples into training tensors with the loss on the answer only."""
from __future__ import annotations

import random

import torch

from .data import completion_text, prompt_text


def encode_example(tok, ex: dict) -> tuple[list[int], list[int]]:
    p = tok(prompt_text(ex["sig"]), add_special_tokens=False)["input_ids"]
    c = tok(completion_text(ex["target"]), add_special_tokens=False)["input_ids"] + [tok.eos_token_id]
    return p + c, [-100] * len(p) + c


def encode_all(tok, examples: list[dict]) -> list[tuple[list[int], list[int]]]:
    return [encode_example(tok, e) for e in examples]


def collate(batch: list[tuple[list[int], list[int]]], pad_id: int) -> dict[str, torch.Tensor]:
    n = max(len(x) for x, _ in batch)
    ids = torch.full((len(batch), n), pad_id, dtype=torch.long)
    labels = torch.full((len(batch), n), -100, dtype=torch.long)
    mask = torch.zeros((len(batch), n), dtype=torch.long)
    for i, (x, y) in enumerate(batch):
        ids[i, :len(x)] = torch.tensor(x)
        labels[i, :len(y)] = torch.tensor(y)
        mask[i, :len(x)] = 1
    return {"input_ids": ids, "attention_mask": mask, "labels": labels}


def batches(data: list, bs: int, steps: int, seed: int):
    """Yield `steps` shuffled batches, reshuffling every pass over the data."""
    rng = random.Random(seed)
    order: list[int] = []
    for _ in range(steps):
        if len(order) < bs:
            fresh = list(range(len(data)))
            rng.shuffle(fresh)
            order.extend(fresh)
        take, order = order[:bs], order[bs:]
        yield [data[i] for i in take]
