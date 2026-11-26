"""LoRA written from scratch, as a reference.

    h = W x + (alpha / r) * B (A x)       W frozen (out x in), A is r x in, B is out x r

A starts random and B starts at zero, so the adapted model equals the base model before training.
Only A and B are trained. After training B @ A can be added into W (`merge`), so inference costs nothing extra.

The experiments in this repo use the PEFT library. This module exists so the mechanics are visible, and
the tests check that it matches PEFT exactly (same outputs, same gradients, same training trajectory).
"""
from __future__ import annotations

import math

import torch
from torch import nn


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r: int, alpha: float, dropout: float = 0.0):
        super().__init__()
        if r <= 0:
            raise ValueError("rank must be positive")
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.r, self.alpha, self.scaling = r, alpha, alpha / r
        self.lora_A = nn.Parameter(torch.empty(r, base.in_features))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, r))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))   # same init PEFT uses
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.merged = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        if self.merged:
            return out
        update = (self.dropout(x) @ self.lora_A.t().to(x.dtype)) @ self.lora_B.t().to(x.dtype)
        return out + update * self.scaling

    @torch.no_grad()
    def delta_weight(self) -> torch.Tensor:
        return (self.lora_B @ self.lora_A) * self.scaling

    @torch.no_grad()
    def merge(self) -> None:
        if not self.merged:
            self.base.weight += self.delta_weight().to(self.base.weight.dtype)
            self.merged = True

    @torch.no_grad()
    def unmerge(self) -> None:
        if self.merged:
            self.base.weight -= self.delta_weight().to(self.base.weight.dtype)
            self.merged = False


def inject_lora(model: nn.Module, targets: tuple[str, ...], r: int, alpha: float, dropout: float = 0.0) -> list[str]:
    """Freeze the whole model, then wrap every nn.Linear whose name ends with one of `targets`."""
    for p in model.parameters():
        p.requires_grad_(False)
    wrapped = []
    for name, module in list(model.named_modules()):
        for child_name, child in list(module.named_children()):
            if isinstance(child, nn.Linear) and child_name in targets:
                setattr(module, child_name, LoRALinear(child, r, alpha, dropout))
                wrapped.append(f"{name}.{child_name}" if name else child_name)
    if not wrapped:
        raise ValueError(f"no nn.Linear layers matched {targets}")
    return wrapped


def lora_parameters(model: nn.Module) -> list[nn.Parameter]:
    return [p for n, p in model.named_parameters() if "lora_A" in n or "lora_B" in n]


def merge_all(model: nn.Module) -> None:
    for m in model.modules():
        if isinstance(m, LoRALinear):
            m.merge()


def lora_param_count(in_features: int, out_features: int, r: int) -> int:
    """Trainable parameters LoRA adds to one linear layer."""
    return r * (in_features + out_features)
