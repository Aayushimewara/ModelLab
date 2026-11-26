"""Command line use of a trained adapter.

    python -m modellab.cli predict "self, path, mode='r', *args" --adapter results/adapters/main-mlora-s0
    python -m modellab.cli predict "a, b=1" --adapter results/adapters/main-mqlora-s0 --qlora
    python -m modellab.cli merge --adapter results/adapters/main-mlora-s0 --out merged_model
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from peft import PeftModel

from .data import prompt_text
from .evaluate import generate
from .finetune import RunSpec, load_base
from .tokenizer import load_tokenizer

ROOT = Path(__file__).resolve().parent.parent


def load(adapter: str | None, base: str, qlora: bool):
    spec = RunSpec(run_id="cli", method="qlora" if qlora else "lora", base=base)
    model = load_base(spec)
    if adapter:
        model = PeftModel.from_pretrained(model, adapter)
    return model.eval(), load_tokenizer(Path(base) / "tokenizer")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="modellab")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("predict")
    p.add_argument("signature", nargs="+", help="one or more argument lists, e.g. \"self, path, mode='r'\"")
    p.add_argument("--adapter", default=None, help="adapter directory (omit to use the base model)")
    p.add_argument("--base", default=str(ROOT / "models" / "base"))
    p.add_argument("--qlora", action="store_true", help="load the base model in 4 bit NF4")
    m = sub.add_parser("merge", help="fold a LoRA adapter into the base weights and save a plain model")
    m.add_argument("--adapter", required=True)
    m.add_argument("--base", default=str(ROOT / "models" / "base"))
    m.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    if args.cmd == "predict":
        model, tok = load(args.adapter, args.base, args.qlora)
        for sig, out in zip(args.signature, generate(model, tok, [prompt_text(s) for s in args.signature])):
            print(f"{sig}\n  -> {out}")
    else:
        model, tok = load(args.adapter, args.base, False)
        merged = model.merge_and_unload()
        merged.save_pretrained(args.out)
        tok.save_pretrained(Path(args.out) / "tokenizer")
        print("saved merged model to", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
