"""Data for the ModelLab study.

Task: read the argument list of a real Python function and say which arguments are
required, which are optional, and which are *args / **kwargs.

    input : self, path, mode='r', *args, buffering=-1, **kw
    output: req: self, path | opt: mode, buffering | var: *args, **kw

All examples come from the Python standard library source on this machine, parsed
with `ast`, so the labels are exact. The split is by top level module (a module goes
wholly to train, val or test, decided by a hash of its name). The pretraining text
for the base model leaves out the val and test modules, so test signatures were never
seen by the base model either.
"""
from __future__ import annotations

import ast
import hashlib
import json
import random
import sysconfig
import warnings
from pathlib import Path

SKIP_DIRS = {"test", "tests", "idlelib", "turtledemo", "site-packages", "__pycache__", "lib-dynload",
             "ensurepip", "lib2to3", "config-3.13-x86_64-linux-gnu", "dist-packages"}
MAX_SIG_CHARS = 160
MAX_PARAMS = 10

PROMPT = "sig: {sig}\nout:"
EOS_TEXT = "\n"


# ------------------------------------------------------------------ splitting
def module_split(name: str) -> str:
    """Deterministic split of a top level module name: ~70% train, 8% val, 22% test."""
    h = int(hashlib.sha1(name.encode()).hexdigest(), 16) % 100
    if h < 8:
        return "val"
    if h < 30:
        return "test"
    return "train"


def stdlib_root() -> Path:
    return Path(sysconfig.get_paths()["stdlib"])


def iter_stdlib_files(root: Path | None = None):
    """Yield (top_level_module_name, path) for every .py file in the stdlib, skipping tests and tools."""
    root = Path(root) if root else stdlib_root()
    for p in sorted(root.rglob("*.py")):
        rel = p.relative_to(root)
        if any(part in SKIP_DIRS for part in rel.parts):
            continue
        top = rel.parts[0][:-3] if len(rel.parts) == 1 else rel.parts[0]
        yield top, p


# ------------------------------------------------------------------ labels
def label_from_args(a: ast.arguments) -> str:
    """Exact label for an argument list."""
    pos = list(a.posonlyargs) + list(a.args)
    n_def = len(a.defaults)
    req: list[str] = []
    opt: list[str] = []
    for i, arg in enumerate(pos):
        (opt if i >= len(pos) - n_def else req).append(arg.arg)
    for arg, d in zip(a.kwonlyargs, a.kw_defaults):
        (req if d is None else opt).append(arg.arg)
    var = []
    if a.vararg:
        var.append("*" + a.vararg.arg)
    if a.kwarg:
        var.append("**" + a.kwarg.arg)
    j = lambda xs: ", ".join(xs) if xs else "-"
    return f"req: {j(req)} | opt: {j(opt)} | var: {j(var)}"


def n_params(a: ast.arguments) -> int:
    return len(a.posonlyargs) + len(a.args) + len(a.kwonlyargs) + bool(a.vararg) + bool(a.kwarg)


def has_complex_default(sig: str) -> bool:
    """True when a default value contains brackets, quotes or a call, which makes comma splitting harder."""
    depth_chars = "([{'\""
    in_default = False
    for ch in sig:
        if ch == "=":
            in_default = True
        elif ch == ",":
            in_default = False
        elif in_default and ch in depth_chars:
            return True
    return False


def extract_from_source(source: str) -> list[dict]:
    """All (signature, label) pairs from one Python source string."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # old source files contain invalid escape sequences in strings
            tree = ast.parse(source)
    except (SyntaxError, ValueError, RecursionError):
        return []
    out = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            a = node.args
            k = n_params(a)
            if k == 0 or k > MAX_PARAMS:
                continue
            try:
                sig = ast.unparse(a)
            except Exception:
                continue
            if "\n" in sig or len(sig) > MAX_SIG_CHARS:
                continue
            out.append({"sig": sig, "target": label_from_args(a), "n_params": k, "complex": has_complex_default(sig)})
    return out


# ------------------------------------------------------------------ building the datasets
def build_task(root: Path | None = None, seed: int = 0, n_train: int = 6000, n_val: int = 400, n_test: int = 800,
               trivial_cap: int = 30) -> dict[str, list[dict]]:
    """Unique signatures per split. A signature string that occurs in more than one split is dropped
    from val and test, so val and test never contain a string the model trained on.
    `trivial_cap` limits how many one-parameter signatures each split keeps, to keep the mix interesting."""
    by_split: dict[str, dict[str, dict]] = {"train": {}, "val": {}, "test": {}}
    for top, path in iter_stdlib_files(root):
        try:
            src = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        split = module_split(top)
        for ex in extract_from_source(src):
            if ex["sig"] not in by_split[split]:
                by_split[split][ex["sig"]] = {**ex, "module": top}
    train_sigs = set(by_split["train"])
    for split in ("val", "test"):
        by_split[split] = {s: e for s, e in by_split[split].items() if s not in train_sigs}
    val_sigs = set(by_split["val"])
    by_split["test"] = {s: e for s, e in by_split["test"].items() if s not in val_sigs}

    rng = random.Random(seed)
    out: dict[str, list[dict]] = {}
    for split, cap in (("train", n_train), ("val", n_val), ("test", n_test)):
        items = sorted(by_split[split].values(), key=lambda e: e["sig"])
        rng.shuffle(items)
        kept, trivial = [], 0
        for e in items:
            if e["n_params"] == 1:
                if trivial >= trivial_cap:
                    continue
                trivial += 1
            kept.append(e)
            if len(kept) >= cap:
                break
        out[split] = kept
    return out


def write_task(splits: dict[str, list[dict]], directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for name, rows in splits.items():
        with (directory / f"{name}.jsonl").open("w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")


def load_task(directory: Path) -> dict[str, list[dict]]:
    return {s: [json.loads(l) for l in (Path(directory) / f"{s}.jsonl").read_text().splitlines() if l]
            for s in ("train", "val", "test")}


def build_pretrain_text(held: set[str], root: Path | None = None, valid_fraction: float = 0.03, seed: int = 0,
                        max_file_bytes: int = 400_000, third_party_bytes: int = 45_000_000) -> tuple[str, str, dict]:
    """Text for pretraining the base model.

    1. every stdlib file from a *train* module (val and test modules are left out);
    2. a seeded random sample of the third party Python code installed next to the stdlib, up to
       `third_party_bytes`. A third party file is dropped if it contains any signature string that is
       one of the val or test examples (`held`), so no pretraining file contains a held out example.

    A small share of whole files is held back as `valid` text from the same distribution, used later to
    measure how much a fine-tuning method damages the base model's general ability (forgetting)."""
    rng = random.Random(seed)
    train_parts, valid_parts = [], []
    stats = {"stdlib_files": 0, "third_party_files": 0, "third_party_dropped_for_leak": 0}

    def add(label: str, text: str) -> None:
        (valid_parts if rng.random() < valid_fraction else train_parts).append(f"# file: {label}\n{text}\n")

    for top, path in iter_stdlib_files(root):
        if module_split(top) != "train":
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        if 0 < len(text) <= max_file_bytes and text.strip():
            stats["stdlib_files"] += 1
            add(f"{top}/{path.name}", text)

    tp_root = (Path(root) if root else stdlib_root()) / "dist-packages"
    if tp_root.is_dir() and third_party_bytes > 0:
        files = [p for p in sorted(tp_root.rglob("*.py")) if "__pycache__" not in p.parts]
        rng.shuffle(files)
        used = 0
        for p in files:
            if used >= third_party_bytes:
                break
            try:
                text = p.read_text(encoding="utf-8")
            except (UnicodeDecodeError, OSError):
                continue
            if not (0 < len(text) <= max_file_bytes) or not text.strip():
                continue
            if held and any(e["sig"] in held for e in extract_from_source(text)):
                stats["third_party_dropped_for_leak"] += 1
                continue
            used += len(text)
            stats["third_party_files"] += 1
            add(f"third_party/{p.name}", text)
    train, valid = "".join(train_parts), "".join(valid_parts)
    stats.update(train_chars=len(train), valid_chars=len(valid), heldout_signatures=len(held))
    return train, valid, stats


# ------------------------------------------------------------------ formatting
def prompt_text(sig: str) -> str:
    return PROMPT.format(sig=sig)


def completion_text(target: str) -> str:
    return " " + target + EOS_TEXT


def few_shot_prompt(shots: list[dict], sig: str) -> str:
    """k worked examples followed by the query, in exactly the fine-tuning format."""
    return "".join(prompt_text(s["sig"]) + completion_text(s["target"]) for s in shots) + prompt_text(sig)


def parse_baseline(sig: str) -> str:
    """Reference solution: parse with ast. It is exact by construction, which shows the task is easy for code.
    The point of the study is how well *fine-tuned language models* learn it, not that it needs a model."""
    tree = ast.parse(f"def f({sig}): pass")
    return label_from_args(tree.body[0].args)


def naive_split_baseline(sig: str) -> str:
    """A quick heuristic that splits on commas. It fails on defaults that contain commas or the bare * marker."""
    req, opt, var = [], [], []
    for part in sig.split(","):
        part = part.strip()
        if not part or part in ("*", "/"):
            continue
        if part.startswith("**") or part.startswith("*"):
            var.append(part.split("=")[0].split(":")[0].strip())
        elif "=" in part:
            opt.append(part.split("=")[0].split(":")[0].strip())
        else:
            req.append(part.split(":")[0].strip())
    j = lambda xs: ", ".join(xs) if xs else "-"
    return f"req: {j(req)} | opt: {j(opt)} | var: {j(var)}"
