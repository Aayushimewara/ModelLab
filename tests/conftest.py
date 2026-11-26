import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from modellab.model import new_model, small_llama_config  # noqa: E402
from modellab.tokenizer import train_tokenizer  # noqa: E402

torch.set_num_threads(2)

CODE = ["def foo(self, a, b=1, *args, **kw):\n    return a + b\n", "class Thing:\n    def run(self, path, mode='r'):\n        pass\n",
        "sig: self, path, mode='r'\nout: req: self, path | opt: mode | var: -\n",
        "for i in range(10):\n    print(i, end='')\n", "import os\nx = os.path.join('a', 'b')\n"] * 40


@pytest.fixture(scope="session")
def tiny_tok(tmp_path_factory):
    d = tmp_path_factory.mktemp("tok")
    return train_tokenizer(CODE, d, vocab_size=300)


def tiny_cfg(vocab=300):
    return small_llama_config(vocab_size=vocab, hidden=64, layers=2, heads=4, inter=128, max_len=128)


@pytest.fixture()
def tiny_model():
    return new_model(tiny_cfg(), seed=0)


EXAMPLES = [
    {"sig": "self, path, mode='r'", "target": "req: self, path | opt: mode | var: -", "n_params": 3, "complex": False, "module": "a"},
    {"sig": "a, b=1, *args, **kw", "target": "req: a | opt: b | var: *args, **kw", "n_params": 4, "complex": False, "module": "a"},
    {"sig": "x", "target": "req: x | opt: - | var: -", "n_params": 1, "complex": False, "module": "b"},
    {"sig": "self, key, default=None", "target": "req: self, key | opt: default | var: -", "n_params": 3, "complex": False, "module": "b"},
] * 6


@pytest.fixture()
def tiny_world(tmp_path, tiny_tok):
    """A complete miniature setup on disk: base model + tokenizer, task files, pretraining text."""
    base = tmp_path / "base"
    new_model(tiny_cfg(), seed=0).save_pretrained(base)
    tiny_tok.save_pretrained(base / "tokenizer")
    task = tmp_path / "task"
    task.mkdir()
    for split in ("train", "val", "test"):
        (task / f"{split}.jsonl").write_text("\n".join(json.dumps(e) for e in EXAMPLES) + "\n")
    pre = tmp_path / "pre"
    pre.mkdir()
    (pre / "valid.txt").write_text("".join(CODE[:5]) * 30)
    return {"base": str(base), "data": str(task), "pretrain_data": str(pre)}
