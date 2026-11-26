import ast
import json
from pathlib import Path

import numpy as np
import pytest

from modellab import data
from modellab.evaluate import aggregate, breakdown, parse_label, pick_shots, score_one
from modellab.tokenize_task import batches, collate, encode_example

ROOT = Path(__file__).resolve().parent.parent
TASK = ROOT / "data" / "task"


def lab(sig):
    return data.label_from_args(ast.parse(f"def f({sig}): pass").body[0].args)


@pytest.mark.parametrize("sig,expected", [
    ("a, b", "req: a, b | opt: - | var: -"),
    ("a, b=1", "req: a | opt: b | var: -"),
    ("self, /, x, y=2", "req: self, x | opt: y | var: -"),
    ("a, *args, k, m=3, **kw", "req: a, k | opt: m | var: *args, **kw"),
    ("*, a, b=None", "req: a | opt: b | var: -"),
    ("a=(1, 2), b=[3, 4]", "req: - | opt: a, b | var: -"),
    ("**kw", "req: - | opt: - | var: **kw"),
])
def test_labels(sig, expected):
    assert lab(sig) == expected


def test_extract_from_source():
    src = ("def f(a, b=1): pass\n"
           "class C:\n    def m(self, x, *y, **z): pass\n"
           "    async def n(self, q): pass\n"
           "def g(): pass\n"
           "h = lambda a, b: a\n"
           "def big(" + ", ".join(f"a{i}" for i in range(12)) + "): pass\n")
    sigs = [e["sig"] for e in data.extract_from_source(src)]
    assert sigs == ["a, b=1", "self, x, *y, **z", "self, q"] or set(sigs) == {"a, b=1", "self, x, *y, **z", "self, q"}
    assert data.extract_from_source("def broken(:") == []


def test_complex_default_detection():
    assert data.has_complex_default("a, b=(1, 2)") and data.has_complex_default("a, b='x'")
    assert not data.has_complex_default("a, b=1, c=None")


def test_module_split_is_deterministic_and_balanced():
    names = [f"module_{i}" for i in range(2000)]
    s = [data.module_split(n) for n in names]
    assert s == [data.module_split(n) for n in names]
    share = {k: s.count(k) / len(s) for k in ("train", "val", "test")}
    assert 0.65 < share["train"] < 0.75 and 0.05 < share["val"] < 0.11 and 0.18 < share["test"] < 0.26


def test_baselines():
    assert data.parse_baseline("self, path, mode='r', *args") == "req: self, path | opt: mode | var: *args"
    assert data.naive_split_baseline("a, b=(1, 2)") != data.parse_baseline("a, b=(1, 2)")


def test_prompt_formats():
    shots = [{"sig": "a, b=1", "target": "req: a | opt: b | var: -"}]
    text = data.few_shot_prompt(shots, "x")
    assert text.startswith("sig: a, b=1\nout: req: a | opt: b | var: -\n") and text.endswith("sig: x\nout:")


# ---------------- the shipped task files
@pytest.mark.skipif(not (TASK / "test.jsonl").is_file(), reason="task files not built")
class TestShippedTask:
    task = None

    @classmethod
    def setup_class(cls):
        cls.task = data.load_task(TASK)

    def test_sizes(self):
        assert len(self.task["train"]) > 1500 and len(self.task["val"]) > 200 and len(self.task["test"]) > 600

    def test_no_signature_is_shared_between_splits(self):
        sets = {k: {e["sig"] for e in v} for k, v in self.task.items()}
        assert not sets["train"] & sets["val"] and not sets["train"] & sets["test"] and not sets["val"] & sets["test"]

    def test_no_module_is_shared_between_splits(self):
        mods = {k: {e["module"] for e in v} for k, v in self.task.items()}
        assert not mods["train"] & mods["val"] and not mods["train"] & mods["test"] and not mods["val"] & mods["test"]
        assert all(data.module_split(m) == s for s, ms in mods.items() for m in ms)

    def test_every_label_matches_the_reference_parser(self):
        for split in self.task.values():
            for e in split:
                assert data.parse_baseline(e["sig"]) == e["target"]

    def test_naive_heuristic_is_good_but_not_perfect(self):
        em = np.mean([data.naive_split_baseline(e["sig"]) == e["target"] for e in self.task["test"]])
        assert 0.8 < em < 1.0


@pytest.mark.skipif(not (ROOT / "data" / "pretrain" / "train.txt").is_file(), reason="pretraining text not built")
def test_no_held_out_signature_in_a_def_line_of_the_pretraining_text():
    import re
    text = (ROOT / "data" / "pretrain" / "train.txt").read_text()
    task = data.load_task(TASK)
    bad = [e["sig"] for s in ("val", "test") for e in task[s] if re.search(r"def\s+\w+\s*" + re.escape("(" + e["sig"] + ")"), text)]
    assert bad == []


# ---------------- scoring
def test_scoring_and_aggregate():
    g = "req: self, path | opt: mode | var: *args"
    assert score_one(g, g)["em"] == 1.0
    partial = score_one("req: self | opt: mode | var: *args", g)
    # prediction has 3 (category, name) pairs, all correct; the gold label has 4
    assert partial["em"] == 0.0 and partial["valid"] == 1.0 and partial["tp"] == 3 and partial["n_pred"] == 3 and partial["n_gold"] == 4
    assert score_one("nonsense", g)["valid"] == 0.0
    agg = aggregate([score_one(g, g), score_one("req: self | opt: mode | var: *args", g)])
    assert agg["em"] == 0.5 and agg["field_recall"] == pytest.approx(7 / 8) and agg["field_precision"] == 1.0
    assert parse_label("req: - | opt: - | var: -") == set()


def test_breakdown_buckets():
    ex = [{"n_params": 1, "complex": False}, {"n_params": 3, "complex": True}, {"n_params": 6, "complex": False}]
    b = breakdown(ex, [1.0, 0.0, 1.0])
    assert b["1-2 params"]["em"] == 1.0 and b["3-4 params"]["em"] == 0.0 and b["complex defaults"]["em"] == 0.0 and b["simple defaults"]["em"] == 1.0


def test_pick_shots():
    train = [{"sig": "self, path, mode='r'", "target": "t1"}, {"sig": "x, y", "target": "t2"}, {"sig": "key, default=None", "target": "t3"}]
    q = [{"sig": "self, path, mode='r'"}]
    assert pick_shots(train, q, 0) == [[]]
    assert pick_shots(train, q, 1, "retrieved")[0][0]["target"] == "t1"
    rnd = pick_shots(train, q * 3, 2, "random", seed=3)
    assert rnd[0] == rnd[1] == rnd[2] and len(rnd[0]) == 2


# ---------------- task tensors
def test_encoding_masks_the_prompt(tiny_tok):
    ids, labels = encode_example(tiny_tok, {"sig": "a, b", "target": "req: a, b | opt: - | var: -"})
    n_prompt = sum(1 for l in labels if l == -100)
    assert 0 < n_prompt < len(ids) and ids[-1] == tiny_tok.eos_token_id and labels[-1] == tiny_tok.eos_token_id
    assert labels[n_prompt:] == ids[n_prompt:]


def test_collate_pads_right_and_masks(tiny_tok):
    b = collate([([5, 6, 7], [-100, 6, 7]), ([5], [-100])], pad_id=0)
    assert b["input_ids"].tolist() == [[5, 6, 7], [5, 0, 0]] and b["attention_mask"].tolist() == [[1, 1, 1], [1, 0, 0]]
    assert b["labels"].tolist() == [[-100, 6, 7], [-100, -100, -100]]


def test_batches_cycle_through_all_data_and_are_seeded():
    data_ = list(range(10))
    seen = [x for b in batches(data_, 5, 4, seed=1) for x in b]
    assert sorted(seen[:10]) == data_ and sorted(seen[10:]) == data_
    assert [list(b) for b in batches(data_, 5, 3, 7)] == [list(b) for b in batches(data_, 5, 3, 7)]
