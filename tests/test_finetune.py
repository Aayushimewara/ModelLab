import json

import numpy as np
import pytest
import torch
from peft import PeftModel

from modellab.finetune import RunSpec, apply_method, load_base, logical_param_counts, lr_at, run, train
from modellab.model import LINEAR_TARGETS, count_params
from modellab.tokenizer import load_tokenizer
from conftest import EXAMPLES


def spec_for(world, **kw):
    base = dict(run_id="t", method="lora", rank=4, lr=3e-3, epochs=3, min_steps=30, bs=8, **world)
    base.update(kw)
    return RunSpec(**base)


def test_lr_schedule_shape():
    assert lr_at(0, 100, 1.0) < lr_at(5, 100, 1.0)
    assert lr_at(6, 100, 1.0) == pytest.approx(1.0, abs=0.01)
    assert lr_at(99, 100, 1.0) == pytest.approx(0.1, abs=0.02)


@pytest.mark.parametrize("method", ["full", "lora", "qlora"])
def test_each_method_learns_the_tiny_task(tiny_world, tiny_tok, method):
    # adapters on a frozen random base learn more slowly than full training, so they get more steps
    spec = spec_for(tiny_world, method=method, lr=3e-3 if method == "full" else 1e-2, min_steps=60 if method == "full" else 100)
    model = apply_method(load_base(spec), spec)
    curve = train(model, tiny_tok, EXAMPLES, spec)["loss_curve"]
    # a rank-4 adapter on a frozen *random* base has little capacity (it plateaus), so it only needs to clearly improve
    needed = 0.8 if method == "full" else 0.92
    assert curve[-1] < curve[0] * needed, f"{method}: loss did not fall enough {curve}"
    assert all(b <= a + 0.3 for a, b in zip(curve, curve[1:])), f"{method}: loss curve is not roughly decreasing {curve}"
    counts = logical_param_counts(model)
    if method == "full":
        assert counts["trainable_params"] == counts["total_params"]
    else:
        assert 0 < counts["trainable_params"] < counts["total_params"] * 0.5


def test_lora_param_count_matches_formula(tiny_world):
    spec = spec_for(tiny_world, rank=4)
    model = apply_method(load_base(spec), spec)
    cfg = model.config
    h, i = cfg.hidden_size, cfg.intermediate_size
    expect = cfg.num_hidden_layers * 4 * (4 * (h + h) + 2 * (h + i) + (i + h))
    assert logical_param_counts(model)["trainable_params"] == expect


def test_qlora_base_is_4bit_and_smaller(tiny_world):
    import bitsandbytes as bnb
    fp = load_base(spec_for(tiny_world, method="lora"))
    q = load_base(spec_for(tiny_world, method="qlora"))
    n4 = sum(1 for m in q.modules() if isinstance(m, bnb.nn.Linear4bit))
    assert n4 == 7 * q.config.num_hidden_layers
    assert logical_param_counts(q)["weight_bytes"] < logical_param_counts(fp)["weight_bytes"] * 0.5
    assert logical_param_counts(q)["total_params"] == logical_param_counts(fp)["total_params"]  # same number of weights, fewer bytes


def test_adapter_roundtrip_and_merge(tiny_world, tiny_tok, tmp_path):
    spec = spec_for(tiny_world, min_steps=20)
    model = apply_method(load_base(spec), spec)
    train(model, tiny_tok, EXAMPLES, spec)
    model.eval()
    x = torch.randint(0, 300, (2, 12))
    with torch.no_grad():
        ref = model(input_ids=x).logits
    model.save_pretrained(tmp_path / "ad")
    fresh = PeftModel.from_pretrained(load_base(spec), tmp_path / "ad").eval()
    with torch.no_grad():
        assert torch.allclose(fresh(input_ids=x).logits, ref, atol=1e-5)
        merged = fresh.merge_and_unload().eval()
        assert torch.allclose(merged(input_ids=x).logits, ref, atol=1e-4)
    assert count_params(merged) == count_params(load_base(spec))   # merging adds no parameters


def test_run_end_to_end_with_a_tiny_world(tiny_world):
    res = run(spec_for(tiny_world, min_steps=40, bench=True))
    for key in ("val", "test", "training", "general_loss_before", "general_loss_after", "forgetting_nats", "peak_rss_mb", "inference"):
        assert key in res, key
    assert len(res["test"]["per_example_em"]) == len(EXAMPLES) and 0.0 <= res["test"]["em"] <= 1.0
    assert set(res["inference"]) == {"adapter_unmerged", "lora_merged"}
    json.dumps(res)   # must be serialisable


def test_baseline_run_needs_no_training(tiny_world):
    res = run(spec_for(tiny_world, method="none", shots=1, eval_val=False))
    assert "training" not in res and res["trainable_params"] == 0 or res["trainable_params"] > 0
    assert res["forgetting_nats"] == 0.0


# ---------------------------------------------------------------------------------------------
# Regression tests for a real bug found while building this project.
# On CPUs with AVX512-BF16, bitsandbytes switches a 4 bit layer to an inference-only packed format the first
# time it runs in eval mode on an input with no gradient. After that the layer has no backward pass, training
# still runs without any error, and the adapters just learn badly. The tests below pin the correct behaviour.
# ---------------------------------------------------------------------------------------------
def _dequantized_twin(tiny_world, quantized):
    import bitsandbytes as bnb
    plain = load_base(spec_for(tiny_world, method="lora"))
    q_modules = dict(quantized.named_modules())
    for name, mod in plain.named_modules():
        q = q_modules.get(name)
        if isinstance(mod, torch.nn.Linear) and isinstance(q, bnb.nn.Linear4bit):
            mod.weight.data.copy_(bnb.functional.dequantize_4bit(q.weight.data, q.weight.quant_state).float())
    return plain


def _adapter_grads(model, spec):
    peft_model = apply_method(model, spec)
    gen = torch.Generator().manual_seed(1)
    for n, p in peft_model.named_parameters():
        if "lora_B" in n:
            p.data = torch.randn(p.shape, generator=gen) * 0.05
    peft_model.train()
    gen = torch.Generator().manual_seed(2)
    ids = torch.randint(3, 290, (4, 24), generator=gen)
    loss = peft_model(input_ids=ids, labels=ids).loss
    loss.backward()
    return {n: p.grad.clone() for n, p in peft_model.named_parameters() if p.grad is not None}


def test_qlora_gradients_match_a_dequantized_reference(tiny_world):
    """Gradients that reach every adapter, including the ones in the lowest layers, must match a
    full-precision model that holds the same (dequantized) weights."""
    spec = spec_for(tiny_world, method="qlora", dropout=0.0)
    quantized = load_base(spec)
    twin = _dequantized_twin(tiny_world, quantized)
    g_q = _adapter_grads(quantized, spec)
    g_ref = _adapter_grads(twin, spec_for(tiny_world, method="lora", dropout=0.0))
    assert g_q.keys() == g_ref.keys() and len(g_q) > 0
    for name in g_q:
        a, b = g_q[name].flatten(), g_ref[name].flatten()
        cos = torch.nn.functional.cosine_similarity(a, b, dim=0).item()
        ratio = (a.norm() / b.norm()).item()
        assert cos > 0.95 and 0.8 < ratio < 1.25, f"{name}: gradient differs from reference (cos {cos:.3f}, norm ratio {ratio:.3f})"


def test_guard_refuses_a_quantized_model_that_was_switched_to_the_inference_format(tiny_world):
    from modellab.finetune import assert_backward_ready
    q = load_base(spec_for(tiny_world, method="qlora"))
    assert_backward_ready(q)                      # a fresh model passes
    q.eval()
    with torch.no_grad():
        q(input_ids=torch.randint(3, 290, (2, 8)))
    switched = any(getattr(getattr(m, "weight", None), "quant_state", None) is not None
                   and getattr(m.weight.quant_state, "packing_format_for_cpu", False) for m in q.modules())
    if not switched:
        pytest.skip("this CPU does not use bitsandbytes' inference-only packed format, so the failure cannot occur here")
    with pytest.raises(RuntimeError, match="inference-only"):
        assert_backward_ready(q)


def test_qlora_run_never_trains_on_a_switched_model(tiny_world):
    # run() measures the base on a throwaway copy; if it did not, train() would raise on AVX512-BF16 CPUs
    res = run(spec_for(tiny_world, method="qlora", min_steps=30, eval_test=False))
    assert res["training"]["steps"] == 30 and res["general_loss_before"] > 0
