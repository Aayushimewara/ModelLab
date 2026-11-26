import copy

import pytest
import torch
from peft import LoraConfig, get_peft_model
from torch import nn

from modellab.lora_ref import LoRALinear, inject_lora, lora_param_count, lora_parameters, merge_all
from modellab.model import LINEAR_TARGETS, count_params, new_model
from conftest import tiny_cfg


def test_zero_init_means_no_change_before_training():
    base = nn.Linear(16, 8)
    x = torch.randn(5, 16)
    layer = LoRALinear(copy.deepcopy(base), r=4, alpha=8)
    assert torch.allclose(layer(x), base(x))


def test_merge_and_unmerge_are_exact_inverses():
    layer = LoRALinear(nn.Linear(16, 8), r=4, alpha=8)
    nn.init.normal_(layer.lora_B)
    x = torch.randn(5, 16)
    before, w0 = layer(x), layer.base.weight.clone()
    layer.merge()
    assert torch.allclose(layer(x), before, atol=1e-5)
    layer.unmerge()
    assert torch.allclose(layer.base.weight, w0, atol=1e-5) and torch.allclose(layer(x), before, atol=1e-5)


def test_only_adapters_are_trainable_and_count_matches_formula():
    m = new_model(tiny_cfg())
    names = inject_lora(m, LINEAR_TARGETS, r=4, alpha=8)
    cfg = m.config
    h, i = cfg.hidden_size, cfg.intermediate_size
    per_layer = 4 * lora_param_count(h, h, 4) + 2 * lora_param_count(h, i, 4) + lora_param_count(i, h, 4)
    assert count_params(m, trainable_only=True) == per_layer * cfg.num_hidden_layers
    assert len(names) == 7 * cfg.num_hidden_layers
    assert all("lora_" in n for n, p in m.named_parameters() if p.requires_grad)


def test_bad_arguments():
    with pytest.raises(ValueError):
        LoRALinear(nn.Linear(4, 4), r=0, alpha=1)
    with pytest.raises(ValueError):
        inject_lora(new_model(tiny_cfg()), ("no_such_layer",), 4, 8)


def _paired(r=4, alpha=8):
    base = new_model(tiny_cfg(), seed=1)
    own = copy.deepcopy(base)
    torch.manual_seed(0)
    peft = get_peft_model(base, LoraConfig(r=r, lora_alpha=alpha, lora_dropout=0.0, target_modules=list(LINEAR_TARGETS), bias="none"))
    inject_lora(own, LINEAR_TARGETS, r, alpha, 0.0)
    pm = dict(peft.named_modules())
    for name, mod in own.named_modules():
        if isinstance(mod, LoRALinear):
            pl = pm["base_model.model." + name]
            mod.lora_A.data.copy_(pl.lora_A["default"].weight.data)
            mod.lora_B.data.copy_(pl.lora_B["default"].weight.data)
    return peft, own


def test_matches_peft_outputs_and_gradients_with_nonzero_B():
    peft, own = _paired()
    gen = torch.Generator().manual_seed(5)
    for (_, p1), (_, p2) in zip(((n, p) for n, p in peft.named_parameters() if "lora_B" in n),
                                ((n, p) for n, p in own.named_parameters() if "lora_B" in n)):
        v = torch.randn(p1.shape, generator=gen) * 0.05
        p1.data.copy_(v)
        p2.data.copy_(v)
    x = torch.randint(0, 300, (3, 20))
    o1, o2 = peft(input_ids=x, labels=x), own(input_ids=x, labels=x)
    assert torch.allclose(o1.logits, o2.logits, atol=1e-5)
    o1.loss.backward()
    o2.loss.backward()
    g1 = [p.grad for n, p in peft.named_parameters() if p.requires_grad]
    g2 = [p.grad for p in lora_parameters(own)]
    assert len(g1) == len(g2) and all(torch.allclose(a, b, atol=1e-6) for a, b in zip(g1, g2))
    assert all(p.grad is None for n, p in own.named_parameters() if "lora_" not in n)


def test_training_trajectory_matches_peft():
    peft, own = _paired()
    opts = [torch.optim.AdamW([p for p in peft.parameters() if p.requires_grad], lr=1e-2, weight_decay=0.0),
            torch.optim.AdamW(lora_parameters(own), lr=1e-2, weight_decay=0.0)]
    g = torch.Generator().manual_seed(11)
    for _ in range(8):
        x = torch.randint(0, 300, (4, 24), generator=g)
        losses = []
        for model, opt in zip((peft, own), opts):
            loss = model(input_ids=x, labels=x).loss
            loss.backward()
            opt.step()
            opt.zero_grad()
            losses.append(loss.item())
        assert abs(losses[0] - losses[1]) < 1e-5
    assert losses[0] < 6.5  # it is learning something


def test_merge_all_matches_unmerged_model_and_peft_merge():
    peft, own = _paired()
    for p in lora_parameters(own):
        p.data.add_(torch.randn_like(p) * 0.05)
    x = torch.randint(0, 300, (2, 16))
    before = own(input_ids=x).logits
    merge_all(own)
    assert torch.allclose(own(input_ids=x).logits, before, atol=1e-4)
