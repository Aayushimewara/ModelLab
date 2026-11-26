import pytest
import torch

from modellab.nf4_ref import NF4_CODE, dequantize_nf4, quantization_error, quantize_nf4

PUBLISHED = [-1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453, -0.28444138169288635, -0.18477343022823334,
             -0.09105003625154495, 0.0, 0.07958029955625534, 0.16093020141529922, 0.24611230194568634, 0.33791524171829224,
             0.44070982933044434, 0.5626170039176941, 0.7229568362236023, 1.0]


def test_code_book_is_the_published_nf4_table():
    assert len(NF4_CODE) == 16 and 0.0 in NF4_CODE.tolist()
    assert torch.allclose(NF4_CODE, torch.tensor(PUBLISHED), atol=1e-6)
    assert (NF4_CODE[1:] > NF4_CODE[:-1]).all()


def test_roundtrip_error_and_size():
    torch.manual_seed(0)
    w = torch.randn(256, 128) * 0.02
    q = quantization_error(w)
    assert 0.05 < q["relative_error"] < 0.15
    assert q["bytes_nf4"] < q["bytes_fp32"] / 6


def test_exact_values_survive_quantization():
    """The largest value in a block and zero are code values, so they are reproduced exactly."""
    w = torch.zeros(64)
    w[0], w[1] = 0.5, -0.5
    packed, absmax, shape = quantize_nf4(w)
    back = dequantize_nf4(packed, absmax, shape)
    assert back[0].item() == pytest.approx(0.5) and back[1].item() == pytest.approx(-0.5) and back[2:].abs().max() == 0


def test_bad_shape_rejected():
    with pytest.raises(ValueError):
        quantize_nf4(torch.randn(10, 7))


def test_matches_bitsandbytes_bit_for_bit():
    bnb = pytest.importorskip("bitsandbytes")
    import bitsandbytes.functional as F
    torch.manual_seed(0)
    w = torch.randn(128, 64) * 0.03
    packed_ref, state = F.quantize_4bit(w.clone(), blocksize=64, quant_type="nf4")
    mine, absmax, shape = quantize_nf4(w, 64)
    assert torch.equal(packed_ref.reshape(-1).cpu(), mine), "packed codes differ from bitsandbytes"
    assert torch.allclose(state.absmax.float().cpu(), absmax)
    ref_back = F.dequantize_4bit(packed_ref, state).float().cpu()
    assert torch.allclose(ref_back.reshape(shape), dequantize_nf4(mine, absmax, shape), atol=1e-6)
