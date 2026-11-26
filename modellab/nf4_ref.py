"""4 bit NormalFloat (NF4) quantization written from scratch, as a reference for the QLoRA base weights.

NF4 (Dettmers et al., 2023): 16 code values placed at quantiles of a standard normal distribution, scaled to
[-1, 1] with an exact zero. A weight tensor is cut into blocks of 64 values, each block is divided by its own
largest absolute value (the "absmax"), and every value is replaced by the index of the nearest code.
Two 4 bit indices are packed into one byte. Storage is about 4.5 bits per weight with float32 absmax.

bitsandbytes does the same thing with fused kernels. The tests check this code against bitsandbytes
bit for bit, so the quantization used in the QLoRA experiments is known to be this scheme.
"""
from __future__ import annotations

import torch
from scipy.stats import norm


def create_normal_map(offset: float = 0.9677083) -> torch.Tensor:
    """The 16 NF4 code values, derived from normal quantiles (8 positive, 7 negative, and zero)."""
    pos = norm.ppf(torch.linspace(offset, 0.5, 9)[:-1].numpy()).tolist()
    neg = (-norm.ppf(torch.linspace(offset, 0.5, 8)[:-1].numpy())).tolist()
    v = torch.tensor(sorted(pos + [0.0] + neg), dtype=torch.float32)
    return v / v.abs().max()


NF4_CODE = create_normal_map()


def quantize_nf4(w: torch.Tensor, blocksize: int = 64) -> tuple[torch.Tensor, torch.Tensor, torch.Size]:
    """Return (packed uint8 codes, absmax per block, original shape). w.numel() must divide by blocksize."""
    shape = w.shape
    flat = w.detach().float().reshape(-1)
    if flat.numel() % blocksize:
        raise ValueError("number of weights must be a multiple of the block size")
    blocks = flat.reshape(-1, blocksize)
    absmax = blocks.abs().amax(dim=1)
    normed = blocks / absmax.clamp_min(1e-30).unsqueeze(1)
    idx = (normed.reshape(-1, 1) - NF4_CODE.reshape(1, -1)).abs().argmin(dim=1).to(torch.uint8)
    packed = (idx[0::2] << 4) | idx[1::2]          # first value in the high nibble
    return packed, absmax, shape


def dequantize_nf4(packed: torch.Tensor, absmax: torch.Tensor, shape: torch.Size, blocksize: int = 64) -> torch.Tensor:
    hi, lo = (packed >> 4).long(), (packed & 0x0F).long()
    idx = torch.stack([hi, lo], dim=1).reshape(-1)
    vals = NF4_CODE[idx].reshape(-1, blocksize) * absmax.unsqueeze(1)
    return vals.reshape(shape)


def quantization_error(w: torch.Tensor, blocksize: int = 64) -> dict:
    packed, absmax, shape = quantize_nf4(w, blocksize)
    back = dequantize_nf4(packed, absmax, shape, blocksize)
    err = (back - w.float())
    return {"mean_abs_error": err.abs().mean().item(), "relative_error": (err.norm() / w.float().norm()).item(),
            "bytes_fp32": w.numel() * 4, "bytes_nf4": packed.numel() + absmax.numel() * 4}
