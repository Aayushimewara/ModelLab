"""The base model: a small Llama style transformer (RoPE, RMSNorm, SwiGLU) from Hugging Face `transformers`.

Using the stock Llama classes means PEFT (LoRA) and bitsandbytes (4 bit NF4) attach to it exactly as they
do to a real Llama, Mistral or Qwen model, and the same code can be pointed at one of those with `--model`.
"""
from __future__ import annotations

import torch
from transformers import LlamaConfig, LlamaForCausalLM

LINEAR_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
ATTN_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj")
MLP_TARGETS = ("gate_proj", "up_proj", "down_proj")


def small_llama_config(vocab_size: int = 4096, hidden: int = 256, layers: int = 6, heads: int = 8,
                       inter: int = 704, max_len: int = 256) -> LlamaConfig:
    return LlamaConfig(vocab_size=vocab_size, hidden_size=hidden, num_hidden_layers=layers,
                       num_attention_heads=heads, num_key_value_heads=heads, intermediate_size=inter,
                       max_position_embeddings=max_len, tie_word_embeddings=True, rms_norm_eps=1e-5,
                       pad_token_id=0, bos_token_id=1, eos_token_id=2)


def new_model(cfg: LlamaConfig, seed: int = 0) -> LlamaForCausalLM:
    torch.manual_seed(seed)
    return LlamaForCausalLM(cfg)


def count_params(model: torch.nn.Module, trainable_only: bool = False) -> int:
    return sum(p.numel() for p in model.parameters() if (p.requires_grad or not trainable_only))


def model_bytes(model: torch.nn.Module) -> int:
    """Bytes held by parameters and buffers, counting packed 4 bit weights at their real stored size."""
    seen, total = set(), 0
    for t in list(model.parameters()) + list(model.buffers()):
        if id(t) in seen:
            continue
        seen.add(id(t))
        total += t.numel() * t.element_size()
    return total
