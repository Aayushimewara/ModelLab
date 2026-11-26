"""Byte level BPE tokenizer trained on the pretraining text, wrapped as a Hugging Face fast tokenizer."""
from __future__ import annotations

from pathlib import Path

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
from transformers import PreTrainedTokenizerFast

SPECIALS = ["<pad>", "<bos>", "<eos>"]


def train_tokenizer(texts, out_dir: str | Path, vocab_size: int = 4096) -> PreTrainedTokenizerFast:
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=vocab_size, special_tokens=SPECIALS, min_frequency=2,
                                  initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), show_progress=False)
    tok.train_from_iterator(texts, trainer)
    fast = PreTrainedTokenizerFast(tokenizer_object=tok, pad_token="<pad>", bos_token="<bos>", eos_token="<eos>")
    fast.padding_side = "left"
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    fast.save_pretrained(out_dir)
    return fast


def load_tokenizer(path: str | Path) -> PreTrainedTokenizerFast:
    tok = PreTrainedTokenizerFast.from_pretrained(str(path))
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    return tok
