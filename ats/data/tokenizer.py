"""Tokenizer wrapper. Supports "tiktoken:<encoding_name>" and "hf:<model_id>"
specs. No custom BPE implementation. Special tokens (pad/eos) are always
placed at vocab_size (or the tokenizer's own reserved ids for HF), never at
id 0, since 0 is frequently a real content token."""

from __future__ import annotations

from typing import List, Literal

from ats.config.schema import ConfigError

TruncationStrategy = Literal["left", "right", "middle"]


class Tokenizer:
    def __init__(self, tokenizer_name: str) -> None:
        self.tokenizer_name = tokenizer_name
        if tokenizer_name.startswith("tiktoken:"):
            self._backend = "tiktoken"
            encoding_name = tokenizer_name.split(":", 1)[1]
            try:
                import tiktoken
            except ImportError as exc:
                raise ConfigError(
                    "data.tokenizer_name uses a tiktoken: spec but the tiktoken package "
                    "is not installed. Fix: pip install tiktoken."
                ) from exc
            try:
                self._enc = tiktoken.get_encoding(encoding_name)
            except ValueError as exc:
                raise ConfigError(
                    f"Unknown tiktoken encoding '{encoding_name}'. "
                    f"Fix: use a valid tiktoken encoding name, e.g. 'cl100k_base'."
                ) from exc
            self.vocab_size = self._enc.n_vocab + 1  # +1 reserved slot for pad/eos
            self.eos_token_id = self._enc.n_vocab  # last valid index: vocab_size - 1
            self.pad_token_id = self._enc.n_vocab

        elif tokenizer_name.startswith("hf:"):
            self._backend = "hf"
            model_id = tokenizer_name.split(":", 1)[1]
            try:
                from transformers import AutoTokenizer
            except ImportError as exc:
                raise ConfigError(
                    "data.tokenizer_name uses an hf: spec but the transformers package "
                    "is not installed. Fix: pip install transformers."
                ) from exc
            self._hf_tok = AutoTokenizer.from_pretrained(model_id)
            self.vocab_size = len(self._hf_tok)
            if self._hf_tok.eos_token_id is None:
                raise ConfigError(
                    f"HuggingFace tokenizer '{model_id}' has no eos_token_id set. "
                    f"Fix: choose a tokenizer with an EOS token, or set one explicitly "
                    f"before loading (tokenizer.add_special_tokens({{'eos_token': '<eos>'}}))."
                )
            self.eos_token_id = self._hf_tok.eos_token_id
            self.pad_token_id = (
                self._hf_tok.pad_token_id if self._hf_tok.pad_token_id is not None
                else self._hf_tok.eos_token_id
            )
        else:
            raise ConfigError(
                f"Unrecognized tokenizer_name '{tokenizer_name}'. "
                f"Fix: use 'tiktoken:<encoding>' (e.g. 'tiktoken:cl100k_base') or "
                f"'hf:<model_id>' (e.g. 'hf:meta-llama/Llama-2-7b-hf')."
            )

    def encode(self, text: str) -> List[int]:
        if self._backend == "tiktoken":
            return self._enc.encode(text, allowed_special="all")
        return self._hf_tok.encode(text, add_special_tokens=False)

    def decode(self, token_ids: List[int]) -> str:
        real_ids = [t for t in token_ids if t < self.vocab_size]
        if self._backend == "tiktoken":
            return self._enc.decode(real_ids)
        return self._hf_tok.decode(real_ids, skip_special_tokens=True)

    def truncate(
        self, token_ids: List[int], max_length: int, strategy: TruncationStrategy = "right",
    ) -> List[int]:
        if max_length <= 0:
            raise ConfigError(f"truncate() max_length must be positive, got {max_length}.")
        if len(token_ids) <= max_length:
            return token_ids

        if strategy == "right":
            return token_ids[:max_length]
        if strategy == "left":
            return token_ids[-max_length:]
        if strategy == "middle":
            if max_length < 3:
                raise ConfigError(
                    f"'middle' truncation requires max_length >= 3 to fit a marker token "
                    f"between the head and tail, got max_length={max_length}."
                )
            marker_id = self.eos_token_id
            keep = max_length - 1
            head = keep // 2
            tail = keep - head
            return token_ids[:head] + [marker_id] + token_ids[len(token_ids) - tail:]
        raise ConfigError(
            f"Unknown truncation strategy '{strategy}'. Fix: use 'left', 'right', or 'middle'."
        )
