"""Chat REPL that works with any ats-v2 checkpoint (dense, SWA, MLA, MoE, MoD,
Mamba, MTP), auto-detecting the architecture from the checkpoint's own saved
config.yaml.

Not stock HuggingFace transformers.generate(): MoE, MoD, MLA, and Mamba
models raise a ConfigError on HF export (see ats/export/huggingface.py), so
no single stock loader can cover every ats-v2 architecture. This uses
ats-v2's own ATSTransformer class directly instead, which already implements
every one of those architectures internally as one class -- the closest
thing to "stock" that actually covers all of them.

Diffusion models (model_type: diffusion) are NOT supported here: their
sample() method (ats/model/diffusion.py) generates unconditionally from pure
noise with no way to condition on a prompt, so there is no chat-style
interaction possible with them in this codebase as written.

These models have no instruction-tuning or chat-format training (ats-v2's
finetune/align commands are unimplemented placeholders), so this continues
raw text the same way a base language model does -- it does not reliably
follow the ### User / ### Assistant structure below.

Usage:
    python chat.py --checkpoint checkpoints/125m/step_1000
    python chat.py --checkpoint checkpoints/7b/step_5000 --config configs/7b.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path

import tiktoken
import torch

from ats.config.loader import load_config
from ats.config.schema import ConfigError
from ats.model.transformer import ATSTransformer
from ats.training.checkpoint import load_model_weights_safetensors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Checkpoint directory, e.g. checkpoints/125m/step_1000",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Config used for training. If omitted, looks for "
        "<checkpoint>/config.yaml (same convention as ats-export).",
    )
    parser.add_argument("--max-new-tokens", type=int, default=150)
    parser.add_argument(
        "--temperature", type=float, default=0.8, help="0 = greedy/deterministic."
    )
    return parser.parse_args()


def resolve_config_path(checkpoint: str, explicit_config: str | None) -> Path:
    if explicit_config is not None:
        return Path(explicit_config)
    candidate = Path(checkpoint) / "config.yaml"
    if candidate.exists():
        return candidate
    raise ConfigError(
        f"No --config was given and no config.yaml was found at {candidate}. "
        f"Fix: pass --config explicitly, pointing at the YAML this checkpoint "
        f"was trained with."
    )


def load_model(config_path: Path, checkpoint_dir: str, device: torch.device) -> ATSTransformer:
    config = load_config(config_path)

    if config.model.model_type == "diffusion":
        raise ConfigError(
            "This checkpoint is model_type=diffusion. Its sample() method "
            "(ats/model/diffusion.py) generates unconditionally from pure "
            "noise with no prompt-conditioning path -- there is no chat "
            "interaction possible with a diffusion checkpoint in this "
            "codebase. Use an autoregressive checkpoint instead."
        )

    model = ATSTransformer(config.model)
    weights = load_model_weights_safetensors(checkpoint_dir)
    missing, unexpected = model.load_state_dict(weights, strict=False)
    if missing or unexpected:
        print(f"[warning] missing keys: {missing}")
        print(f"[warning] unexpected keys: {unexpected}")
    model.to(device)
    model.eval()

    active = [
        name
        for name in ("use_swa", "use_mla", "use_mamba", "use_moe", "use_mod", "use_mtp")
        if getattr(config.model, name)
    ]
    print(f"Detected architecture: {active if active else ['dense']}")
    return model


@torch.no_grad()
def generate(
    model: ATSTransformer,
    tokenizer: tiktoken.Encoding,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    device: torch.device,
) -> str:
    input_ids = torch.tensor(
        [tokenizer.encode(prompt, allowed_special="all")], dtype=torch.long, device=device
    )
    past_key_values = None
    generated: list[int] = []
    eos_token_id = tokenizer.n_vocab

    for _ in range(max_new_tokens):
        output = model(
            input_ids if past_key_values is None else input_ids[:, -1:],
            past_key_values=past_key_values,
            use_cache=True,
        )
        past_key_values = output.past_key_values
        next_token_logits = output.logits[0, -1, :]

        if temperature > 0:
            probs = torch.softmax(next_token_logits / temperature, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
        else:
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

        token_id = int(next_token.item())
        if token_id == eos_token_id:
            break
        generated.append(token_id)
        input_ids = torch.cat([input_ids, next_token.view(1, 1)], dim=1)

    return tokenizer.decode(generated)


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config_path = resolve_config_path(args.checkpoint, args.config)
    print(f"Loading {args.checkpoint} (config: {config_path}) onto {device}...")

    model = load_model(config_path, args.checkpoint, device)
    tokenizer = tiktoken.get_encoding("cl100k_base")

    print("Model loaded. Type a message and press Enter. Ctrl+C to quit.")
    print(
        "(No instruction-tuning has been applied to this model -- it will "
        "often ignore the chat format and just continue text. This is "
        "expected, not a bug in this script.)\n"
    )

    history = ""
    while True:
        try:
            user_msg = input("You: ")
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break
        if not user_msg.strip():
            continue

        history += f"### User:\n{user_msg}\n### Assistant:\n"
        reply = generate(model, tokenizer, history, args.max_new_tokens, args.temperature, device)
        print(f"Assistant: {reply}\n")
        history += reply + "\n"


if __name__ == "__main__":
    main()