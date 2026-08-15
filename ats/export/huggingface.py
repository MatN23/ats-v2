"""Export a dense (non-MoE, non-MoD) ATSTransformer to a HuggingFace-loadable
LlamaForCausalLM checkpoint. ats-v2's dense architecture (GQA, RoPE, SwiGLU,
RMSNorm) is deliberately Llama-compatible, so this is a real weight-name
remap, not a stub.

MoE and MoD models do not have a standard HuggingFace architecture equivalent
and are NOT exported here; export_to_huggingface raises a clear ConfigError
for them rather than emitting a checkpoint that silently loads wrong.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import torch

from ats.config.schema import ConfigError, ModelConfig
from ats.model.transformer import ATSTransformer
from ats.utils.logging_utils import get_logger

logger = get_logger("ats.export.huggingface")


def _build_hf_config(model_config: ModelConfig) -> dict:
    if not model_config.is_resolved():
        raise ConfigError("Cannot export a model whose ModelConfig is not resolved.")
    hf_config = {
        "architectures": ["LlamaForCausalLM"],
        "model_type": "llama",
        "vocab_size": model_config.vocab_size,
        "hidden_size": model_config.hidden_size,
        "intermediate_size": model_config.intermediate_size,
        "num_hidden_layers": model_config.num_layers,
        "num_attention_heads": model_config.num_heads,
        "num_key_value_heads": model_config.num_kv_heads,
        "max_position_embeddings": model_config.max_seq_len,
        "rms_norm_eps": model_config.rms_norm_eps,
        "rope_theta": model_config.rope_theta,
        "tie_word_embeddings": model_config.tie_word_embeddings,
        "torch_dtype": "bfloat16",
    }
    if model_config.use_swa:
        # Mistral-style export: architecture stays "LlamaForCausalLM"-compatible
        # in every field except this one, which HF's Mistral config also uses.
        hf_config["sliding_window"] = model_config.swa_window_size
    return hf_config


def _remap_state_dict(
    ats_state_dict: dict[str, torch.Tensor],
    num_layers: int,
    tie_word_embeddings: bool,
) -> dict[str, torch.Tensor]:
    """ats.model.transformer naming -> HF LlamaForCausalLM naming."""
    hf_state_dict: dict[str, torch.Tensor] = {}
    hf_state_dict["model.embed_tokens.weight"] = ats_state_dict["embed_tokens.weight"]
    hf_state_dict["model.norm.weight"] = ats_state_dict["final_norm.weight"]
    # When tied, ats.model.transformer.ATSTransformer makes lm_head.weight literally
    # the same tensor storage as embed_tokens.weight (see ATSTransformer.__init__).
    # safetensors.torch.save_file refuses to write two keys that share memory (it
    # would silently duplicate the data on disk and risk them diverging on reload),
    # so follow the standard HF convention instead: omit lm_head.weight entirely
    # when tied and let the loader re-tie it from model.embed_tokens.weight based
    # on config.json's tie_word_embeddings, exactly like real Llama checkpoints do.
    if not tie_word_embeddings:
        hf_state_dict["lm_head.weight"] = ats_state_dict["lm_head.weight"]

    per_layer_map = {
        "input_norm.weight": "input_layernorm.weight",
        "post_attention_norm.weight": "post_attention_layernorm.weight",
        "attention.q_proj.weight": "self_attn.q_proj.weight",
        "attention.k_proj.weight": "self_attn.k_proj.weight",
        "attention.v_proj.weight": "self_attn.v_proj.weight",
        "attention.o_proj.weight": "self_attn.o_proj.weight",
        "ffn.down_proj.weight": "mlp.down_proj.weight",
    }

    for layer_idx in range(num_layers):
        ats_prefix = f"layers.{layer_idx}."
        hf_prefix = f"model.layers.{layer_idx}."

        for ats_suffix, hf_suffix in per_layer_map.items():
            key = ats_prefix + ats_suffix
            if key not in ats_state_dict:
                raise ConfigError(
                    f"Missing expected parameter '{key}' while exporting layer {layer_idx}. "
                    f"This model may use MoE or MoD, which are not HuggingFace-exportable."
                )
            hf_state_dict[hf_prefix + hf_suffix] = ats_state_dict[key]

        # SwiGLU packs gate+up into one Linear; HF Llama keeps them separate.
        gate_up_key = ats_prefix + "ffn.gate_up_proj.weight"
        if gate_up_key not in ats_state_dict:
            raise ConfigError(
                f"Missing expected parameter '{gate_up_key}' while exporting layer {layer_idx}. "
                f"This model may use MoE, which is not HuggingFace-exportable."
            )
        gate_up = ats_state_dict[gate_up_key]
        intermediate_size = gate_up.shape[0] // 2
        hf_state_dict[hf_prefix + "mlp.gate_proj.weight"] = gate_up[
            :intermediate_size
        ].clone()
        hf_state_dict[hf_prefix + "mlp.up_proj.weight"] = gate_up[
            intermediate_size:
        ].clone()

    return hf_state_dict


def _build_model_card(model_config: ModelConfig) -> str:
    attention_kind = (
        "MLA" if model_config.use_mla else ("SWA" if model_config.use_swa else "GQA")
    )
    ffn_kind = "MoE" if model_config.use_moe else "SwiGLU (dense)"
    tags = ["ats-v2", model_config.name, "moe" if model_config.use_moe else "dense"]
    if model_config.use_swa:
        tags.append("sliding-window-attention")
    if model_config.use_mod:
        tags.append("mixture-of-depths")

    tag_lines = "\n".join(f"  - {tag}" for tag in tags)
    lines = [
        "---",
        "license: apache-2.0",
        "tags:",
        tag_lines,
        "---",
        "",
        f"# {model_config.name}",
        "",
        "Trained with [ats-v2](https://github.com/anthropics/ats-v2) using:",
        "",
        f"- Architecture: {model_config.name}",
        f"- Hidden size: {model_config.hidden_size}",
        f"- Layers: {model_config.num_layers}",
        f"- Attention heads: {model_config.num_heads} (KV heads: {model_config.num_kv_heads})",
        f"- Attention mechanism: {attention_kind}",
        f"- FFN: {ffn_kind}",
        f"- Vocab size: {model_config.vocab_size}",
        f"- Max sequence length: {model_config.max_seq_len}",
        f"- Tied embeddings: {model_config.tie_word_embeddings}",
    ]
    if model_config.use_swa:
        lines.append(
            f"- Sliding window: {model_config.swa_window_size} tokens "
            f"(full attention every {model_config.swa_full_attention_interval} layers)"
        )
    if model_config.use_moe:
        lines.append(
            f"- Experts: {model_config.num_experts}, top-{model_config.moe_top_k} routing"
        )
    lines.append("")
    lines.append(
        "This checkpoint was exported automatically by `ats.export.huggingface."
        "export_to_huggingface`; the fields above are read directly from the "
        "training config, not hand-written."
    )
    return "\n".join(lines) + "\n"


def export_to_huggingface(
    model: ATSTransformer,
    model_config: ModelConfig,
    output_dir: str,
    tokenizer_dir: str | None = None,
) -> Path:
    if model_config.use_mla:
        raise ConfigError(
            "MLA models cannot be exported to LlamaForCausalLM format because MLA uses "
            "a different attention mechanism. Export to a custom config or disable MLA."
        )
    if model_config.use_mamba:
        raise ConfigError(
            "Mamba/SSM models cannot be exported to LlamaForCausalLM format because "
            "Mamba blocks have no attention mechanism at all, let alone a Llama-compatible "
            "one. Fix: export a dense/SWA checkpoint, or disable use_mamba."
        )
    if model_config.model_type == "diffusion":
        raise ConfigError(
            "Diffusion language models cannot be exported to LlamaForCausalLM format: "
            "they have no autoregressive lm_head/logits path at all. "
            "Fix: export an autoregressive (model_type='autoregressive') checkpoint."
        )
    if model_config.use_moe or model_config.use_mod:
        raise ConfigError(
            "export_to_huggingface only supports dense models (model.use_moe=False and "
            "model.use_mod=False). MoE and Mixture-of-Depths have no standard HuggingFace "
            "architecture to export into. Fix: export a dense checkpoint, or write a custom "
            "HF `trust_remote_code` modeling file for MoE/MoD (not provided by ats-v2)."
        )

    try:
        from safetensors.torch import save_file
    except ImportError as exc:
        raise ConfigError(
            "Exporting to HuggingFace format requires the safetensors package. "
            "Fix: pip install safetensors."
        ) from exc

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    ats_state_dict = {
        k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()
    }
    # model is an already-constructed ATSTransformer, which requires a
    # resolved ModelConfig at construction time, so model_config.num_layers
    # is guaranteed not None here -- mypy can't see across that earlier
    # construction, so narrow again explicitly.
    assert model_config.num_layers is not None
    hf_state_dict = _remap_state_dict(
        ats_state_dict, model_config.num_layers, model_config.tie_word_embeddings
    )

    save_file(hf_state_dict, str(out_path / "model.safetensors"))

    hf_config = _build_hf_config(model_config)
    with open(out_path / "config.json", "w", encoding="utf-8") as f:
        json.dump(hf_config, f, indent=2)

    if tokenizer_dir is not None:
        tokenizer_src = Path(tokenizer_dir)
        if not tokenizer_src.exists():
            raise ConfigError(f"tokenizer_dir does not exist: {tokenizer_src}.")
        for item in tokenizer_src.iterdir():
            if item.is_file():
                shutil.copy2(item, out_path / item.name)

    model_card = _build_model_card(model_config)
    with open(out_path / "README.md", "w", encoding="utf-8") as f:
        f.write(model_card)

    logger.info("Exported HuggingFace checkpoint to %s", out_path)
    return out_path
