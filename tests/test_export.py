"""Tests for ats.export.huggingface: dense export produces real files with
correct shapes, MoE/MoD/MLA models are refused with a clear error."""

from __future__ import annotations

import json

import pytest
import torch

from ats.config.schema import ConfigError, ModelConfig
from ats.export.huggingface import export_to_huggingface
from ats.model.transformer import ATSTransformer

try:
    import safetensors  # noqa: F401
    _SAFETENSORS_AVAILABLE = True
except ImportError:
    _SAFETENSORS_AVAILABLE = False


def _dense_config(**overrides) -> ModelConfig:
    base = dict(
        hidden_size=32, num_layers=2, num_heads=4, num_kv_heads=2,
        intermediate_size=64, vocab_size=50, max_seq_len=32,
    )
    base.update(overrides)
    return ModelConfig(**base)


@pytest.mark.skipif(not _SAFETENSORS_AVAILABLE, reason="safetensors not installed in this environment")
def test_dense_export_creates_expected_files(tmp_path):
    config = _dense_config()
    model = ATSTransformer(config)
    out_dir = export_to_huggingface(model, config, str(tmp_path / "exported"))

    assert (out_dir / "model.safetensors").exists()
    assert (out_dir / "config.json").exists()

    with open(out_dir / "config.json") as f:
        hf_config = json.load(f)
    assert hf_config["hidden_size"] == 32
    assert hf_config["num_hidden_layers"] == 2
    assert hf_config["architectures"] == ["LlamaForCausalLM"]


@pytest.mark.skipif(not _SAFETENSORS_AVAILABLE, reason="safetensors not installed in this environment")
def test_swa_export_includes_sliding_window(tmp_path):
    config = _dense_config(use_swa=True, swa_window_size=128)
    model = ATSTransformer(config)
    out_dir = export_to_huggingface(model, config, str(tmp_path / "exported_swa"))
    with open(out_dir / "config.json") as f:
        hf_config = json.load(f)
    assert hf_config["sliding_window"] == 128


def test_moe_export_raises(tmp_path):
    config = _dense_config(use_moe=True, num_experts=4, moe_top_k=2)
    model = ATSTransformer(config)
    with pytest.raises(ConfigError):
        export_to_huggingface(model, config, str(tmp_path / "exported_moe"))


def test_mod_export_raises(tmp_path):
    config = _dense_config(use_mod=True)
    model = ATSTransformer(config)
    with pytest.raises(ConfigError):
        export_to_huggingface(model, config, str(tmp_path / "exported_mod"))


def test_mla_export_raises(tmp_path):
    config = _dense_config(use_mla=True, mla_latent_dim=8)
    model = ATSTransformer(config)
    with pytest.raises(ConfigError):
        export_to_huggingface(model, config, str(tmp_path / "exported_mla"))


def test_mla_export_error_message_is_specific(tmp_path):
    config = _dense_config(use_mla=True, mla_latent_dim=8)
    model = ATSTransformer(config)
    with pytest.raises(ConfigError, match="MLA"):
        export_to_huggingface(model, config, str(tmp_path / "exported_mla2"))


def test_mamba_export_raises(tmp_path):
    config = _dense_config(use_mamba=True, mamba_every_n_layers=1)
    model = ATSTransformer(config)
    with pytest.raises(ConfigError):
        export_to_huggingface(model, config, str(tmp_path / "exported_mamba"))


def test_diffusion_export_raises(tmp_path):
    config = _dense_config(model_type="diffusion")
    model = ATSTransformer(config)
    with pytest.raises(ConfigError):
        export_to_huggingface(model, config, str(tmp_path / "exported_diffusion"))
