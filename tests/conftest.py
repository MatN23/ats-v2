"""Shared pytest fixtures: a tiny dummy config/model/batch for fast tests."""

from __future__ import annotations

import pytest
import torch

from ats.config.schema import (
    ATSConfig,
    DataConfig,
    DataSource,
    ModelConfig,
    TrainingConfig,
)
from ats.model.transformer import ATSTransformer


@pytest.fixture
def dummy_model_config() -> ModelConfig:
    return ModelConfig(
        hidden_size=64,
        num_layers=2,
        num_heads=4,
        num_kv_heads=2,
        intermediate_size=128,
        vocab_size=100,
        max_seq_len=64,
    )


@pytest.fixture
def dummy_ats_config(dummy_model_config: ModelConfig) -> ATSConfig:
    return ATSConfig(
        model=dummy_model_config,
        training=TrainingConfig(max_steps=20, learning_rate=1e-3, warmup_steps=2),
        data=DataConfig(
            sources=[DataSource(path="dummy.jsonl", weight=1.0)], seq_length=16
        ),
    )


@pytest.fixture
def dummy_model(dummy_model_config: ModelConfig) -> ATSTransformer:
    return ATSTransformer(dummy_model_config)


@pytest.fixture
def dummy_batch(dummy_model_config: ModelConfig) -> dict:
    batch, seq_len = 2, 8
    input_ids = torch.randint(0, dummy_model_config.vocab_size, (batch, seq_len))
    return {
        "input_ids": input_ids,
        "labels": input_ids.clone(),
        "attention_mask": torch.ones(batch, seq_len, dtype=torch.long),
    }
