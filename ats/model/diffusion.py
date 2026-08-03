"""A minimal continuous diffusion language model.

Noise is added in embedding space (following Diffusion-LM style approaches):
token embeddings are treated as the continuous signal that gets noised
according to a cosine schedule, the backbone transformer is trained to
predict the noise (not the tokens) added at a randomly sampled timestep, and
the training loss is MSE between predicted and true noise — never
cross-entropy, since there is no "next token" being predicted here. Sampling
uses DDIM (deterministic reverse process) starting from Gaussian noise in
embedding space, followed by a nearest-embedding lookup to map the final
continuous sample back to discrete tokens.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def cosine_alpha_bar(t: torch.Tensor, s: float = 0.008) -> torch.Tensor:
    """Cosine noise schedule (Nichol & Dhariwal, 2021): returns alpha_bar(t)
    for t in [0, 1], the cumulative product of (1 - beta) up to time t."""
    return torch.cos(((t + s) / (1 + s)) * (math.pi / 2)) ** 2


@dataclass
class DiffusionOutput:
    predicted_noise: torch.Tensor
    loss: torch.Tensor


class DiffusionLM(nn.Module):
    """Wraps an existing ATSTransformer-style backbone (used here purely as
    a sequence-to-sequence noise predictor operating on embeddings, so its
    `lm_head` is not used in the diffusion forward path) with the noising
    process, MSE training objective, and DDIM sampler."""

    def __init__(self, backbone: nn.Module, hidden_size: int, num_timesteps: int = 1000) -> None:
        super().__init__()
        if num_timesteps < 2:
            raise ValueError(f"DiffusionLM requires num_timesteps >= 2, got {num_timesteps}.")
        self.backbone = backbone
        self.hidden_size = hidden_size
        self.num_timesteps = num_timesteps
        # Predicts the noise added to the embedding, conditioned implicitly
        # through the backbone's hidden states plus an explicit timestep
        # embedding added at the input.
        self.time_embed = nn.Sequential(
            nn.Linear(hidden_size, hidden_size), nn.SiLU(), nn.Linear(hidden_size, hidden_size),
        )
        self.noise_pred_head = nn.Linear(hidden_size, hidden_size, bias=False)

    def _sinusoidal_timestep_embedding(self, timesteps: torch.Tensor) -> torch.Tensor:
        half = self.hidden_size // 2
        freqs = torch.exp(
            -math.log(10000.0) * torch.arange(half, device=timesteps.device, dtype=torch.float32) / half
        )
        args = timesteps.float().unsqueeze(-1) * freqs.unsqueeze(0)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if emb.shape[-1] < self.hidden_size:
            emb = F.pad(emb, (0, self.hidden_size - emb.shape[-1]))
        return emb

    def add_noise(
        self, clean_embeddings: torch.Tensor, timesteps: torch.Tensor,
    ) -> "tuple[torch.Tensor, torch.Tensor]":
        """Forward diffusion: x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * noise."""
        t_normalized = timesteps.float() / self.num_timesteps
        alpha_bar = cosine_alpha_bar(t_normalized).clamp(min=1e-5, max=1.0)
        alpha_bar = alpha_bar.view(-1, *([1] * (clean_embeddings.dim() - 1)))
        noise = torch.randn_like(clean_embeddings)
        noisy = torch.sqrt(alpha_bar) * clean_embeddings + torch.sqrt(1 - alpha_bar) * noise
        return noisy, noise

    def forward(
        self, input_ids: torch.Tensor, embed_tokens: nn.Embedding, timesteps: Optional[torch.Tensor] = None,
    ) -> DiffusionOutput:
        if input_ids.dim() != 2:
            raise ValueError(
                f"DiffusionLM expected input_ids of shape [batch, seq_len], got {tuple(input_ids.shape)}."
            )
        batch = input_ids.shape[0]
        if timesteps is None:
            timesteps = torch.randint(0, self.num_timesteps, (batch,), device=input_ids.device)

        clean_embeddings = embed_tokens(input_ids)
        noisy_embeddings, true_noise = self.add_noise(clean_embeddings, timesteps)

        time_emb = self.time_embed(self._sinusoidal_timestep_embedding(timesteps))
        model_input = noisy_embeddings + time_emb.unsqueeze(1)

        backbone_output = self.backbone.forward_hidden(inputs_embeds=model_input)
        predicted_noise = self.noise_pred_head(backbone_output)
        loss = F.mse_loss(predicted_noise, true_noise)
        return DiffusionOutput(predicted_noise=predicted_noise, loss=loss)

    @torch.no_grad()
    def sample(
        self, embed_tokens: nn.Embedding, batch_size: int, seq_len: int,
        num_inference_steps: int = 50, device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """DDIM sampling: deterministic reverse process from Gaussian noise in
        embedding space, then nearest-neighbor lookup against the embedding
        table to recover discrete token ids."""
        if num_inference_steps < 1:
            raise ValueError(f"num_inference_steps must be >= 1, got {num_inference_steps}.")
        device = device or next(self.parameters()).device
        hidden_size = embed_tokens.embedding_dim

        x_t = torch.randn(batch_size, seq_len, hidden_size, device=device)
        step_indices = torch.linspace(self.num_timesteps - 1, 0, num_inference_steps, device=device).long()

        for i, t in enumerate(step_indices):
            t_batch = t.expand(batch_size)
            t_normalized = t_batch.float() / self.num_timesteps
            alpha_bar_t = cosine_alpha_bar(t_normalized).clamp(min=1e-5, max=1.0).view(-1, 1, 1)

            time_emb = self.time_embed(self._sinusoidal_timestep_embedding(t_batch))
            model_input = x_t + time_emb.unsqueeze(1)
            backbone_output = self.backbone.forward_hidden(inputs_embeds=model_input)
            predicted_noise = self.noise_pred_head(backbone_output)

            x0_pred = (x_t - torch.sqrt(1 - alpha_bar_t) * predicted_noise) / torch.sqrt(alpha_bar_t)

            if i + 1 < len(step_indices):
                t_next = step_indices[i + 1].expand(batch_size)
                alpha_bar_next = cosine_alpha_bar(t_next.float() / self.num_timesteps).clamp(
                    min=1e-5, max=1.0
                ).view(-1, 1, 1)
                x_t = torch.sqrt(alpha_bar_next) * x0_pred + torch.sqrt(1 - alpha_bar_next) * predicted_noise
            else:
                x_t = x0_pred

        # Map the final continuous embeddings back to discrete tokens via
        # nearest-neighbor lookup in embedding space.
        vocab_embeddings = embed_tokens.weight  # [vocab_size, hidden_size]
        distances = torch.cdist(x_t.reshape(-1, hidden_size), vocab_embeddings)
        token_ids = distances.argmin(dim=-1).view(batch_size, seq_len)
        return token_ids
