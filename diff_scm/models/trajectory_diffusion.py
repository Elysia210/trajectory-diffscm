"""
Minimal sequence denoiser for the trajectory Diff-SCM prototype.

The first trajectory diffusion target is deliberately narrow: condition on the
observed pair history and denoise/generate the pair future. The tensor layout is
kept compatible with the existing GaussianDiffusion code: [B, F, T_future].
"""

import math

import torch
from torch import nn


def sinusoidal_timestep_embedding(timesteps: torch.Tensor, dim: int) -> torch.Tensor:
    """Create transformer-style sinusoidal embeddings for diffusion timesteps."""
    half_dim = dim // 2
    if half_dim == 0:
        return timesteps.float()[:, None]
    scale = math.log(10000.0) / max(half_dim - 1, 1)
    frequencies = torch.exp(torch.arange(half_dim, device=timesteps.device) * -scale)
    args = timesteps.float()[:, None] * frequencies[None]
    embedding = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if dim % 2 == 1:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


class TrajectoryFutureDenoiser(nn.Module):
    """
    GRU denoiser for conditional future trajectory diffusion.

    Args:
        input_dim: per-timestep trajectory feature dimension, currently 23.
        hidden_dim: recurrent hidden size.
        history_steps: number of observed history steps used as condition.
        future_steps: number of generated future steps.
    """

    def __init__(
        self,
        input_dim: int = 23,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.1,
        history_steps: int = 50,
        future_steps: int = 50,
        time_embed_dim: int = 64,
        history_dim: int = None,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.history_dim = input_dim if history_dim is None else history_dim
        self.history_steps = history_steps
        self.future_steps = future_steps
        self.time_embed_dim = time_embed_dim

        self.history_encoder = nn.GRU(
            input_size=self.history_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
        )
        self.time_mlp = nn.Sequential(
            nn.Linear(time_embed_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        recurrent_dropout = dropout if num_layers > 1 else 0.0
        self.future_denoiser = nn.GRU(
            input_size=input_dim + hidden_dim + hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=recurrent_dropout,
        )
        self.output = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, x: torch.Tensor, timesteps: torch.Tensor, history: torch.Tensor) -> torch.Tensor:
        """
        Predict diffusion noise for a normalized future trajectory.

        x: noisy future in [B, F, T_future], matching GaussianDiffusion layout.
        history: normalized observed history in [B, T_history, F].
        returns: predicted noise in [B, F, T_future].
        """
        noisy_future = x.transpose(1, 2)
        _, history_hidden = self.history_encoder(history)
        history_context = history_hidden[-1]

        time_embedding = sinusoidal_timestep_embedding(timesteps, self.time_embed_dim)
        time_context = self.time_mlp(time_embedding)

        batch_size, future_steps, _ = noisy_future.shape
        history_context = history_context[:, None, :].expand(batch_size, future_steps, -1)
        time_context = time_context[:, None, :].expand(batch_size, future_steps, -1)

        denoiser_input = torch.cat([noisy_future, history_context, time_context], dim=-1)
        denoised, _ = self.future_denoiser(denoiser_input)
        predicted_noise = self.output(denoised)
        return predicted_noise.transpose(1, 2)
