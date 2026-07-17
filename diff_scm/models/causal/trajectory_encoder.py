"""
Trajectory (sequence) encoder for CausalDiffAE — the domain swap for the image
`GaussianConvEncoder` in CausalDiffAE (improved_diffusion/nn.py).

Same role and interface (`.encode(x) -> [mu, var]` with a positive `var`), but the
2D-conv image encoder is replaced by a GRU over the trajectory sequence [B, T, F].
The output latent width is `latent_dim = num_vars * per_var`, so it plugs straight
into `CausalModeling`, which reshapes it to [B, num_vars, per_var].
"""

import torch
from torch import nn
import torch.nn.functional as F


class GaussianTrajectoryEncoder(nn.Module):
    """
    Encode a trajectory into Gaussian latent parameters.

    Args:
        input_dim: per-timestep feature dim (23 for the pair layout).
        latent_dim: total latent width; must equal num_vars * per_var and be
            divisible by num_vars (matches CausalModeling).
        num_vars: number of causal variables (DAG nodes).
        hidden_dim / num_layers / bidirectional: GRU encoder settings.
    """

    def __init__(
        self,
        input_dim: int = 23,
        latent_dim: int = 64,
        num_vars: int = 4,
        hidden_dim: int = 128,
        num_layers: int = 1,
        bidirectional: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()
        assert latent_dim % num_vars == 0, "latent_dim must be divisible by num_vars"
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.num_vars = num_vars

        gru_dropout = dropout if num_layers > 1 else 0.0
        self.encoder = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=gru_dropout,
            bidirectional=bidirectional,
        )
        enc_out = hidden_dim * (2 if bidirectional else 1)
        self.fc_mu = nn.Linear(enc_out, latent_dim)
        self.fc_var = nn.Linear(enc_out, latent_dim)

    def encode(self, trajectory: torch.Tensor):
        """
        trajectory: [B, T, F]  ->  [mu, var], each [B, latent_dim], var > 0.
        """
        _, hidden = self.encoder(trajectory)
        if self.encoder.bidirectional:
            hidden = torch.cat([hidden[-2], hidden[-1]], dim=-1)
        else:
            hidden = hidden[-1]
        mu = self.fc_mu(hidden)
        var = F.softplus(self.fc_var(hidden)) + 1e-8
        return [mu, var]

    def forward(self, trajectory: torch.Tensor):
        return self.encode(trajectory)
