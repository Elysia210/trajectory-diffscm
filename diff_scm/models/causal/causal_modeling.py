"""
Causal latent modeling — ported from CausalDiffAE (improved_diffusion/nn.py:
`CausalModeling`, `MLP`, `reparameterize`).

This is the domain-agnostic heart of CausalDiffAE. It operates purely on a latent
vector shaped [B, num_var, dim_per_var], so it transfers unchanged from the image
domain to trajectories. Given an encoder output `u` (exogenous / noise latents) and a
causal DAG adjacency `A`, it produces causal latents `z` through a structural causal
model:

    z_pre = A^T · u                 # mix each variable into its children (per the DAG)
    z_i   = f_i(z_pre_i) + u_i      # structural equation: nonlinearity(parents) + noise

Kept faithful to the original, with two deliberate changes:
- device-safe (z_post built via torch.stack on u's device; the original built a CPU
  zeros tensor and relied on a later `.to(device)`),
- adjacency `A` is passed in / configured (from our causal graph) instead of being
  hard-coded to a 2- or 4-node matrix inside the model.
"""

from typing import Optional

import torch
from torch import nn


class MLP(nn.Module):
    """Per-variable structural mechanism f_i (a small MLP), as in CausalDiffAE."""

    def __init__(self, latent_dim: int, num_var: int):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_var = num_var
        per_var = latent_dim // num_var
        self.net = nn.Sequential(
            nn.Linear(per_var, latent_dim),
            nn.LeakyReLU(),
            nn.Linear(latent_dim, per_var),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CausalModeling(nn.Module):
    """
    Map exogenous latents to causal latents through a DAG.

    Args:
        latent_dim: total latent width (must be divisible by num_var).
        num_var: number of causal variables (DAG nodes).
        adjacency: [num_var, num_var] adjacency, A[i, j] = 1 means i -> j. Defaults to
            zeros. If `learn` is True it becomes a trainable parameter seeded from this;
            otherwise it is stored as a fixed buffer.
        learn: whether A is learned (True) or fixed (False).
    """

    def __init__(
        self,
        latent_dim: int,
        num_var: int,
        adjacency: Optional[torch.Tensor] = None,
        learn: bool = False,
    ):
        super().__init__()
        assert latent_dim % num_var == 0, "latent_dim must be divisible by num_var"
        self.latent_dim = latent_dim
        self.num_var = num_var
        self.per_var = latent_dim // num_var

        if adjacency is None:
            adjacency = torch.zeros(num_var, num_var)
        adjacency = torch.as_tensor(adjacency, dtype=torch.float32).clone()
        if learn:
            self.A = nn.Parameter(adjacency)
        else:
            self.register_buffer("A", adjacency)

        self.nonlinearities = nn.ModuleDict(
            {str(i): MLP(latent_dim=latent_dim, num_var=num_var) for i in range(num_var)}
        )

    def causal_masking(self, u: torch.Tensor, A: Optional[torch.Tensor] = None) -> torch.Tensor:
        """z_pre = A^T · u — propagate each variable to its children."""
        A = self.A if A is None else A
        u = u.reshape(-1, self.num_var, self.per_var)
        return torch.matmul(A.t().to(u.device), u)

    def nonlinearity_add_back_noise(self, u: torch.Tensor, z_pre: torch.Tensor) -> torch.Tensor:
        """z_i = f_i(z_pre_i) + u_i — apply structural equations, then flatten."""
        u = u.reshape(-1, self.num_var, self.per_var)
        z_post = torch.stack(
            [self.nonlinearities[str(i)](z_pre[:, i, :]) + u[:, i, :] for i in range(self.num_var)],
            dim=1,
        )
        return z_post.reshape(-1, self.num_var * self.per_var)

    def forward(self, u: torch.Tensor, A: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Full exogenous -> causal latent map (masking + structural equations)."""
        z_pre = self.causal_masking(u, A)
        return self.nonlinearity_add_back_noise(u, z_pre)


def reparameterize(mean: torch.Tensor, var: torch.Tensor) -> torch.Tensor:
    """Gaussian reparameterization trick: z = mean + sqrt(var) * eps."""
    eps = torch.randn_like(mean)
    return mean + (var ** 0.5) * eps


def default_causal_adjacency(num_var: int) -> torch.Tensor:
    """
    CausalDiffAE's example DAG adjacency for the common node counts (hard-coded in its
    unet.py). A[i, j] = 1 means node i -> node j. Use as a working starting graph;
    order your factor_nodes so the edges make sense (roots first, sink last).

      2 nodes  : 0 -> 1                            (MorphoMNIST: thickness -> intensity)
      4 nodes  : 0 -> {1,2,3}, 1 -> 3, 2 -> 3      (CausalCircuit: root -> ... -> sink)
    """
    if num_var == 2:
        A = [[0, 1], [0, 0]]
    elif num_var == 4:
        A = [[0, 1, 1, 1], [0, 0, 0, 1], [0, 0, 0, 1], [0, 0, 0, 0]]
    else:
        raise ValueError(f"no default CausalDiffAE adjacency for num_var={num_var}; "
                         f"pass a custom adjacency.")
    return torch.tensor(A, dtype=torch.float32)
