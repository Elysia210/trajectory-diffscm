"""
CausalDiffAE main module, ported to trajectories.

Mirrors CausalDiffAE's UNet representation path (improved_diffusion/unet.py forward):

    mu, var = encoder.encode(x_encode)          # semantic (exogenous) latents
    z_pre   = causal_mask.causal_masking(mu, A) # DAG mixing
    z_post  = causal_mask.nonlinearity_add_back_noise(mu, z_pre)  # structural equations
    z       = reparameterize(z_post, var * var_scale)
    eps     = denoiser(x_t, t, z)               # z conditions the diffusion (DiffAE-style)

The 2D-conv image encoder / UNet decoder are replaced by a GRU sequence encoder and a
GRU step-denoiser; the causal machinery in between (CausalModeling) is unchanged.

`forward` returns predicted noise plus {mu, var, z_post, z} so the training loop can
add the variational (KL) + label-prior terms. For sampling / counterfactuals, encode
once to get z (or intervene on it) and call `.denoiser(x_t, t, z)` in the DDIM loop.
"""

from typing import Optional

import torch
from torch import nn

from diff_scm.models.trajectory_diffusion import sinusoidal_timestep_embedding
from diff_scm.models.causal.causal_modeling import CausalModeling, reparameterize
from diff_scm.models.causal.trajectory_encoder import GaussianTrajectoryEncoder


class ZConditionedTrajectoryDenoiser(nn.Module):
    """
    GRU step denoiser conditioned on a latent z (DiffAE-style: an up-projection of z is
    added to the timestep embedding). Tensor layout matches GaussianDiffusion: the
    noisy input x is [B, C, T], the returned noise is [B, C, T].
    """

    def __init__(
        self,
        input_dim: int = 4,
        latent_dim: int = 64,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.1,
        time_embed_dim: int = 64,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.time_embed_dim = time_embed_dim

        self.time_mlp = nn.Sequential(
            nn.Linear(time_embed_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim)
        )
        self.z_mlp = nn.Sequential(  # "up_emb": lift z into the conditioning space
            nn.Linear(latent_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim)
        )
        recurrent_dropout = dropout if num_layers > 1 else 0.0
        self.denoiser = nn.GRU(
            input_size=input_dim + hidden_dim,
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

    def forward(self, x: torch.Tensor, timesteps: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        noisy = x.transpose(1, 2)  # [B, T, C]
        temb = self.time_mlp(sinusoidal_timestep_embedding(timesteps, self.time_embed_dim))
        cond = temb + self.z_mlp(z)  # DiffAE adds the latent to the time embedding
        cond = cond[:, None, :].expand(noisy.shape[0], noisy.shape[1], -1)
        denoised, _ = self.denoiser(torch.cat([noisy, cond], dim=-1))
        return self.output(denoised).transpose(1, 2)


class CausalTrajDiffAE(nn.Module):
    """
    Encoder + causal latent model + z-conditioned denoiser.

    Args:
        encoder: GaussianTrajectoryEncoder (returns [mu, var]).
        causal_mask: CausalModeling (DAG masking + structural equations).
        denoiser: ZConditionedTrajectoryDenoiser.
        var_scale: scales the encoder variance used for reparameterization (CausalDiffAE
            uses a small value, 0.001, so z stays close to the mean).
        causal_modeling: if False, behaves like plain (Diff)AE (skip the DAG).
    """

    def __init__(
        self,
        encoder: GaussianTrajectoryEncoder,
        causal_mask: CausalModeling,
        denoiser: ZConditionedTrajectoryDenoiser,
        var_scale: float = 0.001,
        causal_modeling: bool = True,
        masking: bool = False,
        drop_prob: float = 0.1,
    ):
        super().__init__()
        self.encoder = encoder
        self.causal_mask = causal_mask
        self.denoiser = denoiser
        self.var_scale = var_scale
        self.causal_modeling = causal_modeling
        self.masking = masking          # classifier-free: randomly drop z during training
        self.drop_prob = drop_prob

    def encode_to_z(self, x_encode: torch.Tensor, A: Optional[torch.Tensor] = None):
        """x_encode: [B, T, F] -> (z, mu, var, z_post)."""
        mu, var = self.encoder.encode(x_encode)
        if self.causal_modeling:
            z_pre = self.causal_mask.causal_masking(mu, A)
            z_post = self.causal_mask.nonlinearity_add_back_noise(mu, z_pre)
            z = reparameterize(z_post, var * self.var_scale)
        else:
            z_post = mu
            z = reparameterize(mu, var * self.var_scale)
        return z, mu, var, z_post

    def forward(self, x: torch.Tensor, timesteps: torch.Tensor,
                x_encode: torch.Tensor, A: Optional[torch.Tensor] = None):
        """
        x: noisy target [B, C, T]; x_encode: sequence the encoder sees [B, T, F].
        Returns (predicted_noise, aux) where aux has mu / var / z_post / z.
        """
        z, mu, var, z_post = self.encode_to_z(x_encode, A)
        # Classifier-free masking: with prob drop_prob zero z so the model also learns an
        # unconditional (z=0) branch, enabling guidance-weighted intervention at sampling.
        mask = None
        if self.masking and self.training:
            keep = torch.bernoulli(torch.full((z.shape[0],), 1.0 - self.drop_prob, device=z.device))
            z = z * keep[:, None]
            z_post = z_post * keep[:, None]
            mask = keep
        eps = self.denoiser(x, timesteps, z)
        return eps, {"mu": mu, "var": var, "z_post": z_post, "z": z, "mask": mask}


def build_causal_traj_diffae(
    target_dim: int = 4,
    encode_dim: int = 23,
    num_vars: int = 4,
    per_var: int = 16,
    hidden_dim: int = 128,
    num_layers: int = 2,
    dropout: float = 0.1,
    time_embed_dim: int = 64,
    adjacency: Optional[torch.Tensor] = None,
    learn_adjacency: bool = False,
    var_scale: float = 0.001,
    causal_modeling: bool = True,
    masking: bool = False,
    drop_prob: float = 0.1,
) -> CausalTrajDiffAE:
    """
    Convenience factory. `target_dim` is what the diffusion generates (e.g. 4 step-delta
    dims); `encode_dim` is what the encoder reads (e.g. the full 23-dim trajectory).
    latent_dim = num_vars * per_var.
    """
    latent_dim = num_vars * per_var
    encoder = GaussianTrajectoryEncoder(
        input_dim=encode_dim, latent_dim=latent_dim, num_vars=num_vars,
        hidden_dim=hidden_dim, num_layers=1, bidirectional=True,
    )
    causal_mask = CausalModeling(
        latent_dim=latent_dim, num_var=num_vars, adjacency=adjacency, learn=learn_adjacency,
    )
    denoiser = ZConditionedTrajectoryDenoiser(
        input_dim=target_dim, latent_dim=latent_dim, hidden_dim=hidden_dim,
        num_layers=num_layers, dropout=dropout, time_embed_dim=time_embed_dim,
    )
    return CausalTrajDiffAE(encoder, causal_mask, denoiser, var_scale=var_scale,
                            causal_modeling=causal_modeling, masking=masking, drop_prob=drop_prob)
