"""
Causal representation losses — ported from CausalDiffAE:
- kl_normal            (improved_diffusion/nn.py)
- prior + representation_loss (improved_diffusion/gaussian_diffusion.py)

These are domain-agnostic (they act on latent vectors + a label vector), so they carry
over from images to trajectories unchanged. The total CausalDiffAE objective is

    loss = diffusion_mse(eps, noise)  +  kl_weight * representation_loss(...)

where representation_loss = KL(encoder posterior || N(0,1))
                          + sum_i KL(causal latent_i || label-conditioned prior_i).
The per-variable label term (unit variances) reduces to 0.5 * ||z_post_i - label_i||^2,
i.e. it pulls each causal latent toward its supervised factor value.
"""

import torch


def kl_normal(qm: torch.Tensor, qv: torch.Tensor, pm: torch.Tensor, pv: torch.Tensor) -> torch.Tensor:
    """Elementwise KL(q||p) between diagonal Gaussians, summed over the last dim -> [B]."""
    element_wise = 0.5 * (torch.log(pv) - torch.log(qv) + qv / pv + (qm - pm).pow(2) / pv - 1)
    return element_wise.sum(dim=-1)


def label_prior_mean(labels: torch.Tensor, per_var: int) -> torch.Tensor:
    """
    Label-conditioned prior mean, one causal variable per label column.

    labels: [B, num_vars], expected pre-normalized to a comparable scale (e.g. ~[0,1]).
    Returns [B, num_vars, per_var] with each variable's prior mean = its label value.
    """
    return labels.unsqueeze(-1).expand(labels.shape[0], labels.shape[1], per_var)


def representation_loss(
    mu: torch.Tensor,
    var: torch.Tensor,
    z_post: torch.Tensor,
    labels: torch.Tensor,
    num_vars: int,
    causal_modeling: bool = True,
    mask: torch.Tensor = None,
) -> torch.Tensor:
    """
    mu / var / z_post: [B, latent_dim]; labels: [B, num_vars].
    If a classifier-free `mask` [B] (1 = conditioned) is given, the label KL is averaged
    over conditioned samples only. Returns a scalar.
    """
    B, latent_dim = mu.shape
    per_var = latent_dim // num_vars
    zero_mean = torch.zeros_like(mu)
    unit_var = torch.ones_like(var)
    prior_mean = label_prior_mean(labels.to(mu.device), per_var)  # [B, num_vars, per_var]

    # Encoder posterior regularized toward a standard Gaussian.
    kld = kl_normal(mu, var, zero_mean, unit_var)

    if causal_modeling:
        z_post_r = z_post.reshape(B, num_vars, per_var)
        unit_r = unit_var.reshape(B, num_vars, per_var)
        for i in range(num_vars):
            kld = kld + kl_normal(
                z_post_r[:, i, :], unit_r[:, i, :], prior_mean[:, i, :], unit_r[:, i, :]
            )

    if mask is not None:
        return (kld * mask).sum() / mask.sum().clamp_min(1.0)
    return kld.mean()
