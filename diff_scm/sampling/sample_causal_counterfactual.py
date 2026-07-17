"""
Counterfactual generation for the trajectory CausalDiffAE via do-interventions.

Ported from CausalDiffAE (scripts/image_causaldae_test.py). Procedure per scene:

  1. encode the real trajectory        -> mu (exogenous causal latents)
  2. do(z_i = v): overwrite variable i's latent block with the (normalized) value v
  3. propagate through the DAG          -> z_pre = A^T·mu,  z_post = f(z_pre)+mu
  4. reparameterize                     -> z
  5. DDIM decode conditioned on z       -> counterfactual future step sequence
  6. reconstruct positions              -> counterfactual trajectory

Also decodes the factual z (no intervention) for comparison. Saves both to .npz.

Example:
  python -m diff_scm.sampling.sample_causal_counterfactual \
    --model-path .../causal_diffae_train/best_model.pt \
    --data-path /mnt/h/trajectory_apr11/Apr11_relaxed_all_archives \
    --intervene collision --value 1.0 --num-samples 16
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.append(str(Path.cwd()))

import numpy as np
import torch
from torch import nn
from torch.utils.data import Subset

from diff_scm.configs import get_config
from diff_scm.datasets.trajectory_dataset import TrajectoryDataset
from diff_scm.datasets.causal_trajectory_dataset import CausalTrajectoryDataset, collate_causal_batch
from diff_scm.models.causal import build_causal_traj_diffae, reparameterize
from diff_scm.guidance.trajectory_preservation import (
    build_position_history,
    denorm_step_delta,
    reconstruct_future_positions,
    recompute_motion_features,
    update_trajectory_positions,
)
from diff_scm.training.trajectory_diffusion_train import split_dataset
from diff_scm.training.trajectory_relative_diffusion_train import TARGET_DIM
from diff_scm.utils.script_util import create_gaussian_diffusion


class _ZDenoiser(nn.Module):
    """Adapt the z-conditioned denoiser to diffusion.ddim_sample_loop's model(x, t) API
    by holding a fixed (possibly intervened) latent z.

    If a guidance weight `w` is given, applies classifier-free guidance
    eps = w * eps(z) + (1 - w) * eps(0), so w > 1 amplifies the intervention.
    """

    def __init__(self, denoiser: nn.Module, z: torch.Tensor, w=None):
        super().__init__()
        self.denoiser = denoiser
        self.z = z
        self.w = w

    def forward(self, x, timesteps, **kwargs):
        if self.w is None:
            return self.denoiser(x, timesteps, self.z)
        eps_cond = self.denoiser(x, timesteps, self.z)
        eps_uncond = self.denoiser(x, timesteps, torch.zeros_like(self.z))
        return self.w * eps_cond + (1.0 - self.w) * eps_uncond


def decode(diffusion, denoiser, z, shape, noise, device, eta, w=None):
    model = _ZDenoiser(denoiser, z, w=w)
    target, _ = diffusion.ddim_sample_loop(
        model, shape=shape, noise=noise, clip_denoised=False, cond_fn=None,
        model_kwargs={}, device=device, progress=False, eta=eta,
    )
    return target


def ddim_invert(diffusion, denoiser, z, x_start):
    """
    Abduction: deterministic DDIM ODE inversion of x_start (conditioned on the factual z)
    up to x_T. Re-decoding x_T with z reconstructs x_start; re-decoding with an intervened
    z gives a minimal-change counterfactual that keeps the same exogenous latent.
    """
    model = _ZDenoiser(denoiser, z)
    x = x_start
    for i in range(diffusion.num_timesteps):
        t = torch.full((x.shape[0],), i, device=x.device, dtype=torch.long)
        with torch.no_grad():
            out = diffusion.ddim_reverse_sample(model, x, t, clip_denoised=False,
                                                model_kwargs={}, eta=0.0)
        x = out["sample"]
    return x


def main(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    config = get_config.file_from_dataset("trajectory")
    config.data.path = Path(args.data_path)
    device = config.device

    ckpt = torch.load(args.model_path, map_location=device)
    factor_nodes = list(ckpt["factor_nodes"])
    num_vars = int(ckpt["num_vars"])
    per_var = int(ckpt["per_var"])
    history_steps = int(ckpt["history_steps"])
    future_steps = int(ckpt["future_steps"])
    target_mean, target_std = ckpt["target_mean"].float().cpu(), ckpt["target_std"].float().cpu()
    feature_mean, feature_std = ckpt["feature_mean"].float().cpu(), ckpt["feature_std"].float().cpu()
    adjacency = ckpt["adjacency"].float()
    factor_ranges = ckpt["factor_ranges"]

    model = build_causal_traj_diffae(
        target_dim=TARGET_DIM, encode_dim=config.trajectory_diffusion.input_dim,
        num_vars=num_vars, per_var=per_var, hidden_dim=config.trajectory_diffusion.hidden_dim,
        num_layers=config.trajectory_diffusion.num_layers, dropout=config.trajectory_diffusion.dropout,
        time_embed_dim=config.trajectory_diffusion.time_embed_dim, adjacency=adjacency,
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    diffusion = create_gaussian_diffusion(config)

    if args.intervene not in factor_nodes:
        raise ValueError(f"--intervene must be one of {factor_nodes}")
    node = factor_nodes.index(args.intervene)
    lo, hi = factor_ranges[args.intervene]
    value_norm = float((args.value - lo) / (hi - lo + 1e-6))

    dataset = TrajectoryDataset(
        data_path=config.data.path, expected_timesteps=config.data.expected_timesteps,
        require_labels=False, recursive=config.data.recursive,
        cache_in_memory=config.data.cache_in_memory, label_candidates=config.data.label_candidates,
    )
    _, val_subset, _ = split_dataset(dataset, config)
    causal_val = CausalTrajectoryDataset(
        val_subset, factor_npz_path=Path(args.factor_npz), factor_nodes=factor_nodes,
        history_steps=history_steps, target_mean=target_mean, target_std=target_std,
        feature_mean=feature_mean, feature_std=feature_std, factor_ranges=factor_ranges,
    )
    batch = collate_causal_batch([causal_val[i] for i in range(min(args.num_samples, len(causal_val)))])
    x_encode = batch["x_encode"].to(device)
    x_start = batch["x_start"].to(device)
    B = x_encode.shape[0]
    shape = (B, TARGET_DIM, future_steps)

    fmean = feature_mean.to(device)
    fstd = feature_std.to(device)
    tmean = target_mean.to(device)
    tstd = target_std.to(device)

    with torch.no_grad():
        # Encode; small deterministic variance as in CausalDiffAE.
        mu, _ = model.encoder.encode(x_encode)
        var = torch.ones_like(mu) * args.var_scale

        # Factual latent (no intervention).
        z_pre = model.causal_mask.causal_masking(mu)
        z_post = model.causal_mask.nonlinearity_add_back_noise(mu, z_pre)
        z_factual = reparameterize(z_post, var)

        # Counterfactual: do(z_node = value_norm), then re-propagate through the DAG.
        mu_cf = mu.clone()
        mu_cf[:, node * per_var:(node + 1) * per_var] = value_norm
        z_pre_cf = model.causal_mask.causal_masking(mu_cf)
        z_post_cf = model.causal_mask.nonlinearity_add_back_noise(mu_cf, z_pre_cf)
        z_cf = reparameterize(z_post_cf, var)

        # Abduction: invert the real trajectory (with factual z) to its exogenous latent
        # x_T, then decode both factual and counterfactual from the SAME x_T. Without
        # abduction, fall back to a shared random-noise start.
        if args.no_abduction:
            x_T = torch.randn(*shape, device=device)
        else:
            x_T = ddim_invert(diffusion, model.denoiser, z_factual, x_start)
        factual_target = decode(diffusion, model.denoiser, z_factual, shape, x_T.clone(), device, args.eta)
        cf_target = decode(diffusion, model.denoiser, z_cf, shape, x_T.clone(), device, args.eta, w=args.w)

    # Reconstruct trajectories (denormalize x_encode back to raw for the history anchor).
    real_trajectory = x_encode * fstd + fmean
    history_xy = build_position_history(real_trajectory, history_steps)

    def to_traj(target):
        step = denorm_step_delta(target.transpose(1, 2), tmean, tstd)
        pos = reconstruct_future_positions(history_xy, step)
        traj = update_trajectory_positions(real_trajectory, pos, history_steps)
        return recompute_motion_features(traj, history_steps, dt=0.1), step

    factual_traj, factual_step = to_traj(factual_target)
    cf_traj, cf_step = to_traj(cf_target)

    out = {
        "scene_id": np.asarray(batch["scene_id"], dtype=object),
        "intervene": np.asarray(args.intervene, dtype=object),
        "value": np.asarray(args.value, dtype=np.float32),
        "value_norm": np.asarray(value_norm, dtype=np.float32),
        "factor_nodes": np.asarray(factor_nodes, dtype=object),
        "adjacency": adjacency.cpu().numpy(),
        "labels": batch["c"].cpu().numpy(),
        "factual_trajectory": factual_traj.cpu().numpy(),
        "counterfactual_trajectory": cf_traj.cpu().numpy(),
        "factual_step": factual_step.cpu().numpy(),
        "counterfactual_step": cf_step.cpu().numpy(),
        "history": real_trajectory[:, :history_steps].cpu().numpy(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output, **out)
    print(f"intervened do({args.intervene}={args.value}) -> normalized {value_norm:.3f}")
    print(f"saved counterfactual trajectories to {args.output}")


def build_argparser():
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", type=str, required=True)
    p.add_argument("--data-path", type=str, required=True)
    p.add_argument("--factor-npz", type=str, default="labels/causaldiffae_factors.npz")
    p.add_argument("--intervene", type=str, required=True, help="factor node to intervene on")
    p.add_argument("--value", type=float, required=True, help="intervention value in raw factor units")
    p.add_argument("--num-samples", type=int, default=16)
    p.add_argument("--var-scale", type=float, default=0.001)
    p.add_argument("--no-abduction", action="store_true",
                   help="Skip DDIM inversion and decode from random noise (weaker counterfactuals).")
    p.add_argument("--w", type=float, default=None,
                   help="Classifier-free guidance weight for the counterfactual (needs a --masking-trained model). "
                        "w=1 plain conditional; w>1 amplifies the intervention (try 2-5).")
    p.add_argument("--eta", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", type=Path,
                   default=Path("results/causal_diffae/counterfactual.npz"))
    return p


if __name__ == "__main__":
    main(build_argparser().parse_args())
