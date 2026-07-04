"""
Sample preservation-aware guided step-delta trajectory diffusion.

This v2 sampler keeps the method intentionally minimal:
- risk guidance still uses normalized, clipped gradients
- local step preservation is enforced against an unguided reference rollout
- preservation gradients keep their raw magnitude structure and are only clipped

The point of this version is to make preservation weights meaningful again.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import re
import sys
from typing import Optional

sys.path.append(str(Path.cwd()))

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from diff_scm.configs import get_config
from diff_scm.datasets.trajectory_dataset import TrajectoryDataset
from diff_scm.guidance.trajectory_preservation import (
    acceleration_hinge_loss,
    apply_feature_mask,
    apply_feature_weight,
    apply_time_mask,
    build_position_history,
    clip_grad_norm_per_sample,
    denorm_step_delta,
    estimate_dt,
    grad_l2_norm_per_sample,
    map_offroad_loss,
    normalize_grad_per_sample,
    recompute_motion_features,
    reconstruct_future_positions,
    step_preservation_loss,
    turn_angle_hinge_loss,
    update_trajectory_positions,
)
from diff_scm.models.trajectory_baseline import TrajectoryGRUBaseline
from diff_scm.models.trajectory_diffusion import TrajectoryFutureDenoiser
from diff_scm.training.trajectory_diffusion_train import split_dataset
from diff_scm.training.trajectory_relative_diffusion_train import TARGET_DIM
from diff_scm.training.trajectory_step_diffusion_train import StepFutureDataset, collate_batch
from diff_scm.utils.script_util import create_gaussian_diffusion


def load_step_model(checkpoint_path: Path, config, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    history_steps = int(checkpoint["history_steps"])
    future_steps = int(checkpoint["future_steps"])
    model = TrajectoryFutureDenoiser(
        input_dim=TARGET_DIM,
        history_dim=config.trajectory_diffusion.input_dim,
        hidden_dim=config.trajectory_diffusion.hidden_dim,
        num_layers=config.trajectory_diffusion.num_layers,
        dropout=config.trajectory_diffusion.dropout,
        history_steps=history_steps,
        future_steps=future_steps,
        time_embed_dim=config.trajectory_diffusion.time_embed_dim,
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, checkpoint


def load_classifier(checkpoint_path: Path, config, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = TrajectoryGRUBaseline(
        input_dim=config.classifier.input_dim,
        hidden_dim=config.classifier.hidden_dim,
        num_layers=config.classifier.num_layers,
        dropout=config.classifier.dropout,
        bidirectional=config.classifier.bidirectional,
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model


def parse_mask(values: str, expected_dim: int, device: torch.device) -> torch.Tensor:
    parsed = [float(value.strip()) for value in values.split(",") if value.strip()]
    if len(parsed) != expected_dim:
        raise ValueError(f"Expected {expected_dim} comma-separated values, got {values!r}")
    return torch.tensor(parsed, dtype=torch.float32, device=device)


def load_map_cost(scene_ids, history_steps, future_steps, device):
    """
    Load per-step agent-centric off-road cost fields and raster_from_world transforms
    for ego (agent 0) and the controlled adversary, over the future steps.

    Cost = distance (in pixels) to the nearest drivable cell (0 on-road, growing
    off-road), precomputed once. Returns four tensors:
      ego_cost/adv_cost: [B, T_future, H, W]; ego_rfw/adv_rfw: [B, T_future, 3, 3].
    """
    import h5py
    try:
        from scipy.ndimage import distance_transform_edt
        have_scipy = True
    except Exception:
        have_scipy = False
        print("[map] scipy not available; falling back to soft (1 - drivable) cost.")

    def cost_from_drivable(d):  # d: [T_future, H, W] bool
        d = d.astype(bool)
        if have_scipy:
            return np.stack([distance_transform_edt(~m) for m in d]).astype(np.float32)
        return (~d).astype(np.float32)

    ego_cost, ego_rfw, adv_cost, adv_rfw = [], [], [], []
    fut = slice(history_steps, history_steps + future_steps)
    for sid in scene_ids:
        file_path, scene_key = sid.rsplit(":", 1)
        adv_idx = int(re.search(r"ctrl_\[(\d+)\]", scene_key).group(1))
        with h5py.File(file_path, "r") as h:
            g = h[scene_key]
            dm = np.asarray(g["drivable_map"])
            rfw = np.asarray(g["raster_from_world"]).astype(np.float32)
        ego_cost.append(cost_from_drivable(dm[0, fut]))
        adv_cost.append(cost_from_drivable(dm[adv_idx, fut]))
        ego_rfw.append(rfw[0, fut])
        adv_rfw.append(rfw[adv_idx, fut])

    to_t = lambda x: torch.from_numpy(np.stack(x)).to(device)
    return to_t(ego_cost), to_t(ego_rfw), to_t(adv_cost), to_t(adv_rfw)


def make_preservation_cond_fn(
    diffusion,
    model: TrajectoryFutureDenoiser,
    classifier: TrajectoryGRUBaseline,
    history: torch.Tensor,
    real_trajectory: torch.Tensor,
    reference_step: torch.Tensor,
    target_mean: torch.Tensor,
    target_std: torch.Tensor,
    history_steps: int,
    guidance_target: str,
    guidance_scale: float,
    lambda_step: float,
    lambda_accel: float,
    accel_max: float,
    lambda_turn: float,
    turn_max: float,
    lambda_map: float,
    ego_cost: Optional[torch.Tensor],
    ego_rfw: Optional[torch.Tensor],
    adv_cost: Optional[torch.Tensor],
    adv_rfw: Optional[torch.Tensor],
    dt: float,
    preservation_dim_weight: Optional[torch.Tensor],
    risk_dim_mask: Optional[torch.Tensor],
    risk_clip_norm: Optional[float],
    step_clip_norm: Optional[float],
    time_mask_start: float,
    time_mask_end: float,
    grad_stats: dict,
):
    """
    Preservation-aware classifier guidance for the DDIM denoising loop.

    Unlike the vanilla sampler, this function reconstructs a denoised x0-style
    step prediction from the current noisy x_t and evaluates both a risk term
    and a local step-preservation term against an unguided factual rollout.
    """
    direction = 1.0 if guidance_target == "collision" else -1.0
    history_xy = build_position_history(real_trajectory, history_steps)

    target_mean = target_mean.to(real_trajectory.device)
    target_std = target_std.to(real_trajectory.device)
    timestep_lookup = None
    if hasattr(diffusion, "timestep_map"):
        timestep_lookup = torch.full(
            (diffusion.original_num_steps,),
            fill_value=-1,
            dtype=torch.long,
            device=real_trajectory.device,
        )
        for compressed_index, original_timestep in enumerate(diffusion.timestep_map):
            timestep_lookup[int(original_timestep)] = compressed_index

    def cond_fn(x, timesteps, **model_kwargs):
        del model_kwargs
        with torch.enable_grad():
            x_t = x.detach().requires_grad_(True)
            with torch.backends.cudnn.flags(enabled=False):
                pred_eps = model(x_t, timesteps, history)

                # In SpacedDiffusion, cond_fn receives timesteps already mapped
                # back to the original diffusion indices. The denoiser expects
                # those original indices, but _predict_xstart_from_eps must use
                # the compressed DDIM timestep index that matches the active
                # diffusion buffers. Convert back before calling the diffusion
                # helper.
                if timestep_lookup is not None:
                    compressed_timesteps = timestep_lookup[timesteps.long()]
                    if (compressed_timesteps < 0).any():
                        raise ValueError("Encountered a timestep outside the active DDIM schedule.")
                else:
                    compressed_timesteps = timesteps.long()

                pred_xstart = diffusion._predict_xstart_from_eps(x_t, compressed_timesteps, pred_eps)

                step_pred = denorm_step_delta(pred_xstart.transpose(1, 2), target_mean, target_std)
                future_positions = reconstruct_future_positions(history_xy, step_pred)
                generated_trajectory = update_trajectory_positions(real_trajectory, future_positions, history_steps)
                # Recompute heading/velocity from the new positions so the classifier
                # evaluates a self-consistent trajectory, not stale reference features.
                generated_trajectory = recompute_motion_features(generated_trajectory, history_steps, dt)
                logits = classifier(generated_trajectory).squeeze(-1)

            risk_obj = direction * logits.mean()
            step_pres_loss = step_preservation_loss(step_pred, reference_step, dim_weight=None)
            accel_loss = acceleration_hinge_loss(step_pred, dt, accel_max)
            turn_loss = turn_angle_hinge_loss(step_pred, turn_max)
            if lambda_map > 0:
                map_loss = map_offroad_loss(
                    future_positions[..., 0:2], ego_rfw, ego_cost,
                    future_positions[..., 2:4], adv_rfw, adv_cost,
                )
            else:
                map_loss = None

            g_risk = torch.autograd.grad(
                risk_obj,
                x_t,
                retain_graph=True,
                create_graph=False,
                allow_unused=False,
            )[0]
            g_step = torch.autograd.grad(
                step_pres_loss,
                x_t,
                retain_graph=True,
                create_graph=False,
                allow_unused=False,
            )[0]
            if lambda_accel > 0:
                g_accel = torch.autograd.grad(
                    accel_loss,
                    x_t,
                    retain_graph=True,
                    create_graph=False,
                    allow_unused=False,
                )[0]
            else:
                g_accel = torch.zeros_like(g_step)
            if lambda_turn > 0:
                g_turn = torch.autograd.grad(
                    turn_loss,
                    x_t,
                    retain_graph=True,
                    create_graph=False,
                    allow_unused=False,
                )[0]
            else:
                g_turn = torch.zeros_like(g_step)
            if map_loss is not None:
                g_map = torch.autograd.grad(
                    map_loss,
                    x_t,
                    retain_graph=False,
                    create_graph=False,
                    allow_unused=False,
                )[0]
            else:
                g_map = torch.zeros_like(g_step)

            raw_risk_norm = grad_l2_norm_per_sample(g_risk).mean().item()
            raw_step_norm = grad_l2_norm_per_sample(g_step).mean().item()
            raw_accel_norm = grad_l2_norm_per_sample(g_accel).mean().item()
            raw_turn_norm = grad_l2_norm_per_sample(g_turn).mean().item()
            raw_map_norm = grad_l2_norm_per_sample(g_map).mean().item()

            # Risk branch: normalize + risk mask + temporal emphasis + clip.
            g_risk = normalize_grad_per_sample(g_risk)
            g_risk = apply_feature_mask(g_risk, risk_dim_mask)
            g_risk = apply_time_mask(g_risk, time_mask_start, time_mask_end)
            g_risk = clip_grad_norm_per_sample(g_risk, risk_clip_norm)

            # Preservation branch: feature weighting + clip only. We
            # intentionally keep the raw magnitude structure here.
            g_step = apply_feature_weight(g_step, preservation_dim_weight)
            g_step = clip_grad_norm_per_sample(g_step, step_clip_norm)

            # Feasibility branches (accel / turn / map): clip only (raw hinge kept).
            g_accel = clip_grad_norm_per_sample(g_accel, step_clip_norm)
            g_turn = clip_grad_norm_per_sample(g_turn, step_clip_norm)
            g_map = clip_grad_norm_per_sample(g_map, step_clip_norm)

            post_risk_norm = grad_l2_norm_per_sample(g_risk).mean().item()
            post_step_norm = grad_l2_norm_per_sample(g_step).mean().item()
            post_accel_norm = grad_l2_norm_per_sample(g_accel).mean().item()
            post_turn_norm = grad_l2_norm_per_sample(g_turn).mean().item()
            post_map_norm = grad_l2_norm_per_sample(g_map).mean().item()

            gradient = (guidance_scale * g_risk - lambda_step * g_step
                        - lambda_accel * g_accel - lambda_turn * g_turn
                        - lambda_map * g_map)
            total_grad_norm = grad_l2_norm_per_sample(gradient).mean().item()

            grad_stats["raw_risk_norm"].append(raw_risk_norm)
            grad_stats["raw_step_norm"].append(raw_step_norm)
            grad_stats["raw_accel_norm"].append(raw_accel_norm)
            grad_stats["raw_turn_norm"].append(raw_turn_norm)
            grad_stats["raw_map_norm"].append(raw_map_norm)
            grad_stats["post_risk_norm"].append(post_risk_norm)
            grad_stats["post_step_norm"].append(post_step_norm)
            grad_stats["post_accel_norm"].append(post_accel_norm)
            grad_stats["post_turn_norm"].append(post_turn_norm)
            grad_stats["post_map_norm"].append(post_map_norm)
            grad_stats["total_norm"].append(total_grad_norm)

            return gradient

    return cond_fn


def build_output_dict(
    scene_ids,
    history_steps: int,
    reference_step: torch.Tensor,
    reference_trajectory: torch.Tensor,
    generated_step: torch.Tensor,
    generated_trajectory: torch.Tensor,
    args,
):
    return {
        "scene_id": np.asarray(scene_ids, dtype=object),
        "history": generated_trajectory[:, :history_steps].cpu().numpy(),
        "generated_future": generated_trajectory[:, history_steps:].cpu().numpy(),
        "generated_trajectory": generated_trajectory.cpu().numpy(),
        "generated_step": generated_step.cpu().numpy(),
        "reference_future": reference_trajectory[:, history_steps:].cpu().numpy(),
        "reference_trajectory": reference_trajectory.cpu().numpy(),
        "reference_step": reference_step.cpu().numpy(),
        "target_type": np.asarray("step_future_xy", dtype=object),
        "guidance_target": np.asarray(args.guidance_target, dtype=object),
        "guidance_scale": np.asarray(args.guidance_scale, dtype=np.float32),
        "lambda_step": np.asarray(args.lambda_step, dtype=np.float32),
        "lambda_accel": np.asarray(args.lambda_accel, dtype=np.float32),
        "accel_max": np.asarray(args.accel_max, dtype=np.float32),
        "lambda_turn": np.asarray(args.lambda_turn, dtype=np.float32),
        "turn_max": np.asarray(args.turn_max, dtype=np.float32),
        "lambda_map": np.asarray(args.lambda_map, dtype=np.float32),
        "risk_dim_mask": np.asarray(args.risk_dim_mask, dtype=object),
        "preservation_dim_weight": np.asarray(args.preservation_dim_weight, dtype=object),
        "time_mask_start": np.asarray(args.time_mask_start, dtype=np.float32),
        "time_mask_end": np.asarray(args.time_mask_end, dtype=np.float32),
        "risk_clip_norm": np.asarray(-1.0 if args.risk_clip_norm is None else args.risk_clip_norm, dtype=np.float32),
        "step_clip_norm": np.asarray(-1.0 if args.step_clip_norm is None else args.step_clip_norm, dtype=np.float32),
    }


def main(args) -> None:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    config = get_config.file_from_dataset("trajectory")
    config.data.path = Path(args.data_path)
    if args.label_map_path is not None:
        config.data.label_map_path = Path(args.label_map_path)
    device = config.device

    diffusion_model, checkpoint = load_step_model(Path(args.model_path), config, device)
    classifier = load_classifier(Path(args.classifier_path), config, device)
    diffusion = create_gaussian_diffusion(config)

    history_steps = int(checkpoint["history_steps"])
    future_steps = int(checkpoint["future_steps"])
    history_mean = checkpoint["history_mean"].float().cpu()
    history_std = checkpoint["history_std"].float().cpu()
    target_mean = checkpoint["target_mean"].float().cpu()
    target_std = checkpoint["target_std"].float().cpu()

    dataset = TrajectoryDataset(
        data_path=config.data.path,
        expected_timesteps=config.data.expected_timesteps,
        require_labels=False,
        recursive=config.data.recursive,
        cache_in_memory=config.data.cache_in_memory,
        label_candidates=config.data.label_candidates,
        label_map_path=config.data.label_map_path,
    )
    train_subset, val_subset, _ = split_dataset(dataset, config)
    sample_indices = val_subset.indices[: args.num_samples] if len(val_subset) > 0 else train_subset.indices[: args.num_samples]
    sample_subset = Subset(dataset, sample_indices)
    sample_dataset = StepFutureDataset(sample_subset, history_mean, history_std, target_mean, target_std, history_steps)
    sample_loader = DataLoader(sample_dataset, batch_size=args.num_samples, shuffle=False, collate_fn=collate_batch)
    batch = next(iter(sample_loader))

    history = batch["history"].to(device)
    real_trajectory = batch["trajectory"].float().to(device)
    shape = (history.shape[0], TARGET_DIM, future_steps)
    initial_noise = torch.randn(*shape, device=device)

    dt = estimate_dt(real_trajectory, history_steps)
    print(f"estimated per-step dt: {dt:.4f}s (used for velocity recompute and acceleration cap)")

    # First roll out the unguided factual/reference sample with the exact same
    # initial noise. This becomes the preservation anchor for the guided pass.
    reference_target, _ = diffusion.ddim_sample_loop(
        diffusion_model,
        shape=shape,
        noise=initial_noise.clone(),
        clip_denoised=False,
        cond_fn=None,
        model_kwargs={"history": history},
        device=device,
        progress=False,
        eta=args.eta,
    )
    reference_step = denorm_step_delta(reference_target.transpose(1, 2), target_mean.to(device), target_std.to(device))
    history_xy = build_position_history(real_trajectory, history_steps)
    reference_positions = reconstruct_future_positions(history_xy, reference_step)
    reference_trajectory = update_trajectory_positions(real_trajectory, reference_positions, history_steps)
    reference_trajectory = recompute_motion_features(reference_trajectory, history_steps, dt)

    preservation_dim_weight = parse_mask(args.preservation_dim_weight, TARGET_DIM, device)
    risk_dim_mask = parse_mask(args.risk_dim_mask, TARGET_DIM, device)
    grad_stats = {
        "raw_risk_norm": [],
        "raw_step_norm": [],
        "raw_accel_norm": [],
        "raw_turn_norm": [],
        "raw_map_norm": [],
        "post_risk_norm": [],
        "post_step_norm": [],
        "post_accel_norm": [],
        "post_turn_norm": [],
        "post_map_norm": [],
        "total_norm": [],
    }
    ego_cost = ego_rfw = adv_cost = adv_rfw = None
    if args.lambda_map > 0:
        print("[map] loading drivable rasters and building off-road cost fields ...")
        ego_cost, ego_rfw, adv_cost, adv_rfw = load_map_cost(
            batch["scene_id"], history_steps, future_steps, device
        )

    cond_fn = make_preservation_cond_fn(
        diffusion=diffusion,
        model=diffusion_model,
        classifier=classifier,
        history=history,
        real_trajectory=real_trajectory,
        reference_step=reference_step.detach(),
        target_mean=target_mean,
        target_std=target_std,
        history_steps=history_steps,
        guidance_target=args.guidance_target,
        guidance_scale=args.guidance_scale,
        lambda_step=args.lambda_step,
        lambda_accel=args.lambda_accel,
        accel_max=args.accel_max,
        lambda_turn=args.lambda_turn,
        turn_max=args.turn_max,
        lambda_map=args.lambda_map,
        ego_cost=ego_cost,
        ego_rfw=ego_rfw,
        adv_cost=adv_cost,
        adv_rfw=adv_rfw,
        dt=dt,
        preservation_dim_weight=preservation_dim_weight,
        risk_dim_mask=risk_dim_mask,
        risk_clip_norm=args.risk_clip_norm,
        step_clip_norm=args.step_clip_norm,
        time_mask_start=args.time_mask_start,
        time_mask_end=args.time_mask_end,
        grad_stats=grad_stats,
    )

    generated_target, _ = diffusion.ddim_sample_loop(
        diffusion_model,
        shape=shape,
        noise=initial_noise.clone(),
        clip_denoised=False,
        cond_fn=cond_fn,
        model_kwargs={"history": history},
        device=device,
        progress=False,
        eta=args.eta,
    )
    generated_step = denorm_step_delta(generated_target.transpose(1, 2), target_mean.to(device), target_std.to(device))
    generated_positions = reconstruct_future_positions(history_xy, generated_step)
    generated_trajectory = update_trajectory_positions(real_trajectory, generated_positions, history_steps)
    generated_trajectory = recompute_motion_features(generated_trajectory, history_steps, dt)

    with torch.no_grad():
        reference_logits = classifier(reference_trajectory).squeeze(-1)
        generated_logits = classifier(generated_trajectory).squeeze(-1)
        reference_probabilities = torch.sigmoid(reference_logits).cpu().numpy()
        generated_probabilities = torch.sigmoid(generated_logits).cpu().numpy()

    output = build_output_dict(
        scene_ids=batch["scene_id"],
        history_steps=history_steps,
        reference_step=reference_step,
        reference_trajectory=reference_trajectory,
        generated_step=generated_step,
        generated_trajectory=generated_trajectory,
        args=args,
    )
    output["reference_collision_probability"] = reference_probabilities
    output["collision_probability"] = generated_probabilities
    output["dt"] = np.asarray(dt, dtype=np.float32)
    output["raw_risk_grad_norm_mean"] = np.asarray(float(np.mean(grad_stats["raw_risk_norm"])), dtype=np.float32)
    output["raw_step_grad_norm_mean"] = np.asarray(float(np.mean(grad_stats["raw_step_norm"])), dtype=np.float32)
    output["post_risk_grad_norm_mean"] = np.asarray(float(np.mean(grad_stats["post_risk_norm"])), dtype=np.float32)
    output["post_step_grad_norm_mean"] = np.asarray(float(np.mean(grad_stats["post_step_norm"])), dtype=np.float32)
    output["total_grad_norm_mean"] = np.asarray(float(np.mean(grad_stats["total_norm"])), dtype=np.float32)
    output["raw_risk_grad_norm_max"] = np.asarray(float(np.max(grad_stats["raw_risk_norm"])), dtype=np.float32)
    output["raw_step_grad_norm_max"] = np.asarray(float(np.max(grad_stats["raw_step_norm"])), dtype=np.float32)
    output["post_risk_grad_norm_max"] = np.asarray(float(np.max(grad_stats["post_risk_norm"])), dtype=np.float32)
    output["post_step_grad_norm_max"] = np.asarray(float(np.max(grad_stats["post_step_norm"])), dtype=np.float32)
    output["total_grad_norm_max"] = np.asarray(float(np.max(grad_stats["total_norm"])), dtype=np.float32)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output, **output)
    print("reference collision probabilities:", reference_probabilities.tolist())
    print("guided collision probabilities:", generated_probabilities.tolist())
    print(
        "mean grad norms:",
        {
            "raw_risk": float(np.mean(grad_stats["raw_risk_norm"])),
            "raw_step": float(np.mean(grad_stats["raw_step_norm"])),
            "raw_accel": float(np.mean(grad_stats["raw_accel_norm"])),
            "raw_turn": float(np.mean(grad_stats["raw_turn_norm"])),
            "raw_map": float(np.mean(grad_stats["raw_map_norm"])),
            "post_risk": float(np.mean(grad_stats["post_risk_norm"])),
            "post_step": float(np.mean(grad_stats["post_step_norm"])),
            "post_accel": float(np.mean(grad_stats["post_accel_norm"])),
            "post_turn": float(np.mean(grad_stats["post_turn_norm"])),
            "post_map": float(np.mean(grad_stats["post_map_norm"])),
            "total": float(np.mean(grad_stats["total_norm"])),
        },
    )
    print(f"saved preservation-aware trajectories to {args.output}")


def build_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--classifier-path", type=str, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/trajectory_step/preservation/trajectory_step_diffusion_preservation_guided.npz"),
    )
    parser.add_argument("--label-map-path", type=str, default=None)
    parser.add_argument("--guidance-target", choices=["collision", "no_collision"], default="collision")
    parser.add_argument("--guidance-scale", type=float, default=2.0)
    parser.add_argument("--lambda-step", type=float, default=0.05)
    parser.add_argument("--lambda-accel", type=float, default=0.0,
                        help="Weight for the acceleration-feasibility hinge guidance term (0 disables it).")
    parser.add_argument("--accel-max", type=float, default=8.0,
                        help="Acceleration cap in m/s^2; only acceleration above this is penalized.")
    parser.add_argument("--lambda-turn", type=float, default=0.0,
                        help="Weight for the turn-angle-feasibility hinge guidance term (0 disables it).")
    parser.add_argument("--turn-max", type=float, default=0.3,
                        help="Per-step turn-angle cap in radians; only heading changes above this are penalized.")
    parser.add_argument("--lambda-map", type=float, default=0.0,
                        help="Weight for the map (off-road) guidance term (0 disables it; needs scipy for best gradients).")
    parser.add_argument(
        "--preservation-dim-weight",
        type=str,
        default="1.5,1.5,1.2,1.2",
        help="Comma-separated weights for [ego_dx, ego_dy, adv_dx, adv_dy] preservation losses.",
    )
    parser.add_argument(
        "--risk-dim-mask",
        type=str,
        default="0.2,0.2,1.0,1.0",
        help="Comma-separated risk-gradient mask for [ego_dx, ego_dy, adv_dx, adv_dy].",
    )
    parser.add_argument("--risk-clip-norm", type=float, default=1.0)
    parser.add_argument("--step-clip-norm", type=float, default=1.0)
    parser.add_argument("--time-mask-start", type=float, default=0.5)
    parser.add_argument("--time-mask-end", type=float, default=1.0)
    parser.add_argument("--num-samples", type=int, default=16)
    parser.add_argument("--eta", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    return parser


if __name__ == "__main__":
    main(build_argparser().parse_args())
