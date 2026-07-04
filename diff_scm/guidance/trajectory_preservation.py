"""
Preservation-aware guidance helpers for trajectory step-delta diffusion.

These utilities keep the sampling-time method narrow and explicit:
- denormalize step-delta predictions
- reconstruct future positions from per-step displacements
- rebuild the classifier trajectory features that depend on positions
- compute local preservation losses
- manipulate risk/preservation gradients before they are injected into DDIM
"""

from __future__ import annotations

from typing import Optional

import torch


EGO_XY = slice(0, 2)
ADV_XY = slice(9, 11)
REL_XY = slice(18, 20)
REL_DISTANCE = 22

# Scalar feature indices in the [T, 23] pair layout (see TrajectoryDataset.build_pair_features).
EGO_COS, EGO_SIN, EGO_VX, EGO_VY = 2, 3, 4, 5
ADV_COS, ADV_SIN, ADV_VX, ADV_VY = 11, 12, 13, 14
REL_VX, REL_VY = 20, 21


def denorm_step_delta(
    x_norm: torch.Tensor,
    target_mean: torch.Tensor,
    target_std: torch.Tensor,
) -> torch.Tensor:
    """
    Convert normalized step-delta predictions to physical units.

    Args:
        x_norm: [B, T_future, 4]
        target_mean/target_std: [4]
    """
    return x_norm * target_std.view(1, 1, -1) + target_mean.view(1, 1, -1)


def build_position_history(trajectory: torch.Tensor, history_steps: int) -> torch.Tensor:
    """
    Extract history xy for ego and adversary in a compact [B, T_hist, 4] form.
    """
    ego_history = trajectory[:, :history_steps, EGO_XY]
    adv_history = trajectory[:, :history_steps, ADV_XY]
    return torch.cat([ego_history, adv_history], dim=-1)


def reconstruct_future_positions(history_xy: torch.Tensor, future_step_delta: torch.Tensor) -> torch.Tensor:
    """
    Reconstruct future absolute positions from per-step displacement.

    Args:
        history_xy: [B, T_hist, 4] with [ego_x, ego_y, adv_x, adv_y]
        future_step_delta: [B, T_future, 4]

    Returns:
        [B, T_future, 4] future absolute positions in the same coordinate order.
    """
    anchor = history_xy[:, -1:, :]
    return anchor + torch.cumsum(future_step_delta, dim=1)


def update_trajectory_positions(
    base_trajectory: torch.Tensor,
    future_positions: torch.Tensor,
    history_steps: int,
) -> torch.Tensor:
    """
    Insert reconstructed future positions back into the full [B, T, 23] tensor.

    Only position-dependent features are updated here. Other future features
    remain copied from the reference trajectory, which is intentional for this
    position-only intervention prototype.
    """
    generated = base_trajectory.clone()
    ego_future = future_positions[:, :, 0:2]
    adv_future = future_positions[:, :, 2:4]

    generated[:, history_steps:, EGO_XY] = ego_future
    generated[:, history_steps:, ADV_XY] = adv_future

    relative_xy = adv_future - ego_future
    generated[:, history_steps:, REL_XY] = relative_xy
    generated[:, history_steps:, REL_DISTANCE] = torch.linalg.norm(relative_xy, dim=-1)
    return generated


def step_preservation_loss(
    step_pred: torch.Tensor,
    step_ref: torch.Tensor,
    dim_weight: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Penalize deviations from the unguided reference in per-step displacement.
    """
    diff = step_pred - step_ref
    if dim_weight is not None:
        diff = diff * dim_weight.view(1, 1, -1)
    return (diff ** 2).mean()


def endpoint_preservation_loss(
    pos_pred: torch.Tensor,
    pos_ref: torch.Tensor,
    dim_weight: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Penalize endpoint drift relative to the unguided reference rollout.
    """
    diff = pos_pred[:, -1] - pos_ref[:, -1]
    if dim_weight is not None:
        diff = diff * dim_weight.view(1, -1)
    return (diff ** 2).mean()


def smoothness_loss(step_pred: torch.Tensor) -> torch.Tensor:
    """
    Penalize high-frequency jitter in the predicted step sequence.
    """
    return ((step_pred[:, 1:] - step_pred[:, :-1]) ** 2).mean()


def estimate_dt(trajectory: torch.Tensor, history_steps: int, eps: float = 1e-3) -> float:
    """
    Estimate the per-step time interval (seconds) from the observed history.

    Uses the identity displacement = speed * dt: for each history step it divides
    the inter-step travel distance by the recorded speed magnitude and takes the
    median over moving steps. Falls back to 0.1s if no usable steps are found.
    """
    ego_xy = trajectory[:, :history_steps, EGO_XY]
    ego_v = trajectory[:, :history_steps, EGO_VX:EGO_VY + 1]
    disp = ego_xy[:, 1:] - ego_xy[:, :-1]
    step_dist = torch.linalg.norm(disp, dim=-1)
    speed = torch.linalg.norm(ego_v[:, :-1], dim=-1)
    mask = speed > eps
    if int(mask.sum()) == 0:
        return 0.1
    return float((step_dist[mask] / speed[mask]).median())


def recompute_motion_features(
    trajectory: torch.Tensor,
    history_steps: int,
    dt: float,
    stationary_eps: float = 1e-2,
) -> torch.Tensor:
    """
    Recompute heading (cos/sin yaw) and velocity (vx/vy) for the FUTURE steps from
    the already-updated future positions, so the classifier sees a self-consistent
    trajectory after a position-only intervention.

    Heading is the direction of per-step displacement; velocity is displacement/dt.
    Near-stationary steps keep their previous heading to avoid an undefined angle.
    Fully differentiable w.r.t. positions, and written without in-place ops so it
    is safe inside the guidance autograd graph.
    """
    history_block = trajectory[:, :history_steps, :]
    future = trajectory[:, history_steps:, :]
    end = trajectory.shape[1]

    def heading_vel(disp, old_cos, old_sin):
        dist = torch.linalg.norm(disp, dim=-1, keepdim=True)
        moving = dist > stationary_eps
        unit = disp / dist.clamp_min(stationary_eps)
        cos = torch.where(moving, unit[..., 0:1], old_cos)
        sin = torch.where(moving, unit[..., 1:2], old_sin)
        v = disp / dt
        return cos, sin, v[..., 0:1], v[..., 1:2]

    ego_disp = trajectory[:, history_steps:, EGO_XY] - trajectory[:, history_steps - 1:end - 1, EGO_XY]
    adv_disp = trajectory[:, history_steps:, ADV_XY] - trajectory[:, history_steps - 1:end - 1, ADV_XY]
    ego_cos, ego_sin, ego_vx, ego_vy = heading_vel(ego_disp, future[:, :, EGO_COS:EGO_COS + 1], future[:, :, EGO_SIN:EGO_SIN + 1])
    adv_cos, adv_sin, adv_vx, adv_vy = heading_vel(adv_disp, future[:, :, ADV_COS:ADV_COS + 1], future[:, :, ADV_SIN:ADV_SIN + 1])

    cols = [future[:, :, i:i + 1] for i in range(future.shape[-1])]
    cols[EGO_COS], cols[EGO_SIN], cols[EGO_VX], cols[EGO_VY] = ego_cos, ego_sin, ego_vx, ego_vy
    cols[ADV_COS], cols[ADV_SIN], cols[ADV_VX], cols[ADV_VY] = adv_cos, adv_sin, adv_vx, adv_vy
    cols[REL_VX], cols[REL_VY] = adv_vx - ego_vx, adv_vy - ego_vy
    future_new = torch.cat(cols, dim=-1)
    return torch.cat([history_block, future_new], dim=1)


def acceleration_hinge_loss(step_pred: torch.Tensor, dt: float, accel_max: float) -> torch.Tensor:
    """
    Feasibility cap on acceleration: only penalize acceleration magnitude above
    accel_max (m/s^2), so normal motion is free and teleport-like jumps are pushed
    back. step_pred is per-step displacement, so velocity = step/dt and
    acceleration = (step_t - step_{t-1}) / dt^2.
    """
    accel = (step_pred[:, 1:] - step_pred[:, :-1]) / (dt * dt)
    ego_accel = torch.linalg.norm(accel[..., 0:2], dim=-1)
    adv_accel = torch.linalg.norm(accel[..., 2:4], dim=-1)
    over = torch.relu(ego_accel - accel_max) ** 2 + torch.relu(adv_accel - accel_max) ** 2
    return over.mean()


def _turn_angle(v: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Unsigned heading change (radians) between consecutive displacement vectors.

    v: [B, T, 2] per-step displacement. Uses atan2(|cross|, dot) which is robust:
    no division, and degenerate (near-zero) steps give ~0 turn. Returns [B, T-1].
    """
    v1, v2 = v[:, :-1], v[:, 1:]
    cross = v1[..., 0] * v2[..., 1] - v1[..., 1] * v2[..., 0]
    dot = (v1 * v2).sum(dim=-1)
    return torch.atan2(cross.abs() + eps, dot)


def turn_angle_hinge_loss(step_pred: torch.Tensor, turn_max: float) -> torch.Tensor:
    """
    Feasibility cap on per-step turn angle (radians): only penalize heading changes
    above turn_max, so gentle steering is free and physically impossible sharp
    spins are pushed back. Operates on ego (0:2) and adversary (2:4) displacements.
    """
    ego_turn = _turn_angle(step_pred[..., 0:2])
    adv_turn = _turn_angle(step_pred[..., 2:4])
    over = torch.relu(ego_turn - turn_max) ** 2 + torch.relu(adv_turn - turn_max) ** 2
    return over.mean()


def _sample_offroad(pos: torch.Tensor, rfw: torch.Tensor, cost: torch.Tensor) -> torch.Tensor:
    """
    Bilinearly sample an off-road cost field at world positions, per future step.

    pos:  [B, T, 2] world xy of one agent's future
    rfw:  [B, T, 3, 3] raster_from_world for that agent/step (pixel = rfw @ [x,y,1])
    cost: [B, T, H, W] off-road cost (0 on the drivable area, growing off-road)
    Returns [B, T] sampled cost. Differentiable w.r.t. pos (via the pixel grid).
    """
    B, T = pos.shape[:2]
    H, W = cost.shape[-2:]
    homo = torch.cat([pos, torch.ones_like(pos[..., :1])], dim=-1)   # [B, T, 3]
    px = torch.einsum("btij,btj->bti", rfw, homo)                     # [B, T, 3]
    xy = px[..., :2] / px[..., 2:3].clamp_min(1e-6)                   # pixel (x, y)
    gx = 2.0 * xy[..., 0] / (W - 1) - 1.0
    gy = 2.0 * xy[..., 1] / (H - 1) - 1.0
    grid = torch.stack([gx, gy], dim=-1).reshape(B * T, 1, 1, 2)
    inp = cost.reshape(B * T, 1, H, W)
    sampled = torch.nn.functional.grid_sample(
        inp, grid, mode="bilinear", padding_mode="border", align_corners=True
    )
    return sampled.reshape(B, T)


def map_offroad_loss(
    ego_pos: torch.Tensor, ego_rfw: torch.Tensor, ego_cost: torch.Tensor,
    adv_pos: torch.Tensor, adv_rfw: torch.Tensor, adv_cost: torch.Tensor,
) -> torch.Tensor:
    """
    Penalize ego/adversary future positions that leave the drivable area, using the
    per-step agent-centric drivable rasters from the dataset. Pushes the trajectory
    back toward the road.
    """
    ego = _sample_offroad(ego_pos, ego_rfw, ego_cost)
    adv = _sample_offroad(adv_pos, adv_rfw, adv_cost)
    return (ego + adv).mean()


def grad_l2_norm_per_sample(g: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """
    Compute a per-sample L2 norm for gradients shaped [B, C, T].
    """
    flat = g.reshape(g.shape[0], -1)
    return torch.sqrt(torch.sum(flat * flat, dim=1) + eps)


def normalize_grad_per_sample(g: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """
    Normalize each sample gradient independently to unit norm.
    """
    norms = grad_l2_norm_per_sample(g, eps=eps).view(g.shape[0], *([1] * (g.ndim - 1)))
    return g / norms


def clip_grad_norm_per_sample(g: torch.Tensor, max_norm: float, eps: float = 1e-12) -> torch.Tensor:
    """
    Clip each sample gradient independently to a maximum L2 norm.
    """
    if max_norm is None or max_norm <= 0:
        return g
    norms = grad_l2_norm_per_sample(g, eps=eps)
    scale = torch.clamp(max_norm / (norms + eps), max=1.0)
    scale = scale.view(g.shape[0], *([1] * (g.ndim - 1)))
    return g * scale


def apply_feature_weight(g: torch.Tensor, weight) -> torch.Tensor:
    """
    Apply per-feature weights to gradients shaped [B, T, D] or [B, D, T].
    """
    if weight is None:
        return g
    w = torch.as_tensor(weight, dtype=g.dtype, device=g.device)
    if g.ndim != 3:
        raise ValueError(f"Expected gradient rank 3, got shape {tuple(g.shape)}")
    if g.shape[-1] == w.numel():
        return g * w.view(1, 1, -1)
    if g.shape[1] == w.numel():
        return g * w.view(1, -1, 1)
    raise ValueError(f"Could not align feature weight of shape {tuple(w.shape)} with gradient shape {tuple(g.shape)}")


def apply_feature_mask(g: torch.Tensor, mask) -> torch.Tensor:
    """
    Apply a per-feature mask to gradients shaped [B, T, D] or [B, D, T].
    """
    return apply_feature_weight(g, mask)


def apply_time_mask(g: torch.Tensor, start: float = 1.0, end: float = 1.0) -> torch.Tensor:
    """
    Apply a linear temporal mask over T steps. Intended for the risk branch.
    """
    if start == 1.0 and end == 1.0:
        return g
    if g.ndim != 3:
        raise ValueError(f"Expected gradient rank 3, got shape {tuple(g.shape)}")
    if g.shape[-1] <= g.shape[1]:
        steps = g.shape[-1]
        tm = torch.linspace(start, end, steps=steps, device=g.device, dtype=g.dtype).view(1, 1, steps)
    else:
        steps = g.shape[1]
        tm = torch.linspace(start, end, steps=steps, device=g.device, dtype=g.dtype).view(1, steps, 1)
    return g * tm


def apply_intervention_mask(
    grad: torch.Tensor,
    dim_mask: Optional[torch.Tensor] = None,
    time_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Backward-compatible helper for selective intervention masking.
    """
    grad = apply_feature_mask(grad, dim_mask)
    if time_mask is not None:
        grad = apply_time_mask(grad, float(time_mask[0]), float(time_mask[-1])) if time_mask.ndim == 1 else grad * time_mask
    return grad


# Backward-compatible aliases used by older samplers.
normalize_grad = normalize_grad_per_sample
