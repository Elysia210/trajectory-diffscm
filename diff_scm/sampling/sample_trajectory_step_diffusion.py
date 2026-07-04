"""
Sample the per-step future-motion trajectory diffusion model.

The model outputs ego/adversary step displacement. Sampling reconstructs
absolute future positions by cumulative sum from the last history position.
Non-position future features are copied from the real future for this targeted
sanity check.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.append(str(Path.cwd()))

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from diff_scm.configs import get_config
from diff_scm.datasets.trajectory_dataset import TrajectoryDataset
from diff_scm.models.trajectory_baseline import TrajectoryGRUBaseline
from diff_scm.models.trajectory_diffusion import TrajectoryFutureDenoiser
from diff_scm.training.trajectory_diffusion_train import split_dataset
from diff_scm.training.trajectory_relative_diffusion_train import ADV_XY, EGO_XY, TARGET_DIM
from diff_scm.training.trajectory_step_diffusion_train import StepFutureDataset, collate_batch
from diff_scm.utils.script_util import create_gaussian_diffusion

REL_XY = slice(18, 20)
REL_DISTANCE = 22


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


def reconstruct_positions(real_trajectory: torch.Tensor, generated_step: torch.Tensor, history_steps: int) -> torch.Tensor:
    """Cumulative-sum generated step deltas into absolute future positions."""
    generated_trajectory = real_trajectory.clone()
    ego_anchor = real_trajectory[:, history_steps - 1, EGO_XY]
    adv_anchor = real_trajectory[:, history_steps - 1, ADV_XY]

    ego_future = ego_anchor[:, None, :] + torch.cumsum(generated_step[:, :, 0:2], dim=1)
    adv_future = adv_anchor[:, None, :] + torch.cumsum(generated_step[:, :, 2:4], dim=1)
    generated_trajectory[:, history_steps:, EGO_XY] = ego_future
    generated_trajectory[:, history_steps:, ADV_XY] = adv_future

    # Keep the position-derived relative features consistent with the generated
    # ego/adversary positions. Other future features are intentionally copied
    # from the real trajectory in this position-only prototype.
    relative_xy = adv_future - ego_future
    generated_trajectory[:, history_steps:, REL_XY] = relative_xy
    generated_trajectory[:, history_steps:, REL_DISTANCE] = torch.linalg.norm(relative_xy, dim=-1)
    return generated_trajectory


def make_step_classifier_cond_fn(
    classifier: TrajectoryGRUBaseline,
    real_trajectory: torch.Tensor,
    target_mean: torch.Tensor,
    target_std: torch.Tensor,
    history_steps: int,
    guidance_target: str,
    classifier_scale: float,
):
    """
    Create classifier guidance for the step-delta DDIM denoising loop.

    The diffusion variable is the normalized future step displacement
    [B, 4, T_future]. At each denoising step we temporarily denormalize it,
    reconstruct absolute future positions by cumulative sum, score the full
    [B, 100, 23] trajectory with the collision classifier, and return the
    gradient with respect to the current normalized step sample.
    """
    target_mean = target_mean.to(real_trajectory.device)
    target_std = target_std.to(real_trajectory.device)
    direction = 1.0 if guidance_target == "collision" else -1.0

    def cond_fn(x, timesteps, **model_kwargs):
        del timesteps, model_kwargs
        with torch.enable_grad():
            normalized_step = x.detach().requires_grad_(True)
            generated_step = normalized_step.transpose(1, 2) * target_std + target_mean
            generated_trajectory = reconstruct_positions(real_trajectory, generated_step, history_steps)

            # cuDNN RNNs do not support eval-mode backward for input gradients.
            # Disabling cuDNN locally preserves deterministic classifier weights
            # while allowing the sampler to receive classifier gradients.
            with torch.backends.cudnn.flags(enabled=False):
                logits = classifier(generated_trajectory).squeeze(-1)
            objective = direction * logits.sum()
            gradient = torch.autograd.grad(objective, normalized_step)[0]
            return classifier_scale * gradient

    return cond_fn


def main(args) -> None:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    config = get_config.file_from_dataset("trajectory")
    config.data.path = Path(args.data_path)
    if args.label_map_path is not None:
        config.data.label_map_path = Path(args.label_map_path)
    device = config.device

    model, checkpoint = load_step_model(Path(args.model_path), config, device)
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
    real_trajectory = batch["trajectory"].float()
    real_trajectory_device = real_trajectory.to(device)
    shape = (history.shape[0], TARGET_DIM, future_steps)

    classifier = None
    cond_fn = None
    if args.classifier_path is not None:
        classifier = load_classifier(Path(args.classifier_path), config, device)
        if args.classifier_scale != 0.0:
            cond_fn = make_step_classifier_cond_fn(
                classifier=classifier,
                real_trajectory=real_trajectory_device,
                target_mean=target_mean,
                target_std=target_std,
                history_steps=history_steps,
                guidance_target=args.guidance_target,
                classifier_scale=args.classifier_scale,
            )
    elif args.classifier_scale != 0.0:
        raise ValueError("--classifier-scale requires --classifier-path.")

    generated_target, _ = diffusion.ddim_sample_loop(
        model,
        shape=shape,
        noise=torch.randn(*shape, device=device),
        clip_denoised=False,
        cond_fn=cond_fn,
        model_kwargs={"history": history},
        device=device,
        progress=False,
        eta=args.eta,
    )

    generated_step = generated_target.transpose(1, 2).cpu() * target_std + target_mean
    generated_trajectory = reconstruct_positions(real_trajectory, generated_step, history_steps)

    output = {
        "scene_id": np.asarray(batch["scene_id"], dtype=object),
        "history": generated_trajectory[:, :history_steps].numpy(),
        "generated_future": generated_trajectory[:, history_steps:].numpy(),
        "generated_trajectory": generated_trajectory.numpy(),
        "generated_step": generated_step.numpy(),
        "target_type": np.asarray("step_future_xy", dtype=object),
        "classifier_scale": np.asarray(args.classifier_scale, dtype=np.float32),
        "guidance_target": np.asarray(args.guidance_target, dtype=object),
    }

    if classifier is not None:
        with torch.no_grad():
            logits = classifier(generated_trajectory.to(device)).squeeze(-1)
            probabilities = torch.sigmoid(logits).cpu().numpy()
        output["collision_probability"] = probabilities
        print("generated collision probabilities:", probabilities.tolist())

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output, **output)
    print(f"saved generated trajectories to {args.output}")


def build_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/trajectory_step/guided/trajectory_step_diffusion_unguided.npz"),
    )
    parser.add_argument("--label-map-path", type=str, default=None)
    parser.add_argument("--classifier-path", type=str, default=None)
    parser.add_argument("--classifier-scale", type=float, default=0.0)
    parser.add_argument("--guidance-target", choices=["collision", "no_collision"], default="collision")
    parser.add_argument("--num-samples", type=int, default=16)
    parser.add_argument("--eta", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    return parser


if __name__ == "__main__":
    main(build_argparser().parse_args())
