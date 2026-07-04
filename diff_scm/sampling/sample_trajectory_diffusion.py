"""
Minimal conditional sampling demo for trajectory diffusion.

This script loads the future diffusion checkpoint, conditions on real observed
history, generates future trajectories, and optionally scores them with the
pair-level collision classifier. It is the first counterfactual-ready sampling
surface for the trajectory Diff-SCM prototype.
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
from diff_scm.training.trajectory_diffusion_train import NormalizedFutureDataset, collate_batch, split_dataset
from diff_scm.utils.script_util import create_gaussian_diffusion


def load_diffusion_model(checkpoint_path: Path, config, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    history_steps = int(checkpoint["history_steps"])
    future_steps = int(checkpoint["future_steps"])
    model = TrajectoryFutureDenoiser(
        input_dim=config.trajectory_diffusion.input_dim,
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


def make_classifier_cond_fn(
    classifier: TrajectoryGRUBaseline,
    history: torch.Tensor,
    feature_mean: torch.Tensor,
    feature_std: torch.Tensor,
    guidance_target: str,
    classifier_scale: float,
):
    """
    Create classifier guidance for the DDIM denoising loop.

    The diffusion variable is only the normalized future [B, F, T_future].
    For guidance, we rebuild the full normalized trajectory by concatenating
    fixed history with the current generated future, denormalize it, run the
    collision classifier, and return d(logit)/d(future). The existing diffusion
    sampler uses this gradient to steer each denoising step.
    """
    feature_mean = feature_mean.to(history.device)
    feature_std = feature_std.to(history.device)
    direction = 1.0 if guidance_target == "collision" else -1.0

    def cond_fn(x, timesteps, **model_kwargs):
        del timesteps, model_kwargs
        with torch.enable_grad():
            future = x.detach().requires_grad_(True)
            full_normalized = torch.cat([history, future.transpose(1, 2)], dim=1)
            full_trajectory = full_normalized * feature_std + feature_mean

            # PyTorch/cuDNN RNNs cannot backpropagate in eval mode for input
            # gradients. Disabling cuDNN here keeps classifier dropout disabled
            # while still allowing guidance gradients through the GRU.
            with torch.backends.cudnn.flags(enabled=False):
                logits = classifier(full_trajectory).squeeze(-1)
            objective = direction * logits.sum()
            gradient = torch.autograd.grad(objective, future)[0]
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

    diffusion_model, checkpoint = load_diffusion_model(Path(args.model_path), config, device)
    diffusion = create_gaussian_diffusion(config)
    history_steps = int(checkpoint["history_steps"])
    feature_mean = checkpoint["feature_mean"].float().cpu()
    feature_std = checkpoint["feature_std"].float().cpu()

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
    sample_dataset = NormalizedFutureDataset(sample_subset, feature_mean, feature_std, history_steps)
    sample_loader = DataLoader(sample_dataset, batch_size=args.num_samples, shuffle=False, collate_fn=collate_batch)
    batch = next(iter(sample_loader))

    history = batch["history"].to(device)
    shape = (history.shape[0], config.trajectory_diffusion.input_dim, config.data.expected_timesteps - history_steps)

    classifier = None
    cond_fn = None
    if args.classifier_path is not None:
        classifier = load_classifier(Path(args.classifier_path), config, device)
        if args.classifier_scale != 0.0:
            cond_fn = make_classifier_cond_fn(
                classifier=classifier,
                history=history,
                feature_mean=feature_mean,
                feature_std=feature_std,
                guidance_target=args.guidance_target,
                classifier_scale=args.classifier_scale,
            )
    elif args.classifier_scale != 0.0:
        raise ValueError("--classifier-scale requires --classifier-path.")

    generated_future, _ = diffusion.ddim_sample_loop(
        diffusion_model,
        shape=shape,
        noise=torch.randn(*shape, device=device),
        clip_denoised=False,
        cond_fn=cond_fn,
        model_kwargs={"history": history},
        device=device,
        progress=False,
        eta=args.eta,
    )

    generated_future = generated_future.transpose(1, 2).cpu()
    history_cpu = history.cpu()
    generated_normalized = torch.cat([history_cpu, generated_future], dim=1)
    generated = generated_normalized * feature_std + feature_mean
    history_denormalized = history_cpu * feature_std + feature_mean
    generated_future_denormalized = generated_future * feature_std + feature_mean

    output = {
        "scene_id": np.asarray(batch["scene_id"], dtype=object),
        "history": history_denormalized.numpy(),
        "generated_future": generated_future_denormalized.numpy(),
        "generated_trajectory": generated.numpy(),
        "classifier_scale": np.asarray(args.classifier_scale, dtype=np.float32),
        "guidance_target": np.asarray(args.guidance_target, dtype=object),
    }

    if classifier is not None:
        with torch.no_grad():
            logits = classifier(generated.to(device)).squeeze(-1)
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
    parser.add_argument("--output", type=Path, default=Path("results/trajectory_absolute/trajectory_diffusion_samples.npz"))
    parser.add_argument("--label-map-path", type=str, default=None)
    parser.add_argument("--classifier-path", type=str, default=None)
    parser.add_argument("--classifier-scale", type=float, default=0.0)
    parser.add_argument("--guidance-target", choices=["collision", "no_collision"], default="collision")
    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument("--eta", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    return parser


if __name__ == "__main__":
    main(build_argparser().parse_args())
