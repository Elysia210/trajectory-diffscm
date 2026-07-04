"""
Sample the relative/local future-motion trajectory diffusion model.

The model generates only future ego/adversary xy deltas. For this first
physical-realism sanity pass, non-position future features are copied from the
real future so we can isolate whether local position diffusion fixes the large
absolute-coordinate drift.
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
from diff_scm.training.trajectory_relative_diffusion_train import (
    ADV_XY,
    EGO_XY,
    TARGET_DIM,
    RelativeFutureDataset,
    collate_batch,
)
from diff_scm.utils.script_util import create_gaussian_diffusion


def load_relative_model(checkpoint_path: Path, config, device: torch.device):
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


def main(args) -> None:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    config = get_config.file_from_dataset("trajectory")
    config.data.path = Path(args.data_path)
    if args.label_map_path is not None:
        config.data.label_map_path = Path(args.label_map_path)
    device = config.device

    model, checkpoint = load_relative_model(Path(args.model_path), config, device)
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
    sample_dataset = RelativeFutureDataset(sample_subset, history_mean, history_std, target_mean, target_std, history_steps)
    sample_loader = DataLoader(sample_dataset, batch_size=args.num_samples, shuffle=False, collate_fn=collate_batch)
    batch = next(iter(sample_loader))

    history = batch["history"].to(device)
    shape = (history.shape[0], TARGET_DIM, future_steps)
    generated_target, _ = diffusion.ddim_sample_loop(
        model,
        shape=shape,
        noise=torch.randn(*shape, device=device),
        clip_denoised=False,
        model_kwargs={"history": history},
        device=device,
        progress=False,
        eta=args.eta,
    )

    generated_delta = generated_target.transpose(1, 2).cpu() * target_std + target_mean
    real_trajectory = batch["trajectory"].float()
    generated_trajectory = real_trajectory.clone()

    ego_anchor = real_trajectory[:, history_steps - 1, EGO_XY]
    adv_anchor = real_trajectory[:, history_steps - 1, ADV_XY]
    generated_trajectory[:, history_steps:, EGO_XY] = ego_anchor[:, None, :] + generated_delta[:, :, 0:2]
    generated_trajectory[:, history_steps:, ADV_XY] = adv_anchor[:, None, :] + generated_delta[:, :, 2:4]

    output = {
        "scene_id": np.asarray(batch["scene_id"], dtype=object),
        "history": generated_trajectory[:, :history_steps].numpy(),
        "generated_future": generated_trajectory[:, history_steps:].numpy(),
        "generated_trajectory": generated_trajectory.numpy(),
        "generated_delta": generated_delta.numpy(),
        "target_type": np.asarray("relative_future_xy", dtype=object),
    }

    if args.classifier_path is not None:
        classifier = load_classifier(Path(args.classifier_path), config, device)
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
    parser.add_argument("--output", type=Path, default=Path("results/trajectory_relative_diffusion_unguided.npz"))
    parser.add_argument("--label-map-path", type=str, default=None)
    parser.add_argument("--classifier-path", type=str, default=None)
    parser.add_argument("--num-samples", type=int, default=16)
    parser.add_argument("--eta", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    return parser


if __name__ == "__main__":
    main(build_argparser().parse_args())
