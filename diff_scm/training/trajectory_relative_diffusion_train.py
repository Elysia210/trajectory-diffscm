"""
Train relative/local future-motion diffusion for trajectory Diff-SCM.

This version fixes the main issue found in the absolute-coordinate prototype:
instead of denoising full future world-state features, it denoises only local
future position deltas for ego and adversary:

    [ego_dx, ego_dy, adv_dx, adv_dy]

Each delta is measured relative to that agent's last observed history position.
The final classifier/evaluation can reconstruct absolute trajectories later.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import random
import sys
from typing import Dict, Tuple

sys.path.append(str(Path.cwd()))

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset

from diff_scm.configs import get_config
from diff_scm.datasets.trajectory_dataset import TrajectoryDataset
from diff_scm.models.resample import UniformSampler
from diff_scm.models.trajectory_diffusion import TrajectoryFutureDenoiser
from diff_scm.training.trajectory_diffusion_train import split_dataset
from diff_scm.utils import logger
from diff_scm.utils.script_util import create_gaussian_diffusion


EGO_XY = slice(0, 2)
ADV_XY = slice(9, 11)
TARGET_DIM = 4


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_relative_target(trajectory: torch.Tensor, history_steps: int) -> torch.Tensor:
    """Return future local position deltas in [T_future, 4]."""
    ego_anchor = trajectory[history_steps - 1, EGO_XY]
    adv_anchor = trajectory[history_steps - 1, ADV_XY]
    future = trajectory[history_steps:]
    ego_delta = future[:, EGO_XY] - ego_anchor
    adv_delta = future[:, ADV_XY] - adv_anchor
    return torch.cat([ego_delta, adv_delta], dim=-1)


class RelativeFutureDataset(Dataset):
    """Wrap trajectory samples as normalized history and relative future targets."""

    def __init__(
        self,
        subset: Subset,
        history_mean: torch.Tensor,
        history_std: torch.Tensor,
        target_mean: torch.Tensor,
        target_std: torch.Tensor,
        history_steps: int,
    ):
        self.subset = subset
        self.history_mean = history_mean.float()
        self.history_std = history_std.float()
        self.target_mean = target_mean.float()
        self.target_std = target_std.float()
        self.history_steps = history_steps

    def __len__(self) -> int:
        return len(self.subset)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        item = self.subset[index]
        trajectory = item["trajectory"].float()
        history = (trajectory[: self.history_steps] - self.history_mean) / self.history_std
        target = build_relative_target(trajectory, self.history_steps)
        normalized_target = (target - self.target_mean) / self.target_std
        return {
            "future": normalized_target.transpose(0, 1),
            "history": history,
            "trajectory": trajectory,
            "scene_id": item["scene_id"],
        }


def compute_stats(dataset: TrajectoryDataset, subset: Subset, history_steps: int):
    histories = []
    targets = []
    for index in subset.indices:
        trajectory = dataset[index]["trajectory"].float()
        histories.append(trajectory[:history_steps])
        targets.append(build_relative_target(trajectory, history_steps))

    history_stack = torch.stack(histories, dim=0)
    target_stack = torch.stack(targets, dim=0)
    history_mean = history_stack.reshape(-1, history_stack.shape[-1]).mean(dim=0)
    history_std = history_stack.reshape(-1, history_stack.shape[-1]).std(dim=0).clamp_min(1e-6)
    target_mean = target_stack.reshape(-1, target_stack.shape[-1]).mean(dim=0)
    target_std = target_stack.reshape(-1, target_stack.shape[-1]).std(dim=0).clamp_min(1e-6)
    return history_mean, history_std, target_mean, target_std


def collate_batch(batch):
    return {
        "future": torch.stack([item["future"] for item in batch], dim=0),
        "history": torch.stack([item["history"] for item in batch], dim=0),
        "trajectory": torch.stack([item["trajectory"] for item in batch], dim=0),
        "scene_id": [item["scene_id"] for item in batch],
    }


def build_loader(dataset: Dataset, batch_size: int, shuffle: bool, num_workers: int) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=shuffle,
        collate_fn=collate_batch,
    )


def evaluate(model, diffusion, schedule_sampler, data_loader, device) -> float:
    model.eval()
    total_loss = 0.0
    total_samples = 0
    with torch.no_grad():
        for batch in data_loader:
            future = batch["future"].to(device)
            history = batch["history"].to(device)
            timesteps, weights = schedule_sampler.sample(future.shape[0], device)
            losses = diffusion.training_losses(model, future, timesteps, model_kwargs={"history": history})
            loss = (losses["loss"] * weights).mean()
            batch_size = future.shape[0]
            total_loss += float(loss.item()) * batch_size
            total_samples += batch_size
    model.train()
    return total_loss / max(total_samples, 1)


def train(args) -> None:
    config = get_config.file_from_dataset("trajectory")
    if args.data_path is not None:
        config.data.path = Path(args.data_path)
    if args.label_map_path is not None:
        config.data.label_map_path = Path(args.label_map_path)
    if args.epochs is not None:
        config.trajectory_diffusion.training.epochs = args.epochs
    if args.batch_size is not None:
        config.trajectory_diffusion.training.batch_size = args.batch_size

    set_seed(config.seed)
    device = config.device

    logger.configure(
        Path(config.experiment_name) / "trajectory_relative_diffusion_train",
        format_strs=["log", "stdout", "csv", "tensorboard"],
    )

    dataset = TrajectoryDataset(
        data_path=config.data.path,
        expected_timesteps=config.data.expected_timesteps,
        require_labels=False,
        recursive=config.data.recursive,
        cache_in_memory=config.data.cache_in_memory,
        label_candidates=config.data.label_candidates,
        label_map_path=config.data.label_map_path,
    )
    if len(dataset) == 0:
        raise RuntimeError("No trajectory samples were found.")

    history_steps = int(config.trajectory_diffusion.history_steps)
    future_steps = config.data.expected_timesteps - history_steps
    train_subset, val_subset, test_subset = split_dataset(dataset, config)
    history_mean, history_std, target_mean, target_std = compute_stats(dataset, train_subset, history_steps)

    train_dataset = RelativeFutureDataset(
        train_subset, history_mean, history_std, target_mean, target_std, history_steps
    )
    val_dataset = RelativeFutureDataset(
        val_subset, history_mean, history_std, target_mean, target_std, history_steps
    )
    train_loader = build_loader(
        train_dataset,
        batch_size=config.trajectory_diffusion.training.batch_size,
        shuffle=True,
        num_workers=config.data.num_workers,
    )
    val_loader = build_loader(
        val_dataset,
        batch_size=config.trajectory_diffusion.training.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
    )

    diffusion = create_gaussian_diffusion(config)
    schedule_sampler = UniformSampler(diffusion)
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
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.trajectory_diffusion.training.lr,
        weight_decay=config.trajectory_diffusion.training.weight_decay,
    )

    logger.log(f"indexed {len(dataset)} samples from {len(dataset.h5_files)} files")
    logger.log(f"train/val/test sizes: {len(train_subset)}/{len(val_subset)}/{len(test_subset)}")
    logger.log(f"relative target dim: {TARGET_DIM}; history/future steps: {history_steps}/{future_steps}")
    logger.log(f"model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    best_val_loss = float("inf")
    best_checkpoint = Path(logger.get_dir()) / "best_model.pt"
    latest_checkpoint = Path(logger.get_dir()) / "last_model.pt"

    for epoch in range(int(config.trajectory_diffusion.training.epochs)):
        model.train()
        running_loss = 0.0
        running_samples = 0
        for batch in train_loader:
            future = batch["future"].to(device)
            history = batch["history"].to(device)
            timesteps, weights = schedule_sampler.sample(future.shape[0], device)
            losses = diffusion.training_losses(model, future, timesteps, model_kwargs={"history": history})
            loss = (losses["loss"] * weights).mean()

            optimizer.zero_grad()
            loss.backward()
            if config.trajectory_diffusion.training.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.trajectory_diffusion.training.grad_clip)
            optimizer.step()

            batch_size = future.shape[0]
            running_loss += float(loss.item()) * batch_size
            running_samples += batch_size

        train_loss = running_loss / max(running_samples, 1)
        val_loss = evaluate(model, diffusion, schedule_sampler, val_loader, device)

        logger.logkv("epoch", epoch + 1)
        logger.logkv("train_loss", train_loss)
        logger.logkv("val_loss", val_loss)
        logger.dumpkvs()

        checkpoint = {
            "model": model.state_dict(),
            "config": config.to_dict(),
            "history_mean": history_mean,
            "history_std": history_std,
            "target_mean": target_mean,
            "target_std": target_std,
            "history_steps": history_steps,
            "future_steps": future_steps,
            "target_dim": TARGET_DIM,
            "target_type": "relative_future_xy",
        }
        torch.save(checkpoint, latest_checkpoint)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(checkpoint, best_checkpoint)

    logger.log(f"best validation relative diffusion loss: {best_val_loss:.6f}")
    logger.log(f"best checkpoint: {best_checkpoint}")
    logger.log(f"latest checkpoint: {latest_checkpoint}")


def build_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, default=None)
    parser.add_argument("--label-map-path", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    return parser


if __name__ == "__main__":
    train(build_argparser().parse_args())
