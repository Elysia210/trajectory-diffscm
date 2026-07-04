"""
Train the first trajectory-domain diffusion component.

This is the smallest step from the classifier baseline toward trajectory
Diff-SCM: condition on observed pair history and learn a diffusion model for
the future pair trajectory. It reuses the repository's GaussianDiffusion noise
process while keeping the sequence model separate from image-domain UNets.
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
from torch.utils.data import DataLoader, Dataset, Subset, random_split

from diff_scm.configs import get_config
from diff_scm.datasets.trajectory_dataset import TrajectoryDataset
from diff_scm.models.resample import UniformSampler
from diff_scm.models.trajectory_diffusion import TrajectoryFutureDenoiser
from diff_scm.utils import logger
from diff_scm.utils.script_util import create_gaussian_diffusion


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class NormalizedFutureDataset(Dataset):
    """
    Wrap TrajectoryDataset samples as normalized history/future diffusion pairs.

    The diffusion target uses [F, T_future] because the existing diffusion code
    expects channel-first tensors. The conditioning history remains [T_history, F].
    """

    def __init__(
        self,
        subset: Subset,
        feature_mean: torch.Tensor,
        feature_std: torch.Tensor,
        history_steps: int,
    ):
        self.subset = subset
        self.feature_mean = feature_mean.float()
        self.feature_std = feature_std.float()
        self.history_steps = history_steps

    def __len__(self) -> int:
        return len(self.subset)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        item = self.subset[index]
        trajectory = item["trajectory"].float()
        normalized = (trajectory - self.feature_mean) / self.feature_std
        history = normalized[: self.history_steps]
        future = normalized[self.history_steps :].transpose(0, 1)
        return {
            "future": future,
            "history": history,
            "scene_id": item["scene_id"],
        }


def split_dataset(dataset: TrajectoryDataset, config) -> Tuple[Subset, Subset, Subset]:
    train_ratio = float(config.data.train_ratio)
    val_ratio = float(config.data.val_ratio)
    test_ratio = float(config.data.test_ratio)
    ratio_sum = train_ratio + val_ratio + test_ratio
    if ratio_sum <= 0:
        raise ValueError("train/val/test ratios must sum to a positive number.")

    normalized = [train_ratio / ratio_sum, val_ratio / ratio_sum, test_ratio / ratio_sum]
    total_size = len(dataset)
    train_size = int(total_size * normalized[0])
    val_size = int(total_size * normalized[1])
    test_size = total_size - train_size - val_size
    if total_size > 0 and train_size == 0:
        train_size = 1
        if val_size > 0:
            val_size -= 1
        elif test_size > 0:
            test_size -= 1

    return random_split(
        dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(config.seed),
    )


def compute_feature_stats(dataset: TrajectoryDataset, subset: Subset) -> Tuple[torch.Tensor, torch.Tensor]:
    trajectories = [dataset[index]["trajectory"].float() for index in subset.indices]
    stacked = torch.stack(trajectories, dim=0)
    mean = stacked.reshape(-1, stacked.shape[-1]).mean(dim=0)
    std = stacked.reshape(-1, stacked.shape[-1]).std(dim=0).clamp_min(1e-6)
    return mean, std


def collate_batch(batch):
    return {
        "future": torch.stack([item["future"] for item in batch], dim=0),
        "history": torch.stack([item["history"] for item in batch], dim=0),
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
        Path(config.experiment_name) / "trajectory_diffusion_train",
        format_strs=["log", "stdout", "csv", "tensorboard"],
    )

    logger.log("building trajectory dataset for future diffusion...")
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
    if history_steps <= 0 or future_steps <= 0:
        raise ValueError("history_steps must leave at least one future step.")

    train_subset, val_subset, test_subset = split_dataset(dataset, config)
    feature_mean, feature_std = compute_feature_stats(dataset, train_subset)
    train_dataset = NormalizedFutureDataset(train_subset, feature_mean, feature_std, history_steps)
    val_dataset = NormalizedFutureDataset(val_subset, feature_mean, feature_std, history_steps)

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
        input_dim=config.trajectory_diffusion.input_dim,
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
    logger.log(f"history/future steps: {history_steps}/{future_steps}")
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
            "feature_mean": feature_mean,
            "feature_std": feature_std,
            "history_steps": history_steps,
            "future_steps": future_steps,
        }
        torch.save(checkpoint, latest_checkpoint)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(checkpoint, best_checkpoint)

    logger.log(f"best validation diffusion loss: {best_val_loss:.6f}")
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
