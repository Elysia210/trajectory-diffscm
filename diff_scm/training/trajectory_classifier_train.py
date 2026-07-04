"""
Train a first-stage trajectory collision classifier.

This script intentionally stays separate from the image-domain classifier path:
- no diffusion noise injection
- pair-level trajectory input only
- weighted BCE for class imbalance
"""

from __future__ import annotations

import argparse
from pathlib import Path
import random
import sys
from typing import Dict, Optional, Tuple

sys.path.append(str(Path.cwd()))

import numpy as np
import torch
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, precision_score, recall_score
from torch import nn
from torch.utils.data import DataLoader, Subset, random_split

from diff_scm.configs import get_config
from diff_scm.datasets.trajectory_dataset import TrajectoryDataset
from diff_scm.models.trajectory_baseline import TrajectoryGRUBaseline
from diff_scm.utils import logger


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def collate_trajectory_batch(batch):
    trajectory = torch.stack([item["trajectory"] for item in batch], dim=0)
    if "y" not in batch[0]:
        raise ValueError("Training requires labeled trajectory samples, but the dataset batch has no 'y' field.")
    labels = torch.stack([item["y"] for item in batch], dim=0)
    return {
        "trajectory": trajectory,
        "y": labels,
        "scene_id": [item["scene_id"] for item in batch],
    }


def split_dataset(dataset: TrajectoryDataset, config) -> Tuple[Subset, Subset, Optional[Subset]]:
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
    if train_size + val_size + test_size != total_size:
        test_size = total_size - train_size - val_size

    train_subset, val_subset, test_subset = random_split(
        dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(config.seed),
    )
    return train_subset, val_subset, test_subset


def labels_from_subset(dataset: TrajectoryDataset, subset: Subset) -> np.ndarray:
    return np.asarray([dataset.records[index].label for index in subset.indices], dtype=np.float32)


def compute_pos_weight(train_labels: np.ndarray) -> torch.Tensor:
    positive = float((train_labels == 1).sum())
    negative = float((train_labels == 0).sum())
    if positive == 0:
        raise ValueError("Training split contains no positive collision labels.")
    if negative == 0:
        return torch.tensor(1.0, dtype=torch.float32)
    return torch.tensor(negative / positive, dtype=torch.float32)


def compute_feature_stats(dataset: TrajectoryDataset, subset: Subset) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Per-feature mean/std over the training split, used to standardize classifier
    inputs. Computed on the train subset only to avoid val/test leakage.
    """
    features = [dataset[index]["trajectory"].float() for index in subset.indices]
    stacked = torch.stack(features, dim=0)            # [N, T, F]
    flat = stacked.reshape(-1, stacked.shape[-1])      # [N*T, F]
    mean = flat.mean(dim=0)
    std = flat.std(dim=0).clamp_min(1e-6)
    return mean, std


def build_dataloader(subset: Subset, batch_size: int, shuffle: bool, num_workers: int) -> DataLoader:
    return DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=shuffle,
        collate_fn=collate_trajectory_batch,
    )


def compute_metrics(logits: torch.Tensor, labels: torch.Tensor, threshold: float) -> Dict[str, float]:
    probabilities = torch.sigmoid(logits).detach().cpu().numpy().reshape(-1)
    label_array = labels.detach().cpu().numpy().reshape(-1)
    predictions = (probabilities >= threshold).astype(np.float32)

    metrics = {
        "accuracy": float(accuracy_score(label_array, predictions)),
        "precision": float(precision_score(label_array, predictions, zero_division=0)),
        "recall": float(recall_score(label_array, predictions, zero_division=0)),
        "f1": float(f1_score(label_array, predictions, zero_division=0)),
    }
    try:
        metrics["pr_auc"] = float(average_precision_score(label_array, probabilities))
    except ValueError:
        metrics["pr_auc"] = float("nan")
    return metrics


def evaluate(
    model: nn.Module,
    data_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    threshold: float,
) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_samples = 0
    all_logits = []
    all_labels = []

    with torch.no_grad():
        for batch in data_loader:
            trajectory = batch["trajectory"].to(device)
            labels = batch["y"].to(device)
            logits = model(trajectory).squeeze(-1)
            loss = criterion(logits, labels)

            batch_size = trajectory.shape[0]
            total_loss += float(loss.item()) * batch_size
            total_samples += batch_size
            all_logits.append(logits)
            all_labels.append(labels)

    if total_samples == 0:
        return {"loss": float("nan"), "accuracy": float("nan"), "precision": float("nan"),
                "recall": float("nan"), "f1": float("nan"), "pr_auc": float("nan")}

    logits = torch.cat(all_logits, dim=0)
    labels = torch.cat(all_labels, dim=0)
    metrics = compute_metrics(logits, labels, threshold)
    metrics["loss"] = total_loss / total_samples
    return metrics


def train(args) -> None:
    config = get_config.file_from_dataset("trajectory")
    if args.data_path is not None:
        config.data.path = Path(args.data_path)
    if args.label_map_path is not None:
        config.data.label_map_path = Path(args.label_map_path)
    if args.epochs is not None:
        config.classifier.training.iterations = args.epochs
    if args.batch_size is not None:
        config.classifier.training.batch_size = args.batch_size
        config.sampling.batch_size = args.batch_size

    set_seed(config.seed)
    device = config.device

    logger.configure(
        Path(config.experiment_name) / "trajectory_classifier_train",
        format_strs=["log", "stdout", "csv", "tensorboard"],
    )

    logger.log("building trajectory dataset...")
    dataset = TrajectoryDataset(
        data_path=config.data.path,
        expected_timesteps=config.data.expected_timesteps,
        require_labels=True,
        recursive=config.data.recursive,
        cache_in_memory=config.data.cache_in_memory,
        label_candidates=config.data.label_candidates,
        label_map_path=config.data.label_map_path,
    )
    logger.log(f"indexed {len(dataset)} labeled samples from {len(dataset.h5_files)} files")
    logger.log(f"dataset stats: {dataset.stats}")
    if len(dataset) == 0:
        raise RuntimeError("No labeled trajectory samples were found. Please check label fields or add label candidates.")

    train_subset, val_subset, test_subset = split_dataset(dataset, config)
    train_labels = labels_from_subset(dataset, train_subset)
    pos_weight = (
        torch.tensor(float(config.classifier.training.pos_weight), dtype=torch.float32)
        if config.classifier.training.pos_weight is not None
        else compute_pos_weight(train_labels)
    )

    train_loader = build_dataloader(
        train_subset,
        batch_size=config.classifier.training.batch_size,
        shuffle=True,
        num_workers=config.data.num_workers,
    )
    val_loader = build_dataloader(
        val_subset,
        batch_size=config.classifier.training.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
    )

    logger.log("creating baseline model...")
    model = TrajectoryGRUBaseline(
        input_dim=config.classifier.input_dim,
        hidden_dim=config.classifier.hidden_dim,
        num_layers=config.classifier.num_layers,
        dropout=config.classifier.dropout,
        bidirectional=config.classifier.bidirectional,
    ).to(device)
    logger.log(f"model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    feature_mean, feature_std = compute_feature_stats(dataset, train_subset)
    model.feature_mean.copy_(feature_mean.to(device))
    model.feature_std.copy_(feature_std.to(device))
    logger.log(f"feature_mean[:4]: {feature_mean[:4].tolist()}")
    logger.log(f"feature_std[:4]: {feature_std[:4].tolist()}")

    logger.log(f"train/val/test sizes: {len(train_subset)}/{len(val_subset)}/{len(test_subset)}")
    logger.log(f"train pos_weight: {float(pos_weight.item()):.4f}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.classifier.training.lr,
        weight_decay=config.classifier.training.weight_decay,
    )
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))

    best_f1 = -1.0
    best_checkpoint = Path(logger.get_dir()) / "best_model.pt"
    latest_checkpoint = Path(logger.get_dir()) / "last_model.pt"

    logger.log("training trajectory classifier...")
    for epoch in range(int(config.classifier.training.iterations)):
        model.train()
        running_loss = 0.0
        running_samples = 0

        for batch in train_loader:
            trajectory = batch["trajectory"].to(device)
            labels = batch["y"].to(device)

            optimizer.zero_grad()
            logits = model(trajectory).squeeze(-1)
            loss = criterion(logits, labels)
            loss.backward()
            if config.classifier.training.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.classifier.training.grad_clip)
            optimizer.step()

            batch_size = trajectory.shape[0]
            running_loss += float(loss.item()) * batch_size
            running_samples += batch_size

        train_loss = running_loss / max(running_samples, 1)
        val_metrics = evaluate(
            model=model,
            data_loader=val_loader,
            criterion=criterion,
            device=device,
            threshold=config.classifier.training.threshold,
        )

        logger.logkv("epoch", epoch + 1)
        logger.logkv("train_loss", train_loss)
        logger.logkv("val_loss", val_metrics["loss"])
        logger.logkv("val_accuracy", val_metrics["accuracy"])
        logger.logkv("val_precision", val_metrics["precision"])
        logger.logkv("val_recall", val_metrics["recall"])
        logger.logkv("val_f1", val_metrics["f1"])
        logger.logkv("val_pr_auc", val_metrics["pr_auc"])
        logger.dumpkvs()

        torch.save({"model": model.state_dict(), "config": config.to_dict()}, latest_checkpoint)
        if val_metrics["f1"] > best_f1:
            best_f1 = val_metrics["f1"]
            torch.save({"model": model.state_dict(), "config": config.to_dict()}, best_checkpoint)

    logger.log(f"best validation f1: {best_f1:.4f}")
    logger.log(f"best checkpoint: {best_checkpoint}")
    logger.log(f"latest checkpoint: {latest_checkpoint}")


def build_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, default=None, help="Trajectory HDF5 root directory.")
    parser.add_argument("--label-map-path", type=str, default=None, help="Optional JSON/CSV label manifest.")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    return parser


if __name__ == "__main__":
    train(build_argparser().parse_args())
