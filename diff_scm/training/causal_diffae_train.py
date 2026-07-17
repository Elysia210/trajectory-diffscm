"""
Train CausalDiffAE on trajectories.

Objective (ported from CausalDiffAE): diffusion MSE on the predicted noise, plus a
representation term = KL(encoder posterior || N(0,1)) + label-conditioned prior KL that
pulls each causal latent toward its factor value. The causal latents are structured by
a DAG `A` over the chosen factor nodes.

Which factors are the DAG nodes is set by --factor-nodes; the adjacency starts at zeros
(or is learned with --learn-adjacency) until Baohua's causal graph is filled in.

Run from repo root:
    python -m diff_scm.training.causal_diffae_train \
        --data-path /mnt/h/trajectory_apr11/Apr11_relaxed_all_archives \
        --factor-npz labels/causaldiffae_factors.npz \
        --factor-nodes ego_speed_mean,adv_speed_mean,min_distance,collision
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.append(str(Path.cwd()))

import numpy as np
import torch
from torch.utils.data import DataLoader

from diff_scm.configs import get_config
from diff_scm.datasets.trajectory_dataset import TrajectoryDataset
from diff_scm.datasets.causal_trajectory_dataset import CausalTrajectoryDataset, collate_causal_batch
from diff_scm.models.resample import UniformSampler
from diff_scm.models.causal import (
    build_causal_traj_diffae,
    representation_loss,
    default_causal_adjacency,
)
from diff_scm.training.trajectory_diffusion_train import split_dataset
from diff_scm.training.trajectory_step_diffusion_train import compute_stats
from diff_scm.training.trajectory_classifier_train import compute_feature_stats
from diff_scm.training.trajectory_relative_diffusion_train import TARGET_DIM
from diff_scm.utils import logger
from diff_scm.utils.script_util import create_gaussian_diffusion


def train(args) -> None:
    config = get_config.file_from_dataset("trajectory")
    if args.data_path is not None:
        config.data.path = Path(args.data_path)
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    device = config.device

    logger.configure(Path(config.experiment_name) / args.run_name,
                     format_strs=["log", "stdout", "csv", "tensorboard"])

    dataset = TrajectoryDataset(
        data_path=config.data.path,
        expected_timesteps=config.data.expected_timesteps,
        require_labels=False,
        recursive=config.data.recursive,
        cache_in_memory=config.data.cache_in_memory,
        label_candidates=config.data.label_candidates,
    )
    history_steps = int(config.trajectory_diffusion.history_steps)
    future_steps = config.data.expected_timesteps - history_steps
    train_subset, val_subset, _ = split_dataset(dataset, config)

    # Normalization stats (train split): step-delta target + full-trajectory features.
    _, _, target_mean, target_std = compute_stats(dataset, train_subset, history_steps)
    feature_mean, feature_std = compute_feature_stats(dataset, train_subset)

    factor_nodes = [s.strip() for s in args.factor_nodes.split(",") if s.strip()]
    num_vars = len(factor_nodes)
    ds_kwargs = dict(
        factor_npz_path=Path(args.factor_npz), factor_nodes=factor_nodes,
        history_steps=history_steps, target_mean=target_mean, target_std=target_std,
        feature_mean=feature_mean, feature_std=feature_std,
    )
    train_ds = CausalTrajectoryDataset(train_subset, **ds_kwargs)
    factor_ranges = {n: (train_ds.fmin[i], train_ds.fmax[i]) for i, n in enumerate(factor_nodes)}
    val_ds = CausalTrajectoryDataset(val_subset, factor_ranges=factor_ranges, **ds_kwargs)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=config.data.num_workers, drop_last=True,
                              collate_fn=collate_causal_batch)

    # DAG adjacency: default to CausalDiffAE's example graph for this node count.
    adjacency = default_causal_adjacency(num_vars)
    logger.log(f"adjacency (A[i,j]=1 means i->j):\n{adjacency.int().tolist()}")

    diffusion = create_gaussian_diffusion(config)
    schedule_sampler = UniformSampler(diffusion)
    model = build_causal_traj_diffae(
        target_dim=TARGET_DIM, encode_dim=config.trajectory_diffusion.input_dim,
        num_vars=num_vars, per_var=args.per_var, hidden_dim=config.trajectory_diffusion.hidden_dim,
        num_layers=config.trajectory_diffusion.num_layers, dropout=config.trajectory_diffusion.dropout,
        time_embed_dim=config.trajectory_diffusion.time_embed_dim,
        adjacency=adjacency, learn_adjacency=args.learn_adjacency, causal_modeling=True,
        masking=args.masking, drop_prob=args.drop_prob,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    logger.log(f"train/val: {len(train_ds)}/{len(val_ds)} | factor nodes ({num_vars}): {factor_nodes}")
    logger.log(f"latent_dim: {num_vars * args.per_var} | history/future: {history_steps}/{future_steps}")
    logger.log(f"model params: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    best_checkpoint = Path(logger.get_dir()) / "best_model.pt"
    latest_checkpoint = Path(logger.get_dir()) / "last_model.pt"
    best_loss = float("inf")

    for epoch in range(args.epochs):
        model.train()
        kl_weight = args.kl_weight * min(1.0, (epoch + 1) / max(args.kl_warmup_epochs, 1))
        run_mse = run_kld = n = 0.0
        for batch in train_loader:
            x_start = batch["x_start"].to(device)      # [B, C, T_future]
            x_encode = batch["x_encode"].to(device)    # [B, T, F]
            c = batch["c"].to(device)                  # [B, num_vars]
            t, _ = schedule_sampler.sample(x_start.shape[0], device)

            noise = torch.randn_like(x_start)
            x_t = diffusion.q_sample(x_start, t, noise=noise)
            eps, aux = model(x_t, diffusion._scale_timesteps(t), x_encode)

            mse = ((noise - eps) ** 2).mean()
            kld = representation_loss(aux["mu"], aux["var"], aux["z_post"], c, num_vars,
                                      causal_modeling=True, mask=aux.get("mask"))
            loss = mse + kl_weight * kld

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            bs = x_start.shape[0]
            run_mse += float(mse.item()) * bs
            run_kld += float(kld.item()) * bs
            n += bs

        train_mse, train_kld = run_mse / n, run_kld / n
        logger.logkv("epoch", epoch + 1)
        logger.logkv("kl_weight", kl_weight)
        logger.logkv("train_mse", train_mse)
        logger.logkv("train_kld", train_kld)
        logger.logkv("train_loss", train_mse + kl_weight * train_kld)
        logger.dumpkvs()

        checkpoint = {
            "model": model.state_dict(), "config": config.to_dict(),
            "target_mean": target_mean, "target_std": target_std,
            "feature_mean": feature_mean, "feature_std": feature_std,
            "history_steps": history_steps, "future_steps": future_steps,
            "factor_nodes": factor_nodes, "per_var": args.per_var, "num_vars": num_vars,
            "factor_ranges": factor_ranges, "adjacency": model.causal_mask.A.detach().cpu(),
        }
        torch.save(checkpoint, latest_checkpoint)
        if train_mse + kl_weight * train_kld < best_loss:
            best_loss = train_mse + kl_weight * train_kld
            torch.save(checkpoint, best_checkpoint)

    logger.log(f"best checkpoint: {best_checkpoint}")


def build_argparser():
    p = argparse.ArgumentParser()
    p.add_argument("--data-path", type=str, default=None)
    p.add_argument("--factor-npz", type=str, default="labels/causaldiffae_factors.npz")
    p.add_argument("--factor-nodes", type=str,
                   default="ego_speed_mean,adv_speed_mean,min_distance,collision")
    p.add_argument("--per-var", type=int, default=16)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--kl-weight", type=float, default=1.0)
    p.add_argument("--kl-warmup-epochs", type=int, default=5)
    p.add_argument("--learn-adjacency", action="store_true")
    p.add_argument("--masking", action="store_true",
                   help="classifier-free training: randomly drop z so an unconditional branch is learned.")
    p.add_argument("--drop-prob", type=float, default=0.1)
    p.add_argument("--run-name", type=str, default="causal_diffae_train")
    return p


if __name__ == "__main__":
    train(build_argparser().parse_args())
