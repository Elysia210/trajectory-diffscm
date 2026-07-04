"""
Visualize unguided and classifier-guided trajectory diffusion samples.

The generated trajectory feature layout follows TrajectoryDataset:
ego x/y are columns 0:2 and adversary x/y are columns 9:11.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


EGO_XY = slice(0, 2)
ADV_XY = slice(9, 11)


def load_sample(path: Path):
    return np.load(path, allow_pickle=True)


def equalize_axis(ax, trajectories):
    points = np.concatenate(trajectories, axis=0)
    min_xy = points.min(axis=0)
    max_xy = points.max(axis=0)
    center = (min_xy + max_xy) * 0.5
    span = max(float((max_xy - min_xy).max()), 1.0)
    pad = span * 0.15
    ax.set_xlim(center[0] - span / 2 - pad, center[0] + span / 2 + pad)
    ax.set_ylim(center[1] - span / 2 - pad, center[1] + span / 2 + pad)
    ax.set_aspect("equal", adjustable="box")


def plot_mode(ax, data, sample_index: int, title: str, history_steps: int = 50):
    trajectory = data["generated_trajectory"][sample_index]
    probability = float(data["collision_probability"][sample_index])

    ego_history = trajectory[:history_steps, EGO_XY]
    adv_history = trajectory[:history_steps, ADV_XY]
    ego_future = trajectory[history_steps:, EGO_XY]
    adv_future = trajectory[history_steps:, ADV_XY]

    ax.plot(ego_history[:, 0], ego_history[:, 1], color="#1f77b4", linewidth=2.0, label="ego history")
    ax.plot(adv_history[:, 0], adv_history[:, 1], color="#d62728", linewidth=2.0, label="adv history")
    ax.plot(ego_future[:, 0], ego_future[:, 1], color="#1f77b4", linestyle="--", linewidth=2.0, label="ego future")
    ax.plot(adv_future[:, 0], adv_future[:, 1], color="#d62728", linestyle="--", linewidth=2.0, label="adv future")

    ax.scatter(ego_history[0, 0], ego_history[0, 1], color="#1f77b4", marker="o", s=18)
    ax.scatter(adv_history[0, 0], adv_history[0, 1], color="#d62728", marker="o", s=18)
    ax.scatter(ego_future[-1, 0], ego_future[-1, 1], color="#1f77b4", marker="x", s=32)
    ax.scatter(adv_future[-1, 0], adv_future[-1, 1], color="#d62728", marker="x", s=32)

    equalize_axis(ax, [ego_history, adv_history, ego_future, adv_future])
    ax.grid(True, linewidth=0.4, alpha=0.35)
    ax.set_title(f"{title}\np(collision)={probability:.3f}", fontsize=10)


def main(args):
    unguided = load_sample(args.unguided)
    collision = load_sample(args.collision)
    no_collision = load_sample(args.no_collision)

    sample_count = min(args.num_samples, len(unguided["scene_id"]))
    fig, axes = plt.subplots(sample_count, 3, figsize=(12, 3.4 * sample_count), squeeze=False)

    for row in range(sample_count):
        scene_id = str(unguided["scene_id"][row]).split(":")[-1]
        plot_mode(axes[row, 0], unguided, row, f"Unguided\n{scene_id}")
        plot_mode(axes[row, 1], collision, row, "Collision guided")
        plot_mode(axes[row, 2], no_collision, row, "No-collision guided")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180)
    print(f"saved visualization to {args.output}")


def build_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--unguided", type=Path, default=Path("results/trajectory_diffusion_unguided.npz"))
    parser.add_argument("--collision", type=Path, default=Path("results/trajectory_diffusion_collision_guided_scale2.npz"))
    parser.add_argument("--no-collision", type=Path, default=Path("results/trajectory_diffusion_no_collision_guided_scale2.npz"))
    parser.add_argument("--output", type=Path, default=Path("results/trajectory_diffusion_guidance_compare.png"))
    parser.add_argument("--num-samples", type=int, default=6)
    return parser


if __name__ == "__main__":
    main(build_argparser().parse_args())
