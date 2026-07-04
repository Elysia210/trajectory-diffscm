"""
Create a simple animated GIF for generated trajectory samples.

This script is intended for quick qualitative review of trajectory diffusion
outputs saved as ``.npz`` bundles. It can compare one or more modes side by
side, for example:

- unguided
- collision guided
- no-collision guided

The trajectory feature layout follows ``TrajectoryDataset``:
- ego x/y are columns 0:2
- adversary x/y are columns 9:11
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
from matplotlib import animation
import matplotlib.pyplot as plt
import numpy as np


EGO_XY = slice(0, 2)
ADV_XY = slice(9, 11)


def load_mode(path: Path):
    return np.load(path, allow_pickle=True)


def infer_history_steps(data, fallback: int) -> int:
    if "history" in data.files:
        history = data["history"]
        if history.ndim >= 3:
            return int(history.shape[1])
    return fallback


def validate_scene_alignment(datasets: Sequence[np.lib.npyio.NpzFile]) -> None:
    if not datasets:
        raise ValueError("No datasets were provided.")
    reference_ids = datasets[0]["scene_id"]
    for index, dataset in enumerate(datasets[1:], start=1):
        if len(dataset["scene_id"]) != len(reference_ids):
            raise ValueError(f"Dataset {index} has a different number of samples.")
        if not np.array_equal(dataset["scene_id"], reference_ids):
            raise ValueError(f"Dataset {index} does not align with the reference scene ordering.")


def centered_trajectory(trajectory: np.ndarray, history_steps: int, origin: np.ndarray) -> np.ndarray:
    centered = trajectory.copy()
    centered[:, EGO_XY] -= origin
    centered[:, ADV_XY] -= origin
    return centered


def compute_axis_limits(
    trajectories: Sequence[np.ndarray],
    history_steps: int,
    centered: bool,
) -> Tuple[float, float, float, float]:
    aligned = []
    for trajectory in trajectories:
        if centered:
            origin = trajectory[history_steps - 1, EGO_XY]
            aligned.append(centered_trajectory(trajectory, history_steps, origin))
        else:
            aligned.append(trajectory)

    points = np.concatenate(
        [np.concatenate([trajectory[:, EGO_XY], trajectory[:, ADV_XY]], axis=0) for trajectory in aligned],
        axis=0,
    )
    min_xy = points.min(axis=0)
    max_xy = points.max(axis=0)
    center = (min_xy + max_xy) * 0.5
    span = max(float((max_xy - min_xy).max()), 1.0)
    pad = span * 0.18
    return (
        center[0] - span / 2 - pad,
        center[0] + span / 2 + pad,
        center[1] - span / 2 - pad,
        center[1] + span / 2 + pad,
    )


def split_segments(trajectory: np.ndarray, history_steps: int, frame_index: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # frame_index is inclusive in the full [T] trajectory timeline.
    if frame_index < history_steps:
        ego_history = trajectory[: frame_index + 1, EGO_XY]
        adv_history = trajectory[: frame_index + 1, ADV_XY]
        ego_future = trajectory[0:0, EGO_XY]
        adv_future = trajectory[0:0, ADV_XY]
    else:
        future_index = frame_index - history_steps + 1
        ego_history = trajectory[:history_steps, EGO_XY]
        adv_history = trajectory[:history_steps, ADV_XY]
        ego_future = trajectory[history_steps : history_steps + future_index, EGO_XY]
        adv_future = trajectory[history_steps : history_steps + future_index, ADV_XY]
    return ego_history, adv_history, ego_future, adv_future


def build_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", type=Path, nargs="+", required=True, help="One or more trajectory npz files.")
    parser.add_argument("--labels", nargs="+", default=None, help="Display labels matching --inputs.")
    parser.add_argument("--output", type=Path, required=True, help="Output animation path. Use .gif for Pillow export.")
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--history-steps", type=int, default=50)
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument("--dpi", type=int, default=140)
    parser.add_argument("--centered", action="store_true", help="Center each mode at the last ego history point.")
    parser.add_argument("--title", type=str, default="Trajectory Diffusion Animation")
    return parser


def main(args) -> None:
    datasets = [load_mode(path) for path in args.inputs]
    validate_scene_alignment(datasets)

    if args.labels is None:
        labels = [path.stem for path in args.inputs]
    else:
        if len(args.labels) != len(args.inputs):
            raise ValueError("--labels must match the number of --inputs.")
        labels = args.labels

    history_steps = infer_history_steps(datasets[0], args.history_steps)
    total_steps = int(datasets[0]["generated_trajectory"].shape[1])

    if args.sample_index < 0 or args.sample_index >= len(datasets[0]["scene_id"]):
        raise IndexError(f"sample index {args.sample_index} is out of range.")

    scene_id = str(datasets[0]["scene_id"][args.sample_index])
    scene_name = scene_id.split(":")[-1]

    trajectories = [dataset["generated_trajectory"][args.sample_index].copy() for dataset in datasets]
    x_min, x_max, y_min, y_max = compute_axis_limits(trajectories, history_steps, args.centered)

    figure, axes = plt.subplots(1, len(datasets), figsize=(4.2 * len(datasets), 4.6), squeeze=False)
    axes = axes[0]

    artists: List[Tuple] = []
    for ax, dataset, label, trajectory in zip(axes, datasets, labels, trajectories):
        probability = dataset["collision_probability"][args.sample_index] if "collision_probability" in dataset else None
        if args.centered:
            origin = trajectory[history_steps - 1, EGO_XY]
            trajectory[:] = centered_trajectory(trajectory, history_steps, origin)

        ego_history_line, = ax.plot([], [], color="#1f77b4", linewidth=2.0)
        adv_history_line, = ax.plot([], [], color="#d62728", linewidth=2.0)
        ego_future_line, = ax.plot([], [], color="#1f77b4", linestyle="--", linewidth=2.0)
        adv_future_line, = ax.plot([], [], color="#d62728", linestyle="--", linewidth=2.0)
        ego_dot, = ax.plot([], [], marker="o", color="#1f77b4", markersize=6)
        adv_dot, = ax.plot([], [], marker="o", color="#d62728", markersize=6)

        title = label
        if probability is not None:
            title = f"{label}\np(collision)={float(probability):.3f}"
        ax.set_title(title, fontsize=10)
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, linewidth=0.4, alpha=0.35)
        ax.set_xlabel("x")
        ax.set_ylabel("y")

        artists.append(
            (
                trajectory,
                ego_history_line,
                adv_history_line,
                ego_future_line,
                adv_future_line,
                ego_dot,
                adv_dot,
            )
        )

    suptitle = figure.suptitle(f"{args.title}\n{scene_name}", fontsize=12)
    figure.tight_layout(rect=(0, 0, 1, 0.92))

    def update(frame_index: int):
        for trajectory, ego_history_line, adv_history_line, ego_future_line, adv_future_line, ego_dot, adv_dot in artists:
            ego_history, adv_history, ego_future, adv_future = split_segments(trajectory, history_steps, frame_index)

            ego_history_line.set_data(ego_history[:, 0], ego_history[:, 1])
            adv_history_line.set_data(adv_history[:, 0], adv_history[:, 1])
            ego_future_line.set_data(ego_future[:, 0], ego_future[:, 1])
            adv_future_line.set_data(adv_future[:, 0], adv_future[:, 1])

            ego_current = trajectory[frame_index, EGO_XY]
            adv_current = trajectory[frame_index, ADV_XY]
            ego_dot.set_data([ego_current[0]], [ego_current[1]])
            adv_dot.set_data([adv_current[0]], [adv_current[1]])

        frame_stage = "history" if frame_index < history_steps else "future"
        suptitle.set_text(f"{args.title}\n{scene_name} | frame {frame_index + 1}/{total_steps} ({frame_stage})")
        return figure.axes

    anim = animation.FuncAnimation(
        figure,
        update,
        frames=total_steps,
        interval=int(1000 / max(args.fps, 1)),
        blit=False,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.suffix.lower() == ".gif":
        writer = animation.PillowWriter(fps=args.fps)
    else:
        writer = animation.FFMpegWriter(fps=args.fps)
    anim.save(str(args.output), writer=writer, dpi=args.dpi)
    plt.close(figure)
    print(f"saved animation to {args.output}")


if __name__ == "__main__":
    main(build_argparser().parse_args())
