"""
Sanity diagnostics for generated trajectory futures.

This script compares real future motion against unguided and guided generated
futures using lightweight displacement metrics. It also writes a centered
trajectory comparison plot where each sample is translated so the last ego
history point is the origin.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Dict, List

sys.path.append(str(Path.cwd()))

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


EGO_XY = slice(0, 2)
ADV_XY = slice(9, 11)
HISTORY_STEPS = 50


def load_npz(path: Path):
    return np.load(path, allow_pickle=True)


def default_output_paths(args):
    """
    Infer a tidy output bundle when the caller does not specify paths.

    For `--single`, diagnostics land next to that NPZ. For trio comparisons,
    they land next to the unguided NPZ so each experiment folder keeps its own
    metrics and plots together.
    """
    if args.output_dir is not None:
        base_dir = args.output_dir
        stem = args.single.stem if args.single is not None else "trajectory_guidance_compare"
    elif args.single is not None:
        base_dir = args.single.parent
        stem = args.single.stem
    else:
        base_dir = args.unguided.parent
        stem = "trajectory_guidance_compare"

    output_csv = args.output_csv or (base_dir / f"{stem}_motion_metrics.csv")
    output_json = args.output_json or (base_dir / f"{stem}_motion_summary.json")
    output_hist = args.output_hist or (base_dir / f"{stem}_motion_hist.png")
    output_centered = args.output_centered or (base_dir / f"{stem}_centered.png")
    return output_csv, output_json, output_hist, output_centered


def split_scene_id(scene_id: str):
    file_path, scene_key = scene_id.split(":", 1)
    return Path(file_path), scene_key


def load_real_trajectory(scene_id: str) -> np.ndarray:
    file_path, scene_key = split_scene_id(scene_id)
    with h5py.File(file_path, "r") as h5_file:
        group = h5_file[scene_key]
        centroid = np.asarray(group["centroid"], dtype=np.float32)
        yaw = np.asarray(group["yaw"], dtype=np.float32)
        speed = np.asarray(group["curr_speed"], dtype=np.float32)
        extent = np.asarray(group["extent"], dtype=np.float32)

    from diff_scm.datasets.trajectory_dataset import TrajectoryDataset

    adversary_index = TrajectoryDataset.parse_adversary_index(scene_key)
    if adversary_index is None:
        raise ValueError(f"Could not parse adversary index from {scene_key}")

    class SceneView:
        def __getitem__(self, key):
            return {
                "centroid": centroid,
                "yaw": yaw,
                "curr_speed": speed,
                "extent": extent,
            }[key]

    return TrajectoryDataset.build_pair_features(SceneView(), adversary_index)


def motion_metrics(trajectory: np.ndarray, history_steps: int = HISTORY_STEPS) -> Dict[str, float]:
    future = trajectory[history_steps:]
    ego_xy = future[:, EGO_XY]
    adv_xy = future[:, ADV_XY]

    ego_step = np.linalg.norm(np.diff(ego_xy, axis=0), axis=-1)
    adv_step = np.linalg.norm(np.diff(adv_xy, axis=0), axis=-1)
    ego_endpoint = np.linalg.norm(ego_xy[-1] - trajectory[history_steps - 1, EGO_XY])
    adv_endpoint = np.linalg.norm(adv_xy[-1] - trajectory[history_steps - 1, ADV_XY])

    return {
        "ego_step_mean": float(ego_step.mean()),
        "ego_step_max": float(ego_step.max()),
        "ego_endpoint_disp": float(ego_endpoint),
        "adv_step_mean": float(adv_step.mean()),
        "adv_step_max": float(adv_step.max()),
        "adv_endpoint_disp": float(adv_endpoint),
    }


def summarize(values: List[float]) -> Dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "p90": float(np.percentile(array, 90)),
        "max": float(array.max()),
    }


def write_metric_csv(rows: List[Dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["sample_index", "scene_id", "mode", "collision_probability"] + [
        "ego_step_mean",
        "ego_step_max",
        "ego_endpoint_disp",
        "adv_step_mean",
        "adv_step_max",
        "adv_endpoint_disp",
    ]
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_metric_histograms(rows: List[Dict[str, object]], output: Path) -> None:
    metrics = ["ego_step_mean", "ego_step_max", "ego_endpoint_disp", "adv_step_mean", "adv_step_max", "adv_endpoint_disp"]
    modes = ["real", "unguided", "collision", "no_collision"]
    colors = {
        "real": "#222222",
        "unguided": "#4c78a8",
        "collision": "#e45756",
        "no_collision": "#54a24b",
    }
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    for ax, metric in zip(axes.reshape(-1), metrics):
        for mode in modes:
            values = [float(row[metric]) for row in rows if row["mode"] == mode]
            ax.hist(values, bins=16, alpha=0.35, color=colors[mode], label=mode)
        ax.set_title(metric)
        ax.grid(True, linewidth=0.4, alpha=0.3)
    axes[0, 0].legend(frameon=False)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)


def plot_centered_examples(datasets: Dict[str, np.lib.npyio.NpzFile], output: Path, num_samples: int) -> None:
    modes = ["unguided", "collision", "no_collision"]
    titles = {"unguided": "Unguided", "collision": "Collision guided", "no_collision": "No-collision guided"}
    sample_count = min(num_samples, len(datasets["unguided"]["scene_id"]))
    fig, axes = plt.subplots(sample_count, 3, figsize=(12, 3.3 * sample_count), squeeze=False)

    for row in range(sample_count):
        origin = datasets["unguided"]["history"][row, -1, EGO_XY]
        for col, mode in enumerate(modes):
            ax = axes[row, col]
            trajectory = datasets[mode]["generated_trajectory"][row].copy()
            trajectory[:, EGO_XY] -= origin
            trajectory[:, ADV_XY] -= origin
            probability = float(datasets[mode]["collision_probability"][row])
            ego_history = trajectory[:HISTORY_STEPS, EGO_XY]
            adv_history = trajectory[:HISTORY_STEPS, ADV_XY]
            ego_future = trajectory[HISTORY_STEPS:, EGO_XY]
            adv_future = trajectory[HISTORY_STEPS:, ADV_XY]

            ax.plot(ego_history[:, 0], ego_history[:, 1], color="#1f77b4", linewidth=1.8)
            ax.plot(adv_history[:, 0], adv_history[:, 1], color="#d62728", linewidth=1.8)
            ax.plot(ego_future[:, 0], ego_future[:, 1], color="#1f77b4", linestyle="--", linewidth=1.8)
            ax.plot(adv_future[:, 0], adv_future[:, 1], color="#d62728", linestyle="--", linewidth=1.8)
            ax.scatter([0.0], [0.0], color="#1f77b4", marker="o", s=22)
            points = np.concatenate([ego_history, adv_history, ego_future, adv_future], axis=0)
            center = (points.min(axis=0) + points.max(axis=0)) * 0.5
            span = max(float((points.max(axis=0) - points.min(axis=0)).max()), 1.0)
            pad = span * 0.15
            ax.set_xlim(center[0] - span / 2 - pad, center[0] + span / 2 + pad)
            ax.set_ylim(center[1] - span / 2 - pad, center[1] + span / 2 + pad)
            ax.set_aspect("equal", adjustable="box")
            ax.grid(True, linewidth=0.4, alpha=0.3)
            scene_key = str(datasets["unguided"]["scene_id"][row]).split(":")[-1]
            title = titles[mode] if col > 0 else f"{titles[mode]}\n{scene_key}"
            ax.set_title(f"{title}\np(collision)={probability:.3f}", fontsize=9)

    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)


def plot_single_centered_examples(
    dataset_name: str,
    data: np.lib.npyio.NpzFile,
    output: Path,
    num_samples: int,
) -> None:
    """Plot real future versus one generated future, centered at last ego history point."""
    sample_count = min(num_samples, len(data["scene_id"]))
    fig, axes = plt.subplots(sample_count, 2, figsize=(8.5, 3.3 * sample_count), squeeze=False)

    for row in range(sample_count):
        scene_id = str(data["scene_id"][row])
        real_trajectory = load_real_trajectory(scene_id)
        generated_trajectory = data["generated_trajectory"][row]
        origin = generated_trajectory[HISTORY_STEPS - 1, EGO_XY]

        for col, (title, trajectory) in enumerate((("Real future", real_trajectory), (dataset_name, generated_trajectory))):
            ax = axes[row, col]
            trajectory = trajectory.copy()
            trajectory[:, EGO_XY] -= origin
            trajectory[:, ADV_XY] -= origin
            ego_history = trajectory[:HISTORY_STEPS, EGO_XY]
            adv_history = trajectory[:HISTORY_STEPS, ADV_XY]
            ego_future = trajectory[HISTORY_STEPS:, EGO_XY]
            adv_future = trajectory[HISTORY_STEPS:, ADV_XY]

            ax.plot(ego_history[:, 0], ego_history[:, 1], color="#1f77b4", linewidth=1.8, label="ego history")
            ax.plot(adv_history[:, 0], adv_history[:, 1], color="#d62728", linewidth=1.8, label="adv history")
            ax.plot(ego_future[:, 0], ego_future[:, 1], color="#1f77b4", linestyle="--", linewidth=1.8, label="ego future")
            ax.plot(adv_future[:, 0], adv_future[:, 1], color="#d62728", linestyle="--", linewidth=1.8, label="adv future")
            ax.scatter([0.0], [0.0], color="#1f77b4", marker="o", s=22)

            points = np.concatenate([ego_history, adv_history, ego_future, adv_future], axis=0)
            center = (points.min(axis=0) + points.max(axis=0)) * 0.5
            span = max(float((points.max(axis=0) - points.min(axis=0)).max()), 1.0)
            pad = span * 0.15
            ax.set_xlim(center[0] - span / 2 - pad, center[0] + span / 2 + pad)
            ax.set_ylim(center[1] - span / 2 - pad, center[1] + span / 2 + pad)
            ax.set_aspect("equal", adjustable="box")
            ax.grid(True, linewidth=0.4, alpha=0.3)

            scene_key = scene_id.split(":")[-1]
            probability = ""
            if col == 1 and "collision_probability" in data:
                probability = f"\np(collision)={float(data['collision_probability'][row]):.3f}"
            heading = f"{title}\n{scene_key}" if col == 0 else title
            ax.set_title(f"{heading}{probability}", fontsize=9)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)


def main(args) -> None:
    output_csv, output_json, output_hist, output_centered = default_output_paths(args)

    if args.single is not None:
        datasets = {args.single_name: load_npz(args.single)}
    else:
        datasets = {
            "unguided": load_npz(args.unguided),
            "collision": load_npz(args.collision),
            "no_collision": load_npz(args.no_collision),
        }
    reference_name = next(iter(datasets))
    sample_count = min(args.num_samples, len(datasets[reference_name]["scene_id"]))

    rows: List[Dict[str, object]] = []
    for index in range(sample_count):
        scene_id = str(datasets[reference_name]["scene_id"][index])
        real_trajectory = load_real_trajectory(scene_id)
        real_row = {
            "sample_index": index,
            "scene_id": scene_id,
            "mode": "real",
            "collision_probability": "",
        }
        real_row.update(motion_metrics(real_trajectory))
        rows.append(real_row)

        for mode, data in datasets.items():
            row = {
                "sample_index": index,
                "scene_id": scene_id,
                "mode": mode,
                "collision_probability": float(data["collision_probability"][index]),
            }
            row.update(motion_metrics(data["generated_trajectory"][index]))
            rows.append(row)

    write_metric_csv(rows, output_csv)
    plot_metric_histograms(rows, output_hist)
    if args.single is None:
        plot_centered_examples(datasets, output_centered, min(args.num_plot_samples, sample_count))
    else:
        plot_single_centered_examples(
            reference_name,
            datasets[reference_name],
            output_centered,
            min(args.num_plot_samples, sample_count),
        )

    summary = {}
    for metric in ["ego_step_mean", "ego_step_max", "ego_endpoint_disp", "adv_step_mean", "adv_step_max", "adv_endpoint_disp"]:
        summary[metric] = {
            mode: summarize([float(row[metric]) for row in rows if row["mode"] == mode])
            for mode in ["real"] + list(datasets.keys())
        }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")

    print(f"saved metrics CSV to {output_csv}")
    print(f"saved summary JSON to {output_json}")
    print(f"saved histogram plot to {output_hist}")
    print(f"saved centered plot to {output_centered}")
    for metric, mode_summary in summary.items():
        print(metric)
        for mode, stats in mode_summary.items():
            print(f"  {mode}: mean={stats['mean']:.3f}, p90={stats['p90']:.3f}, max={stats['max']:.3f}")


def build_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--unguided", type=Path, default=Path("results/trajectory_absolute/trajectory_diffusion_unguided.npz"))
    parser.add_argument("--collision", type=Path, default=Path("results/trajectory_absolute/trajectory_diffusion_collision_guided_scale2.npz"))
    parser.add_argument("--no-collision", type=Path, default=Path("results/trajectory_absolute/trajectory_diffusion_no_collision_guided_scale2.npz"))
    parser.add_argument("--output-dir", type=Path, default=None, help="Optional directory that receives all derived outputs.")
    parser.add_argument("--output-csv", type=Path, default=None)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--output-hist", type=Path, default=None)
    parser.add_argument("--output-centered", type=Path, default=None)
    parser.add_argument("--single", type=Path, default=None, help="Optional single generated npz for real-vs-generated diagnostics.")
    parser.add_argument("--single-name", type=str, default="generated")
    parser.add_argument("--num-samples", type=int, default=16)
    parser.add_argument("--num-plot-samples", type=int, default=6)
    return parser


if __name__ == "__main__":
    main(build_argparser().parse_args())
