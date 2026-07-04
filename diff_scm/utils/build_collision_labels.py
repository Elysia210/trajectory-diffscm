"""
Build binary ego/adversary collision labels from trajectory HDF5 files.

This script is intentionally independent from the training entry points. It
turns raw trajectory state into a CSV label manifest that can be passed to the
first-stage trajectory classifier via ``--label-map-path``.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import re
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

try:
    import h5py
except ImportError:  # pragma: no cover - checked at runtime in the user env.
    h5py = None


CTRL_PATTERN = re.compile(r"ctrl_\[(\d+)\]")
REQUIRED_FIELDS = ("centroid", "yaw", "extent")


def require_h5py() -> None:
    if h5py is None:
        raise ImportError(
            "build_collision_labels.py requires h5py. Run it in the Diff-SCM "
            "environment that can already read the trajectory HDF5 files."
        )


def discover_h5_files(data_path: Path, recursive: bool = True) -> List[Path]:
    if data_path.is_file():
        return [data_path]
    if not data_path.exists():
        raise FileNotFoundError(f"Trajectory data path does not exist: {data_path}")

    patterns = ("**/*.h5", "**/*.hdf5") if recursive else ("*.h5", "*.hdf5")
    files: List[Path] = []
    for pattern in patterns:
        files.extend(path for path in data_path.glob(pattern) if path.is_file())
    return sorted(files)


def parse_adversary_index(scene_key: str) -> Optional[int]:
    match = CTRL_PATTERN.search(scene_key)
    if match is None:
        return None
    return int(match.group(1))


def validate_scene_shape(scene_group: "h5py.Group") -> Tuple[bool, Optional[str]]:
    centroid = scene_group["centroid"]
    yaw = scene_group["yaw"]
    extent = scene_group["extent"]

    if centroid.ndim != 3 or centroid.shape[-1] != 2:
        return False, "skipped_bad_shapes"
    if yaw.ndim != 2:
        return False, "skipped_bad_shapes"
    if extent.ndim != 3 or extent.shape[-1] < 2:
        return False, "skipped_bad_shapes"

    n_agents, timesteps, _ = centroid.shape
    if yaw.shape != (n_agents, timesteps):
        return False, "skipped_bad_shapes"
    if extent.shape[0] != n_agents or extent.shape[1] != timesteps:
        return False, "skipped_bad_shapes"
    return True, None


def validate_pair_values(
    ego_centers: np.ndarray,
    adv_centers: np.ndarray,
    ego_yaws: np.ndarray,
    adv_yaws: np.ndarray,
    ego_sizes: np.ndarray,
    adv_sizes: np.ndarray,
) -> bool:
    arrays = (ego_centers, adv_centers, ego_yaws, adv_yaws, ego_sizes, adv_sizes)
    if not all(np.isfinite(array).all() for array in arrays):
        return False
    if (ego_sizes <= 0).any() or (adv_sizes <= 0).any():
        return False
    return True


def rotated_box_corners(center: np.ndarray, yaw: float, size: np.ndarray) -> np.ndarray:
    """
    Return four BEV rectangle corners for one agent.

    ``size`` is (length, width). The local rectangle is centered at the origin
    with length on the forward x-axis and width on the lateral y-axis, then
    rotated by yaw radians and translated to the world-frame centroid.
    """
    half_length = size[0] * 0.5
    half_width = size[1] * 0.5
    local_corners = np.asarray(
        [
            [half_length, half_width],
            [half_length, -half_width],
            [-half_length, -half_width],
            [-half_length, half_width],
        ],
        dtype=np.float64,
    )

    cos_yaw = np.cos(yaw)
    sin_yaw = np.sin(yaw)
    rotation = np.asarray([[cos_yaw, -sin_yaw], [sin_yaw, cos_yaw]], dtype=np.float64)
    return local_corners @ rotation.T + center


def project_polygon(corners: np.ndarray, axis: np.ndarray) -> Tuple[float, float]:
    projections = corners @ axis
    return float(projections.min()), float(projections.max())


def rotated_rectangles_overlap(corners_a: np.ndarray, corners_b: np.ndarray, epsilon: float = 1e-9) -> bool:
    """
    Test two rotated rectangles with the Separating Axis Theorem.

    For rectangles, it is sufficient to project both boxes onto the two unique
    edge normals from each rectangle. If any axis has disjoint projections, the
    rectangles are separated; otherwise they overlap in BEV.
    """
    axes = []
    for corners in (corners_a, corners_b):
        for edge_index in (0, 1):
            edge = corners[(edge_index + 1) % 4] - corners[edge_index]
            axis = np.asarray([-edge[1], edge[0]], dtype=np.float64)
            norm = np.linalg.norm(axis)
            if norm <= epsilon:
                return False
            axes.append(axis / norm)

    for axis in axes:
        min_a, max_a = project_polygon(corners_a, axis)
        min_b, max_b = project_polygon(corners_b, axis)
        if max_a < min_b - epsilon or max_b < min_a - epsilon:
            return False
    return True


def first_collision_timestep(
    ego_centers: np.ndarray,
    adv_centers: np.ndarray,
    ego_yaws: np.ndarray,
    adv_yaws: np.ndarray,
    ego_sizes: np.ndarray,
    adv_sizes: np.ndarray,
) -> int:
    for timestep in range(ego_centers.shape[0]):
        ego_corners = rotated_box_corners(ego_centers[timestep], float(ego_yaws[timestep]), ego_sizes[timestep])
        adv_corners = rotated_box_corners(adv_centers[timestep], float(adv_yaws[timestep]), adv_sizes[timestep])
        if rotated_rectangles_overlap(ego_corners, adv_corners):
            return timestep
    return -1


def initial_stats() -> Dict[str, int]:
    return {
        "files_scanned": 0,
        "scenes_seen": 0,
        "scenes_processed": 0,
        "scenes_labeled_collision": 0,
        "scenes_labeled_noncollision": 0,
        "skipped_invalid_ctrl": 0,
        "skipped_missing_fields": 0,
        "skipped_bad_shapes": 0,
        "skipped_out_of_range": 0,
        "skipped_bad_values": 0,
    }


def label_scene(scene_group: "h5py.Group", adversary_index: int) -> Tuple[Optional[int], Optional[str]]:
    valid_shape, skip_reason = validate_scene_shape(scene_group)
    if not valid_shape:
        return None, skip_reason

    centroid = np.asarray(scene_group["centroid"], dtype=np.float64)
    yaw = np.asarray(scene_group["yaw"], dtype=np.float64)
    extent = np.asarray(scene_group["extent"], dtype=np.float64)

    n_agents = centroid.shape[0]
    if adversary_index >= n_agents:
        return None, "skipped_out_of_range"

    ego_index = 0
    ego_centers = centroid[ego_index]
    adv_centers = centroid[adversary_index]
    ego_yaws = yaw[ego_index]
    adv_yaws = yaw[adversary_index]
    ego_sizes = extent[ego_index, :, :2]
    adv_sizes = extent[adversary_index, :, :2]

    if not validate_pair_values(ego_centers, adv_centers, ego_yaws, adv_yaws, ego_sizes, adv_sizes):
        return None, "skipped_bad_values"

    return first_collision_timestep(ego_centers, adv_centers, ego_yaws, adv_yaws, ego_sizes, adv_sizes), None


def build_collision_labels(
    data_path: Path,
    output_csv: Path,
    output_json: Path,
    recursive: bool = True,
    max_files: Optional[int] = None,
    max_scenes: Optional[int] = None,
    verbose: bool = False,
    debug_save_failures: Optional[Path] = None,
) -> Dict[str, int]:
    require_h5py()
    stats = initial_stats()
    rows: List[Dict[str, object]] = []
    failures: List[Dict[str, str]] = []

    h5_files = discover_h5_files(data_path, recursive=recursive)
    if max_files is not None:
        h5_files = h5_files[:max_files]

    stop_requested = False
    for file_path in h5_files:
        if stop_requested:
            break
        stats["files_scanned"] += 1
        if verbose:
            print(f"scanning {file_path}")
        with h5py.File(file_path, "r") as h5_file:
            for scene_key in h5_file.keys():
                if max_scenes is not None and stats["scenes_seen"] >= max_scenes:
                    stop_requested = True
                    break

                stats["scenes_seen"] += 1
                scene_group = h5_file[scene_key]

                adversary_index = parse_adversary_index(scene_key)
                if adversary_index is None:
                    stats["skipped_invalid_ctrl"] += 1
                    failures.append({"source_file": str(file_path), "scene_key": scene_key, "reason": "invalid_ctrl"})
                    continue

                if not all(field_name in scene_group for field_name in REQUIRED_FIELDS):
                    stats["skipped_missing_fields"] += 1
                    failures.append({"source_file": str(file_path), "scene_key": scene_key, "reason": "missing_fields"})
                    continue

                collision_timestep, skip_reason = label_scene(scene_group, adversary_index)
                if skip_reason is not None:
                    stats[skip_reason] += 1
                    failures.append({"source_file": str(file_path), "scene_key": scene_key, "reason": skip_reason})
                    continue

                label = 1 if collision_timestep is not None and collision_timestep >= 0 else 0
                stats["scenes_processed"] += 1
                if label == 1:
                    stats["scenes_labeled_collision"] += 1
                else:
                    stats["scenes_labeled_noncollision"] += 1

                rows.append(
                    {
                        "scene_key": scene_key,
                        "label": label,
                        "collision_timestep": collision_timestep,
                        "source_file": str(file_path),
                        "adv_idx": adversary_index,
                    }
                )

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(output_csv, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["scene_key", "label", "collision_timestep", "source_file", "adv_idx"],
        )
        writer.writeheader()
        writer.writerows(rows)

    output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as handle:
        json.dump(stats, handle, indent=2, sort_keys=True)
        handle.write("\n")

    if debug_save_failures is not None:
        debug_save_failures.parent.mkdir(parents=True, exist_ok=True)
        with open(debug_save_failures, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["source_file", "scene_key", "reason"])
            writer.writeheader()
            writer.writerows(failures)

    return stats


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate pair-level collision labels from trajectory HDF5 files.")
    parser.add_argument("--data-path", type=Path, required=True, help="Trajectory HDF5 root directory or file.")
    parser.add_argument("--output-csv", type=Path, required=True, help="CSV manifest path to write.")
    parser.add_argument("--output-json", type=Path, required=True, help="JSON summary path to write.")
    parser.add_argument("--no-recursive", action="store_true", help="Only scan HDF5 files directly under data-path.")
    parser.add_argument("--max-files", type=int, default=None, help="Optional cap for quick debugging.")
    parser.add_argument("--max-scenes", type=int, default=None, help="Optional cap for quick debugging.")
    parser.add_argument("--verbose", action="store_true", help="Print each scanned HDF5 file.")
    parser.add_argument("--debug-save-failures", type=Path, default=None, help="Optional CSV of skipped scenes.")
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    stats = build_collision_labels(
        data_path=args.data_path,
        output_csv=args.output_csv,
        output_json=args.output_json,
        recursive=not args.no_recursive,
        max_files=args.max_files,
        max_scenes=args.max_scenes,
        verbose=args.verbose,
        debug_save_failures=args.debug_save_failures,
    )
    print(json.dumps(stats, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
