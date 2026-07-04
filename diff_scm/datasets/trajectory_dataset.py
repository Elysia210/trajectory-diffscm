"""
Trajectory pair dataset for the first-stage Diff-SCM trajectory prototype.

This module keeps the scope deliberately narrow:
- load HDF5 files from a local directory
- extract an ego/adversary pair per scene
- build a fixed [T, 23] feature tensor
- optionally read binary collision labels when present

The label resolver is intentionally pluggable because different trajectory
exports may encode collision targets under different names.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

try:
    import h5py
except ImportError:  # pragma: no cover - exercised at runtime in user env.
    h5py = None


CTRL_PATTERN = re.compile(r"ctrl_\[(\d+)\]")
FLOAT32 = np.float32


@dataclass
class SceneRecord:
    file_path: Path
    scene_key: str
    adversary_index: int
    label: Optional[float]


def require_h5py():
    if h5py is None:
        raise ImportError(
            "TrajectoryDataset requires the 'h5py' package. "
            "Please install it in the project environment before running data checks or training."
        )


class TrajectoryDataset(Dataset):
    """
    Minimal pair-level trajectory dataset.

    Each sample contains:
    - trajectory: torch.FloatTensor of shape [T, 23]
    - y: torch.FloatTensor scalar if labels are available
    - scene_id / scene_key / file_path metadata for debugging
    """

    REQUIRED_FIELDS = ("centroid", "yaw", "curr_speed", "extent")

    def __init__(
        self,
        data_path: Path,
        expected_timesteps: int = 100,
        require_labels: bool = False,
        recursive: bool = True,
        cache_in_memory: bool = False,
        label_candidates: Optional[Sequence[str]] = None,
        label_map_path: Optional[Path] = None,
    ):
        require_h5py()
        self.data_path = Path(data_path)
        self.expected_timesteps = expected_timesteps
        self.require_labels = require_labels
        self.recursive = recursive
        self.cache_in_memory = cache_in_memory
        self.label_candidates = tuple(label_candidates or ())
        self.label_map_path = Path(label_map_path) if label_map_path is not None else None
        self.external_labels = self._load_external_labels(self.label_map_path)
        self.records: List[SceneRecord] = []
        self.cache: Dict[int, Dict[str, torch.Tensor]] = {}
        self.stats = {
            "files_scanned": 0,
            "scenes_seen": 0,
            "scenes_loaded": 0,
            "skipped_invalid_ctrl": 0,
            "skipped_missing_fields": 0,
            "skipped_bad_shapes": 0,
            "skipped_missing_labels": 0,
            "labels_available": 0,
            "labels_from_manifest": 0,
        }

        self.h5_files = self._discover_h5_files()
        self._index_scenes()

    def _discover_h5_files(self) -> List[Path]:
        if self.data_path.is_file():
            return [self.data_path]

        if not self.data_path.exists():
            raise FileNotFoundError(f"Trajectory data path does not exist: {self.data_path}")

        # Avoid scanning visualisation folders file-by-file; only ask pathlib
        # for HDF5 suffixes that can contain trajectory scenes.
        patterns = ("**/*.h5", "**/*.hdf5") if self.recursive else ("*.h5", "*.hdf5")
        files = []
        for pattern in patterns:
            files.extend(path for path in self.data_path.glob(pattern) if path.is_file())
        return sorted(files)

    def _index_scenes(self) -> None:
        for file_path in self.h5_files:
            self.stats["files_scanned"] += 1
            with h5py.File(file_path, "r") as h5_file:
                for scene_key in h5_file.keys():
                    self.stats["scenes_seen"] += 1
                    scene_group = h5_file[scene_key]
                    adversary_index = self.parse_adversary_index(scene_key)
                    if adversary_index is None:
                        self.stats["skipped_invalid_ctrl"] += 1
                        continue

                    if not self._has_required_fields(scene_group):
                        self.stats["skipped_missing_fields"] += 1
                        continue

                    if not self._is_valid_scene_shape(scene_group, adversary_index):
                        self.stats["skipped_bad_shapes"] += 1
                        continue

                    label = self.resolve_label(scene_group, file_path=file_path, scene_key=scene_key)
                    if label is None and self.require_labels:
                        self.stats["skipped_missing_labels"] += 1
                        continue
                    if label is not None:
                        self.stats["labels_available"] += 1

                    self.records.append(
                        SceneRecord(
                            file_path=file_path,
                            scene_key=scene_key,
                            adversary_index=adversary_index,
                            label=label,
                        )
                    )
                    self.stats["scenes_loaded"] += 1

    def _load_external_labels(self, label_map_path: Optional[Path]) -> Dict[str, float]:
        if label_map_path is None:
            return {}
        if not label_map_path.exists():
            raise FileNotFoundError(f"Label manifest does not exist: {label_map_path}")

        if label_map_path.suffix.lower() == ".json":
            with open(label_map_path, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
            return {str(key): self._validate_label_value(value) for key, value in payload.items()}

        if label_map_path.suffix.lower() == ".csv":
            labels = {}
            with open(label_map_path, "r", encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle)
                for row in reader:
                    key = row.get("scene_id") or row.get("scene_key")
                    if key is None:
                        raise ValueError("CSV label manifest must contain a 'scene_id' or 'scene_key' column.")
                    if "label" not in row:
                        raise ValueError("CSV label manifest must contain a 'label' column.")
                    labels[str(key)] = self._validate_label_value(row["label"])
            return labels

        raise ValueError("Label manifest must be .json or .csv")

    @staticmethod
    def _validate_label_value(value) -> float:
        scalar = float(value)
        if scalar not in (0.0, 1.0):
            raise ValueError(f"Binary labels must be 0 or 1, got {value}")
        return scalar

    @staticmethod
    def parse_adversary_index(scene_key: str) -> Optional[int]:
        match = CTRL_PATTERN.search(scene_key)
        if match is None:
            return None
        return int(match.group(1))

    def _has_required_fields(self, scene_group: "h5py.Group") -> bool:
        return all(field_name in scene_group for field_name in self.REQUIRED_FIELDS)

    def _is_valid_scene_shape(self, scene_group: "h5py.Group", adversary_index: int) -> bool:
        centroid = scene_group["centroid"]
        yaw = scene_group["yaw"]
        speed = scene_group["curr_speed"]
        extent = scene_group["extent"]

        if centroid.ndim != 3 or centroid.shape[-1] != 2:
            return False
        if yaw.ndim != 2 or speed.ndim != 2:
            return False
        if extent.ndim != 3 or extent.shape[-1] < 3:
            return False

        n_agents, timesteps, _ = centroid.shape
        if timesteps != self.expected_timesteps:
            return False
        if yaw.shape != (n_agents, timesteps):
            return False
        if speed.shape != (n_agents, timesteps):
            return False
        if extent.shape[0] != n_agents or extent.shape[1] != timesteps:
            return False
        if n_agents <= adversary_index:
            return False
        return True

    def resolve_label(self, scene_group: "h5py.Group", file_path: Path, scene_key: str) -> Optional[float]:
        """
        Resolve a binary collision label if one exists.

        We keep this conservative:
        - first try an external manifest keyed by full scene_id or scene_key
        - search a small list of common field/attribute names
        - accept only scalar-like values
        - invert "no_collision" style labels so the returned target always means collision=1
        """
        full_scene_id = f"{file_path}:{scene_key}"
        if full_scene_id in self.external_labels:
            self.stats["labels_from_manifest"] += 1
            return self.external_labels[full_scene_id]
        if scene_key in self.external_labels:
            self.stats["labels_from_manifest"] += 1
            return self.external_labels[scene_key]

        for name in self.label_candidates:
            if name in scene_group:
                label = self._coerce_scalar_label(scene_group[name][()])
                if label is not None:
                    return 1.0 - label if "no_collision" in name else label
            if name in scene_group.attrs:
                label = self._coerce_scalar_label(scene_group.attrs[name])
                if label is not None:
                    return 1.0 - label if "no_collision" in name else label
        return None

    @staticmethod
    def _coerce_scalar_label(value) -> Optional[float]:
        array = np.asarray(value)
        if array.size != 1:
            return None
        scalar = float(array.reshape(-1)[0])
        if scalar not in (0.0, 1.0):
            return None
        return scalar

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        if self.cache_in_memory and index in self.cache:
            return self.cache[index]

        record = self.records[index]
        with h5py.File(record.file_path, "r") as h5_file:
            scene_group = h5_file[record.scene_key]
            features = self.build_pair_features(scene_group, record.adversary_index)

        item = {
            "trajectory": torch.from_numpy(features),
            "scene_id": f"{record.file_path}:{record.scene_key}",
            "scene_key": record.scene_key,
            "file_path": str(record.file_path),
        }
        if record.label is not None:
            item["y"] = torch.tensor(record.label, dtype=torch.float32)

        if self.cache_in_memory:
            self.cache[index] = item
        return item

    @staticmethod
    def build_pair_features(scene_group: "h5py.Group", adversary_index: int) -> np.ndarray:
        centroid = np.asarray(scene_group["centroid"], dtype=FLOAT32)
        yaw = np.asarray(scene_group["yaw"], dtype=FLOAT32)
        speed = np.asarray(scene_group["curr_speed"], dtype=FLOAT32)
        extent = np.asarray(scene_group["extent"], dtype=FLOAT32)[..., :3]

        ego_index = 0
        ego_xy = centroid[ego_index]
        adv_xy = centroid[adversary_index]

        ego_yaw = yaw[ego_index]
        adv_yaw = yaw[adversary_index]
        ego_speed = speed[ego_index]
        adv_speed = speed[adversary_index]

        ego_cos = np.cos(ego_yaw).astype(FLOAT32)
        ego_sin = np.sin(ego_yaw).astype(FLOAT32)
        adv_cos = np.cos(adv_yaw).astype(FLOAT32)
        adv_sin = np.sin(adv_yaw).astype(FLOAT32)

        ego_vx = ego_speed * ego_cos
        ego_vy = ego_speed * ego_sin
        adv_vx = adv_speed * adv_cos
        adv_vy = adv_speed * adv_sin

        ego_extent = extent[ego_index]
        adv_extent = extent[adversary_index]

        delta_xy = adv_xy - ego_xy
        delta_vx = (adv_vx - ego_vx)[:, None]
        delta_vy = (adv_vy - ego_vy)[:, None]
        distance = np.linalg.norm(delta_xy, axis=-1, keepdims=True).astype(FLOAT32)

        ego_features = np.concatenate(
            [ego_xy, ego_cos[:, None], ego_sin[:, None], ego_vx[:, None], ego_vy[:, None], ego_extent],
            axis=-1,
        )
        adv_features = np.concatenate(
            [adv_xy, adv_cos[:, None], adv_sin[:, None], adv_vx[:, None], adv_vy[:, None], adv_extent],
            axis=-1,
        )
        relative_features = np.concatenate([delta_xy, delta_vx, delta_vy, distance], axis=-1)
        features = np.concatenate([ego_features, adv_features, relative_features], axis=-1)
        return features.astype(FLOAT32)

    def get_label_distribution(self) -> Optional[Dict[float, int]]:
        labels = [record.label for record in self.records if record.label is not None]
        if not labels:
            return None

        unique_labels, counts = np.unique(np.asarray(labels, dtype=np.float32), return_counts=True)
        return {float(label): int(count) for label, count in zip(unique_labels, counts)}


def summarize_dataset(dataset: TrajectoryDataset, max_examples: int = 5) -> None:
    print(f"data_path: {dataset.data_path}")
    print(f"h5_files: {len(dataset.h5_files)}")
    print(f"samples: {len(dataset)}")
    print(f"stats: {dataset.stats}")

    example_keys = [record.scene_key for record in dataset.records[:max_examples]]
    print(f"example_scene_keys: {example_keys}")

    if len(dataset) > 0:
        sample = dataset[0]
        print(f"sample_shape: {tuple(sample['trajectory'].shape)}")
    else:
        print("sample_shape: unavailable")

    label_distribution = dataset.get_label_distribution()
    if label_distribution is None:
        print("label_distribution: unavailable (no recognized collision labels found)")
    else:
        print(f"label_distribution: {label_distribution}")


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Inspect trajectory HDF5 files for the Diff-SCM pair prototype.")
    parser.add_argument("--data-path", type=Path, required=True, help="Root directory or single .h5/.hdf5 file.")
    parser.add_argument("--expected-timesteps", type=int, default=100)
    parser.add_argument("--recursive", action="store_true", help="Recursively search for HDF5 files under data-path.")
    parser.add_argument("--require-labels", action="store_true", help="Drop scenes without recognized labels.")
    parser.add_argument(
        "--label-map-path",
        type=Path,
        default=None,
        help="Optional JSON/CSV manifest for external binary labels keyed by scene_id or scene_key.",
    )
    parser.add_argument(
        "--label-candidate",
        dest="label_candidates",
        action="append",
        default=None,
        help="Additional scalar field/attribute names to try as binary collision labels.",
    )
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    label_candidates = [
        "collision",
        "is_collision",
        "collided",
        "collision_label",
        "label",
        "target",
        "y",
        "no_collision",
        "is_no_collision",
    ] + (args.label_candidates or [])
    dataset = TrajectoryDataset(
        data_path=args.data_path,
        expected_timesteps=args.expected_timesteps,
        require_labels=args.require_labels,
        recursive=args.recursive,
        label_candidates=label_candidates,
        label_map_path=args.label_map_path,
    )
    summarize_dataset(dataset)


if __name__ == "__main__":
    main()
