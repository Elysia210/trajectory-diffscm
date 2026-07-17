"""
Dataset for CausalDiffAE on trajectories.

Joins each trajectory (from the existing TrajectoryDataset) with its causal factor row
(from preprocess_causaldiffae.py's factor table) by scene_id, and yields the three
tensors CausalDiffAE training needs:

    x_start  : normalized future step-delta target, [C, T_future]   (what diffusion generates)
    x_encode : normalized full trajectory, [T, F]                   (what the encoder reads)
    c        : selected causal-factor labels, [num_vars], min-max normalized to ~[0,1]
               (the DAG nodes; used for the label-conditioned prior)

Which factors become DAG nodes is configurable (`factor_nodes`) so we can slot in
Baohua's causal graph later without touching the data code.
"""

from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset, Subset

from diff_scm.training.trajectory_step_diffusion_train import build_step_target


class CausalTrajectoryDataset(Dataset):
    def __init__(
        self,
        base_subset: Subset,
        factor_npz_path: Path,
        factor_nodes: Sequence[str],
        history_steps: int,
        target_mean: torch.Tensor,
        target_std: torch.Tensor,
        feature_mean: torch.Tensor,
        feature_std: torch.Tensor,
        factor_ranges: Optional[Dict[str, tuple]] = None,
        eps: float = 1e-6,
    ):
        self.base = base_subset
        self.history_steps = history_steps
        self.target_mean = target_mean.float()
        self.target_std = target_std.float()
        self.feature_mean = feature_mean.float()
        self.feature_std = feature_std.float()
        self.factor_nodes = list(factor_nodes)
        self.eps = eps

        table = np.load(factor_npz_path, allow_pickle=True)
        names = [str(n) for n in table["factor_names"]]
        ids = [str(s) for s in table["scene_ids"]]
        factors = np.asarray(table["factors"], dtype=np.float64)
        name_to_col = {n: i for i, n in enumerate(names)}
        missing = [n for n in self.factor_nodes if n not in name_to_col]
        if missing:
            raise ValueError(f"factor_nodes not in table: {missing}. available: {names}")
        self.node_cols = [name_to_col[n] for n in self.factor_nodes]
        self.factor_by_id = {sid: factors[i, self.node_cols] for i, sid in enumerate(ids)}

        # Per-node min/max for [0,1] normalization of the label vector.
        if factor_ranges is None:
            node_vals = factors[:, self.node_cols]
            self.fmin = np.nanmin(node_vals, axis=0)
            self.fmax = np.nanmax(node_vals, axis=0)
        else:
            self.fmin = np.asarray([factor_ranges[n][0] for n in self.factor_nodes])
            self.fmax = np.asarray([factor_ranges[n][1] for n in self.factor_nodes])

        # Keep only base samples that have a factor row.
        self.indices = [i for i in range(len(self.base))
                        if self.base[i]["scene_id"] in self.factor_by_id]

    def __len__(self) -> int:
        return len(self.indices)

    @property
    def num_vars(self) -> int:
        return len(self.factor_nodes)

    def normalize_labels(self, raw: np.ndarray) -> np.ndarray:
        return (raw - self.fmin) / (self.fmax - self.fmin + self.eps)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        item = self.base[self.indices[index]]
        traj = item["trajectory"].float()
        sid = item["scene_id"]

        target = build_step_target(traj, self.history_steps)                # [T_future, 4]
        x_start = ((target - self.target_mean) / self.target_std).transpose(0, 1)  # [4, T_future]
        x_encode = (traj - self.feature_mean) / self.feature_std            # [T, F]

        raw = np.nan_to_num(self.factor_by_id[sid], nan=0.0)
        c = torch.from_numpy(self.normalize_labels(raw)).float()            # [num_vars]

        return {"x_start": x_start, "x_encode": x_encode, "c": c, "scene_id": sid}


def collate_causal_batch(batch):
    return {
        "x_start": torch.stack([b["x_start"] for b in batch], dim=0),
        "x_encode": torch.stack([b["x_encode"] for b in batch], dim=0),
        "c": torch.stack([b["c"] for b in batch], dim=0),
        "scene_id": [b["scene_id"] for b in batch],
    }
