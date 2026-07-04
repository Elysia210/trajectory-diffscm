"""
Quick schema / timestep / label check for new trajectory data before a rerun.

Why this exists:
    TrajectoryDataset silently skips scenes that don't match the expected schema
    (wrong scene-key naming, missing fields, wrong #timesteps, no labels). This
    script surfaces *why* scenes are kept or skipped so you can fix the config
    (mainly data.expected_timesteps) before wasting a full training run.

Run from the repo root, inside the same env that runs training:
    python inspect_new_data.py /mnt/h/trajectory_apr11/Apr11_relaxed_all_archives
    python inspect_new_data.py /mnt/h/trajectory_apr11/Apr11_relaxed_all_archives 50
                                                        ^ optional expected_timesteps

Note: H: is a Windows drive. In WSL it is usually at /mnt/h. If /mnt/h is empty,
mount it once with:  sudo mkdir -p /mnt/h && sudo mount -t drvfs H: /mnt/h
"""

import sys
from pathlib import Path

import h5py
import numpy as np

from diff_scm.datasets.trajectory_dataset import TrajectoryDataset

LABEL_CANDIDATES = (
    "collision", "is_collision", "collided", "collision_label",
    "label", "target", "y", "no_collision", "is_no_collision",
)
REQUIRED_FIELDS = ("centroid", "yaw", "curr_speed", "extent")


def peek(data_path: Path, max_files: int = 3, max_scenes: int = 3):
    files = sorted(set(data_path.glob("**/*.h5")) | set(data_path.glob("**/*.hdf5")))
    print(f"[discover] {len(files)} HDF5 file(s) under {data_path}")
    for f in files[:max_files]:
        print(f"\n=== {f} ===")
        with h5py.File(f, "r") as h5:
            keys = list(h5.keys())
            print(f"  scenes in file: {len(keys)}")
            print(f"  first scene keys: {keys[:max_scenes]}")
            print(f"  (loader keeps only keys matching the 'ctrl_[<n>]' pattern)")
            for k in keys[:max_scenes]:
                g = h5[k]
                print(f"  - scene '{k}': fields={list(g.keys())}")
                for req in REQUIRED_FIELDS:
                    if req in g:
                        print(f"      {req}: shape={g[req].shape}")
                    else:
                        print(f"      {req}: *** MISSING ***")
                present = [c for c in LABEL_CANDIDATES if c in g or c in g.attrs]
                print(f"      label candidates present: {present or 'NONE'}")
    return files


def main():
    if len(sys.argv) < 2:
        print("usage: python inspect_new_data.py <data_dir> [expected_timesteps]")
        sys.exit(1)
    data_path = Path(sys.argv[1])
    expected_T = int(sys.argv[2]) if len(sys.argv) > 2 else 100

    files = peek(data_path)
    if not files:
        print("\n[!] No .h5/.hdf5 files found.")
        print("    Either the path is wrong, or the data is still inside the .tar.zst")
        print("    archives and needs extracting first.")
        return

    print(f"\n[dataset] building TrajectoryDataset(expected_timesteps={expected_T}) ...")
    ds = TrajectoryDataset(
        data_path=data_path,
        expected_timesteps=expected_T,
        require_labels=False,
        recursive=True,
        label_candidates=LABEL_CANDIDATES,
    )
    print(f"  usable samples: {len(ds)}")
    print("  stats (this is the key diagnostic):")
    for k, v in ds.stats.items():
        print(f"      {k}: {v}")

    if len(ds) == 0:
        print("\n[!] 0 usable samples. Read the 'skipped_*' counts above:")
        print("    skipped_invalid_ctrl  -> scene keys aren't named 'ctrl_[n]'")
        print("    skipped_missing_fields-> missing centroid/yaw/curr_speed/extent")
        print("    skipped_bad_shapes    -> usually #timesteps != expected_timesteps")
        print("                             (rerun this script with the real T)")
        return

    sample = ds[0]
    print(f"\n  sample trajectory shape: {tuple(sample['trajectory'].shape)}  (expect [T, 23])")
    print(f"  sample has collision label: {'y' in sample}")
    print(f"  labels available across dataset: {ds.stats['labels_available']} / {len(ds)}")
    if ds.stats["labels_available"] < len(ds):
        print("  [note] classifier training needs labels; if many are missing you'll")
        print("         need build_collision_labels.py or a --label-map-path manifest.")


if __name__ == "__main__":
    main()
