"""
Graph-agnostic preprocessing for CausalDiffAE.

Turns the Apr11 trajectory scenes into a clean table of per-sample *candidate causal
factors* (the quantities a causal graph would draw its nodes from), plus a placeholder
DAG to fill once the graph is finalized. This is deliberately independent of the exact
causal graph: it extracts every plausible node now, and the graph step later just
selects a subset and sets the edges.

Outputs (under --out, default labels/):
  causaldiffae_factors.csv   per-scene factor table (human-inspectable)
  causaldiffae_factors.npz   factors [N,K] float32, factor_names, scene_ids, dag [K,K]
  causaldiffae_meta.json     factor definitions + DAG placeholder + how to fill it

The dataloader (next step) can read each trajectory lazily from HDF5 (via
TrajectoryDataset) and join these factors by scene_id, so we don't dump all
trajectories here.

Run from repo root:
  python preprocess_causaldiffae.py
  python preprocess_causaldiffae.py --data-path /mnt/h/trajectory_apr11/Apr11_relaxed_all_archives \
      --label-csv labels/apr11_collision_labels.csv --limit 500
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from diff_scm.configs import get_config
from diff_scm.datasets.trajectory_dataset import TrajectoryDataset

# Feature indices in the [T, 23] pair layout (see TrajectoryDataset.build_pair_features).
EGO_XY, ADV_XY = slice(0, 2), slice(9, 11)
EGO_V, ADV_V, REL_V = slice(4, 6), slice(13, 15), slice(20, 22)
REL_DIST = 22

# Candidate causal factors and one-line definitions (nodes the graph can pick from).
FACTOR_DEFS = {
    "collision": "binary collision label (geometric SAT overlap)",
    "collision_timestep": "first collision timestep, -1 if none",
    "min_distance": "min ego-adversary distance over the scene",
    "init_distance": "ego-adversary distance at t=0",
    "closing": "init_distance - min_distance (how much they closed)",
    "ego_speed_mean": "mean ego speed",
    "adv_speed_mean": "mean adversary speed",
    "rel_speed_mean": "mean relative speed",
    "ego_max_accel": "max ego acceleration magnitude (m/s^2)",
    "adv_max_accel": "max adversary acceleration magnitude (m/s^2)",
    "ego_max_turn": "max ego per-step turn angle (rad)",
    "adv_max_turn": "max adversary per-step turn angle (rad)",
    "ego_path_len": "ego path length",
    "adv_path_len": "adversary path length",
}
FACTOR_NAMES = list(FACTOR_DEFS.keys())


def _speed(v):
    return np.linalg.norm(v, axis=-1)


def _path_len(xy):
    return float(np.linalg.norm(np.diff(xy, axis=0), axis=-1).sum())


def _max_accel(xy, dt):
    vel = np.diff(xy, axis=0) / dt
    acc = np.diff(vel, axis=0) / dt
    return float(np.linalg.norm(acc, axis=-1).max()) if acc.size else 0.0


def _max_turn(xy):
    d = np.diff(xy, axis=0)
    if d.shape[0] < 2:
        return 0.0
    v1, v2 = d[:-1], d[1:]
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    dot = (v1 * v2).sum(-1)
    return float(np.nan_to_num(np.arctan2(np.abs(cross), dot)).max())


def load_label_map(label_csv):
    """scene_id -> (collision, collision_timestep) from build_collision_labels output."""
    out = {}
    if label_csv is None or not Path(label_csv).exists():
        print(f"[warn] label csv not found ({label_csv}); collision factors will be NaN.")
        return out
    with open(label_csv, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            scene_id = f"{row['source_file']}:{row['scene_key']}"
            out[scene_id] = (int(row["label"]), int(row.get("collision_timestep", -1)))
    return out


def factors_for(traj, label, ctime, dt):
    ego_xy, adv_xy = traj[:, EGO_XY], traj[:, ADV_XY]
    dist = traj[:, REL_DIST]
    return {
        "collision": float(label) if label is not None else np.nan,
        "collision_timestep": float(ctime) if ctime is not None else np.nan,
        "min_distance": float(dist.min()),
        "init_distance": float(dist[0]),
        "closing": float(dist[0] - dist.min()),
        "ego_speed_mean": float(_speed(traj[:, EGO_V]).mean()),
        "adv_speed_mean": float(_speed(traj[:, ADV_V]).mean()),
        "rel_speed_mean": float(_speed(traj[:, REL_V]).mean()),
        "ego_max_accel": _max_accel(ego_xy, dt),
        "adv_max_accel": _max_accel(adv_xy, dt),
        "ego_max_turn": _max_turn(ego_xy),
        "adv_max_turn": _max_turn(adv_xy),
        "ego_path_len": _path_len(ego_xy),
        "adv_path_len": _path_len(adv_xy),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-path", type=str, default=None)
    p.add_argument("--label-csv", type=str, default="labels/apr11_collision_labels.csv")
    p.add_argument("--out", type=Path, default=Path("labels"))
    p.add_argument("--dt", type=float, default=0.1)
    p.add_argument("--limit", type=int, default=None, help="cap #scenes for a quick test")
    args = p.parse_args()

    config = get_config.file_from_dataset("trajectory")
    if args.data_path is not None:
        config.data.path = Path(args.data_path)

    dataset = TrajectoryDataset(
        data_path=config.data.path,
        expected_timesteps=config.data.expected_timesteps,
        require_labels=False,
        recursive=config.data.recursive,
        cache_in_memory=False,
        label_candidates=config.data.label_candidates,
    )
    label_map = load_label_map(args.label_csv)
    n = len(dataset) if args.limit is None else min(args.limit, len(dataset))
    print(f"preprocessing {n} scenes (dt={args.dt}) ...")

    rows, scene_ids, mat = [], [], []
    for i in range(n):
        item = dataset[i]
        sid = item["scene_id"]
        traj = item["trajectory"].numpy().astype(np.float64)
        label, ctime = label_map.get(sid, (None, None))
        fac = factors_for(traj, label, ctime, args.dt)
        rows.append({"scene_id": sid, **fac})
        scene_ids.append(sid)
        mat.append([fac[k] for k in FACTOR_NAMES])
        if (i + 1) % 2000 == 0:
            print(f"  {i + 1}/{n}")

    args.out.mkdir(parents=True, exist_ok=True)
    csv_path = args.out / "causaldiffae_factors.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["scene_id"] + FACTOR_NAMES)
        w.writeheader()
        w.writerows(rows)

    factors = np.asarray(mat, dtype=np.float32)
    dag = np.zeros((len(FACTOR_NAMES), len(FACTOR_NAMES)), dtype=np.int8)  # placeholder
    np.savez(
        args.out / "causaldiffae_factors.npz",
        factors=factors,
        factor_names=np.asarray(FACTOR_NAMES, dtype=object),
        scene_ids=np.asarray(scene_ids, dtype=object),
        dag=dag,
    )
    with open(args.out / "causaldiffae_meta.json", "w", encoding="utf-8") as f:
        json.dump({
            "factor_defs": FACTOR_DEFS,
            "n_samples": int(factors.shape[0]),
            "dag_note": "dag[i,j]=1 means factor i -> factor j. All zeros now; "
                        "fill from Baohua's initial causal graph (or drop unused factors).",
            "dag_order": FACTOR_NAMES,
        }, f, indent=2, ensure_ascii=False)

    print(f"saved {csv_path}")
    print(f"saved {args.out / 'causaldiffae_factors.npz'} (factors {factors.shape})")
    print(f"saved {args.out / 'causaldiffae_meta.json'}")
    # quick sanity
    coll = factors[:, FACTOR_NAMES.index("collision")]
    print(f"collision rate: {np.nanmean(coll):.3f}; "
          f"min_distance median: {np.median(factors[:, FACTOR_NAMES.index('min_distance')]):.2f}")


if __name__ == "__main__":
    main()
