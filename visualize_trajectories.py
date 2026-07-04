"""
Visualize preservation-aware counterfactual trajectories from a sampler .npz.

Produces two figures:
  1. <out>/trajectories.png  — top-K scenes (by collision-prob increase): factual
     (unguided reference) vs guided counterfactual, ego & adversary paths, with the
     closest-approach point marked and reference/guided collision probabilities.
  2. <out>/acceleration.png  — per-step acceleration magnitude of the guided ego/adv
     motion vs the physical cap, to show the feasibility constraint at work.

Usage (from repo root):
  python visualize_trajectories.py results/trajectory_step/preservation/accel_train_mask/regmask_la10.npz
  python visualize_trajectories.py <npz> --out results/trajectory_step/preservation/viz --k 6
"""

import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

EGO_XY = slice(0, 2)
ADV_XY = slice(9, 11)


def closest_approach(traj):
    """Index of the timestep where ego and adversary are nearest."""
    d = np.linalg.norm(traj[:, EGO_XY] - traj[:, ADV_XY], axis=-1)
    return int(d.argmin()), float(d.min())


def plot_trajectories(d, out_path, k, history_steps):
    ref = d["reference_trajectory"]      # [B, T, 23]
    gen = d["generated_trajectory"]
    ref_p = np.asarray(d["reference_collision_probability"], dtype=float)
    gen_p = np.asarray(d["collision_probability"], dtype=float)

    order = np.argsort(gen_p - ref_p)[::-1][:k]   # clearest counterfactuals first
    cols = min(3, len(order))
    rows = int(np.ceil(len(order) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4.2 * rows), squeeze=False)

    for ax_i, b in enumerate(order):
        ax = axes[ax_i // cols][ax_i % cols]
        for traj, ls, lbl in [(ref[b], "--", "factual"), (gen[b], "-", "counterfactual")]:
            ego, adv = traj[:, EGO_XY], traj[:, ADV_XY]
            ax.plot(ego[:, 0], ego[:, 1], ls, color="#1f77b4", lw=2,
                    label=f"ego ({lbl})" if ls == "-" else None)
            ax.plot(adv[:, 0], adv[:, 1], ls, color="#d62728", lw=2,
                    label=f"adv ({lbl})" if ls == "-" else None)
        # history end (intervention onset) and closest approach on the guided traj
        h = history_steps
        ax.scatter(gen[b, h, 0], gen[b, h, 1], c="k", marker="o", s=30, zorder=5)
        ci, cd = closest_approach(gen[b])
        ax.scatter([gen[b, ci, 0], gen[b, ci, 9]], [gen[b, ci, 1], gen[b, ci, 10]],
                   facecolors="none", edgecolors="green", s=120, lw=2, zorder=5)
        # Focus the view on the interaction: ego's full path + the adversary near
        # the closest-approach point, so a far adversary excursion does not squash it.
        ego_p, adv_p = gen[b][:, EGO_XY], gen[b][:, ADV_XY]
        lo, hi = max(0, ci - 12), min(gen.shape[1], ci + 13)
        win = np.concatenate([ego_p, adv_p[lo:hi]], axis=0)
        mn, mx = win.min(0), win.max(0)
        pad = 0.15 * max(mx[0] - mn[0], mx[1] - mn[1], 15.0)
        ax.set_xlim(mn[0] - pad, mx[0] + pad)
        ax.set_ylim(mn[1] - pad, mx[1] + pad)
        ax.set_title(f"scene {b}: collide {ref_p[b]:.2f} → {gen_p[b]:.2f}  (min dist {cd:.1f} m)",
                     fontsize=10)
        ax.set_aspect("equal", adjustable="box")
        ax.tick_params(labelsize=8)
        if ax_i == 0:
            ax.legend(fontsize=8, loc="best")
    for j in range(len(order), rows * cols):
        axes[j // cols][j % cols].axis("off")
    fig.suptitle("Factual (dashed) vs guided counterfactual (solid) — blue=ego, red=adv, "
                 "green=closest approach", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"saved {out_path}")


def plot_acceleration(d, out_path, accel_max, dt, k):
    gen_step = np.asarray(d["generated_step"], dtype=float)   # [B, Tf, 4] displacement
    gen_p = np.asarray(d["collision_probability"], dtype=float)
    ref_p = np.asarray(d["reference_collision_probability"], dtype=float)
    order = np.argsort(gen_p - ref_p)[::-1][:k]

    accel = (gen_step[:, 1:] - gen_step[:, :-1]) / (dt * dt)  # m/s^2
    ego_a = np.linalg.norm(accel[..., 0:2], axis=-1)
    adv_a = np.linalg.norm(accel[..., 2:4], axis=-1)
    t = np.arange(ego_a.shape[1])

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for b in order:
        ax.plot(t, ego_a[b], color="#1f77b4", alpha=0.5, lw=1)
        ax.plot(t, adv_a[b], color="#d62728", alpha=0.5, lw=1)
    ax.axhline(accel_max, color="k", ls="--", lw=1.5, label=f"cap = {accel_max:.0f} m/s²")
    ax.plot([], [], color="#1f77b4", label="ego accel")
    ax.plot([], [], color="#d62728", label="adv accel")
    frac = float(((ego_a[order] > accel_max) | (adv_a[order] > accel_max)).mean())
    ax.set_title(f"Guided acceleration magnitude (top-{len(order)} scenes) — "
                 f"{frac*100:.0f}% of steps over cap")
    ax.set_xlabel("future step"); ax.set_ylabel("acceleration (m/s²)")
    ax.set_ylim(0, max(accel_max * 3, float(np.percentile(np.concatenate([ego_a[order], adv_a[order]]), 99))))
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"saved {out_path}")


def _accel_on_ax(ax, d, k, title):
    gen_step = np.asarray(d["generated_step"], dtype=float)
    gen_p = np.asarray(d["collision_probability"], dtype=float)
    ref_p = np.asarray(d["reference_collision_probability"], dtype=float)
    accel_max = float(d["accel_max"]) if "accel_max" in d.files else 8.0
    dt = float(d["dt"]) if "dt" in d.files else 0.1
    order = np.argsort(gen_p - ref_p)[::-1][:k]
    accel = (gen_step[:, 1:] - gen_step[:, :-1]) / (dt * dt)
    ego_a = np.linalg.norm(accel[..., 0:2], axis=-1)
    adv_a = np.linalg.norm(accel[..., 2:4], axis=-1)
    t = np.arange(ego_a.shape[1])
    for b in order:
        ax.plot(t, ego_a[b], color="#1f77b4", alpha=0.5, lw=1)
        ax.plot(t, adv_a[b], color="#d62728", alpha=0.5, lw=1)
    ax.axhline(accel_max, color="k", ls="--", lw=1.5, label=f"cap = {accel_max:.0f} m/s²")
    frac = float(((ego_a[order] > accel_max) | (adv_a[order] > accel_max)).mean())
    peak = float(max(ego_a[order].max(), adv_a[order].max()))
    ax.set_title(f"{title}\n{frac*100:.0f}% steps over cap, peak {peak:.0f} m/s²", fontsize=10)
    ax.set_xlabel("future step")
    ax.set_ylabel("acceleration (m/s²)")
    ax.legend(fontsize=8)


def plot_accel_compare(npz_a, npz_b, out_path, k, labels):
    da = np.load(npz_a, allow_pickle=True)
    db = np.load(npz_b, allow_pickle=True)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))   # independent y-axes on purpose
    _accel_on_ax(axes[0], da, k, labels[0])
    _accel_on_ax(axes[1], db, k, labels[1])
    fig.suptitle("Acceleration feasibility — without vs with constraint", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"saved {out_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("npz", type=Path)
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--k", type=int, default=6)
    p.add_argument("--history-steps", type=int, default=50)
    p.add_argument("--compare-accel", type=Path, default=None,
                   help="Second .npz; produces a side-by-side acceleration comparison (this=left, that=right).")
    p.add_argument("--compare-labels", type=str, default="without constraint|with constraint")
    args = p.parse_args()

    d = np.load(args.npz, allow_pickle=True)
    out = args.out or args.npz.parent / "viz"
    out.mkdir(parents=True, exist_ok=True)
    accel_max = float(d["accel_max"]) if "accel_max" in d.files else 8.0
    dt = float(d["dt"]) if "dt" in d.files else 0.1

    plot_trajectories(d, out / "trajectories.png", args.k, args.history_steps)
    plot_acceleration(d, out / "acceleration.png", accel_max, dt, args.k)
    if args.compare_accel is not None:
        labels = args.compare_labels.split("|")
        plot_accel_compare(args.npz, args.compare_accel, out / "acceleration_compare.png", args.k, labels)
    print(f"dt={dt:.3f}s, accel_max={accel_max:.1f} m/s²")


if __name__ == "__main__":
    main()
