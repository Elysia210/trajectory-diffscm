"""
Summarize a guidance_scale x lambda_step sweep of the preservation-aware sampler.

For each .npz produced by sample_trajectory_step_diffusion_preservation.py it reports:
- guidance effect : mean guided collision prob vs reference, and the mean increase
- preservation    : per-step displacement deviation from the reference rollout,
                    and final-position (endpoint) drift in meters for ego/adv

Reads guidance_scale / lambda_step straight from each npz so the table is
self-describing regardless of filename.

Usage (from repo root):
    python summarize_preservation_sweep.py results/trajectory_step/preservation/sweep
"""

import glob
import os
import sys

import numpy as np

EGO_XY = slice(0, 2)
ADV_XY = slice(9, 11)


def summarize(path):
    d = np.load(path, allow_pickle=True)
    ref_p = np.asarray(d["reference_collision_probability"], dtype=np.float64)
    gud_p = np.asarray(d["collision_probability"], dtype=np.float64)
    gen_step = np.asarray(d["generated_step"], dtype=np.float64)
    ref_step = np.asarray(d["reference_step"], dtype=np.float64)
    gen_tr = np.asarray(d["generated_trajectory"], dtype=np.float64)
    ref_tr = np.asarray(d["reference_trajectory"], dtype=np.float64)

    ego_drift = np.linalg.norm(gen_tr[:, -1, EGO_XY] - ref_tr[:, -1, EGO_XY], axis=-1)
    adv_drift = np.linalg.norm(gen_tr[:, -1, ADV_XY] - ref_tr[:, -1, ADV_XY], axis=-1)

    # Peak acceleration (m/s^2) of the guided trajectory: step is per-step
    # displacement, so accel = (step_t - step_{t-1}) / dt^2.
    dt = float(d["dt"]) if "dt" in d.files else 0.1
    accel_max = float(d["accel_max"]) if "accel_max" in d.files else 8.0

    def peak_and_frac(step):
        a = (step[:, 1:] - step[:, :-1]) / (dt * dt)
        ego = np.linalg.norm(a[..., 0:2], axis=-1)
        adv = np.linalg.norm(a[..., 2:4], axis=-1)
        return float(max(ego.max(), adv.max())), float(((ego > accel_max) | (adv > accel_max)).mean())

    peak_accel, frac_over = peak_and_frac(gen_step)
    ref_peak, ref_frac = peak_and_frac(ref_step)

    return {
        "g_scale": float(d["guidance_scale"]),
        "lambda": float(d["lambda_step"]),
        "l_accel": float(d["lambda_accel"]) if "lambda_accel" in d.files else 0.0,
        "ref_prob": float(ref_p.mean()),
        "guided_prob": float(gud_p.mean()),
        "delta_prob": float((gud_p - ref_p).mean()),
        "step_dev": float(((gen_step - ref_step) ** 2).mean()),
        "adv_drift_m": float(adv_drift.mean()),
        "peak_accel": peak_accel,
        "frac_over": frac_over,
        "ref_peak": ref_peak,
        "ref_frac": ref_frac,
    }


def main():
    sweep_dir = sys.argv[1] if len(sys.argv) > 1 else "results/trajectory_step/preservation/sweep"
    files = sorted(glob.glob(os.path.join(sweep_dir, "*.npz")))
    if not files:
        print(f"no .npz found under {sweep_dir}")
        return

    rows = [summarize(f) for f in files]
    rows.sort(key=lambda r: (r["g_scale"], r["lambda"], r["l_accel"]))

    cols = ["g_scale", "lambda", "l_accel", "delta_prob",
            "step_dev", "adv_drift_m", "peak_accel", "frac_over", "ref_peak", "ref_frac"]
    print("  ".join(f"{c:>12}" for c in cols))
    for r in rows:
        print("  ".join(f"{r[c]:12.4f}" for c in cols))

    print("\nread: higher lambda      -> smaller step_dev / drift (tighter preservation),")
    print("      higher l_accel     -> smaller peak_accel / frac_over (feasible dynamics),")
    print("      both usually trade against a smaller delta_prob (weaker collision push).")


if __name__ == "__main__":
    main()
