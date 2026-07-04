"""
Inspect the map fields in the trajectory HDF5 so we can implement a map (off-road)
constraint. Prints drivable_map shape/values, the raster_from_world transform, and a
sanity check mapping the ego centroid into raster pixels.

Usage (from repo root):
  python inspect_map_data.py /mnt/h/trajectory_apr11/Apr11_relaxed_all_archives
  python inspect_map_data.py <one_file.hdf5>
"""

import sys
from pathlib import Path

import h5py
import numpy as np


def first_h5(path: Path):
    if path.is_file():
        return path
    files = sorted(set(path.glob("**/*.h5")) | set(path.glob("**/*.hdf5")))
    if not files:
        raise SystemExit(f"no .h5/.hdf5 under {path}")
    return files[0]


def describe(name, a):
    a = np.asarray(a)
    line = f"{name}: shape={a.shape} dtype={a.dtype}"
    if np.issubdtype(a.dtype, np.number):
        line += f" min={a.min():.4g} max={a.max():.4g}"
    print(line)
    return a


def main():
    path = Path(sys.argv[1])
    f = first_h5(path)
    with h5py.File(f, "r") as h:
        scene = list(h.keys())[0]
        g = h[scene]
        print(f"file:  {f}")
        print(f"scene: {scene}")
        print(f"fields: {list(g.keys())}\n")

        dm = describe("drivable_map", g["drivable_map"]) if "drivable_map" in g else None
        if dm is not None:
            print(f"  unique values (first 10): {np.unique(dm)[:10]}")
            print(f"  fraction 'drivable' (==max): {float((dm == dm.max()).mean()):.3f}\n")

        rfw = None
        for name in ("raster_from_world", "raster_from_agent", "world_from_agent",
                     "world_from_raster", "agent_from_world"):
            if name in g:
                arr = describe(name, g[name])
                mat = np.asarray(arr)
                while mat.ndim > 2:        # reduce [n_agents, T, 3, 3] -> [3,3]
                    mat = mat[0]
                print(f"  matrix (agent0, step0):\n{mat}\n")
                if name == "raster_from_world":
                    rfw = arr

        cen = describe("centroid", g["centroid"]) if "centroid" in g else None

        # Sanity: map ego (index 0) positions into raster pixels and check bounds.
        # raster_from_world is per-agent, per-timestep: [n_agents, T, 3, 3].
        if rfw is not None and cen is not None:
            ego = np.asarray(cen[0], dtype=np.float64)        # [T, 2] world xy
            M = np.asarray(rfw[0], dtype=np.float64)          # [T, 3, 3] for ego over time
            homo = np.concatenate([ego, np.ones((ego.shape[0], 1))], axis=1)  # [T, 3]
            if M.ndim == 3:                                   # per-timestep transform
                px = np.einsum("tij,tj->ti", M, homo)         # [T, 3]
            else:
                px = homo @ M.T
            px = px[:, :2] / px[:, 2:3]
            print("ego pixel coords — first 3 and last 3 (x,y):")
            print(np.round(np.concatenate([px[:3], px[-3:]]), 1))
            if dm is not None:
                H, W = dm.shape[-2], dm.shape[-1]
                inside = ((px[:, 0] >= 0) & (px[:, 0] < W) & (px[:, 1] >= 0) & (px[:, 1] < H))
                print(f"map raster size (H,W) = ({H},{W}); ego pixels inside bounds: "
                      f"{int(inside.sum())}/{len(px)}")
                # Is the agent near the raster centre (agent-centric raster)?
                print(f"  mean pixel (x,y) = ({px[:,0].mean():.0f}, {px[:,1].mean():.0f}); centre = ({W//2},{H//2})")
                print("  -> if inside ~= all and mean ~ centre, pixel = raster_from_world[agent,t] @ [x,y,1].")


if __name__ == "__main__":
    main()
