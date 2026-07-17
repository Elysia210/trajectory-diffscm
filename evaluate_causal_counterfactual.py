"""
Evaluate the effectiveness of a CausalDiffAE do-intervention.

Loads the counterfactual .npz (factual vs counterfactual trajectories) and scores both
with the trained collision classifier. If do(collision=1) worked, the counterfactual
collision probability should be higher than the factual one (and lower for do(=0)).
This mirrors CausalDiffAE's "effectiveness" check with an anti-causal classifier.

Usage (from repo root):
  python evaluate_causal_counterfactual.py \
    --npz results/causal_diffae/cf_collision1.npz \
    --classifier-path /home/ruimin/experiment_data/trajectory_diff_scm_baseline/trajectory_classifier_train/best_model.pt
"""

import argparse
from pathlib import Path
import sys

sys.path.append(str(Path.cwd()))

import numpy as np
import torch

from diff_scm.configs import get_config
from diff_scm.models.trajectory_baseline import TrajectoryGRUBaseline


def load_classifier(path, config, device):
    ckpt = torch.load(path, map_location=device)
    model = TrajectoryGRUBaseline(
        input_dim=config.classifier.input_dim,
        hidden_dim=config.classifier.hidden_dim,
        num_layers=config.classifier.num_layers,
        dropout=config.classifier.dropout,
        bidirectional=config.classifier.bidirectional,
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model


def main(args):
    config = get_config.file_from_dataset("trajectory")
    device = config.device
    clf = load_classifier(args.classifier_path, config, device)

    d = np.load(args.npz, allow_pickle=True)
    fac = torch.from_numpy(d["factual_trajectory"]).float().to(device)
    cf = torch.from_numpy(d["counterfactual_trajectory"]).float().to(device)
    intervene = str(d["intervene"])
    value = float(d["value"])

    with torch.no_grad():
        p_fac = torch.sigmoid(clf(fac).squeeze(-1)).cpu().numpy()
        p_cf = torch.sigmoid(clf(cf).squeeze(-1)).cpu().numpy()

    delta = p_cf - p_fac
    direction = "higher" if value >= 0.5 else "lower"
    print(f"do({intervene} = {value})   (expect counterfactual collision prob {direction})")
    print(f"  factual        mean collision prob: {p_fac.mean():.3f}")
    print(f"  counterfactual mean collision prob: {p_cf.mean():.3f}")
    print(f"  mean change: {delta.mean():+.3f}   (median {np.median(delta):+.3f})")
    print(f"  samples moved in intended direction: "
          f"{int((delta > 0).sum() if value >= 0.5 else (delta < 0).sum())}/{len(delta)}")
    print("  per-sample (factual -> counterfactual):")
    for i, (a, b) in enumerate(zip(p_fac, p_cf)):
        print(f"    {i:2d}: {a:.3f} -> {b:.3f}  ({b - a:+.3f})")


def build_argparser():
    p = argparse.ArgumentParser()
    p.add_argument("--npz", type=str, required=True)
    p.add_argument("--classifier-path", type=str, required=True)
    return p


if __name__ == "__main__":
    main(build_argparser().parse_args())
