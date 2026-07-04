"""
Guidance utilities for trajectory-domain Diff-SCM experiments.
"""

from .trajectory_preservation import (
    apply_feature_mask,
    apply_feature_weight,
    apply_time_mask,
    clip_grad_norm_per_sample,
    grad_l2_norm_per_sample,
    normalize_grad_per_sample,
)
