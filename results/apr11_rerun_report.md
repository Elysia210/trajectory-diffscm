# Trajectory Diff-SCM — Apr11 Rerun & Sampling-Side Physical Constraints

_Experiment report — June 2026_

## Summary

The trajectory Diff-SCM baseline was rerun end-to-end on the updated Apr11 dataset
(Yongjie & Ziheng), and two sampling-side physical improvements were added:
self-consistent feature recomputation and an acceleration-feasibility hinge. The
full pipeline (collision classifier → history-conditioned step diffusion →
preservation-aware classifier-guided sampling) now runs on the new data. The
recommended configuration `guidance_scale=5, lambda_step=2, lambda_accel=10`
produces a strong counterfactual collision push (mean collision-probability
increase ≈ 0.31) while keeping the generated motion close to physically feasible
(peak acceleration ≈ 13.5 m/s², ≈ 16% of steps above the 8 m/s² cap, versus ≈ 50%
for the unguided reference). A notable diagnostic finding is that the unguided
diffusion reference itself produces infeasible accelerations, which means the
sampling-time constraints help but the root fix belongs in diffusion training.

## Task

The supervisor's request was to rerun Diff-SCM on the updated trajectory data and,
if possible, add physical constraints during the sampling process. Both parts are
addressed here.

## 1. Data integration (Apr11)

The new export contains 189 HDF5 files. All 14,638 ego/adversary pair-scenes were
indexed with zero skipped: the schema matches the existing loader exactly
(`centroid [n,100,2]`, `yaw`, `curr_speed`, `extent [n,100,3]`; per-sample feature
tensor `[100, 23]`; 100 timesteps as expected). No config change beyond the data
path was needed.

The export carries no explicit collision labels, so binary labels were generated
geometrically with `build_collision_labels.py` (rotated bounding-box overlap via the
Separating Axis Theorem between ego and the controlled adversary), yielding 2,203
collision and 12,435 non-collision scenes.

A correctness issue was caught and fixed at the labeling stage: scene keys are not
unique across the 189 files (14,638 rows but only 4,841 unique `scene_key`, because
the same `scene-####_ctrl_[n]` is reused across cases/policies). Matching labels by
bare `scene_key` would have collapsed and cross-assigned ~10k labels. The manifest
was rekeyed by the globally unique `source_file:scene_key` (the loader's
`full_scene_id`), after which all 14,638 scenes matched their own label
(`labels_from_manifest = 14638`).

## 2. Pipeline fixes

**Classifier input normalization.** The first classifier run did not learn at all:
training loss stayed flat at ≈ 1.17 and validation PR-AUC stayed at ≈ 0.141, i.e.
the positive base rate (chance), with predictions oscillating between all-collision
and all-safe. The cause was that the classifier consumed raw, unnormalized
`[100, 23]` features, whose world-coordinate positions reach hundreds–thousands of
metres (std ≈ 500–580) and saturate the GRU. Per-feature standardization (mean/std
computed on the training split, stored as model buffers so they travel with the
checkpoint and are reused automatically at sampling time) fixed it: validation
PR-AUC rose to 0.994, F1 to 0.965, with training loss decreasing monotonically to
≈ 0.07.

**History-conditioned step diffusion** trained cleanly on the new data (training
loss 0.245 → 0.065, validation loss → 0.052).

## 3. Sampling-side physical methods added

**Feature recomputation.** Previously a position intervention updated only the
ego/adversary positions and the relative position/distance, leaving heading and
velocity copied from the reference — an inconsistent trajectory for the classifier
to score. The generated future now recomputes heading (cos/sin of the displacement
direction), velocity (displacement / dt), and relative velocity from the generated
positions before classifier evaluation. The per-step interval `dt` is estimated from
the history (displacement / speed) and recovers 0.10 s as expected.

**Acceleration hinge.** A feasibility cap that penalizes only acceleration above a
threshold (`relu(|a| − a_max)²`, with `a = (step_t − step_{t-1}) / dt²`), rather than
forcing the generated trajectory to copy the reference. This is complementary to the
existing step-preservation term: one keeps positions near the factual rollout, the
other keeps the dynamics physically feasible.

## 4. Results

**Preservation sweep (guidance_scale × lambda_step).** `guidance_scale` is the main
control on collision promotion (mean collision-probability increase ≈ 0.12 at gs=2
vs ≈ 0.41 at gs=5). `lambda_step` trades collision push for closeness to the
reference. At gs=5, raising `lambda_step` is essentially free (collision push held
while step deviation and ego drift dropped), so `guidance_scale=5, lambda_step=2`
was selected as the first-stage operating point. `lambda_step` alone, however, could
not reduce the large adversary endpoint drift.

**Acceleration sweep (gs=5, lambda_step=2).**

| lambda_accel | Δ collision prob | adv endpoint drift (m) | peak accel (m/s²) | frac. steps > 8 m/s² |
|---:|---:|---:|---:|---:|
| 0  | 0.451 | 26.8 | 2138 | 0.74 |
| 2  | 0.314 | 17.3 | 421  | 0.41 |
| 5  | 0.300 | 12.4 | 19.3 | 0.20 |
| 10 | 0.311 | 12.3 | 13.5 | 0.16 |
| 20 | 0.328 | 14.1 | 11.3 | 0.10 |
| _unguided reference_ | — | — | _223.6_ | _0.496_ |

The hinge is effective beyond `lambda_accel ≈ 5`: peak acceleration and the fraction
of over-threshold steps fall sharply, the adversary drift is finally reduced (which
`lambda_step` alone could not achieve), and the collision push is essentially
preserved (Δ ≈ 0.30–0.33 throughout, even recovering slightly at higher
`lambda_accel` as smoother trajectories collide more cleanly).

## 5. Recommended configuration

`guidance_scale=5, lambda_step=2, lambda_accel=10, accel_max=8` is the recommended
operating point: Δ collision probability ≈ 0.31 (strong counterfactual collision),
peak acceleration ≈ 13.5 m/s², and ≈ 16% over-threshold steps — far better than the
unguided reference's ≈ 50%. `lambda_accel=20` is retained as a stronger-constraint
variant (peak ≈ 11.3 m/s², ≈ 10% over-threshold) for cases where feasibility is
prioritized over push strength.

## 6. Key finding — the model itself is the realism bottleneck

The unguided diffusion reference itself has high-acceleration artifacts (peak
≈ 224 m/s², ≈ 50% of steps above 8 m/s²). Sampling-time constraints clean up the
guided output well below this, but the realism ceiling is set by the diffusion model,
which was trained without any smoothness or acceleration regularization. This motivated
treating the cause, not just the symptom.

## 7. Training-side regularization (root-cause fix)

A smoothness/acceleration regularizer was added to the step-diffusion **training** loss,
acting on the model's predicted step sequence (its first difference = acceleration).

A first attempt with a naive full-timestep regularizer (weight 1.0) backfired: at high
diffusion timesteps the predicted clean sample is essentially noise with huge, meaningless
"acceleration", so the term dominated training, degraded denoising fidelity (validation
loss 0.21 vs the 0.052 baseline) and actually made the model's own samples *worse*
(reference fraction of over-threshold steps rose from 0.50 to 0.99). Lesson: the
regularizer must not come at the cost of fidelity.

Restricting the regularizer to the low-noise regime (only timesteps ≤ 20% of the
schedule, where the predicted trajectory is meaningful) fixed this. With that masking and
weight 1.0:

| | validation loss (fidelity) | reference frac. over 8 m/s² | reference peak accel | collision push (Δ) |
|---|---:|---:|---:|---:|
| baseline diffusion | 0.052 | 0.496 | 223.6 | 0.451 |
| naive reg (failed) | 0.210 | 0.994 | 147.6 | 0.245 |
| **low-noise reg** | **0.054** | **0.310** | **153.8** | **0.443** |

The regularized model preserves denoising fidelity and counterfactual push while making
its **own unguided samples** substantially more feasible (over-threshold fraction
0.50 → 0.31, peak 224 → 154).

**Stacking training + sampling constraints.** Using the regularized model together with
the sampling-side acceleration hinge (`lambda_accel=10`) gives a guided over-threshold
fraction of 0.12 (versus 0.16 on the un-regularized model) at a collision-probability
increase of 0.29 — the two mechanisms are complementary: training lowers the model's
intrinsic infeasibility, sampling enforces a per-sample physical cap.

Remaining headroom: validation fidelity has slack to ≈ 0.08, so the regularizer weight
or the low-noise window can be increased to push the reference feasibility further.

## Appendix — reproduce

```bash
# Labels (geometric) + globally-unique manifest
python -m diff_scm.utils.build_collision_labels \
  --data-path /mnt/h/trajectory_apr11/Apr11_relaxed_all_archives \
  --output-csv labels/apr11_collision_labels.csv \
  --output-json labels/apr11_collision_stats.json
python make_scene_id_manifest.py \
  labels/apr11_collision_labels.csv labels/apr11_collision_labels_scene_id.csv

# Train classifier (normalized) and step diffusion
python -m diff_scm.training.trajectory_classifier_train \
  --label-map-path labels/apr11_collision_labels_scene_id.csv
python -m diff_scm.training.trajectory_step_diffusion_train

# Recommended guided sampling
python -m diff_scm.sampling.sample_trajectory_step_diffusion_preservation \
  --data-path /mnt/h/trajectory_apr11/Apr11_relaxed_all_archives \
  --model-path  /home/ruimin/experiment_data/trajectory_diff_scm_baseline/trajectory_step_diffusion_train/best_model.pt \
  --classifier-path /home/ruimin/experiment_data/trajectory_diff_scm_baseline/trajectory_classifier_train/best_model.pt \
  --guidance-scale 5.0 --lambda-step 2.0 --lambda-accel 10.0 --accel-max 8.0 \
  --output results/trajectory_step/preservation/final_gs5_ls2_la10.npz
```
