# Diff-SCM Trajectory Extension — Detailed Change & Theory Notes

_A line-by-line record of what was changed, what each change does, and how it maps
onto the Diff-SCM / counterfactual / guidance theory. Written so you can re-derive
the conceptual picture tomorrow._

---

## 0. Orientation — how the trajectory code sits inside Diff-SCM theory

Diff-SCM's counterfactual recipe has three stages:

1. **Abduction** — infer the latent noise of a *real* observation (DDIM inversion of
   the real input → latent `z`).
2. **Action** — apply the intervention `do(X = x')`.
3. **Prediction** — run the guided forward/reverse pass from that latent to produce
   the counterfactual, changing only what the intervention forces (minimal change /
   factual preservation).

**Where the trajectory pipeline actually sits today (important):** the trajectory
version does **not** do Abduction. It starts the future from **random noise**, rolls
out an **unguided reference** with the same noise, and uses that reference as a
*preservation anchor*. So formally it is **"Diff-SCM-inspired conditional generation
with preservation"**, not a strict Pearl counterfactual. Everything we added this
session improves the **Prediction** stage (realism, consistency, feasibility); none
of it adds Abduction. That gap is still the single biggest theory item to revisit
(see §8 and §9).

Two other structural facts to keep in mind:

- **Block generation, not autoregressive rollout.** `TrajectoryFutureDenoiser`
  (a GRU) encodes the history once and emits **all** future steps at once
  (`[B, 4, T_future]`). It is not `∏ p(x_i | x_{i-1})`. Pros: no exposure bias /
  error accumulation; cons: no streaming / online intervention.
- **The target is per-step displacement**, not absolute position. Each future step is
  `[ego_dx, ego_dy, adv_dx, adv_dy]` (so `TARGET_DIM = 4`). Positions are
  reconstructed by `cumsum` from the last history position. Displacement ≈ velocity·dt,
  which is why its first difference is acceleration (used below).

The guided sampling update is a **sum of three gradients** injected into the DDIM
reverse loop:

```
gradient = guidance_scale · g_risk  −  lambda_step · g_step  −  lambda_accel · g_accel
            (push toward target)       (stay near reference)     (stay physically feasible)
```

- `g_risk = ∂(direction · classifier_logit)/∂x_t` — classifier guidance (Action+Prediction).
- `g_step = ∂(step_preservation_loss)/∂x_t` — the "trajectory prior" / minimal-change term.
- `g_accel = ∂(acceleration_hinge)/∂x_t` — physical-feasibility term (added this session).

`direction = +1` for `guidance_target=collision`, `−1` for `no_collision`. The `+`
on risk *ascends* the classifier objective; the `−` on the other two *descends* those
penalties.

---

## 1. The pipeline end-to-end (what runs, in order)

1. **Collision classifier** (`trajectory_classifier_train.py` → `TrajectoryGRUBaseline`):
   a bidirectional GRU that maps a full `[T, 23]` pair-trajectory to one collision
   logit. This is the **risk model** that provides the guidance gradient later.
2. **History-conditioned step diffusion** (`trajectory_step_diffusion_train.py` →
   `TrajectoryFutureDenoiser`): denoises the normalized future step sequence
   conditioned on the encoded history.
3. **Preservation-aware guided sampling**
   (`sample_trajectory_step_diffusion_preservation.py`): rolls an unguided reference,
   then runs a guided DDIM pass combining the three gradients above.

---

## 2. Data layer (Apr11)

### 2.1 The 23-dim pair feature layout (from `TrajectoryDataset.build_pair_features`)

| idx | feature | idx | feature | idx | feature |
|---|---|---|---|---|---|
| 0 | ego x | 9  | adv x | 18 | rel dx (adv−ego) |
| 1 | ego y | 10 | adv y | 19 | rel dy |
| 2 | ego cos θ | 11 | adv cos θ | 20 | rel dvx |
| 3 | ego sin θ | 12 | adv sin θ | 21 | rel dvy |
| 4 | ego vx | 13 | adv vx | 22 | distance |
| 5 | ego vy | 14 | adv vy | | |
| 6–8 | ego extent (l,w,h) | 15–17 | adv extent | | |

Heading is stored as `cos/sin θ`; velocity as `vx = speed·cosθ, vy = speed·sinθ`.
Knowing this table is what made the feature-recomputation fix possible.

### 2.2 What we did

- **`inspect_new_data.py`** (new) — runs the data through the real `TrajectoryDataset`
  and prints `stats`. Result: 189 files, 14,638 scenes, **0 skipped**, shape `[100, 23]`,
  100 timesteps → schema matches, **no config change** needed beyond the data path.
- **Labels** — the export has none, so `build_collision_labels.py` generates binary
  collision labels geometrically via **rotated-box Separating-Axis-Theorem overlap**
  between ego and the controlled adversary (`ctrl_[n]`): 2,203 collision / 12,435 not.
- **`make_scene_id_manifest.py`** (new) — fixes a labeling pitfall: scene keys repeat
  across files (14,638 rows but only 4,841 unique `scene_key`). The dataset's
  `resolve_label` checks `full_scene_id = f"{file_path}:{scene_key}"` **before** the
  bare key, so we rekey the manifest by `source_file:scene_key`. After this, all
  14,638 scenes match their own label (`labels_from_manifest = 14638`).

---

## 3. Classifier input normalization (the dead → healthy fix)

**Symptom:** first classifier run did not learn — `train_loss` flat ≈ 1.17,
`val_pr_auc` ≈ 0.141 (= positive base rate = chance), predictions oscillating between
all-collision and all-safe.

**Cause (theory):** the classifier consumed **raw** `[100, 23]` features. World-coordinate
positions reach hundreds–thousands of metres (std ≈ 500–580) while angular features
are O(1). That ~500× scale gap saturates the GRU; gradients can't shape it. (The
diffusion path normalizes; the classifier path did not.)

**Change — `diff_scm/models/trajectory_baseline.py`:** added standardization buffers
and applied them at the very start of `forward`:

```python
self.register_buffer("feature_mean", torch.zeros(input_dim))
self.register_buffer("feature_std",  torch.ones(input_dim))
...
def forward(self, trajectory):
    trajectory = (trajectory - self.feature_mean) / self.feature_std
    ...
```

They default to identity (backward compatible) and are **buffers**, so they are saved
in the checkpoint and reused automatically wherever the classifier is loaded — crucially
**the sampler needs no change**, it self-normalizes during guidance.

**Change — `diff_scm/training/trajectory_classifier_train.py`:** added
`compute_feature_stats(dataset, train_subset)` (per-feature mean/std over the **train
split only**, to avoid val/test leakage) and copied them into the model buffers before
training, with a log line.

**Result:** `val_pr_auc` → 0.994, `val_f1` → 0.965, `train_loss` → 0.07 (monotone).
The high PR-AUC is legitimate, not leakage: the label is geometric (box overlap) and the
relative-distance feature (idx 22) is an input, so "did they collide" is genuinely learnable.

---

## 4. The guidance objective (Prediction stage), in detail

All in `sample_trajectory_step_diffusion_preservation.py`, function
`make_preservation_cond_fn` → inner `cond_fn`.

Per DDIM step:
1. `pred_eps = model(x_t, t, history)` → `pred_xstart = _predict_xstart_from_eps(...)`.
2. `step_pred = denorm(pred_xstart)` (physical metres-per-step).
3. `future_positions = reconstruct_future_positions(history_xy, step_pred)` (cumsum).
4. `generated = update_trajectory_positions(...)` then
   **`generated = recompute_motion_features(generated, history_steps, dt)`** (new).
5. `logits = classifier(generated)` → `risk_obj = direction · logits.mean()`.
6. `step_pres_loss = step_preservation_loss(step_pred, reference_step)`.
7. **`accel_loss = acceleration_hinge_loss(step_pred, dt, accel_max)`** (new).
8. Three `autograd.grad` calls → masked/clipped → combined into `gradient`.

### 4.1 `step_preservation_loss` — the "trajectory prior" / minimal-change term

`mean((step_pred − reference_step)²)`. The **reference** is the unguided rollout from
the same initial noise — i.e. the trajectory prior here is *self-generated*, not external
real motion. This is the soft "stay close to the factual rollout" anchor that
operationalizes **minimal change / factual preservation** in this code.

### 4.2 Feature recomputation (new) — self-consistency of the predicted trajectory

**Why (theory):** the classifier scores the **whole** `[T, 23]` trajectory, including
heading and velocity. Previously a position intervention updated only positions and
relative position/distance, leaving heading/velocity copied from the reference → the
classifier was scoring an **inconsistent** trajectory (new positions, old yaw/speed),
and the risk gradient flowed through stale features.

**Change — `diff_scm/guidance/trajectory_preservation.py`,
`recompute_motion_features(...)`:** for the future steps, recompute from the new positions:
- heading `cos/sin` = direction of per-step displacement (near-stationary steps keep the
  old heading to avoid an undefined angle),
- velocity `vx/vy` = displacement / dt,
- relative velocity `dvx/dvy = adv_v − ego_v`.

Written **without in-place ops** (assembles 23 columns and `cat`s them) so it is safe
and differentiable inside the guidance autograd graph — the risk gradient now flows
through *consistent* features. It is applied in `cond_fn` and for the final
reference/generated trajectories in `main`.

### 4.3 `dt` estimation (new)

`estimate_dt(trajectory, history_steps)` uses `displacement = speed · dt` over the
history: `median(‖Δpos‖ / speed)`. On Apr11 it recovers **0.10 s**, as expected for the
driving sim. `dt` feeds both velocity recompute and the acceleration cap.

### 4.4 `acceleration_hinge_loss` (new) — physical feasibility

**Theory choice (you picked "feasibility cap", not soft-L2):** penalize only
acceleration **above** a physical limit, so normal motion is free and only teleport-like
jumps are pushed back:

```
a       = (step_t − step_{t-1}) / dt²          # step is displacement, so Δstep/dt² = accel
loss    = mean( relu(|a_ego| − a_max)²  +  relu(|a_adv| − a_max)² )
```

This is **complementary** to `step_preservation` (which keeps *positions* near the
reference) and **distinct** from the existing `smoothness_loss` (which is a plain L2 on
Δstep, i.e. it minimizes acceleration *everywhere* rather than capping it). `a_max`
defaults to 8 m/s².

### 4.5 Gradient bookkeeping (signs, masks, clips)

- **Risk branch:** `normalize_grad_per_sample` → `apply_feature_mask(risk_dim_mask =
  0.2,0.2,1.0,1.0)` (so guidance acts mostly on the **adversary** dims) →
  `apply_time_mask(0.5→1.0)` (emphasize later steps) → clip to `risk_clip_norm`.
- **Preservation branch:** `apply_feature_weight(preservation_dim_weight =
  1.5,1.5,1.2,1.2)` (protect ego more) → clip to `step_clip_norm`.
- **Acceleration branch:** clip to `step_clip_norm` (raw hinge structure kept).
- Combined: `guidance_scale·g_risk − lambda_step·g_step − lambda_accel·g_accel`.

Note the interaction this explains: because risk mostly moves the **adversary** and
preservation protects the **ego**, `lambda_step` alone could never rein in the large
adversary drift — only the acceleration term did.

### 4.6 Exact edits to `sample_trajectory_step_diffusion_preservation.py`

- imports: `acceleration_hinge_loss`, `estimate_dt`, `recompute_motion_features`.
- `make_preservation_cond_fn(...)` signature: added `lambda_accel, accel_max, dt`.
- `cond_fn`: feature recompute before classifier; `accel_loss` + `g_accel`; combined
  gradient; `raw/post_accel` stats. (`g_step` now keeps `retain_graph=True` so the accel
  grad can be taken; `g_accel` is zeros when `lambda_accel == 0`.)
- `main`: `dt = estimate_dt(...)` (printed); `recompute_motion_features` for both the
  reference and the generated trajectory; `dt`, `lambda_accel`, `accel_max` saved to npz.
- CLI: `--lambda-accel` (default 0 → off, backward compatible), `--accel-max` (default 8).

---

## 5. Sampling sweep tooling

**`summarize_preservation_sweep.py`** (new) loads every `.npz` in a directory and
reports, per (guidance_scale, lambda_step, lambda_accel): `delta_prob` (mean guided −
reference collision prob), `step_dev`, `adv_drift_m` (endpoint drift), `peak_accel` and
`frac_over` (fraction of steps above `a_max`, in m/s² using the saved `dt`), plus the
**reference** trajectory's `ref_peak` / `ref_frac` for comparison.

---

## 6. Training-side regularization (Phase B — "治本")

**Theory:** sampling-time constraints are corrective; if the diffusion model's own
samples are jittery, guidance is fighting the model. So push feasibility into the
**model** via a training regularizer on its predicted motion.

**Change — `diff_scm/models/gaussian_diffusion.py`:** `training_losses` now also returns
`terms["pred_xstart"]` (computed via `_predict_xstart_from_eps` for the EPSILON model),
so a regularizer can act on the predicted clean step sequence rather than on `ε`.

**Change — `diff_scm/training/trajectory_step_diffusion_train.py`:**
```python
def acceleration_reg(pred_xstart):           # [B, C, T]
    accel = pred_xstart[..., 1:] - pred_xstart[..., :-1]   # Δstep along time = acceleration
    return (accel ** 2).mean()
...
loss = (losses["loss"] * weights).mean()
if accel_reg_weight > 0:
    loss = loss + accel_reg_weight * acceleration_reg(losses["pred_xstart"])
```
Plus: `--accel-reg-weight` (config default 0.0, backward compatible), per-epoch
`train_accel_reg` logging, and **`--run-name`** so a regularized run writes to its own
checkpoint dir and does **not** overwrite the A-baseline diffusion model.

**Note on units:** this regularizer is a **soft L2 on acceleration in normalized step
space** — deliberately different from the sampling-side hinge (which is a hard cap in
physical m/s²). The training one shapes the whole model cheaply; the sampling one
enforces a physical bound per sample. (A possible refinement to discuss: make the
training reg a physical-unit hinge too.)

**Critical subtlety — only regularize the low-noise regime.** `pred_xstart` is the
model's estimate of the clean sample. At **high** diffusion timesteps that estimate is
essentially noise, so its "acceleration" is huge and meaningless. A naive full-timestep
regularizer (weight 1.0) was therefore dominated by high-t garbage: it swamped the
denoising MSE (training loss ≈ 20 vs MSE ≈ 0.05), wrecked fidelity (val 0.21 vs 0.052),
and — because a low-fidelity model produces jittery samples — made the model's own
reference *worse* (over-threshold fraction 0.50 → 0.99). The fix is to apply the
regularizer **only to samples with t ≤ a fraction of the schedule** (default 20%), where
`pred_xstart` is a meaningful trajectory (`--accel-reg-max-t-frac`). With masking, weight
1.0 kept fidelity at 0.054 (≈ baseline) while reducing the reference over-threshold
fraction 0.50 → 0.31. **General lesson for theory: a regularizer on the predicted clean
sample must be gated to the low-noise regime, or it competes with — and loses to — the
denoising objective at high noise.**

**Stacking.** Regularized model + sampling hinge (`lambda_accel=10`) → guided
over-threshold fraction 0.12 (vs 0.16 on the un-regularized model), Δ collision ≈ 0.29.
Training lowers the model's intrinsic infeasibility (the prior); sampling enforces a
per-sample physical cap. They are complementary.

---

## 7. Results, with theory reading

- **Preservation sweep:** `guidance_scale` controls collision push (Δ≈0.12 at 5×… →
  Δ≈0.41 at gs=5); `lambda_step` trades push for closeness. Chosen first stage:
  `gs=5, lambda_step=2`.
- **Acceleration sweep (gs=5, ls=2):** raising `lambda_accel` 0→20 drops peak accel
  2138→11 m/s² and over-threshold fraction 0.74→0.10, finally reduces adversary drift,
  and **barely costs collision push** (Δ≈0.30–0.33 throughout). Recommended:
  `lambda_accel=10` (Δ≈0.31, peak≈13.5, frac_over≈0.16); `=20` as stronger-constraint variant.
- **Root-cause finding:** the **unguided reference** itself is infeasible
  (`ref_peak≈224 m/s²`, `ref_frac≈0.50`). So the realism ceiling is set by the diffusion
  model (no smoothness in training, only 20 epochs) — which is exactly what Phase B targets.

---

## 8. Theory ↔ code status map (use this to re-read the mindmap)

| Mindmap concept | Status | Where in code | Note for re-derivation |
|---|---|---|---|
| **Abduction** (DDIM inversion of real input) | ❌ trajectory version | sampler starts from `torch.randn`; reference = unguided rollout | The deepest gap. Without inverting a *real* future, this isn't a strict Pearl counterfactual. |
| **Action / do-intervention** | ✓ (via classifier guidance) | `g_risk` in `cond_fn` | Intervention = "push toward collision/no-collision" via classifier gradient. |
| **Prediction** | ✓ | DDIM reverse + 3 gradients | Everything we improved lives here. |
| **Minimal change / factual preservation** | ✓ soft | `step_preservation_loss`; + feature recompute | Anchored to the *self-generated* reference, not external real motion. |
| **Classifier guidance** | ✓ | `risk_obj`, `g_risk` | Risk model = the normalized GRU classifier. |
| **Trajectory prior** | ✓ but synthetic | unguided reference rollout | "Prior" = the model's own unguided sample, not a dataset motion prior. |
| **Physical: smoothness / step / endpoint preservation** | ✓ | `trajectory_preservation.py` | Soft regularizers. |
| **Physical: acceleration** | ✓ NEW | hinge (sampling) + reg (training) | Hinge = feasibility cap; reg = soft L2 in training. |
| **Physical: energy / dynamics-consistency** | ❌ | — | Still open. |
| **Feature recomputation (yaw/velocity)** | ✓ NEW | `recompute_motion_features` | Makes the scored trajectory self-consistent. |
| **AR rollout / causal temporal attention / KV cache / streaming** | ❌ | block `TrajectoryFutureDenoiser` | Still block generation. |

---

## 9. Open theoretical questions for tomorrow

1. **Is this a counterfactual or conditional generation?** Without Abduction (inverting a
   real future to its latent), the "preservation against the unguided reference" is closer
   to *conditional generation + regularization* than to Pearl's counterfactual. Worth
   deciding whether to (a) implement trajectory DDIM inversion to make it a true
   counterfactual, or (b) reframe the method honestly as "Diff-SCM-inspired."
2. **What does the "prior" really encode?** The reference is the model's own unguided
   rollout, so preservation regularizes toward the model's prior, not toward real-world
   motion statistics. If the supervisor's intent is "real motion prior," that argues for
   conditioning on / inverting real future trajectories.
3. **Acceleration in normalized vs physical units.** Sampling uses a physical hinge (m/s²);
   training uses normalized L2. Are they consistent enough? Should training also be a
   physical-unit hinge?
4. **Block vs AR.** The `∏ p(x_i|x_{i-1})` story in the theory does not match the block
   generator. Either align the theory to block generation, or implement true AR rollout
   (and then exposure-bias / error-accumulation analysis becomes relevant).
5. **Where should physics live?** This session showed sampling-time constraints work but
   the model itself is infeasible. The principled answer is probably: feasibility in
   **training** (so the prior is feasible) + light correction in **sampling** — quantify
   that trade-off with Phase B.

---

## Appendix — files touched this session

**Created:** `inspect_new_data.py`, `make_scene_id_manifest.py`,
`summarize_preservation_sweep.py`, `results/apr11_rerun_report.md`, this file.

**Edited:**
- `diff_scm/models/trajectory_baseline.py` — classifier input standardization buffers.
- `diff_scm/training/trajectory_classifier_train.py` — `compute_feature_stats` + set buffers.
- `diff_scm/guidance/trajectory_preservation.py` — feature index constants, `estimate_dt`,
  `recompute_motion_features`, `acceleration_hinge_loss`.
- `diff_scm/sampling/sample_trajectory_step_diffusion_preservation.py` — feature recompute,
  acceleration term, `dt`, new CLI args, npz fields.
- `diff_scm/models/gaussian_diffusion.py` — `training_losses` returns `pred_xstart`.
- `diff_scm/configs/trajectory_configs.py` — `accel_reg_weight`.
- `diff_scm/training/trajectory_step_diffusion_train.py` — `acceleration_reg` (low-noise
  timestep-masked), `--accel-reg-weight`, `--accel-reg-max-t-frac`, `--run-name`.
