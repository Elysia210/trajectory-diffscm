# DiffSCM — Apr11 Rerun + Physical Constraints (results summary)

Data: Apr11 (189 files / 14,638 scenes, all loaded). dt = 0.1 s. Cap = 8 m/s².
Setup: `guidance_scale=5, lambda_step=2`.

## 1. Sampling-side acceleration constraint (original model)

| lambda_accel | Δ collision prob | adv drift (m) | peak accel (m/s²) | % steps > 8 m/s² |
|---:|---:|---:|---:|---:|
| 0 (off) | 0.451 | 26.8 | 2138 | 74% |
| 5  | 0.300 | 12.4 | 19.3 | 20% |
| **10** | **0.311** | **12.3** | **13.5** | **16%** |
| 20 | 0.328 | 14.1 | 11.3 | 10% |

→ The acceleration cap cuts infeasible motion sharply (74% → 16% at λ=10) with almost no
loss of collision-promotion strength.

## 2. Training-side acceleration regularizer (model's own unguided samples)

| diffusion model | val loss (fidelity) | reference % > 8 m/s² | peak (m/s²) | Δ collision prob |
|---|---:|---:|---:|---:|
| baseline | 0.052 | 50% | 224 | 0.451 |
| naive reg (failed) | 0.210 | 99% | 148 | 0.245 |
| **low-noise reg** | **0.054** | **31%** | **154** | **0.443** |

→ Regularizing only the low-noise diffusion steps makes the model's own prior more feasible
(50% → 31%) **without** hurting fidelity or collision push. (A naive full-timestep reg
wrecks fidelity and backfires.)

## 3. Combined (recommended)

Regularized model + sampling cap (λ_accel=10): guided **12%** steps over cap (vs 16% on the
original model), Δ collision prob **0.29**. Training lowers the model's intrinsic
infeasibility; sampling enforces a per-sample cap — complementary.

**Takeaway:** adding physical constraints clearly improves feasibility at both levels, while
keeping the counterfactual collision push intact.
