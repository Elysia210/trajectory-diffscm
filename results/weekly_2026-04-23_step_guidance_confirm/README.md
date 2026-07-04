# Weekly Guided Step-Delta Confirmation

## Goal

Confirm that the current step-delta guided trajectory diffusion result is not a
one-off by repeating the main guided sampling experiment across multiple seeds.

## Configuration

- representation: step-delta future diffusion
- classifier: trajectory collision GRU baseline
- seeds: `123`, `231`, `777`, `3407`
- samples per run: `32`
- modes:
  - `unguided`
  - `no_collision_scale1p0`
  - `collision_scale0p5`
  - `collision_scale1p0`
  - `collision_scale2p0`
  - `collision_scale3p0`

## Main files

- `tables/per_run_summary.csv`
  Per-seed summary for every mode.

- `tables/mode_aggregate_summary.csv`
  Mean/std aggregated across seeds for each mode.

- `tables/ordering_checks.json`
  Boolean checks for:
  - `no_collision < unguided < collision`
  - monotonic collision guidance as scale increases

- `plots/seed123_main_trio_centered.png`
  Representative centered trajectory comparison for:
  `no_collision_s1.0`, `unguided`, `collision_s2.0`.

## Current conclusion

Across all four seeds:

- main ordering holds: `no_collision < unguided < collision_s2.0`
- `no_collision < unguided < collision_s1.0` also holds
- collision-guided mean risk increases monotonically from scale `0.5` to `3.0`
- motion metrics remain in the same overall range without severe trajectory
  blow-up
