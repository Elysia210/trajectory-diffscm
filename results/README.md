# Results Layout

This folder is organized by experiment stage so trajectory Diff-SCM outputs are
easier to audit and reuse.

## Folders

- `archive_image_counterfactual/`
  Original image-domain MNIST counterfactual visualizations from the upstream
  Diff-SCM workflow.

- `trajectory_labels/`
  Collision-label manifest and summary built from trajectory geometry.

- `trajectory_absolute/`
  First trajectory diffusion prototype that generated full future absolute
  features. Useful as an ablation/reference because guidance worked but motion
  realism was poor.

- `trajectory_relative/`
  Anchor-relative future-position diffusion outputs. This version fixed the
  largest coordinate drift but still had local step jitter.

- `trajectory_step/baseline/`
  Per-step displacement diffusion baseline outputs before guided sampling became
  the main line.

- `trajectory_step/guided/`
  Default home for step-delta guided sampling outputs from the vanilla
  classifier-guided sampler.

- `trajectory_step/preservation/`
  Default home for preservation-aware step-delta sampling runs, plus their
  follow-up diagnostics when the single-file diagnose script is used.

- `weekly_2026-04-23_step_guidance_confirm/`
  This week's guided step-delta package.
  - `pilot_seed123_n16/`: first guided seed-123 results, plots, and scale sweep.
  - `raw_npz/`: multi-seed confirmation runs (`n=32`) for unguided,
    no-collision guidance, and collision guidance across scales.
  - `logs/`: stdout/stderr logs for each sampling run.
  - `tables/`: aggregated summaries, ordering checks, and motion metrics.
  - `plots/`: centered trajectory comparison and motion histograms for the
    main trio.
