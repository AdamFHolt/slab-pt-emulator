# Session Notes (2026-02-12)

## What Was Added/Changed

- Added unified training entrypoint:
  - `train.py` (YAML config-driven launcher for emulator training)
- Added explicit suite configs:
  - `configs/gp.const-vc.yaml`
  - `configs/gp.ramped-vc.yaml`
  - `configs/rf.const-vc.yaml`
  - `configs/rf.ramped-vc.yaml`
- Updated `README.md`:
  - full from-scratch workflow
  - config-based training usage
- Added root build helpers:
  - `Makefile`
- Added permissive open license:
  - `LICENSE` (MIT)
- Cleaned `requirements.txt` to direct project deps only.
- Updated CI workflow:
  - `.github/workflows/ci.yml`
  - uses `requirements.txt` and upgrades pip
  - checkout hardening (`permissions`, `fetch-depth`, safe.directory)

## Key Commands Now

- Train GP const-vc:
  - `python3 train.py --config configs/gp.const-vc.yaml`
- Train GP ramped-vc:
  - `python3 train.py --config configs/gp.ramped-vc.yaml`
- Train RF const-vc:
  - `python3 train.py --config configs/rf.const-vc.yaml`
- Train RF ramped-vc:
  - `python3 train.py --config configs/rf.ramped-vc.yaml`
- Dry run:
  - `python3 train.py --config configs/gp.const-vc.yaml --dry-run`
- Makefile shortcuts:
  - `make help`
  - `make train-gp-const`
  - `make train-gp-ramped`

## Recent Commits (latest first)

- `64a4431` fixed CI workflow run
- `373f52e` fixed CI workflow run
- `c0dd161` Tweaked README
- `640aeae` chore: add Makefile, MIT license, and clean requirements
- `306c648` chore: split training configs by suite and update docs
- `d79bdf5` feat: add unified config-driven training entrypoint
- `034f33c` docs: add generalized from-scratch workflow for const-vc and ramped-vc

## Suggested Next Steps

1. Optional: add `tests/` with basic smoke tests (`train.py --dry-run`, config validation, tiny synthetic train).
2. Optional: add `configs/*.yaml` variants for common targets (e.g., `dTdt_thermalParam_etaRatio`).
3. Optional: split runtime vs dev dependencies (`requirements-dev.txt`).

## Quick Resume Prompt

Use this next time:

> Continue from `SESSION_NOTES.md`. Focus on [your next task], and check CI + training entrypoint first.

---

## Update (2026-02-12, later)

### What Was Added

- Added smoke/integration tests:
  - `tests/test_train_smoke.py`
  - coverage includes:
    - config loading (`configs/gp.*.yaml`, `configs/rf.*.yaml`)
    - dataset discovery filtering
    - command construction for GP and RF
    - CLI dry run (`train.py --dry-run --datasets ...`)
    - failure paths (invalid suite, invalid dataset mode, invalid model type, missing suite dir)
    - tiny synthetic RF end-to-end training test for `src/emulator/train_emulator.py`
- Updated CI test command:
  - `.github/workflows/ci.yml`
  - now runs `python -m unittest discover -s tests -v` (no silent pass when tests fail)
- Added local test shortcut:
  - `Makefile` target `test`
- Updated docs:
  - `README.md` includes a `## Tests` section with `make test`

### Current Test Status

- `python3 -m unittest discover -s tests -v`
  - 10 tests passing

### Decisions Made

- Deferred optional dev-tooling steps for now:
  - `requirements-dev.txt`
  - `pre-commit`
  - CI lint stage
  - expanded dev-workflow docs
- Rationale: current setup uses stdlib `unittest` and no external lint/format stack yet.

---

## Update (2026-02-12, QC/Plotting pass)

### What Was Added/Changed

- Added pre-ML numerical diagnostics script:
  - `src/qc-numerical-mods/qc_master_dTdt_healthcheck.py`
  - outputs:
    - `*_by-run.csv`
    - `*_missing-heatmap.png` (depth x run missingness)
- Added optional per-run profile heatmaps:
  - `src/qc-numerical-mods/qc_Tprof_depth_time_heatmap.py`
- Added depth-parameter correlation heatmap:
  - `src/qc-numerical-mods/qc_param_depth_correlation_heatmap.py`
  - default behavior:
    - PNG only (CSV optional via `--save-csv`)
    - Pearson heatmap with improved readability (`RdBu_r`, major+minor contour overlays)
    - y-axis labels fixed to `0,10,...,80`
- Added run clustering diagnostic:
  - `src/qc-numerical-mods/qc_run_clustering_map.py`
  - clusters runs using full `dTdt(depth)` profiles
  - plots cluster maps in parameter space + PCA panel
  - optional assignments CSV via `--save-csv`
- Refined existing profile QC script (instead of separate ensemble script):
  - `src/qc-numerical-mods/qc_T_profiles_allmodels_3times.py`
  - now uses:
    - all individual paths in uniform gray
    - median line + 5–95% envelope polygon
- Removed redundant temporary ensemble script:
  - deleted `src/qc-numerical-mods/qc_Tprof_ensemble_envelope.py`

### make_all_plots.sh Retool

- Updated `src/qc-numerical-mods/make_all_plots.sh` to run only core plots by default.
- Core default now includes:
  - `qc_dTdt_vs_params_1depth.py`
  - `qc_dTdt_vs_params_mult-depths.py` (3- and 4-depth panels)
  - `qc_dTdt_vs_depth_mult-depths.py`
  - `qc_hist_dTdt_mult-depths.py`
  - `qc_T_profiles_allmodels_3times.py`
  - `qc_master_dTdt_healthcheck.py`
  - `qc_param_depth_correlation_heatmap.py`
- Optional (commented) blocks include:
  - `qc_pairplot_params.py`
  - `qc_pairplot_params_colored.py`
  - `qc_Tprof_depth_time_heatmap.py`
  - `qc_run_clustering_map.py`

### Recent Commits (QC pass, latest first)

- `589c32d` qc: streamline default plot set and add cluster map tool
- `16e48b3` qc: add depth-parameter correlation heatmap diagnostics
- `b375ff4` qc: merge ensemble styling into existing T profile plot
- `52569b3` qc: add optional Tprof heatmap and ensemble envelope visualizations
- `7227318` qc: extend colored pairplots to 10/50/70 km
- `42d61d5` qc: add pre-ML master healthcheck outputs

---

## Update (2026-02-18, ground-check + ramped-vc refresh)

### Verification Completed

- Re-checked current docs/code alignment:
  - `train.py`
  - `tests/test_train_smoke.py`
  - `src/qc-numerical-mods/make_all_plots.sh`
  - `README.md`
- Confirmed smoke/integration test status:
  - `make test` -> `python3 -m unittest discover -s tests -v`
  - result: 10 tests passing

### Artifacts Refreshed

- Refreshed ramped-vc QC output figures under:
  - `plots/qc-numerical-mods/ramped-vc/`
- Includes updated core outputs and two newly generated diagnostics:
  - `master_DT1-10_healthcheck_missing-heatmap.png`
  - `param_depth_corr_dTdt_pearson.png`

### Tracking Policy Check

- Current repository pattern is to version-control selected QC figures in `plots/`.
- `.gitignore` does not ignore `plots/`, so regenerated core QC PNGs are expected to appear as git changes.

---

## Next Session TODO (added 2026-02-18)

### 1) Add CI quality gate job (Step 3)

- Extend GitHub Actions to:
  - train a small representative emulator subset
  - run `make quality-check-gp-m25`
  - fail CI on quality regressions

### 2) Decide representative CI subset

- Lock in explicit datasets for bounded runtime while sampling depth behavior.
- Candidate default:
  - suites: `const-vc`, `ramped-vc`
  - depths: `10km`, `40km`, `80km`
  - variant: `dTdt`

### 3) Document “Definition of Done” (Step 4)

- Add a concise section in `README.md` covering:
  - required train commands (or subset policy in CI)
  - required pass of `make quality-check-gp-m25`
  - canonical artifact/report locations

---

## Follow-up (added 2026-02-18, branch-protection check names)

- Current state:
  - classic branch protection rule for `main` is in use
  - CI jobs are running on PRs (`CI / test`, `CI / quality-gate`)
- Deferred UI fix (GitHub check-name indexing lag):
  1. Open one tiny PR to `main` and let CI fully complete.
  2. Go to `Settings -> Branches -> main rule -> Edit`.
  3. Under required status checks, add:
     - `CI / test (pull_request)`
     - `CI / quality-gate (pull_request)`
  4. Save and confirm both checks show as required on subsequent PRs.


## Update (2026-02-18, CI gate + DoD finalized)

  ### Completed

  - Added CI `quality-gate` job in `.github/workflows/ci.yml`.
  - PR quality subset locked to:
    - suites: `const-vc`, `ramped-vc`
    - dataset: `40km_dTdt`
  - Added Definition of Done section to `README.md`.
  - `Makefile` quality-check target now supports optional dataset/suite filters.

  ### Policy

  - Branch ruleset should target `main` and require status checks (`CI / Any source`) before merge.

---

## Update (2026-02-20, pragmatic fallback on branch protection)

### Decision

- Branch-protection required-check selection in GitHub UI is currently unreliable for this repo (check context indexing not appearing).
- Stop spending time on UI troubleshooting for now.

### Working Policy (effective immediately)

- CI continues to run on PRs (`CI / test`, `CI / quality-gate`).
- Do not merge PRs to `main` unless both checks are green.
- If a red build is merged accidentally, revert promptly and re-open with fixes.

### Scope note

---

## Update (2026-03-12, profile-PCA defaults + repo layout cleanup)

### Major outcomes

- Profile-PCA representation sweep completed for `t3Myr` across:
  - suites: `const-vc`, `ramped-vc`
  - `k in {4, 6, 8, 10}`
  - score space: `raw`, `whitened`
- Chosen profile-PCA default:
  - time: `t3Myr`
  - `k=10`
  - `score_space=whitened`
- Added profile-PCA quality reporting and validation:
  - `src/emulator/profile_pca/evaluate_profile_pca_quality.py`
  - `src/emulator/profile_pca/validate_profile_pca_quality.py`
- Added profile-PCA sweep + summary tooling:
  - PCA representation sweep
  - GP tuning sweep
- GP tuning sweep completed for:
  - suite: `ramped-vc`
  - dataset: `profileT_pca_t3Myr_k10_whitened`
  - grid size: 24 runs

### GP tuning result

- Best tuned profile-PCA GP setting:
  - kernel: `matern25`
  - restarts: `25`
  - length-scale upper bound: `1e3`
  - noise lower bound: `1e-8`
- This was only a small improvement over the prior default, but it was the best result in the completed sweep.
- Updated active GP defaults to use:
  - `noise_bounds: [1e-8, 1.0]`

### Repo structure cleanup completed

- Emulator workflows are now grouped under:
  - `src/emulator/single_depth/`
  - `src/emulator/profile_pca/`
  - `src/emulator/legacy/`
- Emulator artifacts are now suite-first and workflow-specific:
  - single-depth:
    - `src/emulator/data/single_depth/<suite>/runs/`
    - `src/emulator/models/single_depth/<suite>/runs/`
    - `src/emulator/models/single_depth/<suite>/param_sweep/`
  - profile-PCA:
    - `src/emulator/data/profile_pca/<suite>/runs/`
    - `src/emulator/data/profile_pca/<suite>/pca_sweep/`
    - `src/emulator/models/profile_pca/<suite>/runs/`
    - `src/emulator/models/profile_pca/<suite>/pca_sweep/`
    - `src/emulator/models/profile_pca/<suite>/gp_tuning/`
- Existing outputs were moved into the new structure instead of being regenerated.

### Single-depth QC / plot cleanup

- Single-depth QC plot outputs now live under:
  - `plots/qc-emulator/single_depth/<suite>/`
- Single-depth QC shell entrypoint now lives under:
  - `src/emulator/single_depth/make_qc_emulator_plots.sh`
- Single-depth param-sweep QC is now folded into:
  - `src/emulator/single_depth/make_qc_emulator_plots.sh`
  - enable with `INCLUDE_PARAM_SWEEP=1`

### Tests / validation status

- `python3 -m unittest discover -s tests -v`
  - 24 tests passing after the suite-first layout move
- Dry-run checks passed for:
  - config-driven profile-PCA training
  - profile-PCA sweep runner
  - GP tuning sweep runner

### Important current local state

- The tracked repo structure is updated and pushed.
- There is still expected local churn from regenerated or relocated plot outputs under:
  - `plots/qc-emulator/single_depth/...`
  - and deletions under the old `plots/qc-emulator/<suite>/...` layout
- There may also be local generated model/artifact diffs from ongoing analysis runs.
- Do not blindly commit generated plot/model churn without checking it intentionally.

### Overnight rerun command

Need to rerun the default profile-PCA workflow for:
- suites: `const-vc`, `ramped-vc`
- times: `0.5, 1, 2, 3, 4, 5 Myr`
- default dataset naming (`profileT_pca_t<Myr>_k10`)
- current GP default (`matern25`, `noise_low=1e-8`)

Command:

```bash
make profile-pca-preprocess && make profile-pca-train-gp && make profile-pca-quality-report && make profile-pca-qc
```

### Suggested next steps

1. Let the overnight full profile-PCA rerun finish.
2. Validate the rerun outputs:
   - `make profile-pca-quality-check-gp-m25`
3. Check whether the new `t=0.5,1,2,3,4,5` profile-PCA runs all landed cleanly under:
   - `src/emulator/data/profile_pca/<suite>/runs/`
   - `src/emulator/models/profile_pca/<suite>/runs/`
4. Decide whether to stage/commit the single-depth QC plot relocation:
   - old `plots/qc-emulator/<suite>/...`
   - new `plots/qc-emulator/single_depth/<suite>/...`
5. Move on to science:
   - parameter sensitivity for single-depth emulators
   - parameter sensitivity for profile-PCA emulators
6. For the upcoming short talk:
   - prepare one clean workflow slide
   - one reconstruction/result slide
   - one sensitivity-motivation slide

### Quick resume prompt

Use this next time:

> Continue from `SESSION_NOTES.md`. First check whether the overnight profile-PCA rerun completed cleanly, validate the rerun outputs, then move to parameter sensitivity analysis and talk-figure prep.

- Current maintenance mode is effectively single-maintainer, so manual merge discipline is acceptable as a temporary control.

---

## Next Phase Plan (2026-02-20, functional emulator via PCA on profiles)

### Objective

- Extend emulator capability from per-depth scalar targets (e.g., `dTdt` at fixed depths) to functional targets representing full vertical thermal structure (`T(z)` first, later `T(z,t)`).
- Keep current project architecture (`train.py` + YAML configs + suite split) and add a parallel dataset/mode path rather than replacing existing per-depth workflows.

### Feasibility Decision

- Feasible with current codebase and data layout.
- Recommended implementation path:
  1. Fixed-time profile mode (`T(z)`): implement now.
  2. Compare against current per-depth baseline.
  3. Space-time extension (`T(z,t)`): only after phase-1 quality is established.

### Core Modeling Design

- Use PCA across depth for profile compression.
- For each run (at selected time slice), represent profile by top-`k` PCA scores.
- Train emulator(s) on PCA scores as targets (initially one independent GP per component).
- Reconstruct full profile by inverse PCA (`mean_profile + scores @ components`).

### Why PCA Path (chosen over alternatives)

- Strong dimensionality reduction with smooth vertical structure preservation.
- Better stability than independently emulating each depth.
- Lower complexity than GP over combined `(params, depth)` input.
- Straightforward integration with current single-output training scripts.

### Data Handling Rules (must keep)

- Center profile matrix per depth before PCA (store and reuse mean profile).
- Do not z-score each depth unless explicitly testing that variant (default: centered-only).
- Track and save:
  - `mean_profile`
  - `components`
  - `explained_variance_ratio`
  - `k` and selection rule
  - depth grid metadata
  - time metadata used to extract profiles

### Proposed Repository Layout

- New preprocess script(s) in `src/emulator/`:
  - `preprocess_profile_pca.py`
  - optional helper: `reconstruct_profile_pca.py` (or utility function in existing module)
- New config files:
  - `configs/gp.const-vc.profile-pca.yaml`
  - `configs/gp.ramped-vc.profile-pca.yaml`
  - optional RF variants later
- Data artifacts:
  - `src/emulator/data/<suite>/<dataset_name>/`
  - include:
    - `X_raw.npy`
    - `scores_raw.npy` (N x k target matrix)
    - `train_idx.npy`, `val_idx.npy`
    - `pca_mean_profile.npy`
    - `pca_components.npy` (k x n_depth)
    - `pca_explained_variance_ratio.npy`
    - `metadata.json`
- Model artifacts (follow existing conventions):
  - `src/emulator/models/<suite>/<dataset_name>/<model_tag>/`
  - include component-level reports and aggregate/reconstruction metrics.

### Dataset Naming Convention

- Preserve existing suite-first structure.
- Use explicit mode name in dataset:
  - example: `profileT_pca_t5Myr`
  - or: `Tprofile_pca_fixed-time`
- Avoid ambiguous names like just `profile_pca`.

### `train.py` / Config Integration Plan

- Add a new `dataset.mode` (or equivalent selector) for profile PCA datasets.
- Keep command surface identical:
  - `python train.py --config configs/gp.const-vc.profile-pca.yaml`
- Initial training strategy:
  - train one model per PCA component score (`PC1..PCk`), reusing current single-target training path.
- Add aggregation stage to collect component predictions and reconstruct profiles for evaluation.

### QC/Validation Additions (required for acceptance)

- New QC outputs under:
  - `plots/qc-emulator/profile-pca/default-runs/<suite>/`
- Required plots:
  - cumulative explained variance vs component count
  - reconstruction overlay (true vs reconstructed) on val set for representative runs
  - reconstruction RMSE vs depth
  - predicted vs true PCA scores per component
  - error distribution summary across validation runs

### Quality Gate Additions

- Add thresholds config:
  - `configs/emulator-quality.profile-pca.yaml`
- Candidate gate metrics (validation):
  - reconstruction RMSE (global)
  - reconstruction RMSE by depth percentile (e.g., P90)
  - score-space R2 for each retained component (or macro average)
- Keep existing `gp_m25` gate untouched; add profile-pca gate in parallel.

### Test Plan (must add)

- Unit tests:
  - PCA preprocess output shape and metadata integrity.
  - deterministic split behavior for profile datasets.
  - reconstruction correctness smoke test (`inverse_transform` consistency).
- Integration tests:
  - dry-run command construction for profile-pca configs.
  - tiny synthetic end-to-end training on 2–3 components.
- CI:
  - defer heavy profile training in default PR checks unless runtime is acceptable.
  - add optional or scheduled profile-pca gate job after stabilization.

### Rollout Milestones

1. Scaffold
- Create preprocess script, dataset schema, config stubs.
- Confirm `train.py --dry-run` works for new configs.

2. Trainability
- Train const-vc profile-pca with `k=3` (or variance target threshold).
- Produce initial reports and QC figures.

3. Benchmark
- Compare against current depthwise baseline for comparable target scope.
- Decide default `k` and metric thresholds.

4. Productionize
- Add quality-gate config + Make target.
- Document in README and update Definition of Done for profile-pca changes.

### Open Decisions To Resolve Early

- Fixed-time profile source:
  - exact time index or physical-time interpolation policy.
- PCA `k` policy:
  - fixed `k` (e.g., 4 or 5) vs variance threshold (e.g., 99%).
- Whether to include optional depth weighting in PCA fit.
- Whether to predict raw `T(z)` only first, or include `dTdt(z)` profile mode in same phase.

### Non-goals (for this phase)

- Full multi-output GP with coupled covariance across PCA components.
- Joint depth-time functional PCA (`T(z,t)`) in first implementation.
- Replacing existing per-depth emulator paths.

### Risks and Mitigations

- Risk: components capture artifacts instead of physics.
  - Mitigation: inspect component shapes and reconstruction residual structure vs depth.
- Risk: low-variance components become noise-dominated.
  - Mitigation: cap `k` and monitor per-component validation R2.
- Risk: data leakage via PCA fit on all data.
  - Mitigation: fit PCA on train split only; apply transform to val/test.

### Immediate Next-session Execution Checklist

1. Inspect current emulator preprocess/training interfaces:
  - `src/emulator/preprocess_one_training.py`
  - `src/emulator/train_emulator.py`
  - `train.py`
2. Draft profile-pca metadata schema and dataset naming.
3. Implement `preprocess_profile_pca.py` with train-only PCA fit.
4. Add `gp.*.profile-pca.yaml` configs and dry-run tests.
5. Add first QC plotting script for explained variance + reconstruction overlays.
6. Run on one suite (`const-vc`) and collect baseline metrics.

---

## Update (2026-03-10, profile-pca runs captured)

### What Was Executed

- Preprocessed profile-PCA datasets (train-only PCA fit, depth grid 0..80 km, step 1 km, val split seed 42):
  - `src/emulator/data/const-vc/profileT_pca_t5Myr_k8/`
  - `src/emulator/data/ramped-vc/profileT_pca_t5Myr_k8/`
  - exploratory variants:
    - `src/emulator/data/const-vc/profileT_pca_t0p5Myr_k8/`
    - `src/emulator/data/const-vc/profileT_pca_t5Myr_k8_wht/`
    - `src/emulator/data/const-vc/profileT_pca_t5Myr_k8_wht_id/`
- Trained GP (`gp_m25`) on 5 Myr `k=8` datasets:
  - `src/emulator/models/const-vc/profileT_pca_t5Myr_k8/gp_m25/`
  - `src/emulator/models/ramped-vc/profileT_pca_t5Myr_k8/gp_m25/`
- Generated profile-PCA QC figures:
  - `plots/qc-emulator/profile-pca/default-runs/const-vc/`
  - `plots/qc-emulator/profile-pca/default-runs/ramped-vc/`

### Dataset Snapshot

- `const-vc/profileT_pca_t5Myr_k8`:
  - split: `n_train=332`, `n_val=59`
  - PCA EVR: `PC1=0.9715`, `PC2=0.0188`, `PC3=0.0073`, cumulative `k=8 -> 0.9998`
- `ramped-vc/profileT_pca_t5Myr_k8`:
  - split: `n_train=423`, `n_val=75`
  - PCA EVR: `PC1=0.9671`, `PC2=0.0224`, `PC3=0.0083`, cumulative `k=8 -> 0.9998`
- `const-vc/profileT_pca_t0p5Myr_k8` exploratory preprocess-only:
  - split: `n_train=332`, `n_val=59`
  - PCA EVR: `PC1=0.9410`, `PC2=0.0510`, `PC3=0.0048`, cumulative `k=8 -> 0.9997`

### Baseline GP Metrics (score-space, macro averages)

- `const-vc/profileT_pca_t5Myr_k8/gp_m25`:
  - train: `RMSE=6.3694`, `MAE=4.5701`, `R2=0.9571`
  - val: `RMSE=21.9293`, `MAE=16.0852`, `R2=0.6060`
  - note: 1 component has negative validation `R2` (PC8)
- `ramped-vc/profileT_pca_t5Myr_k8/gp_m25`:
  - train: `RMSE=9.4731`, `MAE=6.3967`, `R2=0.8985`
  - val: `RMSE=28.3043`, `MAE=19.0063`, `R2=0.5911`
  - note: 1 component has negative validation `R2` (PC8)

### Current Interpretation

- Phase-1 pipeline is now functionally end-to-end for `profile-pca` at 5 Myr on both suites.
- Leading components dominate variance strongly (>96% in PC1), and `k=8` captures ~99.98% variance.
- Validation quality is usable but not yet production-gate ready due to weak tail-component generalization.

### Next Session TODO (from current state)

1. Add reconstruction-space metrics to reports (global/profile RMSE, depthwise RMSE, P90 depth RMSE) and log them in `report.json`.
2. Decide `k` policy for default runs (`fixed k=8` vs variance threshold + component quality floor).
3. Evaluate whether to trim/weight tail PCs (e.g., drop components with unstable val `R2`) and compare reconstruction impact.
4. Add/lock `emulator-quality.profile-pca.yaml` thresholds and wire a `make` quality-check target for profile-PCA.
5. Run/record the same baseline for whitened variants if they remain candidates (`*_wht`, `*_wht_id`), otherwise de-scope.

---

## Update (2026-03-11, profile-PCA defaults + GP tuning + repo cleanup)

### What Was Added/Changed

- Profile-PCA workflow was brought up to parity with the single-depth workflow:
  - added profile-PCA smoke tests
  - added synthetic preprocess tests
  - added profile-PCA quality report generation
  - added profile-PCA quality validation against thresholds
- Added profile-PCA model-selection tooling:
  - PCA representation sweep over `k` and `score_space`
  - summary-table generation for sweep results
- Added profile-PCA GP tuning tooling on the chosen PCA representation:
  - resumable sweep runner
  - summary-table generation for GP tuning results
- Refactored emulator scripts into grouped directories:
  - `src/emulator/single_depth/`
  - `src/emulator/profile_pca/`
  - `src/emulator/legacy/`
- Removed transitional flat wrappers after updating references.
- Added repo orientation docs:
  - `docs/repo-map.md`
- Added `.gitignore` coverage so generated profile-PCA datasets/models/sweep outputs stay local-only by default.

### Decisions Made

- Default profile-PCA setup is now:
  - target time: `t3Myr`
  - retained components: `k=10`
  - score space: `whitened`
- CI policy was trimmed to stay practical:
  - `test` remains routine
  - single-depth quality gate remains routine
  - profile-PCA quality gate is manual-only (`workflow_dispatch`)
  - profile-PCA CI representative subset is `ramped-vc` only
- Keep generated analysis artifacts local/untracked unless there is a specific reason to version them.

### Sweep Result Summary

- PCA representation sweep favored `k=10`, `whitened` overall.
- `ramped-vc` had the clearest preference for `k=10`.
- `const-vc` differences between `k=8` and `k=10` were very small, but `k=10_whitened` remained a defensible shared default.
- `raw` vs `whitened` changed score-space metrics strongly, but only weakly changed reconstruction-space metrics.
- Ranking policy used:
  1. `val_profile_rmse`
  2. `val_profile_p95_rmse`

### Current GP Tuning Status

- Added `make profile-pca-gp-tuning-sweep`
- Added `make profile-pca-gp-tuning-summary`
- Default tuning target is:
  - suite: `ramped-vc`
  - dataset: `profileT_pca_t3Myr_k10_whitened`
- Default GP tuning grid:
  - kernels: `matern25`, `matern15`, `rbf`
  - restarts: `10`, `25`
  - length-scale upper bounds: `1e3`, `1e4`
  - noise lower bounds: `1e-6`, `1e-8`
- The tuning sweep is resumable:
  - completed tags with `profile_pca_quality.json` are skipped on rerun
  - use the same command again to continue an interrupted run

### Important Repo State Notes

- The emulator tree is now organized by workflow, but `Makefile`, `README.md`, and `docs/repo-map.md` should be checked again next session after a fresh restart to make sure all wording and references still feel clean after the refactor.
- Generated artifacts were accidentally staged once during cleanup and then explicitly removed from tracking in a follow-up commit; local files were kept on disk.

### Recent Commits (latest first)

- `b6958f3` chore: ignore generated profile PCA artifacts
- `77002ab` refactor: remove transitional emulator wrappers
- `63e2be1` refactor: group emulator workflows by role
- `97169ad` docs: clarify repo map and legacy helpers
- `7ba6d26` feat: resume profile PCA GP tuning sweep
- `4f532fb` feat: add profile PCA GP tuning sweep

### Next Session TODO

1. Resume the interrupted GP tuning sweep:
   - `make profile-pca-gp-tuning-sweep`
2. Summarize GP tuning results:
   - `make profile-pca-gp-tuning-summary`
3. Decide whether GP defaults should change from the current baseline based on:
   - `val_profile_rmse`
   - `val_profile_p95_rmse`
4. Re-check `Makefile`, `README.md`, and `docs/repo-map.md` after the refactor and tighten any remaining naming/workflow wording if needed.
5. After GP tuning is settled, shift from infrastructure work to science-facing use:
   - sensitivity analysis
   - talk figures / summary plots
   - future Bayesian-inference framing

---

## Update (2026-03-13, science plots + workflow cleanup)

### What Was Cleaned Up

- `src/emulator/single_depth/` now has:
  - `core/`
  - `qc/`
  - `science/`
- `src/emulator/profile_pca/` now has:
  - `core/`
  - `qc/`
  - `science/`
  - `sweeps/`
- Added profile-PCA QC wrapper:
  - `src/emulator/profile_pca/make_qc_emulator_plots.sh`
- Profile-PCA QC plot outputs were reorganized under:
  - `plots/qc-emulator/profile-pca/default-runs/<suite>/`
  - `plots/qc-emulator/profile-pca/pca-sweep/`
  - `plots/qc-emulator/profile-pca/gp-tuning/`
- Single-depth science outputs now default to:
  - `plots/science-emulator/single_depth/<suite>/`

### Single-Depth Science Plot Status

- Current single-depth science wrapper:
  - `bash src/emulator/single_depth/make_science_emulator_plots.sh const-vc`
  - `bash src/emulator/single_depth/make_science_emulator_plots.sh ramped-vc`
- Wrapper now generates:
  1. all-depths sensitivity lines
  2. single-depth partial-dependence summaries at `10/40/70 km`
  3. response surfaces at `10/40/70 km`
  4. stacked response-surface figure at `10/40/70 km`
- Preferred all-depths figure is now:
  - `*_sensitivity_lines_all_depths.png`
- The older heatmap-style all-depths figure was removed from the active workflow.

### Profile-PCA Science Plot Status

- Drafted and ran two first-pass science plots for `const-vc`:
  - `plots/science-emulator/profile-pca/const-vc/const-vc_profile_pca_time_summary.png`
  - `plots/science-emulator/profile-pca/const-vc/const-vc_profileT_pca_t3Myr_k10_gp_m25_sensitivity.png`
- New scripts:
  - `src/emulator/profile_pca/science/plot_profile_pca_time_summary.py`
  - `src/emulator/profile_pca/science/plot_profile_pca_sensitivity.py`

### Other Notes

- Profile-PCA default run artifacts exist for both suites and all times:
  - `0.5, 1, 2, 3, 4, 5 Myr`
- Stray duplicate profile-PCA data dirs outside `runs/` were removed.
- QC typography for single-depth plots was cleaned up to use nicer symbols/labels.
- Single-depth pred-vs-true QC now defaults to representative depths:
  - `20, 30, 40, 50, 60, 70 km`

### Recent Commits

- `36f5fd9` feat: add profile PCA science plots
- `1bdecd3` refactor: align emulator workflows and plot layouts
- `17f5d92` refactor: streamline single-depth science plots
- `493cf5e` feat: add single-depth science plot suite

### Suggested Next Steps

1. Refine the two draft profile-PCA science figures for talk use:
   - simplify titles/labels
   - improve legend/text sizing
   - decide whether to keep both or replace one with a cleaner reconstruction snapshot
2. Generate the matching profile-PCA science plots for `ramped-vc`.
3. Decide the talk figure shortlist:
   - single-depth validation
   - single-depth sensitivity
   - profile-PCA reconstruction / time-summary
   - one interaction/surface plot if it clearly helps
4. If needed, add a small wrapper for profile-PCA science plots analogous to the single-depth science wrapper.
