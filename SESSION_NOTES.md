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

---

## Update (2026-03-14, profile-PCA science/path utilities + cleanup)

### What Was Added

- Added profile-PCA science wrapper:
  - `src/emulator/profile_pca/make_science_emulator_plots.sh`
- Added/updated profile-PCA science plot scripts:
  - `src/emulator/profile_pca/science/plot_profile_pca_time_summary.py`
  - `src/emulator/profile_pca/science/plot_profile_pca_sensitivity.py`
  - `src/emulator/profile_pca/science/plot_profile_pca_profile_family.py`
- Added profile-PCA path utilities:
  - `src/emulator/profile_pca/utilities/compute_burial_path.py`
  - `src/emulator/profile_pca/utilities/compute_many_burial_paths.py`
  - `src/emulator/profile_pca/utilities/compute_burial_path_uncertain_parameter.py`
  - `src/emulator/profile_pca/utilities/compute_example_paths.sh`
  - `src/emulator/profile_pca/utilities/PLOT_NOTE.md`

### Current Profile-PCA Science Outputs

- Science plots now live under:
  - `plots/science-emulator/profile-pca/<suite>/`
- Wrapper command:
  - `bash src/emulator/profile_pca/make_science_emulator_plots.sh const-vc`
  - `bash src/emulator/profile_pca/make_science_emulator_plots.sh ramped-vc`
- Current wrapper outputs for one suite:
  - `*_profile_pca_time_summary.png`
  - `*_profile_pca_0p5_3_5Myr_gp_m25_sensitivity.png`
  - `*_profile_pca_0p5_3_5Myr_v_conv_gp_m25_profile_family.png`
  - `*_profile_pca_0p5_3_5Myr_age_SP_gp_m25_profile_family.png`
  - `*_profile_pca_0p5_3_5Myr_age_OP_gp_m25_profile_family.png`

### Current Profile-PCA QC Outputs

- Profile-PCA QC plots now live under:
  - `plots/qc-emulator/profile-pca/default-runs/<suite>/`
  - `plots/qc-emulator/profile-pca/pca-sweep/`
  - `plots/qc-emulator/profile-pca/gp-tuning/`
- Added combined QC reconstruction plot:
  - `src/emulator/profile_pca/qc/plot_profile_pca_combined_reconstruction.py`
- Removed older separate reconstruction plotters:
  - `plot_profile_pca_reconstruction.py`
  - `plot_profile_pca_emulator_reconstruction.py`
- Profile-PCA QC wrapper:
  - `bash src/emulator/profile_pca/make_qc_emulator_plots.sh const-vc`
  - `bash src/emulator/profile_pca/make_qc_emulator_plots.sh ramped-vc`

### Path-Utility Conventions

- Path preview plots are now hardcoded to common presentation bounds:
  - temperature: `-50 to 600 °C`
  - depth: `0 to 60 km`
  - time axis on right panel: `0 to 5 Myr`
- This is documented in:
  - `src/emulator/profile_pca/utilities/PLOT_NOTE.md`
- `compute_burial_path.py` now supports:
  - distinct burial and exhumation rates
  - hold time at max depth (`--exhumation-time-myr`)
  - smoothed burial/hold/exhumation transitions (`--transition-time-myr`)
- Both single-path and uncertain-path utilities now mark the max-temperature point with a star.

### Example Path Workflow

- Helper command:
  - `bash src/emulator/profile_pca/utilities/compute_example_paths.sh`
- For `ramped-vc`:
  - `SUITE=ramped-vc bash src/emulator/profile_pca/utilities/compute_example_paths.sh`
- Current helper runs:
  1. single-path reference case
  2. uncertain `age_SP` case
  3. many-path sweep
- Current reference settings in the helper:
  - `start_time = 0.5 Myr`
  - `burial_rate = 3.0 cm/yr`
  - `exhumation_rate = 1.5 cm/yr`
  - `hold_time = 0.75 Myr`
  - `transition_time = 0.1 Myr`
  - `z_max = 35 km`
  - `age_SP` is intentionally omitted in the reference case so it defaults to the training median
- Current many-path sweep ranges:
  - burial rate: `2 to 6 cm/yr` with 4 values
  - max depth: `20 to 45 km` every `2.5 km`
  - hold time: `0 to 1 Myr` with 4 values

### Cleanup / Repo State

- Removed repo `__pycache__` directories outside `env/`.
- Cleaned stale references in `SESSION_NOTES.md`:
  - old `make_param_sweep_plots.sh`
  - old `plots/qc-emulator/profile-pca/runs/...` paths
- Removed empty placeholder directory:
  - `src/emulator/models/profile_pca/const-vc/gp_tuning`
- `Makefile`, `README.md`, and `docs/repo-map.md` were rechecked and are consistent with the current workflow/layout.

### Important Git Note

- Commit `c45f815` accidentally included generated profile-PCA model artifacts.
- This was corrected immediately in follow-up commit:
  - `d1a364a` `chore: revert generated profile PCA artifacts`
- Nothing from this latest `SESSION_NOTES.md` update is committed yet.

### Suggested Next Steps

1. If needed, generate the `ramped-vc` profile-PCA science plot suite with:
   - `bash src/emulator/profile_pca/make_science_emulator_plots.sh ramped-vc`
2. Decide which profile-PCA talk figures make the final cut:
   - time summary
   - 3-time sensitivity
   - one or more profile-family plots
   - one path utility illustration
3. If the path utilities are now part of the talk story, consider one small wrapper or README snippet focused just on the path examples.

---

## Update (2026-03-15, talk polish + path plot alignment)

### Presentation-Specific Outputs

- Added a temporary presentation variant of the single-depth all-depths sensitivity figure:
  - `plots/science-emulator/single_depth/const-vc/const-vc_dTdt_gp_m25_sensitivity_lines_all_depths_presentation.png`
- In that variant:
  - `v_conv` and `age_OP` retain their original colors
  - all other parameter lines are muted gray
  - axes / axis labels / ticks remain black
- This was generated as a one-off presentation asset without changing the main science plotting script.

### Path Plot Layout Change

- Profile-PCA path preview plots now use:
  - left panel: depth vs time
  - right panel: temperature-depth path
- This was applied consistently to:
  - `compute_burial_path.py`
  - `compute_many_burial_paths.py`
  - `compute_burial_path_uncertain_parameter.py`
- Added a shared fixed figure-layout helper so the three path plot types align much more cleanly for slide overlays.

### Refreshed Example Path Outputs

- Re-ran the `const-vc` path examples after the panel swap / layout alignment:
  - single-path preview
  - uncertain `age_SP` preview
  - many-path preview
- Re-ran the same full example suite for `ramped-vc`:
  - `SUITE=ramped-vc bash src/emulator/profile_pca/utilities/compute_example_paths.sh`

### Notes

- Path plot bounds remain fixed for presentation consistency:
  - temperature: `-50 to 600 °C`
  - depth: `0 to 60 km`
  - time: `0 to 5 Myr`
- The path utilities still emit a harmless `joblib` serial-mode warning in this environment.

### Recent Commit

- `3e14749` `refactor: align profile PCA path plot layouts`

### Suggested Next Steps

1. Decide whether the path plots need one final pass for talk readability:
   - remove or move the many-path range text
   - further simplify colors/line weights if overlaid in slides
2. If the presentation variant of the single-depth sensitivity figure is useful, make an equivalent `ramped-vc` version.
3. Finalize the talk figure shortlist and stop changing infrastructure unless something blocks figure export.

---

## Plan (2026-05-27, return to fundamentals after talk)

### Context

- Talk was delivered and went well.
- Infrastructure phase (workflows, defaults, QC, science plots) is closed.
- Repo housekeeping done in commits `a055979 -> 9175961`:
  - relocated single-depth QC plots under `single_depth/` layout
  - added 5-km-step single-depth runs (5/15/25/35/45/55/65/75 km, both suites)
  - tracked profile-PCA QC plots, pruned stale `.gitignore` patterns
  - committed science plot outputs + schematic removal
  - logged March 15 talk-polish session
- Next phase is scientific extension, not infrastructure.

### Immediate Priority — `const-vc-sh` data generation

- New third suite added: `const-vc-sh` (const-vc with shear heating).
  - directory: `subd-model-runs/const-vc-sh/`
  - sits alongside existing `const-vc` and `ramped-vc` — independent suite, no replacement.
- First task is generating the numerical-model runs for this suite (upstream
  of the emulator).
- Once runs exist, the standard downstream pipeline applies:
  1. QC numerical mods (`src/qc-numerical-mods/`) — health checks, profile QC.
  2. Single-depth preprocessing + training (`src/emulator/single_depth/`).
  3. Profile-PCA preprocessing + training (`src/emulator/profile_pca/`).
  4. QC + science plot regeneration extended to the new suite.
- Eventually, `const-vc-sh` becomes a third option in all wrapper scripts
  that currently iterate over `{const-vc, ramped-vc}`.

### Direction Ranking

Four deferred threads were reviewed. Ranked by value/effort for this project:

| # | Thread | Effort | Payoff | Status |
|---|---|---|---|---|
| 3 | Quantitative parameter sensitivity (Sobol indices etc.) | Low (days) | High, publishable | **Start here** |
| 1 | T(z,t) joint depth-time functional PCA | High (weeks) | High, scientifically central | **Main thrust** |
| 4 | Coupled multi-output GP across PCA components | Medium-high | Moderate, model-quality refinement | **Park** until #1 reconstruction-uncertainty needs it |
| - | Bayesian inversion of observed P-T data | High | High, application-facing | **Park** for future work |

### Why this order

- **#3 first** because it reuses the existing emulator (just samples it heavily via `SALib` or similar), produces quantitative numbers that beat the talk-figure-level sensitivity, and gives a baseline to compare against once #1 lands.
- **#1 as main thrust** because it is the explicit "Phase 2" of the 2026-02-20 plan and the natural next extension of the project's stated goal (emulating the full thermal field, not just fixed-time slices).
- **#4 parked** because PC1 already captures >96% of variance, so the marginal reconstruction gain is bounded. Most concrete benefit is rescuing tail PCs (e.g. PC8 had negative val R² in the March 10 baseline) and producing honest joint profile uncertainty. Reconsider once #1 is in place and Bayesian framing comes back online — that's where joint uncertainty becomes load-bearing.
- **Bayesian parked** explicitly as future work.

### Thread Notes

**#3 — Quantitative sensitivity**
- Tooling candidate: `SALib` for Sobol indices.
- Apply to both single-depth emulators (per depth) and profile-PCA emulators (per PC or after reconstruction).
- Likely a new `src/emulator/{single_depth,profile_pca}/science/` script + a wrapper, output under `plots/science-emulator/.../sensitivity-sobol/` or similar.

**#1 — T(z,t) extension**
- Joint depth-time functional PCA.
- Requires new preprocessing path producing T(z,t) snapshots across the time grid (currently only fixed-time slices are extracted).
- Decide: stack times into one big PCA, or factorize via tensor/2D PCA? Tensor approach preserves smoothness across both axes.
- New dataset naming (e.g. `profileTzt_pca_*`) to keep parallel to existing fixed-time path.
- Will need new QC: reconstruction overlays vs (z,t), error decomposition by time and depth.

**#4 — Coupled multi-output GP (parked)**
- Frameworks: LMC or simpler ICM (separable kernel + task covariance matrix).
- Cost: naive `O((k·n)^3)` vs `O(k·n^3)` for independent. Mitigatable via low-rank ICM.
- Tooling: `sklearn` GP doesn't support multi-output GPs in this sense; would need `GPy` or `GPyTorch`. New dependency.
- Revisit when reconstruction-uncertainty story becomes load-bearing.

### Immediate Next-session Execution Checklist

1. **Generate `const-vc-sh` numerical-model runs.**
   - Decide parameter coverage (mirror existing const-vc or different?).
   - Confirm shear-heating implementation in the model config.
2. **Optional in parallel:** start Thread #3 (Sobol) on existing const-vc + ramped-vc emulators while shear-heating runs are computing.
   - Decide tooling (`SALib` vs custom) and target metrics (first-order S1, total ST, optionally S2).
   - Pick first deliverable: single-depth Sobol for one suite/depth (e.g. const-vc 40 km), validate against existing visual sensitivity.
3. Once `const-vc-sh` runs exist:
   - Run `src/qc-numerical-mods/` health checks; refresh QC outputs.
   - Run single-depth preprocessing + training for the new suite.
   - Run profile-PCA preprocessing + training (use current defaults: `t3Myr, k=10, whitened, matern25`).
   - Regenerate QC + science plots for the new suite.
4. Extend Sobol (Thread #3) to include `const-vc-sh` once emulators exist; compare sensitivity rankings across suites.
5. Update wrapper scripts (`make_science_emulator_plots.sh`, `make_qc_emulator_plots.sh`) to accept `const-vc-sh` and add Makefile shortcuts.
6. Update README / `docs/repo-map.md` once the new suite is integrated.

### Quick Resume Prompt

> Continue from `SESSION_NOTES.md` 2026-05-27 plan. Immediate priority is `const-vc-sh` data generation; in parallel, optionally start Thread #3 (Sobol sensitivity) on existing emulators. T(z,t) extension (#1) and coupled multi-output GP (#4) follow.

---

## Update (2026-05-28, Thread #3 kickoff — Sobol sensitivity for single-depth emulators)

### Context

- `const-vc-sh` data generation is still blocked on the student's shear-heating
  model template, so this session took the parallel track: Thread #3
  (quantitative Sobol sensitivity), which is independent of the new runs.
- Replaces the qualitative one-at-a-time (OAT) sensitivity figure with
  variance-based Sobol indices computed by heavily sampling the existing trained
  single-depth GP emulators.

### What Was Added

- New dependency: `SALib>=1.5` (installed 1.5.2) in `requirements.txt`.
- New scripts under `src/emulator/single_depth/science/`:
  - `_sobol_io.py` — shared loader/predict/label helpers + `build_problem`
    (sampling box from per-feature training quantiles `[0.01, 0.99]`).
  - `compute_sobol_sensitivity.py` — one emulator → Sobol JSON (S1, ST, S2 with
    conf intervals, val metrics, depth). Uses SALib `ProblemSpec`,
    `calc_second_order=True`, fixed seed (deterministic).
  - `plot_sobol_sensitivity.py` — per-emulator figure: ranked S1/ST bars + S2
    interaction heatmap.
  - `plot_sobol_vs_depth.py` — headline S1/ST-vs-depth figure per suite.
  - `make_sobol_plots.sh` — wrapper iterating depths × suites.
- `make sobol-sensitivity` target (override `SOBOL_SUITES`/`DEPTHS`/`N_BASE`).
- `tests/test_sobol_smoke.py` — additive-model assertions + scaler-inversion
  check; skips cleanly if SALib missing. Full suite now 28 tests (was 24).

### Design Decisions (settled with user)

- Scope: single-depth, both suites, `<depth>km_dTdt`, full 5 km grid (5..80 km).
  `_thermalParam` variants excluded from this first pass.
- Sampling box: central training quantiles `[0.01, 0.99]` (keeps Saltelli samples
  inside well-supported GP regions; mirrors the OAT script's quantile policy).
- Indices: S1, ST, **and** S2 (`calc_second_order=True`).
- Code is feature-count agnostic (reads `feature_cols` from metadata):
  `dTdt` const-vc = 5 params; ramped-vc = 7 (`t_conv`, `v_conv_over_tconv`).
  N=1024 base → 12288 evals (const-vc) / 16384 evals (ramped-vc) per emulator.

### Scientific Result

- Clean depth-dependent control of cooling rate:
  - shallow (≲35 km): **overriding-plate age `age_OP`** dominates (plus `age_SP`
    at the very shallowest).
  - deep (≳40 km): **convergence velocity `v_conv`** dominates.
  - distinct crossover near ~40 km in both suites.
- `ramped-vc`: `v_conv` dominates earlier/more strongly; `dip_int` (slab dip)
  emerges at depth.
- `ST − S1` gaps are small at most depths (cooling rate is largely additive),
  widening near the crossover where `v_conv`/`age_OP` interact.
- Sanity check passed: top-ranked ST params at const-vc 40 km (`v_conv`,
  `age_OP`) match the talk's OAT/presentation figure.

### Outputs

- `plots/science-emulator/single_depth/<suite>/sobol/`:
  - 16 per-emulator JSON+PNG pairs per suite.
  - `<suite>_sobol_vs_depth_gp_m25.png` headline figure per suite.
- Layout polish: per-emulator info box moved to mid-right so it no longer
  overlaps the lower-right legend (verified on both 5- and 7-param panels).

### Suggested Next Steps

1. Profile-PCA Sobol (deliberate second pass): per-PC indices + post-
   reconstruction Sobol vs depth, analogous tooling under
   `src/emulator/profile_pca/science/`.
2. Once `const-vc-sh` emulators exist, extend Sobol to it and compare rankings
   across all three suites.
3. Optionally consider S2 only where interactions look real (near the crossover);
   most depths are near-additive.

### Quick Resume Prompt

> Continue from `SESSION_NOTES.md` 2026-05-28 update. Thread #3 (single-depth
> Sobol) is done and committed. Next: profile-PCA Sobol second pass, and/or pick
> up `const-vc-sh` data generation once the shear-heating template arrives.

---

## Direction Review (2026-05-28, publishability / geodynamic-insight synthesis)

Stepping outside the 2026-05-27 thread ranking to ask what actually makes this
publishable. This **updates** that ranking; read the two together.

### Validation status — corrected after checking the code

- **Held-out runs are real and clean.** Every dataset carries `train_idx`/
  `val_idx`; profile-PCA holds out ~15% (`n_val=59` const-vc, `75` ramped). The
  six time-slice datasets (`t0p5..t5Myr`) hold out the **same 59 runs in the same
  order** (verified: `val_idx` identical across all times, `X_raw` rows match), so
  a held-out run's full P–T–t path is reconstructable with no leakage.
- **Physical-unit validation already exists.** `evaluate_profile_pca_quality.py`
  reports reconstruction RMSE in °C, `rmse_by_depth`, and a PCA-truncation
  baseline on held-out runs at each time. (An earlier worry that physical-unit /
  depthwise validation was missing was wrong — it's there.)
- **The one genuine gap: the time interpolation between anchors is untested.**
  All metrics sit exactly on an anchor (t = 0.5/1/2/3/4/5 Myr), but
  `compute_burial_path.py` builds paths by linear interpolation in score space
  *between* anchors (`_interpolate_profile_in_time`), where a rock spends almost
  all its time. Cheap check: hold out t = 2 Myr, predict it by interpolating the
  t = 1 and t = 3 emulators, compare to true t = 2 profiles in °C vs depth. Small
  error → paths trustworthy, de-prioritize T(z,t). Large error near slab top →
  that *is* the quantified motivation for T(z,t).

### Two-paper framing (recommended scope split)

- **Paper 1 (methods + first insight; G-cubed / GMD):** emulator construction,
  the time-interpolation check above, the Sobol result reframed as a scaling law,
  and the shear-heating contrast.
- **Paper 2 (application; EPSL / Nature Geo ambition):** Bayesian inversion of
  observed metamorphic P–T data using the fast surrogate.

### Re-prioritized science moves (highest value first)

1. **Reframe Sobol as a thermal-parameter (Φ ≈ age·v_conv, + dip) scaling law.**
   The `age_OP`→`v_conv` crossover at ~40 km and the `ST−S1` interaction gap near
   it likely *are* the classic Φ coupling (Kirby/Stein/England). Test whether
   emulator output collapses onto Φ. Where it collapses = recovered known physics
   + theory-level validation; where it breaks = the interesting part (shear
   heating, overriding-plate structure, wedge flow). Turns a bar chart into a
   result. Doable now on existing emulators.
2. **`const-vc-sh` shear-heating contrast is probably the headline, not "suite #3."**
   Interface shear heating controls dehydration, the seismic–aseismic transition,
   and the warm-forearc paradox — actively debated. `const-vc` vs `const-vc-sh`
   in both P–T and sensitivity ranking is likely the most novel content. Treat as
   load-bearing science, not housekeeping. (Still gated on the student's template.)
3. **Bring in one observational dataset — even before inversion.** The repo has
   zero contact with the rock record (grep: no Penniston-Dorland / Syracuse /
   metamorphic-compilation references). Overlay a natural subduction P–T
   compilation against the emulator's accessible P–T envelope: does the ensemble
   bracket the data? A systematic miss (e.g. natural rocks hotter than the coldest
   no-shear-heating models) is both a result and the motivation for shear heating
   *and* for inversion. One strong figure; biggest current publishability gap.
4. **Inversion as the methodological payoff** — and two parked items become
   load-bearing the moment it starts:
   - **T(z,t)** (notes' "main thrust"): needed so a rock samples a *continuous*
     field, not the linear-in-time stitch; required to invert path *shape*.
   - **Multi-output GP** (notes' "park"): independent-per-PC GPs give the *wrong
     joint uncertainty* on reconstructed profiles; inversion is all about honest
     posteriors, so un-park it as soon as inversion begins.

### Two sharp cautions

- **Is `dT/dt` the right primary target?** Cooling rate is not directly recorded
  in the rock record (peak P–T and path segments are). Lean toward T(z,t)/paths as
  the deliverable and demote `dT/dt` to a derived diagnostic before building more
  on the cooling-rate target.
- **Box-edge extrapolation.** Sobol box and (future) inversion posteriors can walk
  to parameter-space corners where the GP is unconstrained. Report/gate this or the
  inferences are soft.

### Net change vs the 2026-05-27 ranking

- Validation: **down** from "foundational, do first" to "one afternoon check" (the
  time-interpolation test) — most of it already exists.
- Quantitative sensitivity (#3): keep, but **upgrade** to the Φ-scaling framing.
- T(z,t) (#1) and multi-output GP (#4): keep, but reframed as *prerequisites for
  honest inversion* rather than standalone goals.
- Bayesian inversion: **promote** from "parked future work" to the explicit
  payoff / Paper 2 — it's the reason a fast surrogate exists.
- New, not previously ranked: **observational anchor** (move #3 above) — the
  single biggest lever from "tool" to "insight."

### Immediate doable-now shortlist (no template needed)

1. Time-interpolation hold-out check (settles T(z,t) necessity).
2. ~~Φ-collapse test~~ → transient scaling, see correction below.
3. Scope/sanity pass on whether `dT/dt` or T(z,t)/paths should be the headline
   target.

---

## Correction + First Result (2026-05-28, transient cooling scaling — NOT steady-state Φ)

### Correction

- The thermal parameter Φ = age·v_conv·sinθ is a **steady-state** scaling (depth of
  slab isotherms at equilibrium; Kirby/Stein/England). The emulator target here is
  **transient cooling**: `dTdt_C_per_Myr` = ΔT over a finite ~4.5 Myr window
  (`master_DT1-10.csv`: `T1_C, T2_C, dT_C, dt_Myr`; `dt_Myr ≈ 4.50`, ~constant).
  A naive Φ-collapse is dimensionally/physically wrong. The right move is to build
  a **transient** dimensionless group and let the Sobol indices pick which one
  governs which depth.

### Candidate transient groups

- Diffusive / initial-condition: `δ_OP = √(κ·age_OP)` (overriding-plate conductive
  lid thickness); similarity variable `η = z / √(κ·age_OP)`.
- Advective penetration: `L_adv = v_conv·sinθ·Δt`; variable `ζ = z / L_adv` or
  depth-local Péclet `Pe(z) = v_conv·sinθ·z/κ`.
- (κ = 1e-6 m²/s = 31.5 km²/Myr; z = depth below surface; vertical advection
  ~ v·sinθ — geometry assumptions to confirm against the model setup.)

### First result (const-vc, raw ensemble, exploratory)

Script: `src/emulator/single_depth/science/explore_transient_scaling.py`
(outputs under `plots/science-emulator/single_depth/const-vc/scaling/`).

- **The age_OP → v_conv control crossover sits at the OP conductive-lid thickness.**
  Cross-run correlation of cooling with `age_OP` vs `v·sinθ` crosses at
  **z ≈ 42.7 km**; `√(κ·age_OP_median) = 43.5 km`. Near-exact, with a *standard,
  untuned* κ. The advective scale `L_adv ≈ 155 km` is far deeper — so the crossover
  is set by the **diffusive/initial OP scale, not advective penetration**. This is
  the clean physical interpretation of the Sobol ~40 km crossover.
- **Shallow cooling collapses on `η = z/√(κ·age_OP)`** (diffusive regime); curves
  fan out for `η ≳ 1.5` (advective regime needs `ζ`/Pe instead) — consistent with a
  two-regime similarity picture.
- **Dip co-emerges with v_conv only in the advective (deep) regime** and is inert
  shallow — consistent with both entering via `w = v·sinθ`. NOTE: the raw Sobol
  `ST_dip/ST_vconv` ratio is *not* depth-invariant (rises ~0.005→0.14 with depth),
  but that reflects Sobol's **range-weighting** (dip's sampled range gives less
  variance than v_conv's), not the functional form. Test the grouping by collapsing
  on `w` directly, not via the ST ratio.

### Honest caveats

- The split-by-age_OP **conditional test is confounded** by range restriction
  (correlating against a variable whose range you just truncated) and gave an
  inverted, artifactual result — treat as inconclusive. A valid test of
  `z_cross ∝ √(κ·age_OP)` should be **emulator-based**: hold other params fixed,
  vary age_OP, find where ∂(cooling)/∂age_OP and ∂(cooling)/∂(v·sinθ) cross, and
  check the crossover moves like `√(κ·age_OP)`. This is the right next step before
  claiming a law rather than a median coincidence.
- All exploratory: raw-ensemble correlations, standard κ, assumed geometry.

### Why this matters

- A *transient* forearc-cooling scaling is novel (steady-state Φ is textbook), it is
  built entirely from existing assets (ensemble + Sobol), and the crossover =
  √(κ·age_OP) result is a clean, reportable anchor for Paper 1:
  "Sobol indices reveal a two-regime (diffusive OP-lid vs advective) control on
  transient forearc cooling, with the regime boundary at the OP conductive-lid
  thickness."

### Predictive test — can the groups actually predict ΔT?

Fit the same learner (RandomForest) on different feature sets, split by run
(no depth leakage), held-out R²/RMSE(°C) on ΔT = `dT_C`. Velocity ramp confirmed
from the ASPECT `.prm`: `v(t) = v_conv·min(t/t_conv, 1)` (linear to v_conv over
t_conv, then constant); cooling window ≈ 0.5→5.0 Myr (snapshots 1–10, dt≈4.5).
So cumulative convergence `D(t)=v_conv·t²/(2 t_conv)` (ramping) else
`v_conv·(t−t_conv/2)`, effective velocity `v_eff=D(t_end)/t_end`.

| feature set | const-vc R² | ramped-vc R² |
|---|---|---|
| S2  {η, ζ}                | 0.881 | 0.900 (instantaneous v) |
| S2cum {η, ζ_cum}          | —     | **0.914** |
| S3  {η, ζ, Pe}            | **0.929** | **0.967** (ζ,Pe from v_eff) |
| R3  {age_OP, v_conv, dip, z} | 0.938 | 0.933 |
| R6  {all raw params + z}  | 0.977 | 0.972 |

Findings:
- **Two groups predict ~88–90% of held-out ΔT variance; three groups recover the
  raw ingredients.** S3 ≈ R3 for const-vc (0.929 vs 0.938) — to good approximation
  ΔT is a function of {η, ζ, Pe}. Error is **regime-localized**: residuals ~0 above
  √(κ·age_OP), fanning out below it (advective regime); the depth-colored
  predicted-vs-true scatter shows shallow points on the 1:1 line, deep points
  under-predicted.
- **Velocity history matters (transient signature confirmed).** ramped-vc:
  cumulative > instantaneous (0.914 > 0.900), and S3cum (0.967) **beats** R3 (0.933)
  and nearly equals R6 (0.972) — because `v_eff` encodes the `t_conv` ramp that the
  raw `{age_OP, v_conv, dip, z}` set does not. Steady-state Φ cannot see this.
- **What the scaling can't capture:** R6 − R3 (~0.94→0.97) is the `age_SP` / `eta_UM`
  signal, which the age_OP/convergence scaling omits entirely — a bound on the
  approach, not a defect.
- RMSE: S3 ~30–40 °C on a several-hundred-°C ΔT range — good first cut, not yet
  petrology-grade.
- Note: const-vc crossover 42.7 km ≈ √(κ·age_OP)=43.5 km; ramped-vc crossover is
  shifted shallower (~31 km), plausibly the ramp moving the advective onset — not
  over-interpreted yet.

### Geometry clarified (2026-05-28)

The sampled quantity is the **slab-top (interface) temperature** at vertical depth
`z` — top of the crustal layer — NOT a fixed point in the overriding plate. This
makes the physics the classic transient slab-top problem and explains the Sobol
result exactly:
- shallow: slab top sits against the OP base → controlled by the OP geotherm
  (`age_OP`), scale `√(κ·age_OP)` (= the crossover);
- deep: a parcel reaches `z` by descending the interface in `t_desc ≈ z/w`
  (`w = v·sinθ`); heating en route is set by the **descent Péclet** `Pe = w·z/κ`
  (fast/deep → cold slab top → `v_conv` dominates) — McKenzie / Molnar–England;
- `age_SP` weak because the slab *top* equilibrates to the wedge far faster than
  the slab core.

This corrected the advective variable: the right one is **`Pe = w·z/κ`**, not the
window-penetration `ζ = z/(w·Δt)` used in the first pass.

### Closed-form law — physically grounded (real coefficients)

Anchored on the OP half-space geotherm `T_init = T_m·erf(z/(2√(κ·age_OP))) =
T_m·erf(η/2)`. Fit held-out by run; `_closed_form_scaling()` in
`explore_transient_scaling.py`.

| model / ceiling | const-vc R² | ramped-vc R² |
|---|---|---|
| Model 1  `ΔT = −A·erf(η/2)` (1 const) | 0.561 | 0.481 |
| Model 2  `ΔT = −A·erf(η/2)·(1−e^(−(Pe/p)^q))` (3 const) | 0.606 | 0.671 |
| RF{η} (flexible η-only ceiling) | 0.754 | 0.550 |
| RF{η, Pe} | 0.890 | 0.912 |
| RF{η, Pe, ζ} | 0.929 | 0.967 |

Recovered constants are physical: `A ≈ 536 °C`, `T_m ≈ 1300 °C` (correct mantle
value), efficiency `A/T_m ≈ 0.41`.

Findings (honest):
- **Variables now physically identified and data-confirmed:** η (diffusive, OP lid)
  + **Pe** (advective, slab-descent) — `RF{η,Pe}` (0.89/0.91) ≥ `RF{η,ζ}` (0.88/0.91),
  and Pe is the geometry-derived variable. `ζ` adds the *finite-window transient*
  on top (`RF{η,Pe,ζ}` → 0.93/0.97).
- **`erf(η/2)` is the right leading shape**, and the Pe-efficiency form (Model 2)
  beats the bare geotherm (esp. ramped 0.48→0.67) — removal efficiency rises with
  the descent Péclet, as the slab-top physics predicts.
- **Remaining gap is separability, not physics:** a *separable product*
  `f(η)·g(Pe)` caps at ~0.6–0.67 vs the ~0.89 flexible-2D ceiling on the **same two
  variables**. The true slab-top surface is **non-separable** in (η, Pe) — which is
  exactly what the analytic McKenzie / Molnar–England slab-top solutions are. So the
  principled completion is to adopt that non-separable analytic form, not to keep
  adding empirical knobs to a product.

### Next steps for this thread

1. **Adopt the analytic slab-top solution** (McKenzie 1969 / Molnar–England 1990;
   non-separable in η, Pe) as the closed form; fit its (few, physical) constants and
   target the `RF{η,Pe}` ≈ 0.89–0.91 ceiling.
2. Add the finite-window transient factor (the `ζ` content) for the last ~0.05.
3. Emulator-based conditional crossover test (proper `z_cross ∝ √(κ·age_OP)`).
4. ramped cumulative-convergence gain with CV (cleanest transient-vs-steady signal).
