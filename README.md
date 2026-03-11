# Slab-P-T Emulator

End-to-end pipeline for building surrogate emulators of slab thermal evolution from ASPECT model outputs.

Supported experiment suites:
- `const-vc` (constant convergence rate)
- `ramped-vc` (time-ramped convergence)

## What This Repo Does

1. Generate Latin-hypercube parameter sets.
2. Build ASPECT run inputs (`.prm` + initial temperature/composition grids).
3. Postprocess simulation outputs into cooling targets (`dT`, `dT/dt`) vs depth.
4. Preprocess data into train/validation emulator datasets.
5. Train emulator models (GP by default, RF optional).
6. Produce QC plots for both numerical outputs and emulator performance.

## Prerequisites

- Python 3.10+ recommended
- ASPECT installed and runnable externally (HPC/local)
- ParaView `pvpython` available for postprocessing scripts

## Setup

Run from repo root:

```bash
python3 -m venv env
source env/bin/activate
pip install -r requirements.txt
```

If `python3 -m venv` fails because `python3-venv`/`ensurepip` is unavailable, use:

```bash
python3 -m virtualenv --app-data /tmp/virtualenv-appdata --no-periodic-update --never-download env
source env/bin/activate
pip install -r requirements.txt
```

------------------------------
Standard Depth-Based Training
------------------------------

## Standard Depth-Based Emulator Training

This is the baseline workflow for per-depth targets such as `10km_dTdt`, `40km_dTdt`, and related variants.

Entry point:

```bash
python train.py --config <config.yaml>
```

### GP Commands

Constant-convergence suite (`const-vc`):

```bash
python train.py --config configs/gp.const-vc.yaml
```

Time-ramped suite (`ramped-vc`):

```bash
python train.py --config configs/gp.ramped-vc.yaml
```

Makefile shortcuts:

```bash
make train-gp-const
make train-gp-ramped
```

Dry-run previews:

```bash
python train.py --config configs/gp.const-vc.yaml --dry-run
python train.py --config configs/gp.ramped-vc.yaml --dry-run
```

### Other Standard Depth-Based Training Commands

RF configs use the same depth-based dataset layout:

```bash
python train.py --config configs/rf.const-vc.yaml
python train.py --config configs/rf.ramped-vc.yaml
```

To run only selected depth datasets:

```bash
python train.py --config configs/gp.const-vc.yaml --datasets 10km_dTdt,20km_dTdt
python train.py --config configs/gp.ramped-vc.yaml --datasets 40km_dTdt
```

Standard depth-based GP outputs are written under:

- `src/emulator/models/<suite>/<dataset>/gp_m25/`

RF outputs use the analogous layout:

- `src/emulator/models/<suite>/<dataset>/rf/`

--------------------
Profile-PCA Workflow
--------------------

## Profile-PCA Workflow

This is a separate workflow from the standard depth-based GP training above. It first builds profile-level PCA score datasets at one or more target times, then trains GP emulators on those PCA scores, then makes reconstruction/QC plots.

The Makefile includes profile-PCA workflow targets that run across multiple target times.

Default time set:

- `0.5, 1, 2, 3, 4, 5 Myr`

Default suites:

- `const-vc`, `ramped-vc`

Default PCA components:

- `k=10`

Default PCA score space:

- `whitened`

### Step 1: Preprocess Profile-PCA Datasets

Run preprocess for all default times/suites:

```bash
make profile-pca-preprocess
```

### Step 2: Train GP Models on Profile-PCA Scores

Run GP training for all default times/suites:

```bash
make profile-pca-train-gp
```

### Step 3: Generate Profile-PCA QC Plots

Run profile-PCA QC plots for all default times/suites:

```bash
make profile-pca-qc
```

### Step 4: Write Profile-PCA Quality Reports

Compute score-space and reconstructed-profile quality metrics for all default times/suites:

```bash
make profile-pca-quality-report
```

### Optional: Profile-PCA Sweep And Ranked Summary Tables

Run the first profile-PCA model-selection sweep with fixed `gp_m25` settings:

```bash
make profile-pca-sweep PROFILE_SUITES="const-vc ramped-vc" PROFILE_TIMES="3" PROFILE_SWEEP_KS="4 6 8 10" PROFILE_SWEEP_SCORE_SPACES="raw whitened"
```

Write ranked summary tables from existing sweep reports:

```bash
make profile-pca-sweep-summary PROFILE_SUITES="const-vc ramped-vc" PROFILE_SWEEP_DATASET_PATTERN="profileT_pca_t3Myr"
```

Ranking policy in the summary tables:

- primary sort: validation profile RMSE
- tie-breaker: validation profile p95 RMSE

Common overrides:

```bash
# One suite only
make profile-pca-preprocess PROFILE_SUITES="const-vc"

# Explicit time subset (example: 1..5 Myr)
make profile-pca-train-gp PROFILE_TIMES="1 2 3 4 5"

# Different k and split for QC plots
make profile-pca-qc PROFILE_K=6 PROFILE_QC_SPLIT=train

# One representative CI-style subset
make profile-pca-quality-report PROFILE_SUITES="const-vc ramped-vc" PROFILE_TIMES="3" PROFILE_K=10 PROFILE_SCORE_SPACE=whitened

# Sweep summary for t3Myr only
make profile-pca-sweep-summary PROFILE_SWEEP_DATASET_PATTERN="profileT_pca_t3Myr"
```

Notes:

- Dataset names are generated as `profileT_pca_t<time_label>Myr_k<K>` (e.g., `t0p5Myr`, `t3Myr`, `t5Myr`).
- The default profile-PCA workflow now uses `k=10` and `score_space=whitened`.
- Sweep dataset names include score-space to avoid collisions:
  - `profileT_pca_t3Myr_k10_raw`
  - `profileT_pca_t3Myr_k10_whitened`
- Training configs for this workflow are separate from the standard depth-based GP configs:
  - `configs/gp.const-vc.profile-pca.yaml`
  - `configs/gp.ramped-vc.profile-pca.yaml`
- Preprocessed datasets are written under:
  - `src/emulator/data/<suite>/profileT_pca_t<time_label>Myr_k<K>/`
- `profile-pca-qc` expects corresponding trained model artifacts under:
  - `src/emulator/models/<suite>/<dataset>/gp_m25/`
- `profile-pca-quality-report` writes:
  - `src/emulator/models/<suite>/<dataset>/gp_m25/profile_pca_quality.json`
- `profile-pca-sweep-summary` writes ranked CSV/Markdown tables under:
  - `plots/qc-emulator/profile-pca-sweep/`
- Missing dataset/model folders are skipped with `[WARN]` messages.

--------------------------------------------
From Scratch: Full Numerical Model Pipeline
--------------------------------------------

## From Scratch: Full Numerical Model Pipeline

This section is not the quick-start training path above.

It is the end-to-end workflow for generating new numerical-model outputs from ASPECT, extracting cooling-rate targets, building emulator datasets from those outputs, and then training baseline depth-based emulators.

Use this when you need to rebuild the data products from the raw modeling stage, not when you only want to retrain an emulator from existing prepared datasets.

Generalized for either suite:

Set suite once:

```bash
SUITE=const-vc
# or
# SUITE=ramped-vc
```

### 1) Generate parameter design

```bash
python src/build-numerical-mods/make_lhs.${SUITE}.py
```

Outputs:
- `data/params/params-list.${SUITE}.csv`
- `data/params/params-list.${SUITE}.npy`

### 2) Build ASPECT run inputs

```bash
python src/build-numerical-mods/build_runs.${SUITE}.py
```

Outputs under:
- `subd-model-runs/${SUITE}/run-inputs/run_*/`

### 3) Run ASPECT externally

Run each generated `.prm` model so solutions land under:
- `subd-model-runs/${SUITE}/run-outputs/run_*/solution/...`

This step is external to this repository.

### 4) Extract cooling-rate targets from model outputs

```bash
cd src/postproc-numerical-mods
./extract_cooling-rates_all-mods.sh 1 10 0:10 5:80:5 "${SUITE}"
cd ../..
```

Key output:
- `subd-model-runs/${SUITE}/analysis/master_DT1-10.csv`

### 5) Numerical-model QC plots

```bash
cd src/qc-numerical-mods
./make_all_plots.sh "${SUITE}"
cd ../..
```

Outputs under:
- `plots/qc-numerical-mods/${SUITE}/`

### 6) Preprocess emulator training datasets

```bash
cd src/emulator
./preprocess_all_training.sh "${SUITE}"
cd ../..
```

Outputs under:
- `src/emulator/data/${SUITE}/<depth>km_<variant>/`

### 7) Train baseline depth-based emulator models

```bash
python train.py --config configs/gp.${SUITE}.yaml
```

Outputs under:
- `src/emulator/models/${SUITE}/<dataset>/gp_m25/`

### 8) Emulator QC plots

```bash
cd src/emulator
./make_qc_emulator_plots.sh "${SUITE}" dTdt gp_m25
./make_qc_emulator_plots.sh "${SUITE}" dTdt_thermalParam gp_m25
./make_qc_emulator_plots.sh "${SUITE}" dTdt_thermalParam_etaRatio gp_m25
cd ../..
```

Outputs under:
- `plots/qc-emulator/${SUITE}/`

## Optional: Param Sweep (One Depth/Variant)

```bash
cd src/emulator
./train_one_depth_param-sweep.sh "${SUITE}" 40 dTdt_thermalParam gp_rbf,gp_m15,gp_m25,rf
python plot_param_sweep_compare.py --suite "${SUITE}" --data-name 40km_dTdt_thermalParam
cd ../..
```

Outputs under:
- `src/emulator/models/param-sweep/${SUITE}/40km_dTdt_thermalParam/`
- `plots/qc-emulator/${SUITE}/param-sweep/40km_dTdt_thermalParam/`

----------------------
Tests And CI Policy
----------------------

## Tests

Run smoke tests from repo root:

```bash
make test
```

## Standard Depth-Based GP Quality Gates

This section applies to the standard depth-based `gp_m25` workflow. It does not define a separate quality-gate policy for the profile-PCA workflow.

Per-dataset validation thresholds for baseline depth-based GP models are defined in:

- `configs/emulator-quality.gp_m25.yaml`

This file specifies:

- metric source: `metrics.val._macro_avg` from each model `report.json`
- thresholds per suite + dataset:
  - `r2_min` (minimum acceptable validation R2)
  - `rmse_max` (maximum acceptable validation RMSE)
  - `mae_max` (maximum acceptable validation MAE)

These thresholds are intended to support automated pass/fail checks in CI.

Run validation against existing model reports:

```bash
make quality-check-gp-m25
```

Direct invocation (with optional summary JSON):

```bash
python src/emulator/validate_emulator_quality.py \
  --thresholds configs/emulator-quality.gp_m25.yaml \
  --models-root src/emulator/models \
  --json-out plots/qc-emulator/quality-gates/gp_m25_validation.json
```

## Profile-PCA GP Quality Gates

This section applies to the profile-PCA `gp_m25` workflow.

The current representative threshold spec is:

- `configs/profile-pca-quality.gp_m25.yaml`

This initial gate covers the representative subset:

- suites: `const-vc`, `ramped-vc`
- dataset: `profileT_pca_t3Myr_k10`
- score space: `whitened`

The profile-PCA quality report includes both:

- score-space metrics on predicted PCA targets
- reconstructed-profile metrics in temperature space

Write the quality report JSON for an existing trained model:

```bash
python src/emulator/evaluate_profile_pca_quality.py \
  --dataset-dir src/emulator/data/const-vc/profileT_pca_t3Myr_k10 \
  --model-dir src/emulator/models/const-vc/profileT_pca_t3Myr_k10/gp_m25
```

Validate existing profile-PCA quality reports against thresholds:

```bash
make profile-pca-quality-check-gp-m25
```

Direct invocation:

```bash
python src/emulator/validate_profile_pca_quality.py \
  --thresholds configs/profile-pca-quality.gp_m25.yaml \
  --models-root src/emulator/models \
  --suites const-vc,ramped-vc \
  --datasets profileT_pca_t3Myr_k10
```

## Definition Of Done

A change affecting emulator training, data prep, or model quality is complete when all of the following are true:

- Required checks pass:
  - `make test`
  - `make quality-check-gp-m25` (or CI-equivalent filtered scope)
  - `make profile-pca-quality-check-gp-m25` when the change affects the profile-PCA workflow
- PR CI quality gate passes for representative subset policy:
  - suites: `const-vc`, `ramped-vc`
  - datasets: `40km_dTdt`
  - gate command in CI: `make quality-check-gp-m25 QUALITY_SUITES=const-vc,ramped-vc QUALITY_DATASETS=40km_dTdt`
  - profile-PCA representative subset:
    - suites: `const-vc`, `ramped-vc`
    - dataset: `profileT_pca_t3Myr_k10`
    - gate command in CI:
      `make profile-pca-quality-check-gp-m25 QUALITY_SUITES=const-vc,ramped-vc QUALITY_DATASETS=profileT_pca_t3Myr_k10`
- Canonical artifacts are present in standard locations:
  - training reports: `src/emulator/models/<suite>/<dataset>/<model_tag>/report.json`
  - quality-gate summary (optional JSON): `plots/qc-emulator/quality-gates/gp_m25_validation.json`
  - profile-PCA quality report: `src/emulator/models/<suite>/<dataset>/<model_tag>/profile_pca_quality.json`
  - emulator QC figures: `plots/qc-emulator/<suite>/`
  - numerical QC figures: `plots/qc-numerical-mods/<suite>/`

## Main Scripts By Stage

- Parameter sampling:
  - `src/build-numerical-mods/make_lhs.const-vc.py`
  - `src/build-numerical-mods/make_lhs.ramped-vc.py`
- Run-input generation:
  - `src/build-numerical-mods/build_runs.const-vc.py`
  - `src/build-numerical-mods/build_runs.ramped-vc.py`
- Postprocessing:
  - `src/postproc-numerical-mods/extract_cooling-rates_all-mods.sh`
  - `src/postproc-numerical-mods/extract_cooling-rates_one-mod.py`
- Emulator prep/train:
  - `src/emulator/preprocess_all_training.sh`
  - `src/emulator/train_all_depths.sh`
  - `src/emulator/train_emulator.py`

<!-- ruleset check -->
