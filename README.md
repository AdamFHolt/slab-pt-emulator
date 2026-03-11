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

## Tests

Run smoke tests from repo root:

```bash
make test
```

## Emulator Quality Gates (Step 1)

Per-dataset validation thresholds for baseline GP are defined in:

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

## Definition Of Done (Step 4)

A change affecting emulator training, data prep, or model quality is complete when all of the following are true:

- Required checks pass:
  - `make test`
  - `make quality-check-gp-m25` (or CI-equivalent filtered scope)
- PR CI quality gate passes for representative subset policy:
  - suites: `const-vc`, `ramped-vc`
  - datasets: `40km_dTdt`
  - gate command in CI: `make quality-check-gp-m25 QUALITY_SUITES=const-vc,ramped-vc QUALITY_DATASETS=40km_dTdt`
- Canonical artifacts are present in standard locations:
  - training reports: `src/emulator/models/<suite>/<dataset>/<model_tag>/report.json`
  - quality-gate summary (optional JSON): `plots/qc-emulator/quality-gates/gp_m25_validation.json`
  - emulator QC figures: `plots/qc-emulator/<suite>/`
  - numerical QC figures: `plots/qc-numerical-mods/<suite>/`

## Uniform Training Entry Point

Industry-style config-driven training is available via:

```bash
python train.py --config configs/gp.const-vc.yaml
```

Examples:

```bash
# GP (const-vc)
python train.py --config configs/gp.const-vc.yaml

# GP (ramped-vc)
python train.py --config configs/gp.ramped-vc.yaml

# RF (const-vc)
python train.py --config configs/rf.const-vc.yaml

# RF (ramped-vc)
python train.py --config configs/rf.ramped-vc.yaml

# Preview commands only
python train.py --config configs/gp.const-vc.yaml --dry-run

# Run only selected datasets
python train.py --config configs/gp.const-vc.yaml --datasets 10km_dTdt,20km_dTdt
```

## Profile-PCA Multi-Time Workflow

The Makefile includes profile-PCA workflow targets that run across multiple target times.

Default time set:

- `0.5, 1, 2, 3, 4, 5 Myr`

Default suites:

- `const-vc`, `ramped-vc`

Default PCA components:

- `k=8`

Run preprocess for all default times/suites:

```bash
make profile-pca-preprocess
```

Run GP training for all default times/suites:

```bash
make profile-pca-train-gp
```

Run profile-PCA QC plots for all default times/suites:

```bash
make profile-pca-qc
```

Common overrides:

```bash
# One suite only
make profile-pca-preprocess PROFILE_SUITES="const-vc"

# Explicit time subset (example: 1..5 Myr)
make profile-pca-train-gp PROFILE_TIMES="1 2 3 4 5"

# Different k and split for QC plots
make profile-pca-qc PROFILE_K=6 PROFILE_QC_SPLIT=train
```

Notes:

- Dataset names are generated as `profileT_pca_t<time_label>Myr_k<K>` (e.g., `t0p5Myr`, `t3Myr`, `t5Myr`).
- `profile-pca-qc` expects corresponding trained model artifacts under:
  - `src/emulator/models/<suite>/<dataset>/gp_m25/`
- Missing dataset/model folders are skipped with `[WARN]` messages.

## From Scratch (Generalized For Either Suite)

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

### 7) Train baseline emulator models

```bash
python train.py --config configs/gp.const-vc.yaml
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
