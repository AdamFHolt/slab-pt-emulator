# Repo Map

This file is a short orientation guide for the repository.

## Core Paths

- `README.md`
  - canonical workflow overview
- `Makefile`
  - canonical day-to-day command entrypoint
- `train.py`
  - canonical config-driven training entrypoint
- `configs/`
  - training configs and quality-threshold configs
- `tests/`
  - smoke/integration/unit coverage for the active workflows

## Main Workflow Areas

### 1. Numerical Model Generation

- `src/build-numerical-mods/`
  - Latin-hypercube sampling and ASPECT run construction
- `src/postproc-numerical-mods/`
  - postprocess model outputs into emulator-ready quantities
- `src/qc-numerical-mods/`
  - QC plots and health checks for the numerical ensemble

### 2. Standard Single-Depth Emulator Workflow

- `src/emulator/preprocess_one_training.py`
  - build one single-depth training dataset
- `src/emulator/train_emulator.py`
  - shared model trainer used by both workflows
- `src/emulator/validate_emulator_quality.py`
  - validate standard single-depth model quality against thresholds
- `configs/gp.const-vc.yaml`
- `configs/gp.ramped-vc.yaml`
- `configs/rf.const-vc.yaml`
- `configs/rf.ramped-vc.yaml`

### 3. Profile-PCA Emulator Workflow

- `src/emulator/preprocess_profile_pca.py`
  - build profile-PCA datasets
- `src/emulator/evaluate_profile_pca_quality.py`
  - compute score-space and profile-space quality metrics
- `src/emulator/validate_profile_pca_quality.py`
  - validate profile-PCA reports against thresholds
- `src/emulator/run_profile_pca_sweep.py`
  - sweep PCA representation choices (`k`, `raw` vs `whitened`)
- `src/emulator/run_profile_pca_gp_tuning_sweep.py`
  - sweep GP settings on a fixed chosen PCA representation
- `configs/gp.const-vc.profile-pca.yaml`
- `configs/gp.ramped-vc.profile-pca.yaml`

## Legacy Helpers

These scripts are still tracked for reference, but they are not the preferred
entrypoints for routine work now that the repo uses `Makefile` targets and
config-driven training:

- `src/emulator/preprocess_all_training.sh`
- `src/emulator/train_all_depths.sh`
- `src/emulator/train_one_depth_param-sweep.sh`

If you are not sure which path to use, prefer:

1. `Makefile`
2. `train.py`
3. the dedicated profile-PCA evaluation/sweep scripts
