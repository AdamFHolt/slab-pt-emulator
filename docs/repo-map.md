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

Grouped under `src/emulator/single_depth/` plus the shared trainer in
`src/emulator/train_emulator.py`.

Substructure:
- `src/emulator/single_depth/core/`
- `src/emulator/single_depth/qc/`
- `src/emulator/single_depth/science/`

Default QC figure outputs now live under:
- `plots/qc-emulator/single_depth/<suite>/`
Science-facing figure outputs live under:
- `plots/science-emulator/single_depth/<suite>/`
Default datasets/models live under:
- `src/emulator/data/single_depth/<suite>/runs/`
- `src/emulator/models/single_depth/<suite>/runs/`
- `src/emulator/models/single_depth/<suite>/param_sweep/`

- `src/emulator/single_depth/core/preprocess_single_depth_training.py`
  - build one single-depth training dataset
- `src/emulator/train_emulator.py`
  - shared model trainer used by both workflows
- `src/emulator/single_depth/core/validate_single_depth_quality.py`
  - validate standard single-depth model quality against thresholds
- `configs/gp.const-vc.yaml`
- `configs/gp.ramped-vc.yaml`
- `configs/rf.const-vc.yaml`
- `configs/rf.ramped-vc.yaml`

### 3. Profile-PCA Emulator Workflow

Grouped under `src/emulator/profile_pca/` plus the shared trainer in
`src/emulator/train_emulator.py`.

Substructure:
- `src/emulator/profile_pca/core/`
- `src/emulator/profile_pca/make_qc_emulator_plots.sh`
- `src/emulator/profile_pca/qc/`
- `src/emulator/profile_pca/science/`
- `src/emulator/profile_pca/sweeps/`
- `src/emulator/profile_pca/utilities/`

Default QC figure outputs live under:
- `plots/qc-emulator/profile-pca/default-runs/<suite>/`
- `plots/qc-emulator/profile-pca/pca-sweep/`
- `plots/qc-emulator/profile-pca/gp-tuning/`
Default datasets/models live under:
- `src/emulator/data/profile_pca/<suite>/runs/`
- `src/emulator/data/profile_pca/<suite>/pca_sweep/`
- `src/emulator/models/profile_pca/<suite>/runs/`
- `src/emulator/models/profile_pca/<suite>/pca_sweep/`
- `src/emulator/models/profile_pca/<suite>/gp_tuning/`

- `src/emulator/profile_pca/core/preprocess_profile_pca.py`
  - build profile-PCA datasets
- `src/emulator/profile_pca/core/evaluate_profile_pca_quality.py`
  - compute score-space and profile-space quality metrics
- `src/emulator/profile_pca/core/validate_profile_pca_quality.py`
  - validate profile-PCA reports against thresholds
- `src/emulator/profile_pca/sweeps/run_profile_pca_sweep.py`
  - sweep PCA representation choices (`k`, `raw` vs `whitened`)
- `src/emulator/profile_pca/sweeps/run_profile_pca_gp_tuning_sweep.py`
  - sweep GP settings on a fixed chosen PCA representation
- `src/emulator/profile_pca/utilities/compute_burial_path.py`
  - compute one burial/exhumation path from the profile-PCA emulator
- `src/emulator/profile_pca/utilities/compute_burial_path_uncertain_parameter.py`
  - compute one path with an uncertainty envelope from one varying emulator parameter
- `src/emulator/profile_pca/utilities/compute_many_burial_paths.py`
  - compute and overlay many paths across burial-rate / depth / hold-time ranges
- `src/emulator/profile_pca/utilities/compute_example_paths.sh`
  - run the current reference path examples used for demos/talk figures
- `configs/gp.const-vc.profile-pca.yaml`
- `configs/gp.ramped-vc.profile-pca.yaml`

## Legacy Helpers

These scripts are still tracked for reference, but they are not the preferred
entrypoints for routine work now that the repo uses `Makefile` targets and
config-driven training. They now live under `src/emulator/legacy/`:

- `src/emulator/legacy/preprocess_all_training.sh`
- `src/emulator/legacy/train_all_depths.sh`
- `src/emulator/legacy/train_one_depth_param-sweep.sh`

If you are not sure which path to use, prefer:

1. `Makefile`
2. `train.py`
3. the dedicated profile-PCA evaluation/sweep scripts
