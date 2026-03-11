SHELL := /bin/bash

.PHONY: help setup test train-gp-const train-gp-ramped train-rf-const train-rf-ramped \
        train-gp-const-dry train-gp-ramped-dry qc-num-const qc-num-ramped quality-check-gp-m25 \
        env-status env-ensure env-doctor profile-pca-preprocess profile-pca-train-gp profile-pca-qc \
        profile-pca-quality-report profile-pca-quality-check-gp-m25 \
        profile-pca-sweep profile-pca-sweep-summary

help:
	@echo "Targets:"
	@echo "  setup               - create venv and install Python deps"
	@echo "  test                - run smoke tests"
	@echo "  train-gp-const      - train GP models for const-vc"
	@echo "  train-gp-ramped     - train GP models for ramped-vc"
	@echo "  train-rf-const      - train RF models for const-vc"
	@echo "  train-rf-ramped     - train RF models for ramped-vc"
	@echo "  train-gp-const-dry  - dry-run GP training commands (const-vc)"
	@echo "  train-gp-ramped-dry - dry-run GP training commands (ramped-vc)"
	@echo "  qc-num-const        - make numerical QC plots for const-vc"
	@echo "  qc-num-ramped       - make numerical QC plots for ramped-vc"
	@echo "  quality-check-gp-m25 - validate gp_m25 reports vs thresholds (optional QUALITY_SUITES/QUALITY_DATASETS filters)"
	@echo "  env-status          - show active python/pip/env context"
	@echo "  env-ensure          - create env (venv or virtualenv fallback) and install deps"
	@echo "  env-doctor          - verify core Python imports"
	@echo "  profile-pca-preprocess - build profile-PCA datasets for PROFILE_TIMES (default: 0.5 1 2 3 4 5)"
	@echo "  profile-pca-train-gp - train GP profile-PCA models for PROFILE_TIMES"
	@echo "  profile-pca-qc      - generate profile-PCA QC plots for PROFILE_TIMES"
	@echo "  profile-pca-quality-report - write profile-PCA quality JSON reports for PROFILE_TIMES"
	@echo "  profile-pca-quality-check-gp-m25 - validate profile-PCA gp_m25 quality reports (optional QUALITY_SUITES/QUALITY_DATASETS filters)"
	@echo "  profile-pca-sweep   - run profile-PCA sweep over PROFILE_SWEEP_KS and PROFILE_SWEEP_SCORE_SPACES"
	@echo "  profile-pca-sweep-summary - write ranked summary tables for profile-PCA sweep results"

setup:
	python3 -m venv env
	source env/bin/activate && \
	pip install -r requirements.txt

test:
	python3 -m unittest discover -s tests -v

train-gp-const:
	python3 train.py --config configs/gp.const-vc.yaml

train-gp-ramped:
	python3 train.py --config configs/gp.ramped-vc.yaml

train-rf-const:
	python3 train.py --config configs/rf.const-vc.yaml

train-rf-ramped:
	python3 train.py --config configs/rf.ramped-vc.yaml

train-gp-const-dry:
	python3 train.py --config configs/gp.const-vc.yaml --dry-run

train-gp-ramped-dry:
	python3 train.py --config configs/gp.ramped-vc.yaml --dry-run

qc-num-const:
	cd src/qc-numerical-mods && ./make_all_plots.sh const-vc

qc-num-ramped:
	cd src/qc-numerical-mods && ./make_all_plots.sh ramped-vc

quality-check-gp-m25:
	python3 src/emulator/validate_emulator_quality.py \
		--thresholds configs/emulator-quality.gp_m25.yaml \
		--models-root src/emulator/models \
		$(if $(QUALITY_SUITES),--suites $(QUALITY_SUITES),) \
		$(if $(QUALITY_DATASETS),--datasets $(QUALITY_DATASETS),)

env-status:
	./dev-env.sh status

env-ensure:
	./dev-env.sh ensure

env-doctor:
	./dev-env.sh doctor

# Profile-PCA workflow knobs (override at runtime as needed).
# Example:
#   make profile-pca-preprocess PROFILE_SUITES="const-vc" PROFILE_TIMES="1 2 3 4 5" PROFILE_K=8
PROFILE_SUITES ?= const-vc ramped-vc
PROFILE_TIMES ?= 0.5 1 2 3 4 5
PROFILE_K ?= 8
PROFILE_SCORE_SPACE ?= raw
PROFILE_QC_SPLIT ?= val
PROFILE_MODEL_TAG ?= gp_m25
PROFILE_SWEEP_KS ?= 4 6 8 10
PROFILE_SWEEP_SCORE_SPACES ?= raw whitened
PROFILE_SWEEP_DATASET_PATTERN ?= profileT_pca_t3Myr

profile-pca-preprocess:
	@set -euo pipefail; \
	for suite in $(PROFILE_SUITES); do \
		for t in $(PROFILE_TIMES); do \
			tlabel="$$(echo "$$t" | sed 's/\./p/g')"; \
			dname="profileT_pca_t$${tlabel}Myr_k$(PROFILE_K)"; \
			echo "[RUN] preprocess suite=$$suite time=$$t dataset=$$dname"; \
			python3 src/emulator/preprocess_profile_pca.py \
				--suite "$$suite" \
				--target-time-myr "$$t" \
				--k "$(PROFILE_K)" \
				--score-space "$(PROFILE_SCORE_SPACE)" \
				--dataset-name "$$dname"; \
		done; \
	done

profile-pca-train-gp:
	@set -euo pipefail; \
	for suite in $(PROFILE_SUITES); do \
		cfg="configs/gp.$${suite}.profile-pca.yaml"; \
		for t in $(PROFILE_TIMES); do \
			tlabel="$$(echo "$$t" | sed 's/\./p/g')"; \
			dname="profileT_pca_t$${tlabel}Myr_k$(PROFILE_K)"; \
			echo "[RUN] train suite=$$suite dataset=$$dname"; \
			python3 train.py --config "$$cfg" --datasets "$$dname"; \
		done; \
	done

profile-pca-qc:
	@set -euo pipefail; \
	for suite in $(PROFILE_SUITES); do \
		for t in $(PROFILE_TIMES); do \
			tlabel="$$(echo "$$t" | sed 's/\./p/g')"; \
			dname="profileT_pca_t$${tlabel}Myr_k$(PROFILE_K)"; \
			ds="src/emulator/data/$${suite}/$${dname}"; \
			md="src/emulator/models/$${suite}/$${dname}/gp_m25"; \
			outdir="plots/qc-emulator/$${suite}/profile-pca"; \
			if [ ! -d "$$ds" ]; then \
				echo "[WARN] skip missing dataset $$ds"; \
				continue; \
			fi; \
			if [ ! -d "$$md" ]; then \
				echo "[WARN] skip missing model $$md"; \
				continue; \
			fi; \
			mkdir -p "$$outdir"; \
			prefix="$$outdir/$${dname}"; \
			echo "[RUN] qc suite=$$suite dataset=$$dname split=$(PROFILE_QC_SPLIT)"; \
			python3 src/emulator/plot_profile_pca_reconstruction.py \
				--dataset-dir "$$ds" \
				--split "$(PROFILE_QC_SPLIT)" \
				--out "$${prefix}_true-vs-recon.png"; \
			python3 src/emulator/plot_profile_pca_emulator_reconstruction.py \
				--dataset-dir "$$ds" \
				--model-dir "$$md" \
				--split "$(PROFILE_QC_SPLIT)" \
				--out "$${prefix}_raw-vs-pca-vs-emu.png"; \
			python3 src/emulator/plot_profile_pca_score_diagnostics.py \
				--dataset-dir "$$ds" \
				--model-dir "$$md" \
				--split "$(PROFILE_QC_SPLIT)" \
				--out-prefix "$$prefix"; \
		done; \
	done

profile-pca-quality-report:
	@set -euo pipefail; \
	for suite in $(PROFILE_SUITES); do \
		for t in $(PROFILE_TIMES); do \
			tlabel="$$(echo "$$t" | sed 's/\./p/g')"; \
			dname="profileT_pca_t$${tlabel}Myr_k$(PROFILE_K)"; \
			ds="src/emulator/data/$${suite}/$${dname}"; \
			md="src/emulator/models/$${suite}/$${dname}/$(PROFILE_MODEL_TAG)"; \
			if [ ! -d "$$ds" ]; then \
				echo "[WARN] skip missing dataset $$ds"; \
				continue; \
			fi; \
			if [ ! -d "$$md" ]; then \
				echo "[WARN] skip missing model $$md"; \
				continue; \
			fi; \
			echo "[RUN] quality-report suite=$$suite dataset=$$dname model=$(PROFILE_MODEL_TAG)"; \
			python3 src/emulator/evaluate_profile_pca_quality.py \
				--dataset-dir "$$ds" \
				--model-dir "$$md"; \
		done; \
	done

profile-pca-quality-check-gp-m25:
	python3 src/emulator/validate_profile_pca_quality.py \
		--thresholds configs/profile-pca-quality.gp_m25.yaml \
		--models-root src/emulator/models \
		$(if $(QUALITY_SUITES),--suites $(QUALITY_SUITES),) \
		$(if $(QUALITY_DATASETS),--datasets $(QUALITY_DATASETS),)

profile-pca-sweep:
	python3 src/emulator/run_profile_pca_sweep.py \
		--suites "$(PROFILE_SUITES)" \
		--times "$(PROFILE_TIMES)" \
		--ks "$(PROFILE_SWEEP_KS)" \
		--score-spaces "$(PROFILE_SWEEP_SCORE_SPACES)"

profile-pca-sweep-summary:
	python3 src/emulator/summarize_profile_pca_sweep.py \
		--models-root src/emulator/models \
		--suites "$(PROFILE_SUITES)" \
		--dataset-pattern "$(PROFILE_SWEEP_DATASET_PATTERN)"
