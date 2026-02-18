SHELL := /bin/bash

.PHONY: help setup test train-gp-const train-gp-ramped train-rf-const train-rf-ramped \
        train-gp-const-dry train-gp-ramped-dry qc-num-const qc-num-ramped quality-check-gp-m25

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
