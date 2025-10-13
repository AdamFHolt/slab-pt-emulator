#!/bin/bash

python preprocess_training.py --masters ../../subd-model-runs/run-outputs/analysis/master_*km_DT1-6.csv --targets dTdt_C_per_Myr --max-dTdt -20

