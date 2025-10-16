#!/bin/bash

python preprocess_training.py --master ../../subd-model-runs/run-outputs/analysis/master_25km_DT1-6.csv --targets dTdt_C_per_Myr --max-dTdt -20 
python preprocess_training.py --master ../../subd-model-runs/run-outputs/analysis/master_50km_DT1-6.csv --targets dTdt_C_per_Myr --max-dTdt -20 
python preprocess_training.py --master ../../subd-model-runs/run-outputs/analysis/master_75km_DT1-6.csv --targets dTdt_C_per_Myr --max-dTdt -20 

python preprocess_training.py --master ../../subd-model-runs/run-outputs/analysis/master_25km_DT1-6.csv --targets dTdt_C_per_Myr --max-dTdt -20 --add-thermal-param
python preprocess_training.py --master ../../subd-model-runs/run-outputs/analysis/master_50km_DT1-6.csv --targets dTdt_C_per_Myr --max-dTdt -20 --add-thermal-param
python preprocess_training.py --master ../../subd-model-runs/run-outputs/analysis/master_75km_DT1-6.csv --targets dTdt_C_per_Myr --max-dTdt -20 --add-thermal-param

python preprocess_training.py --master ../../subd-model-runs/run-outputs/analysis/master_25km_DT1-6.csv --targets dTdt_C_per_Myr --max-dTdt -20 --add-thermal-param --add-eta-ratio
python preprocess_training.py --master ../../subd-model-runs/run-outputs/analysis/master_50km_DT1-6.csv --targets dTdt_C_per_Myr --max-dTdt -20 --add-thermal-param --add-eta-ratio
python preprocess_training.py --master ../../subd-model-runs/run-outputs/analysis/master_75km_DT1-6.csv --targets dTdt_C_per_Myr --max-dTdt -20 --add-thermal-param --add-eta-ratio


