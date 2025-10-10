#!/bin/bash

# plot cooling rates compilations (for all models at two depths)
d1="25"; d2="50"
python3 qc_cooling-rates_all-mods_2-depths.py     --params ../../data/params-list.csv  \
     --master1 ../../subd-model-runs/run-outputs/analysis/master_"$d1"km_DT1-6.csv --master2 ../../subd-model-runs/run-outputs/analysis/master_"$d2"km_DT1-6.csv  \
     --y  dTdt_C_per_Myr   --out ../../plots/numerical-mods/DT_vs_params_"$d1"-"$d2"km
python3 qc_cooling-rates_all-mods.py              --params ../../data/params-list.csv  \
     --master ../../subd-model-runs/run-outputs/analysis/master_"$d1"km_DT1-6.csv   --y  dTdt_C_per_Myr   --out ../../plots/numerical-mods/DT_vs_params_"$d1"
python3 qc_cooling-rates_all-mods.py              --params ../../data/params-list.csv  \
     --master ../../subd-model-runs/run-outputs/analysis/master_"$d2"km_DT1-6.csv   --y  dTdt_C_per_Myr   --out ../../plots/numerical-mods/DT_vs_params_"$d2"

# ensemble qc plots
python3 qc_pairplot_params.py  --params ../../data/params-list.csv --out ../../plots/numerical-mods/qc_pairplot
python3 qc_dt_vs_depth.py --masters ../../subd-model-runs/run-outputs/analysis/master_"$d1"km_DT1-6.csv ../../subd-model-runs/run-outputs/analysis/master_"$d2"km_DT1-6.csv  \
     --y dTdt_C_per_Myr --out ../../plots/numerical-mods/qc_all-DT_"$d1"-and-"$d2"km
python3 qc_histograms.py --params ../../data/params-list.csv --master ../../subd-model-runs/run-outputs/analysis/master_"$d1"km_DT1-6.csv --out ../../plots/numerical-mods/qc_histograms_"$d1"km
python3 qc_histograms.py --params ../../data/params-list.csv --master ../../subd-model-runs/run-outputs/analysis/master_"$d2"km_DT1-6.csv --out ../../plots/numerical-mods/qc_histograms_"$d2"km