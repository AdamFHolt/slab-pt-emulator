#!/bin/bash

d1="25"; d2="50"

python3 plot_cooling-rates_all-mods_2-depths.py   --params ../../data/params-list.csv  \
     --master1 ../../subd-model-runs/run-outputs/analysis/master_"$d1"km_DT1-6.csv --master2 ../../subd-model-runs/run-outputs/analysis/master_"$d2"km_DT1-6.csv  \
     --y  dTdt_C_per_Myr   --out ../../subd-model-runs/run-outputs/analysis/figs/DT_vs_params_"$d1"-"$d2"km

python3 plot_cooling-rates_all-mods.py   --params ../../data/params-list.csv  \
     --master ../../subd-model-runs/run-outputs/analysis/master_"$d1"km_DT1-6.csv   --y  dTdt_C_per_Myr   --out ../../subd-model-runs/run-outputs/analysis/figs/DT_vs_params_"$d1"

python3 plot_cooling-rates_all-mods.py   --params ../../data/params-list.csv  \
     --master ../../subd-model-runs/run-outputs/analysis/master_"$d2"km_DT1-6.csv   --y  dTdt_C_per_Myr   --out ../../subd-model-runs/run-outputs/analysis/figs/DT_vs_params_"$d2"

python3 qc_pairplot_params.py  --params ../../data/params-list.csv --out ../../subd-model-runs/run-outputs/analysis/figs/pairplot_params.png

