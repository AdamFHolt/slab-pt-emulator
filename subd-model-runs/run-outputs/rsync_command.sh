#!/usr/bin/env bash

rsync -azP --prune-empty-dirs \
  --include='run_*/' \
  --include='run_*/run_*.prm' \
  --include='run_*/outputs/' \
  --include='run_*/outputs/run_*/' \
  --include='run_*/outputs/run_*/parameters.*' \
  --include='run_*/outputs/run_*/original.prm' \
  --include='run_*/outputs/run_*/log.txt' \
  --include='run_*/outputs/run_*/solution.pvd' \
  --include='run_*/outputs/run_*/solution.visit' \
  --include='run_*/outputs/run_*/statistics/***' \
  --include='run_*/outputs/run_*/solution/***' \
  --exclude='run_*/outputs/run_*/restart*' \
  --exclude='run_*/outputs/run_*/restart*/***' \
  --exclude='run_*/inputs/' \
  --exclude='run_*/inputs/*.txt' \
  --exclude='*' \
  adamholt@stampede3.tacc.utexas.edu:/scratch/04714/adamholt/aspect_work/ .
