#!/usr/bin/env python3
from pathlib import Path
import runpy

runpy.run_path(
    str(Path(__file__).resolve().parent / "single_depth" / "plot_param_sweep_compare.py"),
    run_name="__main__",
)
