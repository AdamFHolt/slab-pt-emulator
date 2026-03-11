#!/usr/bin/env python3
from pathlib import Path
import runpy

runpy.run_path(
    str(Path(__file__).resolve().parent / "profile_pca" / "plot_profile_pca_score_diagnostics.py"),
    run_name="__main__",
)
