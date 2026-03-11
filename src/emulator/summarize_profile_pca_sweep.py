#!/usr/bin/env python3
from pathlib import Path
import runpy

runpy.run_path(
    str(Path(__file__).resolve().parent / "profile_pca" / "summarize_profile_pca_sweep.py"),
    run_name="__main__",
)
