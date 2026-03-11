#!/usr/bin/env python3
from pathlib import Path
import runpy

runpy.run_path(
    str(Path(__file__).resolve().parent / "profile_pca" / "preprocess_profile_pca.py"),
    run_name="__main__",
)
