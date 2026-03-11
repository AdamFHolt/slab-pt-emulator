import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]


class ProfilePCAQualityReportTests(unittest.TestCase):
    # This test module verifies the new quality-report script that sits between
    # "a trained profile-PCA model exists" and "we can eventually gate this in
    # CI with thresholds".
    #
    # The test is intentionally end-to-end:
    # 1. build a tiny synthetic profile-PCA dataset
    # 2. train a very small RF model on that dataset
    # 3. run evaluate_profile_pca_quality.py
    # 4. inspect the output JSON structure
    #
    # Using an RF keeps the test fast. The quality-report script itself only
    # cares about saved predictions and metadata, not about whether the model
    # was GP or RF.

    def _write_profile_csv(self, path: Path, time_myr: float, depths_km: np.ndarray, temps_c: np.ndarray) -> None:
        pd.DataFrame(
            {
                "time_Myr": np.full(depths_km.shape, time_myr, dtype=float),
                "depth_km": depths_km,
                "T_C": temps_c,
            }
        ).to_csv(path, index=False)

    def _build_inputs(self, root: Path, suite: str) -> tuple[Path, Path]:
        params_path = root / f"params-list.{suite}.csv"
        analysis_root = root / "analysis"
        analysis_root.mkdir(parents=True, exist_ok=True)

        rows = []
        depths = np.array([0.0, 10.0, 20.0, 30.0], dtype=float)
        for i in range(8):
            row = {
                "run_id": f"{i:03d}",
                "v_conv": 1.5 + 0.3 * i,
                "age_SP": 50.0 + i,
                "age_OP": 15.0 + 2.0 * i,
                "dip_int": 25.0 + 4.0 * i,
                "eta_UM": 5.0e19 + 0.8e19 * i,
            }
            if suite == "ramped-vc":
                row["t_conv"] = 0.4 + 0.1 * i
            rows.append(row)

            run_dir = analysis_root / f"run_{i:03d}"
            run_dir.mkdir(parents=True, exist_ok=True)

            # Use a simple but nontrivial family of profiles so PCA sees real
            # variation across runs and depth.
            base = 780.0 - 6.0 * depths
            run_signal = 8.0 * np.sin((i + 1) * 0.25) + 3.0 * i
            bend = 0.05 * depths ** 2

            near_profile = base + run_signal + bend
            self._write_profile_csv(run_dir / f"Tprof_near_{i:03d}.csv", 3.01, depths, near_profile)
            self._write_profile_csv(run_dir / f"Tprof_far_{i:03d}.csv", 3.30, depths, near_profile + 20.0)

        pd.DataFrame(rows).to_csv(params_path, index=False)
        return params_path, analysis_root

    def test_quality_report_contains_score_and_profile_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            params_path, analysis_root = self._build_inputs(tmp_path, suite="const-vc")
            dataset_root = tmp_path / "datasets"
            model_root = tmp_path / "models"
            dataset_name = "profileT_pca_t3Myr_k2_quality"

            # Step 1: preprocess a synthetic profile-PCA dataset.
            preprocess = subprocess.run(
                [
                    sys.executable,
                    "src/emulator/preprocess_profile_pca.py",
                    "--suite",
                    "const-vc",
                    "--params",
                    str(params_path),
                    "--analysis-root",
                    str(analysis_root),
                    "--target-time-myr",
                    "3.0",
                    "--time-tol-myr",
                    "0.05",
                    "--depth-min-km",
                    "0",
                    "--depth-max-km",
                    "30",
                    "--depth-step-km",
                    "10",
                    "--k",
                    "2",
                    "--val-frac",
                    "0.25",
                    "--seed",
                    "5",
                    "--dataset-name",
                    dataset_name,
                    "--outdir",
                    str(dataset_root),
                ],
                cwd=REPO_ROOT,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(0, preprocess.returncode, msg=preprocess.stdout + "\n" + preprocess.stderr)

            dataset_dir = dataset_root / dataset_name

            # Step 2: train a tiny model on the synthetic PCA-score targets.
            train_proc = subprocess.run(
                [
                    sys.executable,
                    "src/emulator/train_emulator.py",
                    "--data-root",
                    str(dataset_root),
                    "--data-name",
                    dataset_name,
                    "--model",
                    "rf",
                    "--rf-trees",
                    "10",
                    "--rf-max-depth",
                    "3",
                    "--rf-jobs",
                    "1",
                    "--seed",
                    "0",
                    "--out",
                    str(model_root),
                ],
                cwd=REPO_ROOT,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(0, train_proc.returncode, msg=train_proc.stdout + "\n" + train_proc.stderr)

            model_dir = model_root / dataset_name / "rf"
            report_path = model_dir / "profile_pca_quality.json"

            # Step 3: compute the richer profile-PCA quality report.
            eval_proc = subprocess.run(
                [
                    sys.executable,
                    "src/emulator/evaluate_profile_pca_quality.py",
                    "--dataset-dir",
                    str(dataset_dir),
                    "--model-dir",
                    str(model_dir),
                ],
                cwd=REPO_ROOT,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(0, eval_proc.returncode, msg=eval_proc.stdout + "\n" + eval_proc.stderr)
            self.assertTrue(report_path.exists())

            with open(report_path, "r", encoding="utf-8") as f:
                report = json.load(f)

            self.assertEqual(1, report["schema_version"])
            self.assertEqual("profile-pca", report["dataset_mode"])
            self.assertEqual("const-vc", report["suite"])
            self.assertEqual(dataset_name, report["dataset_name"])
            self.assertEqual("rf", report["model_type"])
            self.assertEqual("raw", report["score_space"])

            # We want both levels of quality information:
            # - score-space accuracy on the predicted PCA targets
            # - reconstructed profile accuracy in temperature space
            for split_name in ("train", "val"):
                self.assertIn(split_name, report["metrics"])

                split_metrics = report["metrics"][split_name]
                self.assertIn("score_space", split_metrics)
                self.assertIn("profile_space", split_metrics)

                score_metrics = split_metrics["score_space"]
                self.assertIn("_macro_avg", score_metrics)
                self.assertIn("per_run_rmse", score_metrics)
                self.assertIn("PC1", score_metrics)
                self.assertIn("PC2", score_metrics)

                profile_metrics = split_metrics["profile_space"]
                self.assertIn("emulator_reconstruction", profile_metrics)
                self.assertIn("pca_truncation_baseline", profile_metrics)

                emu = profile_metrics["emulator_reconstruction"]
                baseline = profile_metrics["pca_truncation_baseline"]

                for metrics_block in (emu, baseline):
                    self.assertIn("rmse", metrics_block)
                    self.assertIn("mae", metrics_block)
                    self.assertIn("r2", metrics_block)
                    self.assertIn("per_run_rmse", metrics_block)
                    self.assertIn("rmse_by_depth", metrics_block)
                    self.assertEqual(4, len(metrics_block["rmse_by_depth"]))


if __name__ == "__main__":
    unittest.main()
