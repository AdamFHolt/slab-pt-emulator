import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]


class SummarizeProfilePCASweepTests(unittest.TestCase):
    def _write_quality_report(
        self,
        path: Path,
        *,
        suite: str,
        dataset_name: str,
        score_space: str,
        val_profile_rmse: float,
        val_profile_p95_rmse: float,
        val_score_rmse: float,
        val_score_r2: float,
        val_pca_only_profile_rmse: float,
    ) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        report = {
            "schema_version": 1,
            "dataset_mode": "profile-pca",
            "suite": suite,
            "dataset_name": dataset_name,
            "model_type": "gp",
            "model_kernel": "matern25",
            "score_space": score_space,
            "metrics": {
                "val": {
                    "score_space": {
                        "_macro_avg": {
                            "rmse": val_score_rmse,
                            "mae": 0.0,
                            "r2": val_score_r2,
                        }
                    },
                    "profile_space": {
                        "emulator_reconstruction": {
                            "rmse": val_profile_rmse,
                            "mae": 0.0,
                            "r2": 0.0,
                            "per_run_rmse": {
                                "mean": val_profile_rmse,
                                "median": val_profile_rmse,
                                "p90": val_profile_p95_rmse,
                                "p95": val_profile_p95_rmse,
                                "max": val_profile_p95_rmse,
                            },
                            "rmse_by_depth": [val_profile_rmse, val_profile_rmse],
                        },
                        "pca_truncation_baseline": {
                            "rmse": val_pca_only_profile_rmse,
                            "mae": 0.0,
                            "r2": 0.0,
                            "per_run_rmse": {
                                "mean": val_pca_only_profile_rmse,
                                "median": val_pca_only_profile_rmse,
                                "p90": val_pca_only_profile_rmse,
                                "p95": val_pca_only_profile_rmse,
                                "max": val_pca_only_profile_rmse,
                            },
                            "rmse_by_depth": [val_pca_only_profile_rmse, val_pca_only_profile_rmse],
                        },
                    },
                }
            },
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

    def test_summary_tables_rank_by_profile_rmse_then_p95(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            models_root = tmp_path / "models"
            outdir = tmp_path / "out"

            # Create three sweep entries. Two belong to const-vc so we can test
            # the ranking logic directly within one suite.
            self._write_quality_report(
                models_root / "const-vc" / "profileT_pca_t3Myr_k4_raw" / "gp_m25" / "profile_pca_quality.json",
                suite="const-vc",
                dataset_name="profileT_pca_t3Myr_k4_raw",
                score_space="raw",
                val_profile_rmse=9.0,
                val_profile_p95_rmse=14.0,
                val_score_rmse=20.0,
                val_score_r2=0.70,
                val_pca_only_profile_rmse=2.0,
            )
            self._write_quality_report(
                models_root / "const-vc" / "profileT_pca_t3Myr_k8_whitened" / "gp_m25" / "profile_pca_quality.json",
                suite="const-vc",
                dataset_name="profileT_pca_t3Myr_k8_whitened",
                score_space="whitened",
                val_profile_rmse=8.0,
                val_profile_p95_rmse=16.0,
                val_score_rmse=19.0,
                val_score_r2=0.72,
                val_pca_only_profile_rmse=1.8,
            )
            self._write_quality_report(
                models_root / "ramped-vc" / "profileT_pca_t3Myr_k6_raw" / "gp_m25" / "profile_pca_quality.json",
                suite="ramped-vc",
                dataset_name="profileT_pca_t3Myr_k6_raw",
                score_space="raw",
                val_profile_rmse=7.5,
                val_profile_p95_rmse=12.0,
                val_score_rmse=18.0,
                val_score_r2=0.75,
                val_pca_only_profile_rmse=1.6,
            )

            proc = subprocess.run(
                [
                    sys.executable,
                    "src/emulator/profile_pca/summarize_profile_pca_sweep.py",
                    "--models-root",
                    str(models_root),
                    "--suites",
                    "const-vc,ramped-vc",
                    "--dataset-pattern",
                    "profileT_pca_t3Myr",
                    "--outdir",
                    str(outdir),
                ],
                cwd=REPO_ROOT,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(0, proc.returncode, msg=proc.stdout + "\n" + proc.stderr)

            combined = pd.read_csv(outdir / "profile_pca_sweep_summary.csv")
            const_only = pd.read_csv(outdir / "const-vc_profile_pca_sweep_summary.csv")

            # Overall best should be the row with the lowest validation profile RMSE.
            self.assertEqual("ramped-vc", combined.iloc[0]["suite"])
            self.assertEqual("profileT_pca_t3Myr_k6_raw", combined.iloc[0]["dataset_name"])

            # Within const-vc, the whitened k=8 run wins because it has the
            # smaller validation profile RMSE, even though its p95 is worse.
            self.assertEqual("profileT_pca_t3Myr_k8_whitened", const_only.iloc[0]["dataset_name"])
            self.assertEqual("profileT_pca_t3Myr_k4_raw", const_only.iloc[1]["dataset_name"])

            self.assertTrue((outdir / "profile_pca_sweep_summary.md").exists())
            self.assertTrue((outdir / "const-vc_profile_pca_sweep_summary.md").exists())


if __name__ == "__main__":
    unittest.main()
