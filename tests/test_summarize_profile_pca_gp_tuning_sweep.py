import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]


class SummarizeProfilePCAGPTuningSweepTests(unittest.TestCase):
    def _write_quality_report(
        self,
        path: Path,
        *,
        suite: str,
        dataset_name: str,
        model_kernel: str,
        val_profile_rmse: float,
        val_profile_p95_rmse: float,
        val_score_rmse: float,
        val_score_r2: float,
        val_pca_only_profile_rmse: float,
    ) -> None:
        # These tests synthesize only the fields the summary script actually
        # reads. That keeps the fixture short while still protecting the ranking
        # and output-table behavior we care about.
        path.parent.mkdir(parents=True, exist_ok=True)
        report = {
            "schema_version": 1,
            "dataset_mode": "profile-pca",
            "suite": suite,
            "dataset_name": dataset_name,
            "model_type": "gp",
            "model_kernel": model_kernel,
            "score_space": "whitened",
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
            sweep_root = tmp_path / "sweep"
            outdir = tmp_path / "out"

            # Two runs are tied on mean RMSE, so the p95 tie-breaker should
            # choose the safer run first.
            self._write_quality_report(
                sweep_root
                / "ramped-vc"
                / "gp_tuning"
                / "profileT_pca_t3Myr_k10_whitened"
                / "gp_matern25_r10_lsu1e3_nlow1e-6"
                / "profile_pca_quality.json",
                suite="ramped-vc",
                dataset_name="profileT_pca_t3Myr_k10_whitened",
                model_kernel="matern25",
                val_profile_rmse=8.05,
                val_profile_p95_rmse=14.50,
                val_score_rmse=0.55,
                val_score_r2=0.56,
                val_pca_only_profile_rmse=1.22,
            )
            self._write_quality_report(
                sweep_root
                / "ramped-vc"
                / "gp_tuning"
                / "profileT_pca_t3Myr_k10_whitened"
                / "gp_rbf_r10_lsu1e3_nlow1e-6"
                / "profile_pca_quality.json",
                suite="ramped-vc",
                dataset_name="profileT_pca_t3Myr_k10_whitened",
                model_kernel="rbf",
                val_profile_rmse=8.05,
                val_profile_p95_rmse=14.20,
                val_score_rmse=0.60,
                val_score_r2=0.52,
                val_pca_only_profile_rmse=1.22,
            )
            self._write_quality_report(
                sweep_root
                / "ramped-vc"
                / "gp_tuning"
                / "profileT_pca_t3Myr_k10_whitened"
                / "gp_matern15_r25_lsu1e4_nlow1e-8"
                / "profile_pca_quality.json",
                suite="ramped-vc",
                dataset_name="profileT_pca_t3Myr_k10_whitened",
                model_kernel="matern15",
                val_profile_rmse=8.20,
                val_profile_p95_rmse=14.00,
                val_score_rmse=0.58,
                val_score_r2=0.54,
                val_pca_only_profile_rmse=1.22,
            )

            proc = subprocess.run(
                [
                    sys.executable,
                    "src/emulator/profile_pca/sweeps/summarize_profile_pca_gp_tuning_sweep.py",
                    "--sweep-root",
                    str(sweep_root),
                    "--suites",
                    "ramped-vc",
                    "--dataset-pattern",
                    "profileT_pca_t3Myr_k10_whitened",
                    "--outdir",
                    str(outdir),
                ],
                cwd=REPO_ROOT,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(0, proc.returncode, msg=proc.stdout + "\n" + proc.stderr)

            combined = pd.read_csv(outdir / "profile_pca_gp_tuning_summary.csv")

            self.assertEqual("gp_rbf_r10_lsu1e3_nlow1e-6", combined.iloc[0]["tag"])
            self.assertEqual("rbf", combined.iloc[0]["kernel"])
            self.assertEqual(10, combined.iloc[0]["restarts"])
            self.assertEqual("1000.0", str(combined.iloc[0]["ls_high_tag"]))
            self.assertEqual("1e-06", str(combined.iloc[0]["noise_low_tag"]))
            self.assertEqual("gp_matern25_r10_lsu1e3_nlow1e-6", combined.iloc[1]["tag"])

            self.assertTrue((outdir / "profile_pca_gp_tuning_summary.md").exists())
            self.assertTrue((outdir / "ramped-vc_profile_pca_gp_tuning_summary.md").exists())


if __name__ == "__main__":
    unittest.main()
