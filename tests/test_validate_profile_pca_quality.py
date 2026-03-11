import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]


class ValidateProfilePCAQualityTests(unittest.TestCase):
    def _write_report(self, path: Path, score_r2: float, score_rmse: float, profile_rmse: float, profile_p95: float) -> None:
        # Keep the synthetic report as small as possible while still matching
        # the fields the validator actually consumes from the real report.
        report = {
            "schema_version": 1,
            "dataset_mode": "profile-pca",
            "suite": "const-vc",
            "dataset_name": "profileT_pca_t3Myr_k8",
            "model_type": "gp",
            "score_space": "raw",
            "metrics": {
                "val": {
                    "score_space": {
                        "_macro_avg": {
                            "r2": score_r2,
                            "rmse": score_rmse,
                            "mae": 0.0,
                        }
                    },
                    "profile_space": {
                        "emulator_reconstruction": {
                            "rmse": profile_rmse,
                            "mae": 0.0,
                            "r2": 0.0,
                            "per_run_rmse": {
                                "mean": profile_rmse,
                                "median": profile_rmse,
                                "p90": profile_p95,
                                "p95": profile_p95,
                                "max": profile_p95,
                            },
                            "rmse_by_depth": [profile_rmse, profile_rmse],
                        }
                    },
                }
            },
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

    def _write_thresholds(self, path: Path) -> None:
        cfg = {
            "model_tag": "gp_m25",
            "thresholds": {
                "const-vc": {
                    "profileT_pca_t3Myr_k8": {
                        "score_r2_min": 0.90,
                        "score_rmse_max": 1.50,
                        "profile_rmse_max": 12.0,
                        "profile_p95_rmse_max": 16.0,
                    }
                }
            },
        }
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f, sort_keys=False)

    def test_validator_passes_when_all_thresholds_are_met(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            report_dir = tmp_path / "models" / "const-vc" / "profileT_pca_t3Myr_k8" / "gp_m25"
            report_dir.mkdir(parents=True, exist_ok=True)

            self._write_report(
                report_dir / "profile_pca_quality.json",
                score_r2=0.95,
                score_rmse=1.20,
                profile_rmse=9.5,
                profile_p95=14.0,
            )
            thresholds_path = tmp_path / "thresholds.yaml"
            self._write_thresholds(thresholds_path)

            proc = subprocess.run(
                [
                    sys.executable,
                    "src/emulator/profile_pca/validate_profile_pca_quality.py",
                    "--thresholds",
                    str(thresholds_path),
                    "--models-root",
                    str(tmp_path / "models"),
                ],
                cwd=REPO_ROOT,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(0, proc.returncode, msg=proc.stdout + "\n" + proc.stderr)
            self.assertIn("[PASS] const-vc/profileT_pca_t3Myr_k8", proc.stdout)

    def test_validator_fails_when_profile_rmse_exceeds_threshold(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            report_dir = tmp_path / "models" / "const-vc" / "profileT_pca_t3Myr_k8" / "gp_m25"
            report_dir.mkdir(parents=True, exist_ok=True)

            self._write_report(
                report_dir / "profile_pca_quality.json",
                score_r2=0.95,
                score_rmse=1.20,
                profile_rmse=20.0,
                profile_p95=25.0,
            )
            thresholds_path = tmp_path / "thresholds.yaml"
            self._write_thresholds(thresholds_path)

            proc = subprocess.run(
                [
                    sys.executable,
                    "src/emulator/profile_pca/validate_profile_pca_quality.py",
                    "--thresholds",
                    str(thresholds_path),
                    "--models-root",
                    str(tmp_path / "models"),
                ],
                cwd=REPO_ROOT,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertNotEqual(0, proc.returncode)
            self.assertIn("[FAIL] const-vc/profileT_pca_t3Myr_k8", proc.stdout)


if __name__ == "__main__":
    unittest.main()
