import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]


class PreprocessProfilePCATests(unittest.TestCase):
    # These tests use a tiny synthetic analysis directory so we can exercise the
    # real profile-PCA preprocessing script end-to-end without relying on any
    # checked-in ASPECT outputs.
    #
    # The goal is to validate the contract of the script:
    # - what input directory layout it expects
    # - which output files it writes
    # - what metadata fields it records
    # - how score-space choices affect the saved targets/scalers

    def _write_params_csv(self, path: Path, suite: str, n_runs: int) -> None:
        # The script expects different default feature columns by suite.
        # For these tests we populate the exact columns it will look for so the
        # resulting dataset is as realistic as possible.
        rows = []
        for i in range(n_runs):
            row = {
                "run_id": f"{i:03d}",
                "v_conv": 2.0 + 0.25 * i,
                "age_SP": 40.0 + 2.0 * i,
                "age_OP": 20.0 + 1.5 * i,
                "dip_int": 30.0 + 3.0 * i,
                "eta_UM": 5.0e19 + 1.0e19 * i,
            }
            if suite == "ramped-vc":
                row["t_conv"] = 0.5 + 0.2 * i
            rows.append(row)

        pd.DataFrame(rows).to_csv(path, index=False)

    def _write_profile_csv(self, path: Path, time_myr: float, depths_km: np.ndarray, temps_c: np.ndarray) -> None:
        # Each Tprof file represents one time slice for one run. The preprocess
        # script requires exactly these three columns.
        pd.DataFrame(
            {
                "time_Myr": np.full(depths_km.shape, time_myr, dtype=float),
                "depth_km": depths_km,
                "T_C": temps_c,
            }
        ).to_csv(path, index=False)

    def _build_synthetic_analysis_tree(self, root: Path, suite: str, n_runs: int = 6) -> tuple[Path, Path]:
        # Create the two top-level inputs consumed by the script:
        # 1. a params CSV
        # 2. an analysis directory with run_XXX/Tprof_*.csv files
        #
        # We write two profile times per run:
        # - one very close to the target time we will ask for
        # - one farther away
        #
        # This lets the test verify that "nearest time within tolerance" logic
        # is actually being used.
        params_path = root / f"params-list.{suite}.csv"
        analysis_root = root / "analysis"
        analysis_root.mkdir(parents=True, exist_ok=True)

        self._write_params_csv(params_path, suite=suite, n_runs=n_runs)

        depths = np.array([0.0, 10.0, 20.0, 30.0], dtype=float)
        target_time = 3.0
        near_time = 3.01
        far_time = 3.20

        for i in range(n_runs):
            run_dir = analysis_root / f"run_{i:03d}"
            run_dir.mkdir(parents=True, exist_ok=True)

            # Build simple but non-identical temperature profiles per run.
            # They vary with both depth and run index so PCA has real structure
            # to fit rather than duplicate rows.
            base = 800.0 - 7.0 * depths
            run_offset = 5.0 * i
            curvature = 0.08 * (depths ** 2) / 10.0

            near_profile = base + run_offset + curvature
            far_profile = base + run_offset + curvature + 15.0

            self._write_profile_csv(run_dir / f"Tprof_targetish_{i:03d}.csv", near_time, depths, near_profile)
            self._write_profile_csv(run_dir / f"Tprof_far_{i:03d}.csv", far_time, depths, far_profile)

        # Keep the target time close to the "near" files so every run should
        # contribute one profile to the dataset.
        self.assertEqual(3.0, target_time)
        return params_path, analysis_root

    def test_preprocess_profile_pca_builds_expected_dataset_files(self) -> None:
        # This is the main end-to-end smoke test for the PCA preprocess script.
        # It checks that a valid synthetic input tree produces a dataset folder
        # with the expected array artifacts and rich metadata.
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            params_path, analysis_root = self._build_synthetic_analysis_tree(tmp_path, suite="const-vc")
            out_root = tmp_path / "datasets"
            dataset_name = "profileT_pca_t3Myr_k3_test"

            proc = subprocess.run(
                [
                    sys.executable,
                    "src/emulator/profile_pca/preprocess_profile_pca.py",
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
                    "3",
                    "--val-frac",
                    "0.33",
                    "--seed",
                    "7",
                    "--dataset-name",
                    dataset_name,
                    "--outdir",
                    str(out_root),
                ],
                cwd=REPO_ROOT,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(0, proc.returncode, msg=proc.stdout + "\n" + proc.stderr)

            ds_dir = out_root / dataset_name
            self.assertTrue(ds_dir.exists())

            # These are the core saved artifacts the downstream training code
            # expects to find in a preprocessed profile-PCA dataset.
            for rel in (
                "X_raw.npy",
                "X_std.npy",
                "Y_raw.npy",
                "Y_std.npy",
                "scores_raw.npy",
                "scores_whitened.npy",
                "pca_score_scale.npy",
                "train_idx.npy",
                "val_idx.npy",
                "pca_mean_profile.npy",
                "pca_components.npy",
                "pca_explained_variance_ratio.npy",
                "metadata.json",
            ):
                self.assertTrue((ds_dir / rel).exists(), msg=f"missing output file: {rel}")

            x_raw = np.load(ds_dir / "X_raw.npy")
            y_raw = np.load(ds_dir / "Y_raw.npy")
            y_std = np.load(ds_dir / "Y_std.npy")
            train_idx = np.load(ds_dir / "train_idx.npy")
            val_idx = np.load(ds_dir / "val_idx.npy")
            pca_components = np.load(ds_dir / "pca_components.npy")
            pca_mean_profile = np.load(ds_dir / "pca_mean_profile.npy")

            # We built 6 runs, each with 4 depth points and 5 const-vc features.
            self.assertEqual((6, 5), x_raw.shape)
            self.assertEqual((6, 3), y_raw.shape)
            self.assertEqual((6, 3), y_std.shape)
            self.assertEqual(4, pca_mean_profile.shape[0])
            self.assertEqual((3, 4), pca_components.shape)
            self.assertEqual(6, train_idx.size + val_idx.size)
            self.assertGreaterEqual(train_idx.size, 2)
            self.assertGreaterEqual(val_idx.size, 1)

            with open(ds_dir / "metadata.json", "r", encoding="utf-8") as f:
                meta = json.load(f)

            self.assertEqual("const-vc", meta["suite"])
            self.assertEqual("profile-pca", meta["dataset_mode"])
            self.assertEqual(dataset_name, meta["dataset_name"])
            self.assertEqual("profile_pca_scores", meta["target"]["target_kind"])
            self.assertEqual("raw", meta["target"]["score_space"])
            self.assertEqual(3, meta["pca"]["k"])
            self.assertEqual("train_split_only", meta["pca"]["fit_on"])
            self.assertEqual(4, meta["profile"]["n_depth"])
            self.assertEqual(["PC1", "PC2", "PC3"], meta["target"]["target_cols"])
            self.assertEqual(6, len(meta["profile"]["run_ids"]))

            # The selected times in metadata should come from the "near" files,
            # not the farther-away alternatives we also placed in each run dir.
            self.assertTrue(all(abs(t - 3.01) < 1e-9 for t in meta["profile"]["time_selected_myr"]))

    def test_preprocess_profile_pca_whitened_targets_keep_identity_scaler(self) -> None:
        # The script has special logic for score_space=whitened:
        # it should *not* standardize those targets a second time.
        #
        # This test locks down that behavior by checking both:
        # - metadata scaler values are identity-like
        # - Y_raw and Y_std are exactly the same arrays on disk
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            params_path, analysis_root = self._build_synthetic_analysis_tree(tmp_path, suite="ramped-vc")
            out_root = tmp_path / "datasets"
            dataset_name = "profileT_pca_t3Myr_k2_whitened"

            proc = subprocess.run(
                [
                    sys.executable,
                    "src/emulator/profile_pca/preprocess_profile_pca.py",
                    "--suite",
                    "ramped-vc",
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
                    "--score-space",
                    "whitened",
                    "--val-frac",
                    "0.33",
                    "--seed",
                    "11",
                    "--dataset-name",
                    dataset_name,
                    "--outdir",
                    str(out_root),
                ],
                cwd=REPO_ROOT,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(0, proc.returncode, msg=proc.stdout + "\n" + proc.stderr)

            ds_dir = out_root / dataset_name
            y_raw = np.load(ds_dir / "Y_raw.npy")
            y_std = np.load(ds_dir / "Y_std.npy")

            with open(ds_dir / "metadata.json", "r", encoding="utf-8") as f:
                meta = json.load(f)

            self.assertEqual("whitened", meta["target"]["score_space"])
            self.assertEqual(2, meta["pca"]["k"])
            self.assertEqual(["v_conv", "t_conv", "age_SP", "age_OP", "dip_int", "eta_UM"], meta["feature_cols"])
            self.assertEqual([0.0, 0.0], meta["scalers"]["Y"]["mean"])
            self.assertEqual([1.0, 1.0], meta["scalers"]["Y"]["std"])
            self.assertTrue(np.array_equal(y_raw, y_std))


if __name__ == "__main__":
    unittest.main()
