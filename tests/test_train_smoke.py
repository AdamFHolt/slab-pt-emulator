import subprocess
import sys
import tempfile
import unittest
import json
from pathlib import Path

import numpy as np
import train
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]


class TrainConfigSmokeTests(unittest.TestCase):
    def test_configs_load(self) -> None:
        # These are the original depth-based training configs.
        # Loading them here protects against YAML syntax issues and accidental
        # drift in the required top-level fields.
        for cfg_rel in (
            "configs/gp.const-vc.yaml",
            "configs/gp.ramped-vc.yaml",
            "configs/rf.const-vc.yaml",
            "configs/rf.ramped-vc.yaml",
        ):
            cfg = train._load_config(REPO_ROOT / cfg_rel)
            self.assertIsInstance(cfg, dict)
            self.assertIn(cfg.get("suite"), {"const-vc", "ramped-vc"})

    def test_profile_pca_configs_load(self) -> None:
        # The profile-PCA workflow uses separate configs from the standard
        # depth-based GP workflow. This test makes sure those files still load
        # and keep the expected dataset discovery mode.
        for cfg_rel in (
            "configs/gp.const-vc.profile-pca.yaml",
            "configs/gp.ramped-vc.profile-pca.yaml",
        ):
            cfg = train._load_config(REPO_ROOT / cfg_rel)
            self.assertIsInstance(cfg, dict)
            self.assertIn(cfg.get("suite"), {"const-vc", "ramped-vc"})
            self.assertEqual("profile-pca", cfg.get("dataset", {}).get("mode"))

    def test_discover_datasets_variant_filter(self) -> None:
        cfg = train._load_config(REPO_ROOT / "configs/gp.const-vc.yaml")
        names = train._discover_datasets(
            REPO_ROOT / "src" / "emulator" / "data" / "const-vc",
            cfg["dataset"],
        )
        self.assertTrue(names)
        self.assertTrue(all(name.endswith("_dTdt") for name in names))
        self.assertEqual("10km_dTdt", names[0])

    def test_discover_datasets_profile_pca_mode_with_prefix(self) -> None:
        # We build a tiny fake suite directory so this test does not depend on
        # checked-in profile-PCA data being present in the repo.
        #
        # The behavior we want to lock down is:
        # - profile-pca mode ignores normal depth datasets
        # - only profile-PCA-like dataset folders are returned
        # - names are returned in sorted order
        with tempfile.TemporaryDirectory() as tmp:
            suite_dir = Path(tmp)
            for name in (
                "40km_dTdt",
                "profileT_pca_t5Myr_k8",
                "profileT_pca_t0p5Myr_k8",
                "profileT_pca_t3Myr_k6",
            ):
                (suite_dir / name).mkdir(parents=True, exist_ok=True)

            names = train._discover_datasets(
                suite_dir,
                {"mode": "profile-pca", "prefix": "profileT_pca_"},
            )

            self.assertEqual(
                [
                    "profileT_pca_t0p5Myr_k8",
                    "profileT_pca_t3Myr_k6",
                    "profileT_pca_t5Myr_k8",
                ],
                names,
            )

    def test_build_train_cmd_gp(self) -> None:
        cfg = train._load_config(REPO_ROOT / "configs/gp.const-vc.yaml")
        cmd = train._build_train_cmd(
            cfg,
            REPO_ROOT / "src" / "emulator" / "data" / "const-vc",
            REPO_ROOT / "src" / "emulator" / "models" / "const-vc",
            "10km_dTdt",
        )
        cmd_text = " ".join(cmd)
        self.assertIn("--model gp", cmd_text)
        self.assertIn("--kernel matern25", cmd_text)
        self.assertIn("--data-name 10km_dTdt", cmd_text)

    def test_build_train_cmd_rf(self) -> None:
        cfg = train._load_config(REPO_ROOT / "configs/rf.const-vc.yaml")
        cmd = train._build_train_cmd(
            cfg,
            REPO_ROOT / "src" / "emulator" / "data" / "const-vc",
            REPO_ROOT / "src" / "emulator" / "models" / "const-vc",
            "10km_dTdt_thermalParam",
        )
        cmd_text = " ".join(cmd)
        self.assertIn("--model rf", cmd_text)
        self.assertIn("--rf-trees 600", cmd_text)
        self.assertIn("--rf-max-depth 40", cmd_text)

    def test_build_train_cmd_profile_pca_gp(self) -> None:
        # Profile-PCA training still funnels through the same unified training
        # entrypoint, but it should use the dedicated profile-PCA config and a
        # dataset name that looks nothing like the usual 10km_dTdt pattern.
        cfg = train._load_config(REPO_ROOT / "configs/gp.const-vc.profile-pca.yaml")
        cmd = train._build_train_cmd(
            cfg,
            REPO_ROOT / "src" / "emulator" / "data" / "const-vc",
            REPO_ROOT / "src" / "emulator" / "models" / "const-vc",
            "profileT_pca_t3Myr_k8",
        )
        cmd_text = " ".join(cmd)
        self.assertIn("--model gp", cmd_text)
        self.assertIn("--kernel matern25", cmd_text)
        self.assertIn("--data-name profileT_pca_t3Myr_k8", cmd_text)

    def test_cli_dry_run_one_dataset(self) -> None:
        proc = subprocess.run(
            [
                sys.executable,
                "train.py",
                "--config",
                "configs/gp.const-vc.yaml",
                "--dry-run",
                "--datasets",
                "10km_dTdt",
            ],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(0, proc.returncode, msg=proc.stdout + "\n" + proc.stderr)
        self.assertIn("[RUN]", proc.stdout)
        self.assertIn("[OK] dry-run complete.", proc.stdout)

    def test_cli_dry_run_profile_pca_one_dataset(self) -> None:
        # Use an explicit --datasets override so this test validates the PCA
        # command path without requiring real profile-PCA datasets to already
        # exist on disk in the repository.
        proc = subprocess.run(
            [
                sys.executable,
                "train.py",
                "--config",
                "configs/gp.const-vc.profile-pca.yaml",
                "--dry-run",
                "--datasets",
                "profileT_pca_t0p5Myr_k8",
            ],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(0, proc.returncode, msg=proc.stdout + "\n" + proc.stderr)
        self.assertIn("[RUN]", proc.stdout)
        self.assertIn("profileT_pca_t0p5Myr_k8", proc.stdout)
        self.assertIn("[OK] dry-run complete.", proc.stdout)

    def test_cli_dry_run_profile_pca_sweep(self) -> None:
        # The sweep runner is just orchestration, so a dry-run test is the
        # right level here: it validates dataset naming, command construction,
        # and step ordering without launching an expensive real sweep.
        proc = subprocess.run(
            [
                sys.executable,
                "src/emulator/run_profile_pca_sweep.py",
                "--suites",
                "const-vc",
                "--times",
                "3",
                "--ks",
                "4",
                "--score-spaces",
                "raw,whitened",
                "--dry-run",
            ],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(0, proc.returncode, msg=proc.stdout + "\n" + proc.stderr)
        self.assertIn("profileT_pca_t3Myr_k4_raw", proc.stdout)
        self.assertIn("profileT_pca_t3Myr_k4_whitened", proc.stdout)
        self.assertIn("[OK] dry-run complete.", proc.stdout)

    def test_invalid_dataset_mode_raises(self) -> None:
        with self.assertRaisesRegex(ValueError, "dataset.mode must be 'auto', 'list', or 'profile-pca'."):
            train._discover_datasets(
                REPO_ROOT / "src" / "emulator" / "data" / "const-vc",
                {"mode": "bad-mode"},
            )

    def test_invalid_model_type_raises(self) -> None:
        cfg = train._load_config(REPO_ROOT / "configs/gp.const-vc.yaml")
        cfg["model"]["type"] = "xgboost"
        with self.assertRaisesRegex(ValueError, "model.type must be 'gp' or 'rf'."):
            train._build_train_cmd(
                cfg,
                REPO_ROOT / "src" / "emulator" / "data" / "const-vc",
                REPO_ROOT / "src" / "emulator" / "models" / "const-vc",
                "10km_dTdt",
            )

    def test_missing_suite_directory_raises(self) -> None:
        missing = REPO_ROOT / "src" / "emulator" / "data" / "does-not-exist"
        with self.assertRaisesRegex(FileNotFoundError, "Suite data directory not found"):
            train._discover_datasets(missing, {"mode": "auto"})

    def test_invalid_suite_cli_fails(self) -> None:
        bad_cfg = {
            "suite": "bad-suite",
            "data_root": "src/emulator/data",
            "out_root": "src/emulator/models",
            "dataset": {"mode": "list", "names": ["10km_dTdt"]},
            "model": {"type": "gp"},
            "execution": {"dry_run": True},
        }
        with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as tf:
            yaml.safe_dump(bad_cfg, tf)
            cfg_path = Path(tf.name)
        try:
            proc = subprocess.run(
                [sys.executable, "train.py", "--config", str(cfg_path)],
                cwd=REPO_ROOT,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertNotEqual(0, proc.returncode)
            self.assertIn("suite must be 'const-vc' or 'ramped-vc'.", proc.stderr)
        finally:
            cfg_path.unlink(missing_ok=True)

    def test_tiny_rf_training_integration(self) -> None:
        rng = np.random.default_rng(7)
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            data_root = tmp_path / "data"
            out_root = tmp_path / "models"
            data_name = "tinyset"
            ds_dir = data_root / data_name
            ds_dir.mkdir(parents=True, exist_ok=True)

            x_raw = rng.normal(size=(12, 3))
            y_raw = (0.5 * x_raw[:, 0] - 0.25 * x_raw[:, 1] + 0.1).reshape(-1, 1)

            x_mu = x_raw.mean(axis=0)
            x_sd = x_raw.std(axis=0)
            x_sd[x_sd == 0.0] = 1.0
            y_mu = y_raw.mean(axis=0)
            y_sd = y_raw.std(axis=0)
            y_sd[y_sd == 0.0] = 1.0

            x_std = (x_raw - x_mu) / x_sd
            y_std = (y_raw - y_mu) / y_sd

            train_idx = np.arange(0, 9, dtype=int)
            val_idx = np.arange(9, 12, dtype=int)

            np.save(ds_dir / "X_raw.npy", x_raw)
            np.save(ds_dir / "Y_raw.npy", y_raw)
            np.save(ds_dir / "X_std.npy", x_std)
            np.save(ds_dir / "Y_std.npy", y_std)
            np.save(ds_dir / "train_idx.npy", train_idx)
            np.save(ds_dir / "val_idx.npy", val_idx)

            metadata = {
                "feature_cols": ["f0", "f1", "f2"],
                "target": {"target_cols": ["y0"]},
                "scalers": {
                    "Y": {
                        "mean": y_mu.tolist(),
                        "std": y_sd.tolist(),
                    }
                },
            }
            with open(ds_dir / "metadata.json", "w", encoding="utf-8") as f:
                json.dump(metadata, f)

            proc = subprocess.run(
                [
                    sys.executable,
                    "src/emulator/train_emulator.py",
                    "--data-root",
                    str(data_root),
                    "--data-name",
                    data_name,
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
                    str(out_root),
                ],
                cwd=REPO_ROOT,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(0, proc.returncode, msg=proc.stdout + "\n" + proc.stderr)

            model_dir = out_root / data_name / "rf"
            self.assertTrue((model_dir / "model.joblib").exists())
            self.assertTrue((model_dir / "yhat_train.npy").exists())
            self.assertTrue((model_dir / "yhat_val.npy").exists())
            self.assertTrue((model_dir / "report.json").exists())


if __name__ == "__main__":
    unittest.main()
