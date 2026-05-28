import sys
import unittest
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SCIENCE_DIR = REPO_ROOT / "src" / "emulator" / "single_depth" / "science"
sys.path.insert(0, str(SCIENCE_DIR))

try:
    from SALib import ProblemSpec  # noqa: E402

    import _sobol_io  # noqa: E402

    _HAVE_SALIB = True
except ImportError:
    _HAVE_SALIB = False


@unittest.skipUnless(_HAVE_SALIB, "SALib not installed")
class SobolSmokeTests(unittest.TestCase):
    """Validate the Sobol machinery on a known additive model.

    For y = 3*x0 + 1*x1 + 0*x2 over a box, first-order indices must dominate
    (the model is purely additive), ST must rank x0 > x1 > x2, and the
    second-order matrix must be present with the expected shape.
    """

    def setUp(self) -> None:
        rng = np.random.default_rng(0)
        # Synthetic "training" data so build_problem has realistic spread.
        self.x_train = rng.uniform(
            low=[0.0, 0.0, 0.0], high=[1.0, 1.0, 1.0], size=(400, 3)
        )
        self.feature_cols = ["x0", "x1", "x2"]

    def _coef_model(self, X: np.ndarray) -> np.ndarray:
        return 3.0 * X[:, 0] + 1.0 * X[:, 1] + 0.0 * X[:, 2]

    def test_build_problem_bounds(self) -> None:
        problem = _sobol_io.build_problem(self.feature_cols, self.x_train, 0.01, 0.99)
        self.assertEqual(problem["num_vars"], 3)
        self.assertEqual(problem["names"], self.feature_cols)
        self.assertEqual(len(problem["bounds"]), 3)
        for lo, hi in problem["bounds"]:
            self.assertLess(lo, hi)

    def test_build_problem_widens_degenerate_feature(self) -> None:
        x = self.x_train.copy()
        x[:, 2] = 5.0  # constant column
        problem = _sobol_io.build_problem(self.feature_cols, x, 0.01, 0.99)
        lo, hi = problem["bounds"][2]
        self.assertLess(lo, hi)

    def test_sobol_indices_additive_model(self) -> None:
        problem = _sobol_io.build_problem(self.feature_cols, self.x_train, 0.01, 0.99)
        sp = ProblemSpec(problem)
        sp.sample_sobol(256, calc_second_order=True, seed=0)
        sp.evaluate(self._coef_model)
        sp.analyze_sobol(calc_second_order=True, seed=0)
        res = sp.analysis

        s1 = np.asarray(res["S1"], dtype=float)
        st = np.asarray(res["ST"], dtype=float)
        s2 = np.asarray(res["S2"], dtype=float)

        self.assertEqual(s1.shape, (3,))
        self.assertEqual(st.shape, (3,))
        self.assertEqual(s2.shape, (3, 3))

        # Total effect must rank with the coefficients: x0 > x1 > x2.
        self.assertGreater(st[0], st[1])
        self.assertGreater(st[1], st[2])

        # Purely additive model: total effect ~ first-order (no interactions).
        np.testing.assert_allclose(st, s1, atol=0.05)

        # First-order indices should sum to ~1 for an additive model.
        self.assertAlmostEqual(float(s1.sum()), 1.0, delta=0.1)

    def test_predict_raw_inverts_scaler(self) -> None:
        # predict_raw should undo standardization for a trivial identity model.
        class _Identity:
            def predict(self, x):
                return x[:, 0]

        x_raw = np.array([[2.0, 0.0], [4.0, 0.0]])
        x_mean = np.array([1.0, 0.0])
        x_std = np.array([2.0, 1.0])
        y_mean = np.array([10.0])
        y_std = np.array([3.0])
        # standardized x0 = (x-1)/2 -> [0.5, 1.5]; *3 + 10 -> [11.5, 14.5]
        out = _sobol_io.predict_raw(_Identity(), x_raw, x_mean, x_std, y_mean, y_std)
        np.testing.assert_allclose(out, [11.5, 14.5])


if __name__ == "__main__":
    unittest.main()
