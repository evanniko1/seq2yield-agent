"""G7 — predictive calibration via train-residual prediction intervals."""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from seq2yield.statistics.calibration import residual_interval_coverage  # noqa: E402


def test_well_calibrated_model_covers_near_nominal():
    rng = np.random.default_rng(0)
    ytr = rng.normal(0, 1, 2000); ptr = ytr + rng.normal(0, 0.3, 2000)
    yte = rng.normal(0, 1, 2000); pte = yte + rng.normal(0, 0.3, 2000)   # same noise -> calibrated
    r = residual_interval_coverage(ytr, ptr, yte, pte, nominal=0.9)
    assert abs(r["empirical_coverage"] - 0.9) <= 0.1 and r["verdict"] == "calibrated"


def test_overconfident_when_test_noisier_than_train():
    rng = np.random.default_rng(1)
    ytr = rng.normal(0, 1, 2000); ptr = ytr + rng.normal(0, 0.1, 2000)   # tiny train residuals
    yte = rng.normal(0, 1, 2000); pte = yte + rng.normal(0, 1.0, 2000)   # big test error
    r = residual_interval_coverage(ytr, ptr, yte, pte, nominal=0.9)
    assert r["empirical_coverage"] < 0.9 and r["verdict"] == "over_confident"
