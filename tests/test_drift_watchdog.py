import numpy as np

from drift_injector import inject_drift
from drift_watchdog import FeatureDriftWatchdog

rng = np.random.default_rng(42)
X_ref = rng.normal(size=(500, 20))
FEATURE_NAMES = [f"f{i}" for i in range(20)]


def test_no_drift_on_same_distribution():
    watchdog = FeatureDriftWatchdog(feature_names=FEATURE_NAMES).fit(X_ref)
    X_same_dist = rng.normal(size=(500, 20))
    report = watchdog.detect(X_same_dist)
    assert report.n_drifted == 0


def test_detects_drift_on_strongly_shifted_data():
    watchdog = FeatureDriftWatchdog(feature_names=FEATURE_NAMES).fit(X_ref)
    X_shifted = inject_drift(X_ref, intensity=1.0, kind="mean_shift", seed=7)
    report = watchdog.detect(X_shifted)
    assert report.n_drifted > 0
    assert set(report.drifted_features).issubset(set(FEATURE_NAMES))


def test_corrected_p_value_is_bonferroni_adjusted():
    watchdog = FeatureDriftWatchdog(feature_names=FEATURE_NAMES, base_p_value=0.01)
    assert watchdog.corrected_p_value == 0.01 / len(FEATURE_NAMES)


def test_detect_before_fit_raises():
    import pytest

    watchdog = FeatureDriftWatchdog(feature_names=FEATURE_NAMES)
    with pytest.raises(RuntimeError):
        watchdog.detect(X_ref)
