import numpy as np

from drift_injector import inject_drift

X = np.random.default_rng(0).normal(size=(200, 10))


def test_zero_intensity_returns_unchanged_data():
    out = inject_drift(X, intensity=0.0, kind="mean_shift")
    assert np.allclose(out, X)


def test_none_kind_returns_unchanged_data_even_with_intensity():
    out = inject_drift(X, intensity=1.0, kind="none")
    assert np.allclose(out, X)


def test_mean_shift_changes_some_feature_means():
    out = inject_drift(X, intensity=1.0, kind="mean_shift", seed=1)
    changed_features = np.sum(~np.isclose(out.mean(axis=0), X.mean(axis=0)))
    assert changed_features > 0


def test_noise_increases_variance():
    out = inject_drift(X, intensity=1.0, kind="noise", seed=1)
    assert out.var() > X.var()


def test_variance_scale_widens_spread_on_affected_features():
    out = inject_drift(X, intensity=1.0, kind="variance_scale", seed=1)
    assert out.std(axis=0).max() > X.std(axis=0).max()


def test_sensor_dropout_zeroes_some_features():
    out = inject_drift(X, intensity=1.0, kind="sensor_dropout", seed=1)
    zeroed_features = np.sum(np.all(out == 0.0, axis=0))
    assert zeroed_features > 0


def test_unknown_kind_raises():
    import pytest

    with pytest.raises(ValueError):
        inject_drift(X, intensity=0.5, kind="not_a_real_kind")
