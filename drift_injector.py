"""Synthetic drift injection at an adjustable intensity in [0, 1].
Each kind models a real failure mode: mean_shift (a sensor/gain change),
noise (a noisier input channel), variance_scale (a more diverse incoming
population), sensor_dropout (a broken upstream feature)."""

import numpy as np

DRIFT_KINDS = ["none", "mean_shift", "noise", "variance_scale", "sensor_dropout"]


def inject_drift(X: np.ndarray, intensity: float, kind: str = "mean_shift", seed: int = 0) -> np.ndarray:
    if kind not in DRIFT_KINDS:
        raise ValueError(f"unknown drift kind: {kind!r}, expected one of {DRIFT_KINDS}")

    X = X.copy().astype(float)
    if intensity <= 0 or kind == "none":
        return X

    rng = np.random.default_rng(seed)
    n_features = X.shape[1]
    feature_std = X.std(axis=0)
    feature_std[feature_std == 0] = 1.0

    if kind == "mean_shift":
        n_affected = max(1, int(0.3 * n_features))
        affected = rng.choice(n_features, size=n_affected, replace=False)
        X[:, affected] += intensity * 2.0 * feature_std[affected]
    elif kind == "noise":
        X += rng.normal(0.0, intensity * feature_std, size=X.shape)
    elif kind == "variance_scale":
        n_affected = max(1, int(0.3 * n_features))
        affected = rng.choice(n_features, size=n_affected, replace=False)
        means = X[:, affected].mean(axis=0)
        X[:, affected] = means + (X[:, affected] - means) * (1.0 + intensity * 3.0)
    elif kind == "sensor_dropout":
        n_affected = max(1, int(0.1 * n_features))
        affected = rng.choice(n_features, size=n_affected, replace=False)
        X[:, affected] = 0.0

    return X
