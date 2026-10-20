"""Per-feature Kolmogorov-Smirnov drift detection, using ml3-drift's
KSAlgorithm with a Bonferroni-corrected p-value per feature (testing 180
features independently at one threshold would flag some as "drifted" by
chance alone) - keeps each feature's result so the dashboard can show
which ones actually drifted."""

from dataclasses import dataclass, field

import numpy as np
from ml3_drift.monitoring.algorithms.batch.ks import KSAlgorithm


@dataclass
class DriftReport:
    n_features: int
    n_drifted: int
    drifted_features: list[str]
    corrected_p_value: float

    @property
    def fraction_drifted(self) -> float:
        return self.n_drifted / self.n_features if self.n_features else 0.0


@dataclass
class FeatureDriftWatchdog:
    feature_names: list[str]
    base_p_value: float = 0.01
    _algorithms: list = field(default_factory=list, init=False, repr=False)

    @property
    def corrected_p_value(self) -> float:
        return self.base_p_value / len(self.feature_names)

    def fit(self, X_ref: np.ndarray) -> "FeatureDriftWatchdog":
        p = self.corrected_p_value
        self._algorithms = []
        for i in range(X_ref.shape[1]):
            algo = KSAlgorithm(p_value=p)
            algo.fit(X_ref[:, i : i + 1])
            self._algorithms.append(algo)
        return self

    def detect(self, X_new: np.ndarray) -> DriftReport:
        if not self._algorithms:
            raise RuntimeError("Call .fit(X_ref) before .detect(X_new).")
        drifted = []
        for i, algo in enumerate(self._algorithms):
            output = algo.detect(X_new[:, i : i + 1])[0]
            if output.drift_detected:
                drifted.append(self.feature_names[i])
        return DriftReport(
            n_features=len(self.feature_names),
            n_drifted=len(drifted),
            drifted_features=drifted,
            corrected_p_value=self.corrected_p_value,
        )
