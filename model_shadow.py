"""Loads the trained classifier and its feature table, split into a
reference set and a holdout set that drift gets injected into.

Note: the model was originally trained on the full table, so the
undrifted baseline reflects train-set performance, not generalization -
fine for tracking relative decay under drift, not a benchmark claim."""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

MODEL_PATH = Path(__file__).parent / "model_snapshot" / "emotion_classifier.joblib"
DATA_PATH = Path(__file__).parent / "model_snapshot" / "Audio_db.pkl"


def load_model():
    return joblib.load(MODEL_PATH)


def load_reference_and_holdout(test_size: float = 0.3, seed: int = 42):
    df = pd.read_pickle(DATA_PATH)
    feature_cols = [c for c in df.columns if c != "target"]
    X = df[feature_cols].values
    y = df["target"].values
    X_ref, X_hold, y_ref, y_hold = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=seed
    )
    return (X_ref, y_ref), (X_hold, y_hold), feature_cols


def score_model(model: Pipeline, X: np.ndarray, y, feature_cols: list[str]) -> float:
    """model.score() with column names attached, so sklearn doesn't warn
    that a plain ndarray doesn't match the names the scaler was fit with."""
    return float(model.score(pd.DataFrame(X, columns=feature_cols), y))
