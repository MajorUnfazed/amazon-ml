"""Probability calibration module for LinkSure."""

import numpy as np
from sklearn.isotonic import IsotonicRegression
from typing import Optional

class ProbabilityCalibrator:
    """
    Fits Isotonic Regression on out-of-fold model predictions
    to produce honest posterior probabilities P(match).
    """
    def __init__(self):
        self.calibrator = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        self.is_fitted = False

    def fit(self, raw_probs: np.ndarray, y_true: np.ndarray):
        """Fits isotonic regression on probability scores."""
        raw_probs = np.asarray(raw_probs, dtype=float).clip(0.0, 1.0)
        y_true = np.asarray(y_true, dtype=float)
        self.calibrator.fit(raw_probs, y_true)
        self.is_fitted = True

    def transform(self, raw_probs: np.ndarray) -> np.ndarray:
        """Transforms raw probabilities to calibrated probabilities."""
        if not self.is_fitted:
            return np.asarray(raw_probs, dtype=float).clip(0.0, 1.0)
        raw_probs = np.asarray(raw_probs, dtype=float).clip(0.0, 1.0)
        return self.calibrator.predict(raw_probs).clip(0.0, 1.0)
