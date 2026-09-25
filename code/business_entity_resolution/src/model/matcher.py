"""LightGBM pair matching model for LinkSure."""

import warnings
warnings.filterwarnings("ignore")
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from typing import Dict, List, Tuple, Optional, Any
from .calibrate import ProbabilityCalibrator

class EntityResolutionMatcher:
    """
    LightGBM tabular classifier for candidate pair resolution.
    Includes GroupKFold out-of-fold training, feature importance reporting,
    and isotonic probability calibration.
    """
    def __init__(self, lgb_params: Optional[Dict[str, Any]] = None):
        self.params = lgb_params or {
            "objective": "binary",
            "metric": "binary_logloss",
            "boosting_type": "gbdt",
            "learning_rate": 0.05,
            "num_leaves": 31,
            "max_depth": -1,
            "min_child_samples": 20,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "n_estimators": 400,
            "random_state": 42,
            "n_jobs": -1,
            "verbose": -1
        }
        self.model = lgb.LGBMClassifier(**self.params)
        self.calibrator = ProbabilityCalibrator()
        self.feature_names: List[str] = []
        self.is_fitted = False

    def fit(self, X: pd.DataFrame, y: np.ndarray):
        """Fits final model on full training dataset."""
        self.feature_names = list(X.columns)
        self.model.fit(X, y)
        self.is_fitted = True

    def predict_proba(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns (p_raw, p_calibrated).
        """
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted.")

        X_in = X[self.feature_names]
        p_raw = self.model.predict_proba(X_in)[:, 1]
        p_cal = self.calibrator.transform(p_raw)
        return p_raw, p_cal

    def train_cv(
        self,
        features_df: pd.DataFrame,
        feature_cols: List[str],
        label_col: str = "label",
        n_splits: int = 5
    ) -> Tuple[pd.DataFrame, Dict[str, float]]:
        """
        Performs 5-fold GroupKFold on source1_entity_id.
        Produces out-of-fold calibrated predictions and fits calibrator.
        """
        self.feature_names = feature_cols
        groups = features_df["source1_entity_id"].values
        X = features_df[feature_cols].copy()
        y = features_df[label_col].values.astype(int)

        # Adapt min_child_samples if training set is small
        if len(features_df) < 100:
            self.params["min_child_samples"] = max(2, len(features_df) // (n_splits * 2))

        # Adjust n_splits if unique groups < n_splits
        unique_groups = len(np.unique(groups))
        actual_splits = min(n_splits, unique_groups)

        gkf = GroupKFold(n_splits=actual_splits)
        oof_raw = np.zeros(len(features_df), dtype=float)

        importances = np.zeros(len(feature_cols), dtype=float)

        callbacks = [lgb.early_stopping(stopping_rounds=30, verbose=False)] if len(features_df) >= 100 else None

        for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups=groups), 1):
            X_train, y_train = X.iloc[train_idx], y[train_idx]
            X_val, y_val = X.iloc[val_idx], y[val_idx]

            fold_model = lgb.LGBMClassifier(**self.params)
            if callbacks:
                fold_model.fit(
                    X_train, y_train,
                    eval_set=[(X_val, y_val)],
                    callbacks=callbacks
                )
            else:
                fold_model.fit(X_train, y_train)

            val_preds = fold_model.predict_proba(X_val)[:, 1]
            oof_raw[val_idx] = val_preds
            importances += fold_model.feature_importances_ / actual_splits

        # Fit calibrator on complete OOF predictions
        self.calibrator.fit(oof_raw, y)

        # Fit final model on all data
        self.model = lgb.LGBMClassifier(**self.params)
        self.fit(X, y)

        oof_cal = self.calibrator.transform(oof_raw)

        oof_df = features_df[["source1_entity_id", "partner_entity_id", label_col]].copy()
        oof_df["p_raw"] = oof_raw
        oof_df["p_cal"] = oof_cal

        feat_imp_dict = dict(zip(feature_cols, importances.tolist()))
        feat_imp_sorted = dict(sorted(feat_imp_dict.items(), key=lambda item: item[1], reverse=True))

        return oof_df, feat_imp_sorted
