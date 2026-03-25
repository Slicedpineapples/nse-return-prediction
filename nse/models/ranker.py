"""
LightGBM-based daily alpha ranker.

Model design
------------
Primary task: regression on label_3d_net_alpha (cross-sectional net alpha).
Secondary task: classification on label_top_bucket.

Features used: all numeric columns in features_daily excluding label_* and dq_*.

The ranker outputs a score per (date, ticker); higher is better.
Scores are then normalised to [0, 1] per day for use in portfolio construction.
"""

from __future__ import annotations

import json
import logging
import pickle
from datetime import date
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from nse.config import MODEL, MODELS_DIR

log = logging.getLogger(__name__)

# Feature columns: all numeric non-label, non-dq columns
_EXCLUDE_PREFIXES = ("label_", "dq_", "id", "date", "ticker")

# LightGBM hyperparameters
_LGB_PARAMS: dict[str, Any] = {
    "objective": "regression",
    "metric": "rmse",
    "learning_rate": 0.05,
    "num_leaves": 31,
    "min_child_samples": 10,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "lambda_l1": 0.1,
    "lambda_l2": 0.1,
    "n_estimators": 500,
    "early_stopping_rounds": 50,
    "verbose": -1,
    "n_jobs": -1,
    "seed": 42,
}

_LGB_CLASSIFIER_PARAMS: dict[str, Any] = {
    **_LGB_PARAMS,
    "objective": "binary",
    "metric": "binary_logloss",
}


class NSERanker:
    """
    Wraps LightGBM regressor (net alpha) + optional classifier (top-bucket).

    Usage
    -----
        ranker = NSERanker()
        ranker.fit(train_df, val_df)
        scores = ranker.predict(today_df)
        ranker.save("2024-01")
        ranker = NSERanker.load("2024-01")
    """

    def __init__(self, run_id: str = "default") -> None:
        self.run_id = run_id
        self.regressor: lgb.LGBMRegressor | None = None
        self.classifier: lgb.LGBMClassifier | None = None
        self.scaler = StandardScaler()
        self.feature_cols: list[str] = []
        self.is_fitted = False

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame | None = None,
        primary_label: str = MODEL["primary_label"],
    ) -> "NSERanker":
        """
        Train on train_df; use val_df for early stopping.

        Both DataFrames must have been through run_feature_pipeline and
        compute_and_store_labels.
        """
        feature_cols = _get_feature_cols(train_df)
        self.feature_cols = feature_cols

        X_train, y_train = _prepare_xy(train_df, feature_cols, primary_label)
        if X_train.empty or y_train.isna().all():
            raise ValueError("Insufficient training data after filtering")

        # Fit scaler
        X_train_scaled = pd.DataFrame(
            self.scaler.fit_transform(X_train),
            columns=feature_cols,
            index=X_train.index,
        )

        eval_set = None
        if val_df is not None and not val_df.empty:
            X_val, y_val = _prepare_xy(val_df, feature_cols, primary_label)
            X_val_scaled = pd.DataFrame(
                self.scaler.transform(X_val),
                columns=feature_cols,
                index=X_val.index,
            )
            eval_set = [(X_val_scaled, y_val)]

        # Regressor
        log.info("Fitting LightGBM regressor on %d rows", len(X_train))
        self.regressor = lgb.LGBMRegressor(**_LGB_PARAMS)
        fit_kwargs: dict[str, Any] = {}
        if eval_set:
            fit_kwargs["eval_set"] = eval_set
        self.regressor.fit(X_train_scaled, y_train, **fit_kwargs)

        # Optional classifier
        if "label_top_bucket" in train_df.columns:
            y_cls = train_df.loc[X_train.index, "label_top_bucket"].fillna(0)
            log.info("Fitting LightGBM classifier on %d rows", len(X_train))
            self.classifier = lgb.LGBMClassifier(**_LGB_CLASSIFIER_PARAMS)
            cls_kwargs: dict[str, Any] = {}
            if eval_set and val_df is not None:
                y_val_cls = val_df.loc[X_val.index, "label_top_bucket"].fillna(0)
                cls_kwargs["eval_set"] = [(X_val_scaled, y_val_cls)]
            self.classifier.fit(X_train_scaled, y_cls, **cls_kwargs)

        self.is_fitted = True
        log.info("Model fitted. Run ID: %s", self.run_id)
        return self

    # ------------------------------------------------------------------
    # Predict
    # ------------------------------------------------------------------

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Score rows in df.

        Returns a DataFrame with columns: date, ticker, score, top_bucket_prob.
        Score is the predicted net alpha (regressor output), normalised
        cross-sectionally per day.
        """
        if not self.is_fitted:
            raise RuntimeError("Call .fit() before .predict()")

        X, _ = _prepare_xy(df, self.feature_cols, label_col=None)
        if X.empty:
            return pd.DataFrame(columns=["date", "ticker", "score", "top_bucket_prob"])

        X_scaled = pd.DataFrame(
            self.scaler.transform(X),
            columns=self.feature_cols,
            index=X.index,
        )

        scores = self.regressor.predict(X_scaled)
        result = df.loc[X.index, ["date", "ticker"]].copy()
        result["raw_score"] = scores

        # Normalise per day to [0, 1]
        result["score"] = result.groupby("date")["raw_score"].transform(
            lambda s: (s - s.min()) / (s.max() - s.min() + 1e-10)
        )

        if self.classifier is not None:
            result["top_bucket_prob"] = self.classifier.predict_proba(X_scaled)[:, 1]
        else:
            result["top_bucket_prob"] = np.nan

        return result[["date", "ticker", "score", "top_bucket_prob"]].reset_index(drop=True)

    # ------------------------------------------------------------------
    # Feature importance
    # ------------------------------------------------------------------

    def feature_importance(self) -> pd.DataFrame:
        if not self.is_fitted or self.regressor is None:
            raise RuntimeError("Model not fitted")
        importance = self.regressor.feature_importances_
        return (
            pd.DataFrame({"feature": self.feature_cols, "importance": importance})
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, label: str | None = None) -> Path:
        label = label or self.run_id
        path = MODELS_DIR / f"ranker_{label}.pkl"
        with open(path, "wb") as f:
            pickle.dump(self, f)
        log.info("Model saved to %s", path)
        return path

    @classmethod
    def load(cls, label: str) -> "NSERanker":
        path = MODELS_DIR / f"ranker_{label}.pkl"
        with open(path, "rb") as f:
            model = pickle.load(f)
        log.info("Model loaded from %s", path)
        return model


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_feature_cols(df: pd.DataFrame) -> list[str]:
    """Return list of usable numeric feature columns."""
    excluded = set()
    for col in df.columns:
        if any(col.startswith(pfx) for pfx in _EXCLUDE_PREFIXES):
            excluded.add(col)
    return [c for c in df.columns if c not in excluded and pd.api.types.is_numeric_dtype(df[c])]


def _prepare_xy(
    df: pd.DataFrame,
    feature_cols: list[str],
    label_col: str | None = "label_3d_net_alpha",
) -> tuple[pd.DataFrame, pd.Series]:
    """Filter rows, fill NaNs, return X and y."""
    df = df.copy()

    # Drop rows with no valid label if label requested
    if label_col and label_col in df.columns:
        df = df.dropna(subset=[label_col])
        y = df[label_col]
    else:
        y = pd.Series(dtype=float)

    # Only keep feature cols that exist
    available = [c for c in feature_cols if c in df.columns]
    X = df[available].copy()

    # Fill NaN with column median (robust, avoids leakage since median is per-column)
    X = X.fillna(X.median())

    return X, y
