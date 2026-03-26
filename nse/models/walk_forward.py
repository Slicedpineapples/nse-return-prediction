"""
Walk-forward cross-validation for the NSE ranker.

Implements strict temporal separation:
  - No future data leaks into any training window.
  - Preprocessing (scaling) is fit only on each training slice.
  - Validation metrics are computed on out-of-sample (OOS) periods.

Schedule (from config)
----------------------
  train: 2019-2022
  validate: 2023
  test: 2024
  live shadow: 2025+

Monthly expanding-window retraining:
  Each month M, train on [train_start, M-1], validate on [M-3, M-1],
  score on month M.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date
from typing import Iterator

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from nse.config import MODEL
from nse.features.pipeline import load_features
from nse.models.ranker import NSERanker

log = logging.getLogger(__name__)


@dataclass
class WalkForwardSlice:
    train_start: date
    train_end: date
    val_start: date
    val_end: date
    test_start: date
    test_end: date


@dataclass
class WalkForwardResult:
    slice_id: str
    val_ic: float          # Spearman IC on validation
    test_ic: float         # Spearman IC on test
    val_hit_rate: float    # fraction of days top-1 name was top-bucket
    test_hit_rate: float
    n_train: int
    n_val: int
    n_test: int
    feature_importance: pd.DataFrame = field(default_factory=pd.DataFrame)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_walk_forward(
    features: pd.DataFrame | None = None,
    n_months: int = 12,
) -> list[WalkForwardResult]:
    """
    Run walk-forward training and evaluation.

    Parameters
    ----------
    features : pre-loaded features DataFrame. If None, loads from DB.
    n_months : number of monthly steps to evaluate.

    Returns
    -------
    List of WalkForwardResult, one per monthly window.
    """
    if features is None:
        features = load_features(
            start=_parse_date(MODEL["train_start"]),
            end=date.today(),
            tradable_only=False,
        )

    if features.empty:
        log.warning("No features available for walk-forward")
        return []

    results = []
    for slc in _generate_slices(features, n_months):
        log.info(
            "Walk-forward slice: train=%s–%s val=%s–%s test=%s–%s",
            slc.train_start, slc.train_end,
            slc.val_start, slc.val_end,
            slc.test_start, slc.test_end,
        )
        result = _run_one_slice(features, slc)
        results.append(result)
        log.info(
            "  val_IC=%.3f  test_IC=%.3f  val_hit=%.1f%%  test_hit=%.1f%%",
            result.val_ic, result.test_ic,
            result.val_hit_rate * 100, result.test_hit_rate * 100,
        )

    return results


def train_production_model(
    features: pd.DataFrame | None = None,
    recency_half_life_days: int | None = None,
    champion_eval_days: int = 60,
) -> NSERanker:
    """
    Train the production model on all data up to validate_start.
    Uses validate period as early-stopping set.

    Champion/challenger: the new (challenger) model is evaluated against the
    current incumbent (champion) on the last `champion_eval_days` of labelled
    data.  The challenger is deployed only if its IC exceeds the champion's.

    Parameters
    ----------
    recency_half_life_days : passed to NSERanker.fit() for exponential
        sample weighting.  None = uniform weights.
    champion_eval_days     : number of days of recent labelled data to use
        for champion vs challenger comparison.

    Returns fitted NSERanker (champion or challenger), saved to disk as
    'production'.  If a challenger wins, the previous champion is archived as
    'production_prev'.
    """
    if features is None:
        features = load_features(tradable_only=False)

    validate_start = _parse_date(MODEL["validate_start"])
    test_start = _parse_date(MODEL["test_start"])
    label = MODEL["primary_label"]

    train_df = features[features["date"].dt.date < validate_start].copy()
    val_df = features[
        (features["date"].dt.date >= validate_start)
        & (features["date"].dt.date < test_start)
    ].copy()

    # Challenger: new model trained on all available data
    challenger = NSERanker(run_id="challenger")
    challenger.fit(train_df, val_df, recency_half_life_days=recency_half_life_days)

    # Evaluation window: most recent labelled data
    eval_df = features.dropna(subset=[label]).copy()
    if not eval_df.empty:
        eval_cutoff = eval_df["date"].max() - pd.Timedelta(days=champion_eval_days)
        eval_df = eval_df[eval_df["date"] >= eval_cutoff]

    challenger_ic, _ = _eval_ic(challenger, eval_df, label)

    # Try loading existing champion
    from nse.config import MODELS_DIR
    champion_path = MODELS_DIR / "ranker_production.pkl"
    if champion_path.exists() and not eval_df.empty:
        try:
            champion = NSERanker.load("production")
            champion_ic, _ = _eval_ic(champion, eval_df, label)
            log.info(
                "Champion/challenger: champion_IC=%.4f  challenger_IC=%.4f (eval_days=%d)",
                champion_ic, challenger_ic, champion_eval_days,
            )
            if challenger_ic > champion_ic:
                log.info("Challenger wins — deploying as new production model")
                import shutil
                shutil.copy(champion_path, MODELS_DIR / "ranker_production_prev.pkl")
            else:
                log.info("Champion holds — keeping existing production model")
                return champion
        except Exception as exc:
            log.warning("Could not load/eval champion: %s — deploying challenger", exc)
    else:
        log.info("No incumbent champion found — deploying challenger directly")

    challenger.run_id = "production"
    challenger.save("production")
    return challenger


# ---------------------------------------------------------------------------
# Slice generation
# ---------------------------------------------------------------------------

def _generate_slices(
    features: pd.DataFrame, n_months: int
) -> Iterator[WalkForwardSlice]:
    """Generate monthly walk-forward slices."""
    validate_start = _parse_date(MODEL["validate_start"])
    test_start = _parse_date(MODEL["test_start"])
    train_start = _parse_date(MODEL["train_start"])

    # Generate monthly test periods starting from test_start
    from dateutil.relativedelta import relativedelta

    current = test_start
    for _ in range(n_months):
        test_end_dt = current + relativedelta(months=1) - relativedelta(days=1)
        if features["date"].max().date() < current:
            break

        # Validation: last 3 months before test
        val_end = current - relativedelta(days=1)
        val_start = current - relativedelta(months=3)

        if MODEL["window_type"] == "expanding":
            t_start = train_start
        else:
            t_start = current - relativedelta(years=MODEL["rolling_window_years"])

        yield WalkForwardSlice(
            train_start=t_start,
            train_end=val_start - relativedelta(days=1),
            val_start=val_start,
            val_end=val_end,
            test_start=current,
            test_end=min(test_end_dt, features["date"].max().date()),
        )
        current += relativedelta(months=1)


# ---------------------------------------------------------------------------
# One slice
# ---------------------------------------------------------------------------

def _run_one_slice(
    features: pd.DataFrame, slc: WalkForwardSlice
) -> WalkForwardResult:
    train_df = _slice_df(features, slc.train_start, slc.train_end)
    val_df = _slice_df(features, slc.val_start, slc.val_end)
    test_df = _slice_df(features, slc.test_start, slc.test_end)

    # Drop rows without labels
    label = MODEL["primary_label"]
    train_df = train_df.dropna(subset=[label])
    val_df = val_df.dropna(subset=[label])
    test_df_with_labels = test_df.dropna(subset=[label])

    if train_df.empty or len(train_df) < 50:
        log.warning("Insufficient training data for slice %s", slc.train_start)
        return _empty_result(slc)

    slice_id = f"{slc.test_start.strftime('%Y-%m')}"
    ranker = NSERanker(run_id=slice_id)
    try:
        ranker.fit(train_df, val_df if not val_df.empty else None)
    except Exception as exc:
        log.error("Fit failed for slice %s: %s", slice_id, exc)
        return _empty_result(slc)

    ranker.save(slice_id)

    val_ic, val_hit = _eval_ic(ranker, val_df.dropna(subset=[label]), label)
    test_ic, test_hit = _eval_ic(ranker, test_df_with_labels, label)

    return WalkForwardResult(
        slice_id=slice_id,
        val_ic=val_ic,
        test_ic=test_ic,
        val_hit_rate=val_hit,
        test_hit_rate=test_hit,
        n_train=len(train_df),
        n_val=len(val_df),
        n_test=len(test_df),
        feature_importance=ranker.feature_importance(),
    )


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _eval_ic(
    ranker: NSERanker, df: pd.DataFrame, label: str
) -> tuple[float, float]:
    """Return (Spearman IC, hit rate) for the given DataFrame."""
    if df.empty:
        return np.nan, np.nan

    scores = ranker.predict(df)
    df = df.merge(scores[["date", "ticker", "score"]], on=["date", "ticker"], how="inner")
    if df.empty:
        return np.nan, np.nan

    # Average cross-sectional Spearman IC per day
    ics = []
    for _, grp in df.groupby("date"):
        if len(grp) < 3:
            continue
        ic, _ = spearmanr(grp["score"], grp[label])
        ics.append(ic)
    mean_ic = float(np.nanmean(ics)) if ics else np.nan

    # Hit rate: fraction of days where highest-scored stock had positive alpha
    hit = []
    for _, grp in df.groupby("date"):
        best = grp.sort_values("score", ascending=False).iloc[0]
        hit.append(int(best[label] > 0))
    hit_rate = float(np.mean(hit)) if hit else np.nan

    return mean_ic, hit_rate


def _slice_df(df: pd.DataFrame, start: date, end: date) -> pd.DataFrame:
    return df[
        (df["date"].dt.date >= start) & (df["date"].dt.date <= end)
    ].copy()


def _empty_result(slc: WalkForwardSlice) -> WalkForwardResult:
    return WalkForwardResult(
        slice_id=slc.test_start.strftime("%Y-%m"),
        val_ic=np.nan, test_ic=np.nan,
        val_hit_rate=np.nan, test_hit_rate=np.nan,
        n_train=0, n_val=0, n_test=0,
    )


def _parse_date(s: str) -> date:
    from datetime import datetime

    return datetime.strptime(s, "%Y-%m-%d").date()
