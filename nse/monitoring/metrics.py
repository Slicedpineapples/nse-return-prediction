"""
Live IC monitoring and performance metrics.

Spearman IC is the primary signal-quality metric: measures rank correlation
between model scores and actual 3-day net alphas.
"""

from __future__ import annotations

import logging
from datetime import date, timedelta

import numpy as np
import pandas as pd
from scipy import stats

from nse.monitoring.ledger import get_ledger

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_rolling_ic(
    ledger: pd.DataFrame | None = None,
    window: int = 20,
    start: date | None = None,
    end: date | None = None,
) -> pd.DataFrame:
    """
    Compute rolling Spearman IC between score and actual_net_alpha_3d.

    Returns a DataFrame indexed by signal_date with columns:
        ic, ic_pvalue, n_obs
    """
    if ledger is None:
        ledger = get_ledger(start=start, end=end)

    df = ledger.dropna(subset=["score", "actual_net_alpha_3d"]).copy()
    if df.empty:
        return pd.DataFrame(columns=["signal_date", "ic", "ic_pvalue", "n_obs"])

    dates = sorted(df["signal_date"].unique())
    records = []

    for i, d in enumerate(dates):
        window_dates = dates[max(0, i - window + 1): i + 1]
        sub = df[df["signal_date"].isin(window_dates)]
        if len(sub) < 5:
            continue
        ic, pval = stats.spearmanr(sub["score"], sub["actual_net_alpha_3d"])
        records.append({"signal_date": d, "ic": ic, "ic_pvalue": pval, "n_obs": len(sub)})

    return pd.DataFrame(records)


def compute_hit_rate(
    ledger: pd.DataFrame | None = None,
    window: int = 20,
    start: date | None = None,
    end: date | None = None,
) -> pd.DataFrame:
    """
    Rolling hit rate: fraction of approved signals with positive actual_net_alpha_3d.
    """
    if ledger is None:
        ledger = get_ledger(start=start, end=end, approved_only=True)

    df = ledger.dropna(subset=["actual_net_alpha_3d"]).copy()
    if df.empty:
        return pd.DataFrame(columns=["signal_date", "hit_rate", "n_obs"])

    df["hit"] = (df["actual_net_alpha_3d"] > 0).astype(int)
    dates = sorted(df["signal_date"].unique())
    records = []

    for i, d in enumerate(dates):
        window_dates = dates[max(0, i - window + 1): i + 1]
        sub = df[df["signal_date"].isin(window_dates)]
        if len(sub) < 3:
            continue
        records.append({
            "signal_date": d,
            "hit_rate": sub["hit"].mean(),
            "n_obs": len(sub),
        })

    return pd.DataFrame(records)


def compute_score_calibration(
    ledger: pd.DataFrame | None = None,
    n_buckets: int = 5,
    start: date | None = None,
    end: date | None = None,
) -> pd.DataFrame:
    """
    Bucket scores into quintiles and show mean actual alpha per bucket.
    Useful for checking monotonicity of model scores.
    """
    if ledger is None:
        ledger = get_ledger(start=start, end=end)

    df = ledger.dropna(subset=["score", "actual_net_alpha_3d"]).copy()
    if len(df) < n_buckets * 3:
        return pd.DataFrame()

    df["bucket"] = pd.qcut(df["score"], n_buckets, labels=False, duplicates="drop")
    summary = (
        df.groupby("bucket")
        .agg(
            mean_score=("score", "mean"),
            mean_alpha=("actual_net_alpha_3d", "mean"),
            n_obs=("actual_net_alpha_3d", "count"),
        )
        .reset_index()
    )
    return summary


def get_summary_stats(
    start: date | None = None,
    end: date | None = None,
    window: int = 20,
) -> dict:
    """
    One-call summary: overall IC, hit rate, and calibration table.
    Also checks whether recent IC has degraded vs historical baseline.
    """
    end = end or date.today()
    start = start or (end - timedelta(days=365))

    ledger = get_ledger(start=start, end=end)
    if ledger.empty:
        return {"status": "no_data", "message": "No signal history found for the period."}

    filled = ledger.dropna(subset=["actual_net_alpha_3d"])
    n_filled = len(filled)
    n_total = len(ledger)

    if n_filled < 5:
        return {
            "status": "insufficient_data",
            "n_signals": n_total,
            "n_with_outcomes": n_filled,
        }

    ic_overall, pval = stats.spearmanr(filled["score"], filled["actual_net_alpha_3d"])

    approved = filled[filled["approved"] == True]
    hit_rate = (approved["actual_net_alpha_3d"] > 0).mean() if len(approved) > 0 else None

    # Recent IC (last window days)
    recent_cutoff = end - timedelta(days=window * 2)
    recent = filled[filled["signal_date"] >= pd.Timestamp(recent_cutoff)]
    if len(recent) >= 5:
        ic_recent, _ = stats.spearmanr(recent["score"], recent["actual_net_alpha_3d"])
    else:
        ic_recent = None

    rolling_ic = compute_rolling_ic(ledger=filled, window=window)
    degraded = False
    if ic_recent is not None and ic_overall > 0.05:
        degraded = ic_recent < ic_overall * 0.5  # recent IC < 50% of historical

    mean_alpha_bps = filled["actual_net_alpha_3d"].mean() * 10_000
    mean_pred_error_bps = (
        filled["prediction_error"].dropna().mean() * 10_000
        if "prediction_error" in filled.columns
        else None
    )

    return {
        "status": "ok",
        "period": {"start": str(start), "end": str(end)},
        "n_signals": n_total,
        "n_with_outcomes": n_filled,
        "ic_overall": round(ic_overall, 4),
        "ic_pvalue": round(pval, 4),
        "ic_recent": round(ic_recent, 4) if ic_recent is not None else None,
        "hit_rate": round(hit_rate, 4) if hit_rate is not None else None,
        "mean_alpha_bps": round(mean_alpha_bps, 2),
        "mean_pred_error_bps": round(mean_pred_error_bps, 2) if mean_pred_error_bps is not None else None,
        "signal_degraded": degraded,
        "rolling_ic": rolling_ic,
        "calibration": compute_score_calibration(ledger=filled),
    }
