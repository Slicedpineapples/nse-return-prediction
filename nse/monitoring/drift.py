"""
Feature drift detection using Population Stability Index (PSI).

PSI measures how much a feature distribution has shifted between a
reference period (training window) and a recent production window.

PSI interpretation
------------------
< 0.10  : no significant change
0.10–0.25 : moderate change, monitor
> 0.25  : significant shift — consider retraining
"""

from __future__ import annotations

import logging
from datetime import date, timedelta

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# Features to monitor for drift
MONITORED_FEATURES = [
    "ret_20d",
    "vol_20d",
    "kes_usd_ret_1d",
    "price_band_proximity_pct",
    "volume_zscore_20d",
    "amihud_20d",
]

PSI_WARN = 0.10
PSI_ALERT = 0.25


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_psi(
    reference: pd.Series,
    current: pd.Series,
    n_bins: int = 10,
) -> float:
    """
    Compute Population Stability Index between two distributions.

    Parameters
    ----------
    reference : feature values from the reference period (training)
    current   : feature values from the recent production period
    n_bins    : number of equal-frequency bins (based on reference quantiles)

    Returns
    -------
    PSI score (float, non-negative)
    """
    reference = reference.dropna()
    current = current.dropna()

    if len(reference) < 20 or len(current) < 5:
        return np.nan

    # Build bins from reference quantiles
    quantiles = np.linspace(0, 100, n_bins + 1)
    breakpoints = np.percentile(reference, quantiles)
    breakpoints[0] = -np.inf
    breakpoints[-1] = np.inf

    ref_counts = np.histogram(reference, bins=breakpoints)[0]
    cur_counts = np.histogram(current, bins=breakpoints)[0]

    ref_pct = ref_counts / len(reference)
    cur_pct = cur_counts / len(current)

    # Avoid log(0): clip to small value
    ref_pct = np.clip(ref_pct, 1e-4, None)
    cur_pct = np.clip(cur_pct, 1e-4, None)

    psi = np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct))
    return float(psi)


def check_feature_drift(
    features: pd.DataFrame | None = None,
    reference_start: date | None = None,
    reference_end: date | None = None,
    current_start: date | None = None,
    current_end: date | None = None,
    monitored: list[str] | None = None,
) -> pd.DataFrame:
    """
    Compute PSI for each monitored feature between a reference window
    and a recent production window.

    If features is None, loads from the DB.

    Default windows:
      reference : 180–30 days ago
      current   : last 30 days

    Returns a DataFrame with columns: feature, psi, status
    """
    if features is None:
        from nse.features.pipeline import load_features
        today = date.today()
        ref_end = reference_end or (today - timedelta(days=30))
        ref_start = reference_start or (today - timedelta(days=210))
        cur_end = current_end or today
        cur_start = current_start or (today - timedelta(days=30))
        features = load_features(start=ref_start, end=cur_end, tradable_only=False)
    else:
        today = date.today()
        ref_end = reference_end or (today - timedelta(days=30))
        ref_start = reference_start or features["date"].min().date()
        cur_end = current_end or today
        cur_start = current_start or (today - timedelta(days=30))

    if features.empty:
        return pd.DataFrame(columns=["feature", "psi", "status"])

    features["date"] = pd.to_datetime(features["date"])
    ref_mask = (features["date"].dt.date >= ref_start) & (features["date"].dt.date <= ref_end)
    cur_mask = (features["date"].dt.date >= cur_start) & (features["date"].dt.date <= cur_end)

    ref_df = features[ref_mask]
    cur_df = features[cur_mask]

    cols = monitored or MONITORED_FEATURES
    records = []
    for col in cols:
        if col not in features.columns:
            continue
        psi = compute_psi(ref_df[col], cur_df[col])
        if np.isnan(psi):
            status = "insufficient_data"
        elif psi < PSI_WARN:
            status = "stable"
        elif psi < PSI_ALERT:
            status = "warning"
        else:
            status = "alert"
        records.append({"feature": col, "psi": round(psi, 4) if not np.isnan(psi) else None, "status": status})

    result = pd.DataFrame(records)
    if not result.empty:
        result = result.sort_values("psi", ascending=False, na_position="last")
    return result.reset_index(drop=True)
