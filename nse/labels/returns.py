"""
Label creation for the NSE ranking model.

Primary label
-------------
label_3d_net_alpha = future_3d_return - benchmark_3d_return - estimated_costs

Where:
  future_3d_return   = close[t+3] / close[t] - 1  (close-to-close)
  benchmark_return   = NSE 20 index return over same period
  estimated_costs    = round-trip cost in bps (base scenario) converted to fraction

Leakage controls
----------------
- Labels are computed from future prices; they are ONLY attached to the training
  slice and must NEVER be used for the signal that drives live inference.
- The pipeline sets dq_labels_ok = True only once future prices are available.
- Forward-filling or look-ahead in corporate actions are explicitly blocked.
"""

from __future__ import annotations

import logging
from datetime import date

import numpy as np
import pandas as pd
from sqlalchemy.dialects.postgresql import insert

from nse.config import COSTS
from nse.db.schema import FeaturesDaily, session_scope
from nse.ingest.market_live import load_market
from nse.ingest.prices_live import load_prices

log = logging.getLogger(__name__)

# Cost fraction for label calculation (base scenario, round-trip)
BASE_COST_FRACTION = COSTS["scenario_base_bps"] / 10_000


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_and_store_labels(
    start: date,
    end: date,
    tickers: list[str] | None = None,
) -> pd.DataFrame:
    """
    Compute net alpha labels for [start, end] and upsert into features_daily.

    Only rows where future prices exist (i.e., end + 5 trading days have data)
    will receive valid labels.

    Returns the label DataFrame.
    """
    # Need extra future data to form labels
    from datetime import timedelta

    extended_end = end + timedelta(days=15)  # buffer for weekends/holidays

    prices = load_prices(tickers=tickers, start=start, end=extended_end)
    market = load_market(start=start, end=extended_end)

    if prices.empty:
        log.warning("No price data to compute labels")
        return pd.DataFrame()

    label_df = _compute_labels(prices, market, start, end)
    _upsert_labels(label_df)
    log.info("Stored labels for %d rows", len(label_df))
    return label_df


def load_labels(
    start: date | None = None,
    end: date | None = None,
) -> pd.DataFrame:
    """Load label columns from features_daily."""
    from nse.features.pipeline import load_features

    df = load_features(start=start, end=end, tradable_only=False)
    if df.empty:
        return pd.DataFrame()
    label_cols = [c for c in df.columns if c.startswith("label_")]
    return df[["date", "ticker"] + label_cols].copy()


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------

def _compute_labels(
    prices: pd.DataFrame,
    market: pd.DataFrame,
    start: date,
    end: date,
) -> pd.DataFrame:
    """Build label columns for all tickers in [start, end]."""
    prices = prices.copy()
    prices["date"] = pd.to_datetime(prices["date"])
    market = market.copy()
    market["date"] = pd.to_datetime(market["date"])

    # Build NSE 20 benchmark returns per trading day.
    # If nse_20 index data is unavailable, fall back to zero benchmark so that
    # label_Nd_net_alpha = future_return - costs (absolute, not market-adjusted).
    nse20_raw = market[["date", "nse_20"]].dropna().sort_values("date").copy()
    if nse20_raw.empty:
        log.warning("nse_20 index data not available; using zero benchmark for labels")
        all_dates = prices["date"].drop_duplicates().sort_values()
        nse20 = pd.DataFrame({
            "date": all_dates,
            "bench_1d": 0.0,
            "bench_3d": 0.0,
            "bench_5d": 0.0,
        })
    else:
        nse20 = nse20_raw.copy()
        nse20["bench_1d"] = nse20["nse_20"].pct_change(1)
        nse20["bench_3d"] = nse20["nse_20"].pct_change(3)
        nse20["bench_5d"] = nse20["nse_20"].pct_change(5)

    frames = []
    for ticker, grp in prices.groupby("ticker"):
        grp = grp.sort_values("date").reset_index(drop=True)
        grp["fut_ret_1d"] = grp["close"].pct_change(1).shift(-1)
        grp["fut_ret_3d"] = grp["close"].pct_change(3).shift(-3)
        grp["fut_ret_5d"] = grp["close"].pct_change(5).shift(-5)

        grp = grp.merge(
            nse20[["date", "bench_1d", "bench_3d", "bench_5d"]],
            on="date",
            how="left",
        )

        # Future benchmark return: shifted so we compare the same future window
        grp["fut_bench_1d"] = grp["bench_1d"].shift(-1)
        grp["fut_bench_3d"] = grp["bench_3d"].shift(-3)
        grp["fut_bench_5d"] = grp["bench_5d"].shift(-5)

        # Net alpha = future return - benchmark - round-trip cost fraction
        grp["label_1d_net_alpha"] = grp["fut_ret_1d"] - grp["fut_bench_1d"] - BASE_COST_FRACTION
        grp["label_3d_net_alpha"] = grp["fut_ret_3d"] - grp["fut_bench_3d"] - BASE_COST_FRACTION
        grp["label_5d_net_alpha"] = grp["fut_ret_5d"] - grp["fut_bench_5d"] - BASE_COST_FRACTION

        # Binary: was net alpha positive?
        grp["label_binary"] = (grp["label_3d_net_alpha"] > 0).astype(int)

        # Top bucket: top 20% by 3d net alpha on that day (computed cross-sectionally below)
        grp["label_top_bucket"] = 0  # filled after merge

        grp["ticker"] = ticker
        frames.append(
            grp[
                [
                    "date",
                    "ticker",
                    "label_1d_net_alpha",
                    "label_3d_net_alpha",
                    "label_5d_net_alpha",
                    "label_binary",
                    "label_top_bucket",
                ]
            ]
        )

    if not frames:
        return pd.DataFrame()

    all_labels = pd.concat(frames, ignore_index=True)

    # Cross-sectional top-bucket label
    all_labels["label_top_bucket"] = all_labels.groupby("date")[
        "label_3d_net_alpha"
    ].transform(
        lambda s: (s >= s.quantile(0.80)).astype(int)
    )

    # Filter to requested range and drop rows with no future prices
    all_labels = all_labels[
        (all_labels["date"] >= pd.Timestamp(start))
        & (all_labels["date"] <= pd.Timestamp(end))
    ]
    all_labels = all_labels.dropna(subset=["label_3d_net_alpha"])
    return all_labels.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Upsert
# ---------------------------------------------------------------------------

def _upsert_labels(df: pd.DataFrame) -> None:
    """Write label columns into existing features_daily rows."""
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.date
    df["dq_labels_ok"] = True

    with session_scope() as s:
        for _, row in df.iterrows():
            s.query(FeaturesDaily).filter(
                FeaturesDaily.date == row["date"],
                FeaturesDaily.ticker == row["ticker"],
            ).update(
                {
                    "label_1d_net_alpha": row.get("label_1d_net_alpha"),
                    "label_3d_net_alpha": row.get("label_3d_net_alpha"),
                    "label_5d_net_alpha": row.get("label_5d_net_alpha"),
                    "label_binary": row.get("label_binary"),
                    "label_top_bucket": row.get("label_top_bucket"),
                    "dq_labels_ok": True,
                }
            )
