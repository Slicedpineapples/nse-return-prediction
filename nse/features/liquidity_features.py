"""
Liquidity and microstructure features.

These are particularly important for NSE because:
- Some names have sporadic trading and stale prices.
- Price bands (daily price movement limits) can cause clustering near the band.
- Single-unit trading (enabled 2025) changes microstructure for small orders.
- Amihud illiquidity captures price impact relative to volume.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from nse.config import LIQUIDITY


def compute_liquidity_features(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Compute liquidity and microstructure features for all tickers.

    Parameters
    ----------
    prices : tidy long DataFrame with columns:
             date, ticker, close, volume, value_traded, number_of_deals,
             high, low, open, previous_close

    Returns
    -------
    Wide DataFrame keyed on (date, ticker).
    """
    prices = prices.copy()
    prices["date"] = pd.to_datetime(prices["date"])

    frames = []
    for ticker, grp in prices.groupby("ticker"):
        grp = grp.sort_values("date").reset_index(drop=True)
        feat = _ticker_liquidity_features(grp, ticker)
        frames.append(feat)

    if not frames:
        return pd.DataFrame()

    return pd.concat(frames, ignore_index=True)


def compute_tradability(prices: pd.DataFrame, as_of: "date") -> pd.Series:
    """
    Return a boolean Series indexed by ticker indicating tradability on as_of.

    Applies the daily tradability filter from config.LIQUIDITY.
    """
    import pandas as pd

    prices = prices.copy()
    prices["date"] = pd.to_datetime(prices["date"])
    lookback = LIQUIDITY["lookback_days"]
    cutoff = pd.Timestamp(as_of) - pd.Timedelta(days=lookback * 1.5)

    results = {}
    for ticker, grp in prices.groupby("ticker"):
        recent = grp[grp["date"] <= pd.Timestamp(as_of)].tail(lookback)
        if recent.empty:
            results[ticker] = False
            continue

        # Median daily value traded
        med_vt = recent["value_traded"].median()
        if pd.isna(med_vt) or med_vt < LIQUIDITY["min_median_value_traded_kes"]:
            results[ticker] = False
            continue

        # Session fraction
        n_traded = (recent["volume"].fillna(0) > 0).sum()
        if n_traded / len(recent) < LIQUIDITY["min_session_fraction"]:
            results[ticker] = False
            continue

        # Spread proxy on latest available day
        today_row = grp[grp["date"] == pd.Timestamp(as_of)]
        if not today_row.empty:
            row = today_row.iloc[0]
            if row["close"] and row["close"] > 0:
                spread_proxy = (row["high"] - row["low"]) / row["close"]
                if spread_proxy > LIQUIDITY["max_spread_proxy_pct"]:
                    results[ticker] = False
                    continue

        results[ticker] = True

    return pd.Series(results, name="tradable")


# ---------------------------------------------------------------------------
# Per-ticker computation
# ---------------------------------------------------------------------------

def _ticker_liquidity_features(grp: pd.DataFrame, ticker: str) -> pd.DataFrame:
    df = grp.copy()

    # --- Median value traded ---
    df["median_value_20d"] = df["value_traded"].rolling(20, min_periods=10).median()
    df["median_value_60d"] = df["value_traded"].rolling(60, min_periods=30).median()

    # --- Zero-volume fraction ---
    df["zero_vol_20d"] = (df["volume"].fillna(0) == 0).astype(float)
    df["zero_vol_fraction_20d"] = df["zero_vol_20d"].rolling(20, min_periods=5).mean()

    # --- Amihud illiquidity: |ret| / value_traded (daily) ---
    df["abs_ret"] = df["close"].pct_change(1).abs()
    df["amihud_daily"] = df["abs_ret"] / df["value_traded"].replace(0, np.nan)
    df["amihud_20d"] = df["amihud_daily"].rolling(20, min_periods=10).mean()

    # --- Overnight gap frequency (proxy for price jumps / thin market) ---
    df["has_gap"] = (
        ((df["open"] / df["close"].shift(1) - 1).abs() > 0.01)
        & df["open"].notna()
        & df["close"].shift(1).notna()
    ).astype(float)
    df["gap_frequency_20d"] = df["has_gap"].rolling(20, min_periods=5).mean()

    # --- Stale price flag: close == previous_close for 2+ days ---
    df["flat_close"] = (df["close"] == df["close"].shift(1)).astype(int)
    df["stale_price_flag"] = (
        df["flat_close"].rolling(LIQUIDITY["stale_price_max_days"], min_periods=2).sum()
        >= LIQUIDITY["stale_price_max_days"] - 1
    ).astype(int)

    # --- Large move on low volume flag ---
    df["large_move"] = (df["close"].pct_change(1).abs() > 0.05).astype(float)
    vol_25pct = df["volume"].rolling(20, min_periods=10).quantile(0.25)
    df["low_volume"] = (df["volume"] < vol_25pct).astype(float)
    df["large_move_low_vol_flag"] = (df["large_move"].astype(bool) & df["low_volume"].astype(bool)).astype(int)

    # --- Days since last "normal" liquidity day ---
    normal_threshold = df["value_traded"].rolling(60, min_periods=20).median() * 0.5
    df["normal_liq_day"] = (df["value_traded"] >= normal_threshold).astype(int)
    df["days_since_normal_liquidity"] = _days_since_last_true(df["normal_liq_day"])

    # --- Price band proximity ---
    # NSE uses ±10% daily price movement limit relative to previous close.
    BAND_PCT = 0.10
    df["upper_band"] = df["previous_close"] * (1 + BAND_PCT)
    df["lower_band"] = df["previous_close"] * (1 - BAND_PCT)
    df["price_band_proximity_pct"] = np.minimum(
        (df["upper_band"] - df["close"]) / df["upper_band"].replace(0, np.nan),
        (df["close"] - df["lower_band"]) / df["lower_band"].abs().replace(0, np.nan),
    )

    df["ticker"] = ticker
    feature_cols = [
        "date", "ticker",
        "median_value_20d", "median_value_60d",
        "zero_vol_fraction_20d",
        "amihud_20d",
        "gap_frequency_20d",
        "stale_price_flag",
        "large_move_low_vol_flag",
        "days_since_normal_liquidity",
        "price_band_proximity_pct",
    ]
    return df[[c for c in feature_cols if c in df.columns]]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _days_since_last_true(series: pd.Series) -> pd.Series:
    """For each position, count days since last 1 (True) in series."""
    result = pd.Series(index=series.index, dtype=float)
    count = np.nan
    for i, val in enumerate(series):
        if val == 1:
            count = 0
        elif not np.isnan(count):
            count += 1
        result.iloc[i] = count
    return result
