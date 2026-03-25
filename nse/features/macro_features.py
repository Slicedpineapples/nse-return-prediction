"""
Macro features derived from macro_daily and market_daily.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_macro_features(
    prices: pd.DataFrame,
    macro: pd.DataFrame,
    market: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute macro features joined to each (date, ticker) row.

    Parameters
    ----------
    prices  : tidy long DataFrame with columns: date, ticker
    macro   : DataFrame from macro_daily
    market  : DataFrame from market_daily

    Returns
    -------
    Wide DataFrame keyed on (date, ticker).
    """
    prices = prices.copy()
    prices["date"] = pd.to_datetime(prices["date"])
    macro = macro.copy()
    macro["date"] = pd.to_datetime(macro["date"])
    market = market.copy()
    market["date"] = pd.to_datetime(market["date"])

    # Compute macro time-series features
    macro_feat = _compute_macro_time_features(macro, market)

    # Join to all ticker-date combinations
    base = prices[["date", "ticker"]].drop_duplicates()
    result = base.merge(macro_feat, on="date", how="left")
    return result


# ---------------------------------------------------------------------------
# Macro time-series features
# ---------------------------------------------------------------------------

def _compute_macro_time_features(macro: pd.DataFrame, market: pd.DataFrame) -> pd.DataFrame:
    df = macro.sort_values("date").copy()

    # Coerce all non-date columns to float so None → NaN and .diff() works
    for col in df.columns:
        if col != "date":
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # --- KES/USD changes ---
    df["kes_usd_ret_1d"] = df["kes_usd"].pct_change(1)
    df["kes_usd_ret_5d"] = df["kes_usd"].pct_change(5)

    # --- T-bill spread changes (91d - 364d; steepening/flattening) ---
    df["tbill_spread"] = df["tbill_91d"] - df["tbill_364d"]
    df["tbill_spread_chg_1d"] = df["tbill_spread"].diff(1)

    # --- Policy rate change flag ---
    df["policy_rate_chg_flag"] = (df["policy_rate"].diff(1) != 0).astype(int)

    # --- Inflation surprise proxy: current vs 3m MA ---
    df["inflation_3m_ma"] = df["inflation_latest"].rolling(3, min_periods=1).mean()
    df["inflation_surprise_proxy"] = df["inflation_latest"] - df["inflation_3m_ma"]

    # --- Market turnover regime ---
    # Join market turnover
    df = df.merge(
        market[["date", "total_market_turnover"]],
        on="date",
        how="left",
    )
    df["total_market_turnover"] = df["total_market_turnover"].ffill()
    df["turnover_ma_20d"] = df["total_market_turnover"].rolling(20, min_periods=10).mean()
    df["market_turnover_regime"] = (
        df["total_market_turnover"] / df["turnover_ma_20d"].replace(0, np.nan)
    )

    # --- Sector regime (NSE All-Share vs 20d MA) ---
    df = df.merge(
        market[["date", "nse_all_share"]].dropna(),
        on="date",
        how="left",
    )
    df["nse_all_share"] = df["nse_all_share"].ffill()
    df["nasi_ma_20d"] = df["nse_all_share"].rolling(20, min_periods=10).mean()
    df["sector_regime"] = df["nse_all_share"] / df["nasi_ma_20d"].replace(0, np.nan)

    # --- Macro event flag (from source) ---
    if "macro_event_flag" not in df.columns:
        df["macro_event_flag"] = 0

    feature_cols = [
        "date",
        "kes_usd_ret_1d", "kes_usd_ret_5d",
        "tbill_spread_chg_1d",
        "policy_rate_chg_flag",
        "inflation_surprise_proxy",
        "market_turnover_regime",
        "sector_regime",
        "macro_event_flag",
    ]
    return df[[c for c in feature_cols if c in df.columns]]
