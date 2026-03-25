"""
Price and volume features.

All features are computed from the prices_daily and market_daily tables.
Input is a tidy long DataFrame (columns: date, ticker, close, ...).
Output is a wide DataFrame keyed on (date, ticker) with all price features.

Design note: all lookback windows use only past data to avoid leakage.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_price_features(
    prices: pd.DataFrame,
    market: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute price/volume features for all tickers.

    Parameters
    ----------
    prices : tidy long DataFrame with columns:
             date, ticker, open, high, low, close, volume, value_traded,
             number_of_deals, previous_close
    market : daily DataFrame with columns: date, nse_20, nse_all_share

    Returns
    -------
    Wide DataFrame keyed on (date, ticker) with feature columns.
    """
    prices = prices.copy()
    prices["date"] = pd.to_datetime(prices["date"])
    if market.empty:
        market = pd.DataFrame(columns=["date"])
    else:
        market = market.copy()
        market["date"] = pd.to_datetime(market["date"])

    frames = []
    for ticker, grp in prices.groupby("ticker"):
        grp = grp.sort_values("date").reset_index(drop=True)
        feat = _ticker_price_features(grp, market, ticker)
        frames.append(feat)

    if not frames:
        return pd.DataFrame()

    result = pd.concat(frames, ignore_index=True)
    return result


# ---------------------------------------------------------------------------
# Per-ticker computation
# ---------------------------------------------------------------------------

def _ticker_price_features(
    grp: pd.DataFrame, market: pd.DataFrame, ticker: str
) -> pd.DataFrame:
    """Compute all price/volume features for a single ticker time series."""
    df = grp.copy()

    # --- Returns ---
    df["ret_1d"] = df["close"].pct_change(1)
    df["ret_3d"] = df["close"].pct_change(3)
    df["ret_5d"] = df["close"].pct_change(5)
    df["ret_20d"] = df["close"].pct_change(20)

    # Overnight gap: today's open vs yesterday's close
    df["overnight_gap"] = (df["open"] / df["close"].shift(1) - 1).where(
        df["open"].notna() & df["close"].shift(1).notna()
    )

    # --- Volatility ---
    df["vol_5d"] = df["ret_1d"].rolling(5, min_periods=3).std()
    df["vol_20d"] = df["ret_1d"].rolling(20, min_periods=10).std()

    # --- Volume z-scores ---
    vol_mean_20 = df["volume"].rolling(20, min_periods=10).mean()
    vol_std_20 = df["volume"].rolling(20, min_periods=10).std()
    df["volume_zscore_20d"] = (df["volume"] - vol_mean_20) / vol_std_20.replace(0, np.nan)

    vt_mean_20 = df["value_traded"].rolling(20, min_periods=10).mean()
    vt_std_20 = df["value_traded"].rolling(20, min_periods=10).std()
    df["value_traded_zscore_20d"] = (df["value_traded"] - vt_mean_20) / vt_std_20.replace(0, np.nan)

    deals_mean_20 = df["number_of_deals"].rolling(20, min_periods=10).mean()
    deals_std_20 = df["number_of_deals"].rolling(20, min_periods=10).std()
    df["deals_zscore_20d"] = (
        (df["number_of_deals"] - deals_mean_20) / deals_std_20.replace(0, np.nan)
    )

    # --- Intraday range ---
    df["intraday_range_pct"] = (df["high"] - df["low"]) / df["close"].replace(0, np.nan)

    # --- Momentum reversal ---
    # Simple mean-reversion: 5d return vs 20d return direction
    df["momentum_reversal_5d"] = np.where(
        df["ret_5d"] * df["ret_20d"] < 0, 1, 0
    )

    # --- Beta to NSE 20 ---
    df = df.merge(
        market[["date", "nse_20"]].rename(columns={"nse_20": "nse20_close"}),
        on="date",
        how="left",
    )
    df["nse20_ret"] = df["nse20_close"].pct_change(1)
    df["beta_nse20_60d"] = _rolling_beta(df["ret_1d"], df["nse20_ret"], window=60)

    # --- Correlation to market ---
    df["corr_sector_60d"] = (
        df["ret_1d"]
        .rolling(60, min_periods=30)
        .corr(df["nse20_ret"])
    )

    df["ticker"] = ticker
    feature_cols = [
        "date", "ticker",
        "ret_1d", "ret_3d", "ret_5d", "ret_20d",
        "overnight_gap",
        "vol_5d", "vol_20d",
        "volume_zscore_20d", "value_traded_zscore_20d", "deals_zscore_20d",
        "intraday_range_pct",
        "momentum_reversal_5d",
        "beta_nse20_60d",
        "corr_sector_60d",
    ]
    return df[[c for c in feature_cols if c in df.columns]]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _rolling_beta(returns: pd.Series, market_rets: pd.Series, window: int) -> pd.Series:
    """Rolling OLS beta of returns on market_rets."""
    betas = pd.Series(index=returns.index, dtype=float)
    for i in range(window, len(returns)):
        y = returns.iloc[i - window : i].dropna()
        x = market_rets.iloc[i - window : i][y.index]
        combined = pd.concat([y, x], axis=1).dropna()
        if len(combined) < window // 2:
            continue
        y_vals = combined.iloc[:, 0].values
        x_vals = combined.iloc[:, 1].values
        if x_vals.std() == 0:
            continue
        betas.iloc[i] = np.cov(y_vals, x_vals)[0, 1] / np.var(x_vals)
    return betas
