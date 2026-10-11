"""
Intraday microstructure features derived from live_stock_data snapshots.

These features are computed from prices_intraday and joined into the daily
feature pipeline when intraday data is available (dates >= 2026-03-31).

Features
--------
Order book (EOD snapshot):
  bid_ask_imbalance        — (bid_qty - offer_qty) / (bid_qty + offer_qty)
  order_book_depth_ratio   — total_bid_qty / total_offer_qty (>1 = buy-side heavier)
  spread_bps               — (offer_price - bid_price) / bid_price * 10000

Execution quality:
  wap_vs_close_bps         — (close - wap) / wap * 10000
                             positive → close above WAP (buying pressure into close)

Foreign flows (cumulative for the session):
  foreign_net_flow_pct     — (foreign_buy - foreign_sell) / turnover * 100
  foreign_participation_pct — (foreign_buy + foreign_sell) / turnover * 100

Session dynamics (from intraday time series):
  session_return_pct       — (close - open) / open * 100
  morning_return_pct       — return through session midpoint (~noon EAT)
  afternoon_return_pct     — return from midpoint to close
  volume_morning_pct       — fraction of daily volume in first half of session
  intraday_high_low_range_pct — (high - low) / open * 100
  session_snapshot_count   — number of raw snapshots (proxy for data completeness)
"""

from __future__ import annotations

import logging
from datetime import date, time

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# NSE session: 09:00 – 15:30 EAT.  Mid-session = ~12:15.
_SESSION_START = time(9, 0)
_SESSION_MID = time(12, 15)
_SESSION_END = time(15, 30)


def compute_intraday_features(
    intraday: pd.DataFrame,
    prices_daily: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute intraday features for all (date, ticker) pairs present in intraday.

    Parameters
    ----------
    intraday      : prices_intraday DataFrame (from load_intraday)
    prices_daily  : prices_daily DataFrame — used for open/close/turnover when
                    the intraday snapshot values are 0 or missing

    Returns
    -------
    Wide DataFrame keyed on (date, ticker) with intraday feature columns.
    """
    if intraday.empty:
        return pd.DataFrame()

    intraday = intraday.copy()
    intraday["scraped_at"] = pd.to_datetime(intraday["scraped_at"])
    intraday["date"] = pd.to_datetime(intraday["date"])

    # EAT local time for session-phase bucketing
    intraday["local_time"] = intraday["scraped_at"].dt.tz_convert("Africa/Nairobi").dt.time

    frames = []
    for (d, ticker), grp in intraday.groupby(["date", "ticker"]):
        grp = grp.sort_values("scraped_at").reset_index(drop=True)

        # Reference daily row for fallbacks
        daily_row = _get_daily_row(prices_daily, ticker, d)

        feat = _session_features(grp, ticker, d, daily_row)
        frames.append(pd.DataFrame([feat]))

    if not frames:
        return pd.DataFrame()

    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# Per-session computation
# ---------------------------------------------------------------------------

def _session_features(
    grp: pd.DataFrame,
    ticker: str,
    d: pd.Timestamp,
    daily_row: pd.Series | None,
) -> dict:
    rec: dict = {"date": d, "ticker": ticker}

    last = grp.iloc[-1]
    first = grp.iloc[0]
    n = len(grp)
    rec["session_snapshot_count"] = n

    # --- Order book (last snapshot of session) ---
    bid_qty = float(last.get("bid_qty") or 0)
    offer_qty = float(last.get("offer_qty") or 0)
    total_bid = float(last.get("total_bid_qty") or 0)
    total_offer = float(last.get("total_offer_qty") or 0)
    bid_price = float(last.get("bid_price") or 0)
    offer_price = float(last.get("offer_price") or 0)

    denom = bid_qty + offer_qty
    rec["bid_ask_imbalance"] = (bid_qty - offer_qty) / denom if denom > 0 else np.nan

    rec["order_book_depth_ratio"] = (
        total_bid / total_offer if total_offer > 0 else np.nan
    )

    if bid_price > 0 and offer_price > bid_price:
        rec["spread_bps"] = (offer_price - bid_price) / bid_price * 10_000
    else:
        rec["spread_bps"] = np.nan

    # --- WAP vs close ---
    wap = float(last.get("wap") or 0)
    close = float(last.get("last_price") or 0)
    if daily_row is not None and (close == 0 or np.isnan(close)):
        close = float(daily_row.get("close") or 0)
    rec["wap_vs_close_bps"] = (close - wap) / wap * 10_000 if wap > 0 else np.nan

    # --- Foreign flows ---
    turnover = float(last.get("turnover") or 0)
    if daily_row is not None and turnover == 0:
        turnover = float(daily_row.get("value_traded") or 0)
    foreign_buy = float(last.get("foreign_buy") or 0)
    foreign_sell = float(last.get("foreign_sell") or 0)

    if turnover > 0:
        rec["foreign_net_flow_pct"] = (foreign_buy - foreign_sell) / turnover * 100
        rec["foreign_participation_pct"] = (foreign_buy + foreign_sell) / turnover * 100
    else:
        rec["foreign_net_flow_pct"] = np.nan
        rec["foreign_participation_pct"] = np.nan

    # --- Session returns ---
    open_price = float(first.get("last_price") or 0)
    if daily_row is not None and open_price == 0:
        open_price = float(daily_row.get("open") or 0)

    if open_price > 0 and close > 0:
        rec["session_return_pct"] = (close - open_price) / open_price * 100
    else:
        rec["session_return_pct"] = np.nan

    # Morning vs afternoon split
    morning = grp[grp["local_time"] <= _SESSION_MID]
    afternoon = grp[grp["local_time"] > _SESSION_MID]

    noon_price = float(morning.iloc[-1]["last_price"]) if not morning.empty else np.nan
    if open_price > 0 and noon_price and noon_price > 0:
        rec["morning_return_pct"] = (noon_price - open_price) / open_price * 100
    else:
        rec["morning_return_pct"] = np.nan

    if noon_price and noon_price > 0 and close > 0:
        rec["afternoon_return_pct"] = (close - noon_price) / noon_price * 100
    else:
        rec["afternoon_return_pct"] = np.nan

    # --- Volume profile ---
    total_vol = float(last.get("volume") or 0)
    morning_vol = float(morning.iloc[-1]["volume"]) if not morning.empty else 0.0
    if total_vol > 0 and morning_vol > 0:
        rec["volume_morning_pct"] = morning_vol / total_vol
    else:
        rec["volume_morning_pct"] = np.nan

    # --- Intraday range ---
    day_high = grp["day_high"].max()
    day_low_nonzero = grp["day_low"].replace(0, np.nan).min()
    if open_price > 0 and pd.notna(day_high) and pd.notna(day_low_nonzero):
        rec["intraday_high_low_range_pct"] = (day_high - day_low_nonzero) / open_price * 100
    else:
        rec["intraday_high_low_range_pct"] = np.nan

    return rec


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_daily_row(
    prices_daily: pd.DataFrame, ticker: str, d: pd.Timestamp
) -> pd.Series | None:
    if prices_daily.empty:
        return None
    mask = (prices_daily["ticker"] == ticker) & (prices_daily["date"] == d)
    rows = prices_daily[mask]
    return rows.iloc[0] if not rows.empty else None
