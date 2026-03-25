"""
Event and calendar features.

Uses issuer_events table (both rule-based and LLM-derived fields).

Key design: all event lookforward flags (earnings_in_Nd) check if an event
of that type is scheduled within N calendar days AHEAD of the feature date.
This requires either a known announcement calendar or the event to have been
pre-announced.  For past events, we use backward-looking windows.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_event_features(
    prices: pd.DataFrame,
    events: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute event/calendar features for all tickers.

    Parameters
    ----------
    prices  : tidy long DataFrame with columns: date, ticker
    events  : DataFrame from issuer_events with columns:
              timestamp_published, ticker, event_type, llm_sentiment,
              llm_surprise, llm_materiality, llm_confidence

    Returns
    -------
    Wide DataFrame keyed on (date, ticker).
    """
    if events.empty:
        # Return zeros when no event data
        df = prices[["date", "ticker"]].copy()
        df["date"] = pd.to_datetime(df["date"])
        for col in _EVENT_FEATURE_COLS:
            df[col] = 0
        return df

    prices = prices.copy()
    prices["date"] = pd.to_datetime(prices["date"])
    events = events.copy()
    events["pub_date"] = pd.to_datetime(events["timestamp_published"]).dt.normalize()

    frames = []
    for ticker, price_grp in prices.groupby("ticker"):
        evt_grp = events[events["ticker"] == ticker].copy()
        feat = _ticker_event_features(price_grp, evt_grp, ticker)
        frames.append(feat)

    if not frames:
        return pd.DataFrame()

    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# Per-ticker
# ---------------------------------------------------------------------------

_EVENT_FEATURE_COLS = [
    "earnings_in_1d", "earnings_in_3d", "earnings_in_5d",
    "dividend_flag", "agm_flag", "investor_briefing_flag",
    "days_since_last_announcement",
    "month_of_year", "reporting_season_flag", "corporate_action_window_flag",
    "llm_max_materiality_today",
    "llm_weighted_sentiment_3d",
    "llm_weighted_surprise_5d",
    "llm_announcement_count_20d",
    "llm_event_earnings_flag",
    "llm_event_dividend_flag",
    "llm_event_agm_flag",
    "llm_event_mgmt_flag",
    "llm_event_regulatory_flag",
    "llm_event_financing_flag",
]


def _ticker_event_features(
    price_grp: pd.DataFrame, evt_grp: pd.DataFrame, ticker: str
) -> pd.DataFrame:
    df = price_grp[["date", "ticker"]].copy()
    df = df.sort_values("date").reset_index(drop=True)

    # Initialise feature columns to 0 / NaN
    for col in _EVENT_FEATURE_COLS:
        df[col] = 0

    if evt_grp.empty:
        df["days_since_last_announcement"] = np.nan
        df["llm_max_materiality_today"] = np.nan
        df["llm_weighted_sentiment_3d"] = np.nan
        df["llm_weighted_surprise_5d"] = np.nan
        df["llm_announcement_count_20d"] = 0
        return df

    evt_grp = evt_grp.sort_values("pub_date")
    pub_dates = evt_grp["pub_date"]

    results = []
    for _, row in df.iterrows():
        d = row["date"]
        rec: dict = {"date": d, "ticker": ticker}

        # --- Forward-looking flags (earnings announced in advance) ---
        for horizon, col in [(1, "earnings_in_1d"), (3, "earnings_in_3d"), (5, "earnings_in_5d")]:
            future_window = (pub_dates >= d) & (pub_dates <= d + pd.Timedelta(days=horizon))
            future_earnings = evt_grp[future_window & (evt_grp["event_type"] == "earnings")]
            rec[col] = int(len(future_earnings) > 0)

        # --- Today's event flags ---
        today_evts = evt_grp[pub_dates == d]
        rec["dividend_flag"] = int((today_evts["event_type"] == "dividend").any())
        rec["agm_flag"] = int((today_evts["event_type"] == "agm").any())
        rec["investor_briefing_flag"] = int(
            (today_evts["event_type"] == "investor_briefing").any()
        )

        # --- Days since last announcement ---
        past_evts = evt_grp[pub_dates <= d]
        if past_evts.empty:
            rec["days_since_last_announcement"] = np.nan
        else:
            rec["days_since_last_announcement"] = (d - past_evts["pub_date"].max()).days

        # --- Calendar features ---
        rec["month_of_year"] = d.month
        rec["reporting_season_flag"] = int(d.month in (2, 3, 4, 8, 9))  # H1/H2 results season
        rec["corporate_action_window_flag"] = int(
            (today_evts["source"] == "corporate_action").any()
        )

        # --- LLM-derived aggregations ---
        # Max materiality: today only
        rec["llm_max_materiality_today"] = (
            today_evts["llm_materiality"].max() if len(today_evts) > 0
            and "llm_materiality" in today_evts.columns
            else np.nan
        )

        # Weighted sentiment: last 3 days (decay weight by recency)
        rec["llm_weighted_sentiment_3d"] = _weighted_score(
            evt_grp, pub_dates, d, days=3, score_col="llm_sentiment"
        )

        # Weighted surprise: last 5 days
        rec["llm_weighted_surprise_5d"] = _weighted_score(
            evt_grp, pub_dates, d, days=5, score_col="llm_surprise"
        )

        # Count of announcements in last 20 days
        window_20 = (pub_dates >= d - pd.Timedelta(days=20)) & (pub_dates <= d)
        rec["llm_announcement_count_20d"] = int(window_20.sum())

        # Event-type one-hot aggregates (last 5 days)
        window_5 = evt_grp[(pub_dates >= d - pd.Timedelta(days=5)) & (pub_dates <= d)]
        for etype, col in [
            ("earnings", "llm_event_earnings_flag"),
            ("dividend", "llm_event_dividend_flag"),
            ("agm", "llm_event_agm_flag"),
            ("management", "llm_event_mgmt_flag"),
            ("regulatory", "llm_event_regulatory_flag"),
            ("financing", "llm_event_financing_flag"),
        ]:
            rec[col] = int((window_5["event_type"] == etype).any())

        results.append(rec)

    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _weighted_score(
    evt_grp: pd.DataFrame,
    pub_dates: pd.Series,
    as_of: pd.Timestamp,
    days: int,
    score_col: str,
) -> float:
    """Decay-weighted average of score_col over the last `days` days."""
    if score_col not in evt_grp.columns:
        return np.nan
    window = evt_grp[
        (pub_dates >= as_of - pd.Timedelta(days=days)) & (pub_dates <= as_of)
    ].copy()
    if window.empty:
        return np.nan
    window = window.dropna(subset=[score_col])
    if window.empty:
        return np.nan
    # Decay: 1.0 for today, 0.5 per day back
    window["lag_days"] = (as_of - window["pub_date"]).dt.days.clip(lower=0)
    window["weight"] = 0.5 ** window["lag_days"]
    return (window[score_col] * window["weight"]).sum() / window["weight"].sum()
