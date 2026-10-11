"""
Main feature pipeline.

Orchestrates: price → liquidity → event → macro → merge → upsert features_daily.

All feature computation respects the train/test boundary passed in: features
are computed only using data available on or before the feature date.
"""

from __future__ import annotations

import logging
from datetime import date

import pandas as pd
from sqlalchemy.dialects.postgresql import insert

from nse.db.schema import FeaturesDaily, session_scope
from nse.features.event_features import compute_event_features
from nse.features.intraday_features import compute_intraday_features
from nse.features.liquidity_features import compute_liquidity_features, compute_tradability
from nse.features.macro_features import compute_macro_features
from nse.features.price_features import compute_price_features
from nse.ingest.announcements_live import load_events
from nse.ingest.live_prices import LIVE_DATA_START, load_intraday
from nse.ingest.macro import load_macro
from nse.ingest.market_live import load_market
from nse.ingest.prices_live import load_prices

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_feature_pipeline(
    start: date,
    end: date,
    tickers: list[str] | None = None,
    force: bool = False,
) -> pd.DataFrame:
    """
    Build features_daily for all tickers in [start, end].

    Returns the merged feature DataFrame (also upserted into DB).
    """
    log.info("Loading raw data for feature pipeline: %s to %s", start, end)

    # Add extra history for rolling windows (need up to 60 days of lookback)
    lookback_start = _offset_date(start, -90)

    prices = load_prices(tickers=tickers, start=lookback_start, end=end)
    market = load_market(start=lookback_start, end=end)
    macro = load_macro(start=lookback_start, end=end)
    events = load_events(tickers=tickers, start=lookback_start, end=end)

    if prices.empty:
        log.warning("No price data found for %s to %s", start, end)
        return pd.DataFrame()

    log.info("Computing price features")
    price_feat = compute_price_features(prices, market)

    log.info("Computing liquidity features")
    liq_feat = compute_liquidity_features(prices)

    log.info("Computing event features")
    event_feat = compute_event_features(prices, events)

    log.info("Computing macro features")
    macro_feat = compute_macro_features(prices, macro, market)

    # Intraday features — only available for dates >= LIVE_DATA_START
    intraday_feat = pd.DataFrame()
    if end >= LIVE_DATA_START:
        intraday_start = max(start, LIVE_DATA_START)
        log.info("Loading intraday snapshots for %s → %s", intraday_start, end)
        intraday = load_intraday(tickers=tickers, start=intraday_start, end=end)
        if not intraday.empty:
            log.info("Computing intraday features (%d snapshots)", len(intraday))
            intraday_feat = compute_intraday_features(intraday, prices)

    log.info("Merging all features")
    merged = _merge_all(price_feat, liq_feat, event_feat, macro_feat, intraday_feat)

    # Filter to requested date range only
    merged = merged[
        (merged["date"] >= pd.Timestamp(start))
        & (merged["date"] <= pd.Timestamp(end))
    ].copy()

    # Add tradability flags
    for d in merged["date"].dt.date.unique():
        tradable = compute_tradability(prices, d)
        mask = merged["date"].dt.date == d
        merged.loc[mask, "dq_tradable"] = merged.loc[mask, "ticker"].map(tradable)

    # Data-quality flag: has minimum required features
    required = ["ret_1d", "vol_20d", "amihud_20d"]
    merged["dq_features_ok"] = merged[required].notna().all(axis=1)

    log.info("Upserting %d feature rows", len(merged))
    _upsert_features(merged)

    return merged


def load_features(
    start: date | None = None,
    end: date | None = None,
    tickers: list[str] | None = None,
    tradable_only: bool = True,
) -> pd.DataFrame:
    """Load features_daily from DB."""
    from nse.db.schema import get_session

    with get_session() as s:
        q = s.query(FeaturesDaily)
        if tickers:
            q = q.filter(FeaturesDaily.ticker.in_(tickers))
        if start:
            q = q.filter(FeaturesDaily.date >= start)
        if end:
            q = q.filter(FeaturesDaily.date <= end)
        if tradable_only:
            q = q.filter(FeaturesDaily.dq_tradable.is_(True))
        rows = q.all()

    if not rows:
        return pd.DataFrame()

    records = []
    for r in rows:
        rec = {c.name: getattr(r, c.name) for c in FeaturesDaily.__table__.columns}
        records.append(rec)

    df = pd.DataFrame(records)
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values(["date", "ticker"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Merge
# ---------------------------------------------------------------------------

def _merge_all(
    price_feat: pd.DataFrame,
    liq_feat: pd.DataFrame,
    event_feat: pd.DataFrame,
    macro_feat: pd.DataFrame,
    intraday_feat: pd.DataFrame | None = None,
) -> pd.DataFrame:
    key = ["date", "ticker"]
    df = price_feat.merge(liq_feat, on=key, how="outer")
    df = df.merge(event_feat, on=key, how="outer")
    df = df.merge(macro_feat, on=key, how="left")
    if intraday_feat is not None and not intraday_feat.empty:
        intraday_feat = intraday_feat.copy()
        intraday_feat["date"] = pd.to_datetime(intraday_feat["date"])
        df = df.merge(intraday_feat, on=key, how="left")
    df["date"] = pd.to_datetime(df["date"])
    return df


# ---------------------------------------------------------------------------
# Upsert
# ---------------------------------------------------------------------------

def _upsert_features(df: pd.DataFrame) -> None:
    """Upsert rows into features_daily."""
    import math
    from sqlalchemy import Integer

    valid_cols = {c.name for c in FeaturesDaily.__table__.columns if c.name != "id"}
    int_cols = {
        c.name for c in FeaturesDaily.__table__.columns
        if isinstance(c.type, Integer) and c.name != "id"
    }
    df_copy = df.copy()
    df_copy["date"] = df_copy["date"].dt.date  # store as date, not timestamp

    records = df_copy[[c for c in df_copy.columns if c in valid_cols]].to_dict(
        orient="records"
    )
    # Replace float NaN with None for integer columns (Postgres rejects NaN in int fields)
    for rec in records:
        for col in int_cols:
            if col in rec and isinstance(rec[col], float) and math.isnan(rec[col]):
                rec[col] = None
    with session_scope() as s:
        for rec in records:
            stmt = (
                insert(FeaturesDaily)
                .values(**rec)
                .on_conflict_do_update(
                    index_elements=["date", "ticker"],
                    set_={k: v for k, v in rec.items() if k not in ("date", "ticker")},
                )
            )
            s.execute(stmt)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _offset_date(d: date, days: int) -> date:
    from datetime import timedelta

    return d + timedelta(days=days)
