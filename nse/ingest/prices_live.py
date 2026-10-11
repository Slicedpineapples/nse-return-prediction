"""
Ingest daily OHLCV prices into prices_daily, routing by date:

  before STOCKSDATA_START  → nse_historical_daily_data (low/high/close/volume)
  from   STOCKSDATA_START  → stocksdata TimescaleDB hypertable (full OHLCV, 10s ticks)

stocksdata column mapping
-------------------------
open         = first latest_price tick of the day
high         = MAX(high) across all ticks
low          = MIN(low) across all ticks
close        = last latest_price tick of the day
volume       = MAX(volume) — cumulative intraday; last value = EOD total
value_traded = MAX(avg_price) * MAX(volume) — approximation
prev_close   = prev_close from the first tick of the day

nse_historical_daily_data column mapping
-----------------------------------------
open         = NULL (not available)
high         = highest
low          = lowest
close        = closing
volume       = volume
prev_close   = previous
value_traded = NULL (not available)
"""

from __future__ import annotations

import logging
from datetime import date, timedelta

import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.dialects.postgresql import insert

from nse.config import SEED_UNIVERSE, get_source_db_url
from nse.db.schema import PriceDaily, session_scope

log = logging.getLogger(__name__)

_TZ = "Africa/Nairobi"

# Date from which stocksdata is available; historical table used before this.
STOCKSDATA_START = date(2025, 8, 1)

# Date from which live_stock_data is available (richer: order book, foreign flows, WAP).
LIVE_DATA_START = date(2026, 3, 31)

# ---------------------------------------------------------------------------
# SQL queries
# ---------------------------------------------------------------------------

_STOCKSDATA_SQL = """
WITH ticks AS (
    SELECT
        (time AT TIME ZONE :tz)::date   AS date,
        symbol,
        latest_price,
        high,
        low,
        volume,
        avg_price,
        prev_close,
        ROW_NUMBER() OVER (
            PARTITION BY symbol, (time AT TIME ZONE :tz)::date
            ORDER BY time ASC
        ) AS rn_asc,
        ROW_NUMBER() OVER (
            PARTITION BY symbol, (time AT TIME ZONE :tz)::date
            ORDER BY time DESC
        ) AS rn_desc
    FROM stocksdata
    WHERE time >= :start_ts AND time < :end_ts
)
SELECT
    date,
    symbol                                              AS ticker,
    MAX(CASE WHEN rn_asc  = 1 THEN latest_price END)   AS open,
    MAX(high)                                           AS high,
    MIN(low)                                            AS low,
    MAX(CASE WHEN rn_desc = 1 THEN latest_price END)    AS close,
    MAX(volume)                                         AS volume,
    MAX(avg_price) * MAX(volume)                        AS value_traded,
    MAX(CASE WHEN rn_asc  = 1 THEN prev_close END)      AS previous_close
FROM ticks
GROUP BY date, symbol
ORDER BY date, symbol
"""

_HISTORICAL_SQL = """
SELECT
    date,
    code        AS ticker,
    NULL::numeric AS open,
    highest     AS high,
    lowest      AS low,
    closing     AS close,
    volume,
    NULL::numeric AS value_traded,
    previous    AS previous_close
FROM nse_historical_daily_data
WHERE date BETWEEN :start_date AND :end_date
ORDER BY date, code
"""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def ingest_prices_from_stocksdata(
    start: date,
    end: date,
    force: bool = False,
) -> int:
    """
    Ingest daily prices for [start, end], routing to the appropriate source table:
      before STOCKSDATA_START  → nse_historical_daily_data
      STOCKSDATA_START to LIVE_DATA_START-1 → stocksdata
      >= LIVE_DATA_START       → live_stock_data (richer: order book, foreign flows)

    Returns the number of rows upserted.
    """
    from nse.ingest.live_prices import ingest_live_prices

    total = 0

    # Historical segment
    hist_end = min(end, STOCKSDATA_START - timedelta(days=1))
    if start <= hist_end:
        total += _ingest_historical(start, hist_end)

    # stocksdata segment
    sd_start = max(start, STOCKSDATA_START)
    sd_end = min(end, LIVE_DATA_START - timedelta(days=1))
    if sd_start <= sd_end:
        total += _ingest_from_stocksdata(sd_start, sd_end)

    # live_stock_data segment (EOD aggregate only; intraday stored separately)
    live_start = max(start, LIVE_DATA_START)
    if live_start <= end:
        n_daily, _ = ingest_live_prices(live_start, end, force=force, intraday=False)
        total += n_daily

    return total


# ---------------------------------------------------------------------------
# Internal
# ---------------------------------------------------------------------------

def _ingest_historical(start: date, end: date) -> int:
    engine = _source_engine()
    log.info("Querying nse_historical_daily_data %s → %s", start, end)
    with engine.connect() as conn:
        df = pd.read_sql(
            text(_HISTORICAL_SQL),
            conn,
            params={"start_date": start.isoformat(), "end_date": end.isoformat()},
        )
    engine.dispose()

    if df.empty:
        log.warning("No historical rows for %s–%s", start, end)
        return 0

    df["ticker"] = df["ticker"].str.strip().str.upper()
    df = df[df["ticker"].isin(SEED_UNIVERSE)].copy()
    if df.empty:
        log.warning("No SEED_UNIVERSE tickers in historical data for %s–%s", start, end)
        return 0

    df["date"] = pd.to_datetime(df["date"]).dt.date
    df["corporate_action_adjustment_factor"] = 1.0
    df["tradable_flag"] = True

    n = _upsert_prices(df)
    log.info("Upserted %d historical price rows (%s–%s)", n, start, end)
    return n


def _ingest_from_stocksdata(start: date, end: date) -> int:
    engine = _source_engine()
    start_ts = f"{start.isoformat()} 00:00:00+03:00"
    end_ts = f"{(end + timedelta(days=1)).isoformat()} 00:00:00+03:00"

    log.info("Querying stocksdata %s → %s", start, end)
    with engine.connect() as conn:
        df = pd.read_sql(
            text(_STOCKSDATA_SQL),
            conn,
            params={"tz": _TZ, "start_ts": start_ts, "end_ts": end_ts},
        )
    engine.dispose()

    if df.empty:
        log.warning("No stocksdata rows for %s–%s", start, end)
        return 0

    df["ticker"] = df["ticker"].str.strip().str.upper()
    df = df[df["ticker"].isin(SEED_UNIVERSE)].copy()
    if df.empty:
        log.warning("No SEED_UNIVERSE tickers in stocksdata for %s–%s", start, end)
        return 0

    df["date"] = pd.to_datetime(df["date"]).dt.date
    df["corporate_action_adjustment_factor"] = 1.0
    df["tradable_flag"] = True

    n = _upsert_prices(df)
    log.info("Upserted %d stocksdata price rows (%s–%s)", n, start, end)
    return n


def _source_engine():
    return create_engine(
        get_source_db_url(),
        pool_pre_ping=True,
        pool_size=2,
        max_overflow=2,
    )


def _upsert_prices(df: pd.DataFrame) -> int:
    schema_cols = {c.name for c in PriceDaily.__table__.columns}
    records = df.to_dict(orient="records")
    with session_scope() as s:
        for rec in records:
            filtered = {k: v for k, v in rec.items() if k in schema_cols}
            stmt = (
                insert(PriceDaily)
                .values(**filtered)
                .on_conflict_do_update(
                    index_elements=["date", "ticker"],
                    set_={k: v for k, v in filtered.items() if k not in ("date", "ticker")},
                )
            )
            s.execute(stmt)
    return len(records)


def load_prices(
    tickers: list[str] | None = None,
    start: date | None = None,
    end: date | None = None,
) -> pd.DataFrame:
    """Load prices_daily from the ml_nse DB into a tidy DataFrame."""
    from nse.db.schema import get_session

    with get_session() as s:
        q = s.query(PriceDaily)
        if tickers:
            q = q.filter(PriceDaily.ticker.in_(tickers))
        if start:
            q = q.filter(PriceDaily.date >= start)
        if end:
            q = q.filter(PriceDaily.date <= end)
        rows = q.all()

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(
        [
            {
                "date": r.date,
                "ticker": r.ticker,
                "open": r.open,
                "high": r.high,
                "low": r.low,
                "close": r.close,
                "volume": r.volume,
                "value_traded": r.value_traded,
                "number_of_deals": r.number_of_deals,
                "previous_close": r.previous_close,
                "adj_factor": r.corporate_action_adjustment_factor,
                "tradable_flag": r.tradable_flag,
            }
            for r in rows
        ]
    )
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values(["ticker", "date"]).reset_index(drop=True)
