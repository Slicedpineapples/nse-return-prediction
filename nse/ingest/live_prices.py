"""
Ingest from live_stock_data table in DB_NAME2 (nse database).

live_stock_data provides richer intraday data than the legacy stocksdata table:
  - Order book: bid/offer prices and quantities, total queue depth
  - Foreign investor flows: foreign_buy, foreign_sell (cumulative KES per session)
  - Weighted average price (WAP)
  - Market capitalisation
  - Total number of trades in session

Two outputs per run:
  1. prices_intraday  — all raw snapshots (10-30s cadence) for the requested range
  2. prices_daily     — EOD aggregate per ticker per day (last snapshot values)

Routing in prices_live.py:
  dates >= LIVE_DATA_START  → this module
  dates <  LIVE_DATA_START  → legacy stocksdata / nse_historical_daily_data
"""

from __future__ import annotations

import logging
from datetime import date, timedelta

import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.dialects.postgresql import insert

from nse.config import SEED_UNIVERSE, get_source_db_url
from nse.db.schema import PriceDaily, PriceIntraday, session_scope

log = logging.getLogger(__name__)

_TZ = "Africa/Nairobi"

# First date covered by live_stock_data (richer source than stocksdata).
LIVE_DATA_START = date(2026, 3, 31)

# ---------------------------------------------------------------------------
# SQL
# ---------------------------------------------------------------------------

# All snapshots for the date range — stored verbatim in prices_intraday.
_SNAPSHOTS_SQL = """
SELECT
    scraped_at,
    (scraped_at AT TIME ZONE :tz)::date  AS date,
    symbol                               AS ticker,
    last_price,
    bid_price,
    bid_qty,
    offer_price,
    offer_qty,
    total_bid_qty,
    total_offer_qty,
    total_trades,
    volume,
    turnover,
    foreign_buy,
    foreign_sell,
    wap,
    market_cap,
    day_open,
    day_high,
    day_low,
    day_close,
    prev_close,
    net_change
FROM live_stock_data
WHERE scraped_at >= :start_ts
  AND scraped_at <  :end_ts
  AND symbol = ANY(:tickers)
ORDER BY scraped_at
"""

# EOD aggregate: last snapshot of each session + session-level stats.
_EOD_SQL = """
WITH ranked AS (
    SELECT *,
        ROW_NUMBER() OVER (
            PARTITION BY symbol, (scraped_at AT TIME ZONE :tz)::date
            ORDER BY scraped_at DESC
        ) AS rn_desc,
        ROW_NUMBER() OVER (
            PARTITION BY symbol, (scraped_at AT TIME ZONE :tz)::date
            ORDER BY scraped_at ASC
        ) AS rn_asc
    FROM live_stock_data
    WHERE scraped_at >= :start_ts
      AND scraped_at <  :end_ts
      AND symbol = ANY(:tickers)
)
SELECT
    (scraped_at AT TIME ZONE :tz)::date                              AS date,
    symbol                                                           AS ticker,
    MAX(CASE WHEN rn_asc  = 1 THEN last_price END)                   AS open,
    MAX(day_high)                                                    AS high,
    MIN(NULLIF(day_low, 0))                                          AS low,
    MAX(CASE WHEN rn_desc = 1 THEN last_price END)                   AS close,
    MAX(volume)                                                      AS volume,
    MAX(turnover)                                                    AS value_traded,
    MAX(CASE WHEN rn_asc  = 1 THEN prev_close END)                   AS previous_close,
    MAX(CASE WHEN rn_desc = 1 THEN foreign_buy  END)                 AS foreign_buy_kes,
    MAX(CASE WHEN rn_desc = 1 THEN foreign_sell END)                 AS foreign_sell_kes,
    MAX(CASE WHEN rn_desc = 1 THEN total_trades END)                 AS total_trades_count,
    MAX(CASE WHEN rn_desc = 1 THEN wap         END)                  AS wap,
    MAX(CASE WHEN rn_desc = 1 THEN market_cap  END)                  AS market_cap_kes,
    MAX(CASE WHEN rn_desc = 1 THEN bid_price   END)                  AS bid_price_eod,
    MAX(CASE WHEN rn_desc = 1 THEN offer_price END)                  AS offer_price_eod,
    MAX(CASE WHEN rn_desc = 1 THEN total_bid_qty   END)              AS total_bid_qty,
    MAX(CASE WHEN rn_desc = 1 THEN total_offer_qty END)              AS total_offer_qty
FROM ranked
GROUP BY date, symbol
ORDER BY date, symbol
"""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def ingest_live_prices(
    start: date,
    end: date,
    force: bool = False,
    intraday: bool = True,
) -> tuple[int, int]:
    """
    Ingest from live_stock_data for [start, end] (inclusive).

    Parameters
    ----------
    start, end  : date range (inclusive)
    force       : re-ingest even if rows already exist
    intraday    : if True, also store raw snapshots in prices_intraday

    Returns
    -------
    (n_daily_rows, n_intraday_rows) — rows upserted into each table
    """
    engine = _source_engine()
    start_ts = f"{start.isoformat()} 00:00:00+03:00"
    end_ts = f"{(end + timedelta(days=1)).isoformat()} 00:00:00+03:00"
    tickers = list(SEED_UNIVERSE)

    params = {"tz": _TZ, "start_ts": start_ts, "end_ts": end_ts, "tickers": tickers}

    # EOD aggregate → prices_daily
    log.info("Querying live_stock_data EOD %s → %s", start, end)
    with engine.connect() as conn:
        eod_df = pd.read_sql(text(_EOD_SQL), conn, params=params)

    n_daily = 0
    if not eod_df.empty:
        eod_df["ticker"] = eod_df["ticker"].str.strip().str.upper()
        eod_df = eod_df[eod_df["ticker"].isin(SEED_UNIVERSE)].copy()
        eod_df["date"] = pd.to_datetime(eod_df["date"]).dt.date
        eod_df["corporate_action_adjustment_factor"] = 1.0
        eod_df["tradable_flag"] = True
        # Zero-volume days: close stays as last_price, but filter out zero closes
        eod_df = eod_df[eod_df["close"].notna() & (eod_df["close"] > 0)]
        n_daily = _upsert_daily(eod_df)
        log.info("Upserted %d live EOD rows into prices_daily (%s–%s)", n_daily, start, end)

    # Raw snapshots → prices_intraday
    n_intraday = 0
    if intraday:
        log.info("Querying live_stock_data snapshots %s → %s", start, end)
        with engine.connect() as conn:
            snap_df = pd.read_sql(text(_SNAPSHOTS_SQL), conn, params=params)

        if not snap_df.empty:
            snap_df["ticker"] = snap_df["ticker"].str.strip().str.upper()
            snap_df = snap_df[snap_df["ticker"].isin(SEED_UNIVERSE)].copy()
            snap_df["date"] = pd.to_datetime(snap_df["date"]).dt.date
            n_intraday = _upsert_intraday(snap_df)
            log.info(
                "Upserted %d intraday snapshots into prices_intraday (%s–%s)",
                n_intraday, start, end,
            )

    engine.dispose()
    return n_daily, n_intraday


def load_intraday(
    tickers: list[str] | None = None,
    start: date | None = None,
    end: date | None = None,
) -> pd.DataFrame:
    """Load prices_intraday from ml_nse DB."""
    from nse.db.schema import get_session

    with get_session() as s:
        q = s.query(PriceIntraday)
        if tickers:
            q = q.filter(PriceIntraday.ticker.in_(tickers))
        if start:
            q = q.filter(PriceIntraday.date >= start)
        if end:
            q = q.filter(PriceIntraday.date <= end)
        rows = q.all()

    if not rows:
        return pd.DataFrame()

    cols = [c.name for c in PriceIntraday.__table__.columns]
    df = pd.DataFrame([{c: getattr(r, c) for c in cols} for r in rows])
    df["scraped_at"] = pd.to_datetime(df["scraped_at"])
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values(["ticker", "scraped_at"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Upsert helpers
# ---------------------------------------------------------------------------

def _upsert_daily(df: pd.DataFrame) -> int:
    schema_cols = {c.name for c in PriceDaily.__table__.columns}
    records = df.to_dict(orient="records")
    with session_scope() as s:
        for rec in records:
            filtered = {k: v for k, v in rec.items() if k in schema_cols}
            # Cast total_trades_count to int if present
            if "total_trades_count" in filtered and filtered["total_trades_count"] is not None:
                try:
                    filtered["total_trades_count"] = int(filtered["total_trades_count"])
                except (ValueError, TypeError):
                    filtered["total_trades_count"] = None
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


def _upsert_intraday(df: pd.DataFrame) -> int:
    schema_cols = {c.name for c in PriceIntraday.__table__.columns if c.name != "id"}
    records = df[[c for c in df.columns if c in schema_cols]].to_dict(orient="records")
    with session_scope() as s:
        for rec in records:
            stmt = (
                insert(PriceIntraday)
                .values(**rec)
                .on_conflict_do_nothing()
            )
            s.execute(stmt)
    return len(records)


def _source_engine():
    return create_engine(
        get_source_db_url(),
        pool_pre_ping=True,
        pool_size=2,
        max_overflow=5,
    )
