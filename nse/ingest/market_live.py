"""
Ingest daily market aggregate statistics from stocksdata (nse DB).

Derives per-day totals across all symbols in the hypertable:
  total_shares_traded    = SUM of end-of-day cumulative volume per symbol
  total_market_turnover  = SUM of avg_price * eod_volume per symbol (KES approx)

Index values (nse_20, nse_all_share, sector indices) are not available in
stocksdata and will be left NULL until another source is wired up.
"""

from __future__ import annotations

import logging
from datetime import date, timedelta

import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.dialects.postgresql import insert

from nse.config import get_source_db_url
from nse.db.schema import MarketDaily, session_scope

log = logging.getLogger(__name__)

_TZ = "Africa/Nairobi"

_MARKET_AGG_SQL = """
WITH eod AS (
    SELECT
        (time AT TIME ZONE :tz)::date  AS date,
        symbol,
        MAX(volume)                    AS eod_volume,
        MAX(avg_price)                 AS avg_price
    FROM stocksdata
    WHERE time >= :start_ts AND time < :end_ts
    GROUP BY date, symbol
)
SELECT
    date,
    SUM(eod_volume)              AS total_shares_traded,
    SUM(avg_price * eod_volume)  AS total_market_turnover
FROM eod
GROUP BY date
ORDER BY date
"""


def ingest_market_from_stocksdata(
    start: date,
    end: date,
    force: bool = False,
    stocksdata_start: date = date(2025, 8, 1),
) -> int:
    """
    Aggregate daily market stats from stocksdata and upsert into market_daily.

    Returns the number of rows upserted.
    """
    source_engine = create_engine(
        get_source_db_url(),
        pool_pre_ping=True,
        pool_size=2,
        max_overflow=2,
    )

    start_ts = f"{start.isoformat()} 00:00:00+03:00"
    end_ts = f"{(end + timedelta(days=1)).isoformat()} 00:00:00+03:00"

    # Market stats can only be derived from stocksdata (historical table has no index data)
    if end < stocksdata_start:
        log.info("No stocksdata coverage before %s; skipping market stats for %s–%s",
                 stocksdata_start, start, end)
        return 0
    start = max(start, stocksdata_start)

    log.info("Querying stocksdata for market stats %s → %s", start, end)
    with source_engine.connect() as conn:
        df = pd.read_sql(
            text(_MARKET_AGG_SQL),
            conn,
            params={"tz": _TZ, "start_ts": start_ts, "end_ts": end_ts},
        )
    source_engine.dispose()

    if df.empty:
        log.warning("No stocksdata rows for market stats %s–%s", start, end)
        return 0

    df["date"] = pd.to_datetime(df["date"]).dt.date

    n = _upsert_market(df)
    log.info("Upserted %d market stat rows (%s–%s)", n, start, end)
    return n


def _upsert_market(df: pd.DataFrame) -> int:
    valid_cols = {c.name for c in MarketDaily.__table__.columns}
    records = df.to_dict(orient="records")
    with session_scope() as s:
        for rec in records:
            filtered = {k: v for k, v in rec.items() if k in valid_cols}
            stmt = (
                insert(MarketDaily)
                .values(**filtered)
                .on_conflict_do_update(
                    index_elements=["date"],
                    set_={k: v for k, v in filtered.items() if k != "date"},
                )
            )
            s.execute(stmt)
    return len(records)


def load_market(start: "date | None" = None, end: "date | None" = None) -> "pd.DataFrame":
    """Load market_daily from the ml_nse DB."""
    from nse.db.schema import get_session

    with get_session() as s:
        q = s.query(MarketDaily)
        if start:
            q = q.filter(MarketDaily.date >= start)
        if end:
            q = q.filter(MarketDaily.date <= end)
        rows = q.all()

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(
        [
            {
                "date": r.date,
                "nse_20": r.nse_20,
                "nse_all_share": r.nse_all_share,
                "total_market_turnover": r.total_market_turnover,
                "total_shares_traded": r.total_shares_traded,
                "total_equity_deals": r.total_equity_deals,
                "sector_banking": r.sector_banking,
                "sector_telecom": r.sector_telecom,
            }
            for r in rows
        ]
    )
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").reset_index(drop=True)
