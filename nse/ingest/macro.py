"""
Macro data: KES/USD, T-bill yields, CBK policy rate, inflation.

Ingestion reads only from a manual CSV drop-in (data/raw/macro/macro.csv).
All HTTP scraping has been removed; data is sourced from the database.

To update macro data, place a CSV with columns:
  date, kes_usd, tbill_91d, tbill_182d, tbill_364d, policy_rate, inflation_latest
into data/raw/macro/macro.csv and run nse-ingest.
"""

from __future__ import annotations

import logging
from datetime import date, timedelta
from typing import Iterator

import pandas as pd
from sqlalchemy.dialects.postgresql import insert

from nse.config import RAW_DIR
from nse.db.schema import MacroDaily, session_scope

log = logging.getLogger(__name__)

MACRO_RAW = RAW_DIR / "macro"
MACRO_RAW.mkdir(parents=True, exist_ok=True)


def ingest_macro(start: date, end: date, force: bool = False) -> int:
    """
    Load macro data from data/raw/macro/macro.csv and upsert into macro_daily.

    Returns rows upserted. If no CSV is present, skips silently (data already
    in DB from previous runs).
    """
    df = _load_csv()
    if df.empty:
        log.info("No macro CSV found — skipping macro ingest (using existing DB data)")
        return 0

    df = df[(df["date"] >= start) & (df["date"] <= end)]
    if df.empty:
        return 0

    return _upsert_macro(df)


def load_macro(start: date | None = None, end: date | None = None) -> pd.DataFrame:
    """Load macro_daily from the ml_nse DB."""
    from nse.db.schema import get_session

    with get_session() as s:
        q = s.query(MacroDaily)
        if start:
            q = q.filter(MacroDaily.date >= start)
        if end:
            q = q.filter(MacroDaily.date <= end)
        rows = q.all()

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(
        [
            {
                "date": r.date,
                "kes_usd": r.kes_usd,
                "short_rate": r.short_rate,
                "tbill_91d": r.tbill_91d,
                "tbill_182d": r.tbill_182d,
                "tbill_364d": r.tbill_364d,
                "policy_rate": r.policy_rate,
                "inflation_latest": r.inflation_latest,
                "macro_event_flag": r.macro_event_flag,
            }
            for r in rows
        ]
    )
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Internal
# ---------------------------------------------------------------------------

def _load_csv() -> pd.DataFrame:
    path = MACRO_RAW / "macro.csv"
    if not path.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(path, parse_dates=["date"])
        df["date"] = df["date"].dt.date
        log.info("Loaded macro CSV (%d rows)", len(df))
        return df
    except Exception as exc:
        log.warning("Failed to load macro CSV: %s", exc)
        return pd.DataFrame()


def _upsert_macro(df: pd.DataFrame) -> int:
    valid_cols = {c.name for c in MacroDaily.__table__.columns}
    records = df[[c for c in df.columns if c in valid_cols]].to_dict(orient="records")
    with session_scope() as s:
        for rec in records:
            stmt = (
                insert(MacroDaily)
                .values(**rec)
                .on_conflict_do_update(
                    index_elements=["date"],
                    set_={k: v for k, v in rec.items() if k != "date"},
                )
            )
            s.execute(stmt)
    return len(records)
