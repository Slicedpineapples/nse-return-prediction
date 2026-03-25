"""
Ingest corporate actions from the live nse database into issuer_events.

Reads from corporate_actions in the nse DB and maps to issuer_events in ml_nse.
The news table is excluded for now.

Column mapping
--------------
corporate_actions.id              → issuer_events.event_id
corporate_actions.announcement_date → issuer_events.timestamp_published (noon EAT)
  fallback: event_date, then created_at
corporate_actions.symbol          → issuer_events.ticker
corporate_actions.type            → issuer_events.event_type  (normalised)
corporate_actions.action_subtype  → issuer_events.event_subtype
corporate_actions.details (jsonb) + amount, venue, etc. → issuer_events.raw_text (JSON)
source = 'corporate_action'
"""

from __future__ import annotations

import json
import logging
from datetime import date, datetime, timedelta, timezone
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd

import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.dialects.postgresql import insert

from nse.config import SEED_UNIVERSE, get_source_db_url
from nse.db.schema import IssuerEvent, session_scope

log = logging.getLogger(__name__)

# Map raw type strings → canonical event_type used by feature engineering
_TYPE_MAP: dict[str, str] = {
    "dividend":         "dividend",
    "div":              "dividend",
    "agm":              "agm",
    "annual general":   "agm",
    "egm":              "agm",
    "extraordinary":    "agm",
    "bonus":            "corporate_action",
    "rights":           "corporate_action",
    "split":            "corporate_action",
    "buyback":          "corporate_action",
    "earnings":         "earnings",
    "results":          "earnings",
    "profit":           "earnings",
    "investor":         "investor_briefing",
    "briefing":         "investor_briefing",
}


def _normalise_type(raw: str | None) -> str:
    if not raw:
        return "other"
    lower = raw.lower()
    for key, canonical in _TYPE_MAP.items():
        if key in lower:
            return canonical
    return "other"


_QUERY_SQL = """
SELECT
    id,
    type,
    symbol,
    company,
    ex_date,
    record_date,
    payment_date,
    event_date,
    event_time,
    venue,
    amount,
    currency,
    details,
    action_subtype,
    announcement_date,
    books_closure_date,
    created_at
FROM corporate_actions
WHERE
    COALESCE(announcement_date, event_date, created_at::date)
        BETWEEN :start_date AND :end_date
"""


def ingest_announcements_from_db(
    start: date,
    end: date,
    force: bool = False,
) -> int:
    """
    Read corporate_actions from nse DB and upsert into issuer_events.

    Returns the number of rows upserted.
    """
    source_engine = create_engine(
        get_source_db_url(),
        pool_pre_ping=True,
        pool_size=2,
        max_overflow=2,
    )

    log.info("Querying corporate_actions %s → %s", start, end)
    with source_engine.connect() as conn:
        df = pd.read_sql(
            text(_QUERY_SQL),
            conn,
            params={"start_date": start.isoformat(), "end_date": end.isoformat()},
        )
    source_engine.dispose()

    if df.empty:
        log.info("No corporate actions found for %s–%s", start, end)
        return 0

    # Only keep rows with a known ticker in SEED_UNIVERSE; skip NULL / unknown symbols
    df = df[df["symbol"].notna()].copy()
    df["symbol"] = df["symbol"].str.strip().str.upper()
    df = df[df["symbol"].isin(SEED_UNIVERSE)].copy()

    if df.empty:
        log.info("No SEED_UNIVERSE corporate actions for %s–%s", start, end)
        return 0

    total = 0
    for _, row in df.iterrows():
        rec = _build_event(row)
        total += _upsert_event(rec)

    log.info("Upserted %d corporate action rows (%s–%s)", total, start, end)
    return total


def _build_event(row: pd.Series) -> dict:
    # Resolve published timestamp: announcement_date → event_date → created_at
    pub_date = row.get("announcement_date") or row.get("event_date")
    if pub_date is not None and not pd.isna(pub_date):
        # Use noon EAT (UTC+3) on the announcement date
        ts = datetime(pub_date.year, pub_date.month, pub_date.day, 12, 0, 0,
                      tzinfo=timezone.utc)
    else:
        ts = row["created_at"]
        if hasattr(ts, "to_pydatetime"):
            ts = ts.to_pydatetime()

    # Serialise details jsonb + scalar fields into raw_text
    details = row.get("details") or {}
    if isinstance(details, str):
        try:
            details = json.loads(details)
        except Exception:
            details = {"raw": details}

    def _safe(v):
        """Return v if JSON-serialisable scalar, else None."""
        if v is None:
            return None
        if isinstance(v, float) and (v != v):  # NaN check
            return None
        if isinstance(v, (str, int, float, bool)):
            return v
        return str(v)

    raw = {
        **{k: _safe(v) for k, v in details.items()},
        "type": _safe(row.get("type")),
        "action_subtype": _safe(row.get("action_subtype")),
        "amount": float(row["amount"]) if pd.notna(row.get("amount")) else None,
        "currency": _safe(row.get("currency")),
        "ex_date": str(row["ex_date"]) if pd.notna(row.get("ex_date")) else None,
        "payment_date": str(row["payment_date"]) if pd.notna(row.get("payment_date")) else None,
        "books_closure_date": str(row["books_closure_date"]) if pd.notna(row.get("books_closure_date")) else None,
        "venue": _safe(row.get("venue")),
        "company": _safe(row.get("company")),
    }

    return {
        "event_id": str(row["id"]),
        "timestamp_published": ts,
        "ticker": row.get("symbol"),
        "source": "corporate_action",
        "event_type": _normalise_type(row.get("type")),
        "event_subtype": row.get("action_subtype"),
        "raw_text": json.dumps({k: v for k, v in raw.items() if v is not None}),
    }


def _upsert_event(rec: dict) -> int:
    valid_cols = {c.name for c in IssuerEvent.__table__.columns}
    filtered = {k: v for k, v in rec.items() if k in valid_cols}
    with session_scope() as s:
        stmt = (
            insert(IssuerEvent)
            .values(**filtered)
            .on_conflict_do_update(
                index_elements=["event_id"],
                set_={k: v for k, v in filtered.items() if k != "event_id"},
            )
        )
        s.execute(stmt)
    return 1


def get_unprocessed_events(batch_size: int = 50) -> list[dict]:
    """Return issuer_events that have not yet been processed by the LLM."""
    from nse.db.schema import get_session

    with get_session() as s:
        rows = (
            s.query(IssuerEvent)
            .filter(IssuerEvent.llm_processed_at.is_(None))
            .filter(IssuerEvent.raw_text.isnot(None))
            .order_by(IssuerEvent.timestamp_published.desc())
            .limit(batch_size)
            .all()
        )

    return [
        {
            "event_id": r.event_id,
            "ticker": r.ticker,
            "event_type": r.event_type,
            "raw_text": r.raw_text,
            "timestamp_published": r.timestamp_published,
        }
        for r in rows
    ]


def load_events(
    tickers: list[str] | None = None,
    start: date | None = None,
    end: date | None = None,
    llm_only: bool = False,
) -> "pd.DataFrame":
    """Load issuer_events from the ml_nse DB into a DataFrame."""
    import pandas as pd
    from nse.db.schema import get_session

    with get_session() as s:
        q = s.query(IssuerEvent)
        if tickers:
            q = q.filter(IssuerEvent.ticker.in_(tickers))
        if start:
            q = q.filter(
                IssuerEvent.timestamp_published >= datetime.combine(start, datetime.min.time())
            )
        if end:
            end_dt = datetime.combine(end + timedelta(days=1), datetime.min.time())
            q = q.filter(IssuerEvent.timestamp_published < end_dt)
        if llm_only:
            q = q.filter(IssuerEvent.llm_processed_at.isnot(None))
        rows = q.all()

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(
        [
            {
                "event_id": r.event_id,
                "timestamp_published": r.timestamp_published,
                "ticker": r.ticker,
                "source": r.source,
                "event_type": r.event_type,
                "event_subtype": r.event_subtype,
                "raw_text": r.raw_text,
                "llm_sentiment": r.llm_sentiment,
                "llm_materiality": r.llm_materiality,
                "llm_surprise": r.llm_surprise,
                "llm_horizon": r.llm_horizon,
                "llm_processed_at": r.llm_processed_at,
            }
            for r in rows
        ]
    )
