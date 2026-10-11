"""
Prediction ledger: save daily signals and backfill actual outcomes.

Flow
----
1. Daily pipeline calls save_signals() after generating the ranked list.
2. A cron/CLI job calls backfill_actuals() each morning to fill in
   actual_return_3d / actual_net_alpha_3d for rows whose outcome window
   has closed (signal_date + 3 trading days ago).
"""

from __future__ import annotations

import logging
from datetime import date, datetime, timedelta
from typing import TYPE_CHECKING

import pandas as pd
from sqlalchemy.dialects.postgresql import insert

from nse.config import COSTS
from nse.db.schema import SignalHistory, session_scope

if TYPE_CHECKING:
    pass

log = logging.getLogger(__name__)

BASE_COST_FRACTION = COSTS["scenario_base_bps"] / 10_000


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def save_signals(
    signal_date: date,
    ranked: pd.DataFrame,
    model_label: str = "lgbm",
) -> int:
    """
    Upsert the day's ranked signals into signal_history.

    Parameters
    ----------
    signal_date : the date signals were generated for
    ranked      : DataFrame with columns: ticker, rank, score,
                  predicted_edge_bps (optional), approved (optional)
    model_label : version string for the model that produced the scores
    """
    if ranked.empty:
        return 0

    rows = []
    now = datetime.utcnow()
    for _, r in ranked.iterrows():
        rows.append({
            "signal_date": signal_date,
            "ticker": r["ticker"],
            "rank": int(r["rank"]) if "rank" in r else None,
            "score": float(r.get("score", r.get("model_score", 0.0))),
            "predicted_edge_bps": float(r["predicted_edge_bps"]) if "predicted_edge_bps" in r else None,
            "approved": bool(r.get("approved", False)),
            "model_label": model_label,
            "created_at": now,
        })

    with session_scope() as s:
        for rec in rows:
            stmt = (
                insert(SignalHistory)
                .values(**rec)
                .on_conflict_do_update(
                    index_elements=["signal_date", "ticker"],
                    set_={k: v for k, v in rec.items() if k not in ("signal_date", "ticker")},
                )
            )
            s.execute(stmt)

    log.info("Saved %d signals for %s (model=%s)", len(rows), signal_date, model_label)
    return len(rows)


def backfill_actuals(
    as_of: date | None = None,
    lookback_days: int = 10,
) -> int:
    """
    Fill actual_return_3d / actual_net_alpha_3d for signals whose 3-day
    window has closed (i.e. signal_date + 3 TRADING days of data exists).

    Uses prices_daily and market_daily from the DB.

    FIXED: Changed from calendar days (3) to proper trading-day wait.
           Signals are backfilled only once we have 3 trading days of subsequent prices.

    Returns number of rows updated.
    """
    from nse.ingest.market_live import load_market
    from nse.ingest.prices_live import load_prices

    if as_of is None:
        as_of = date.today()

    # signals that need actuals: outcome_filled_at is NULL and old enough
    cutoff = as_of - timedelta(days=lookback_days)

    # FIXED: Load market data to identify trading days, not just use calendar days
    market = load_market(start=cutoff, end=as_of)
    if market.empty:
        log.warning("No market data available for trading-day detection")
        trading_dates = []
    else:
        market["date"] = pd.to_datetime(market["date"]).dt.date
        trading_dates = sorted(set(market["date"]))

    with session_scope() as s:
        orm_rows = (
            s.query(SignalHistory)
            .filter(SignalHistory.outcome_filled_at.is_(None))
            .filter(SignalHistory.signal_date >= cutoff)
            .all()
        )
        # Materialise into plain dicts before session closes
        rows = [
            {c.name: getattr(r, c.name) for c in SignalHistory.__table__.columns}
            for r in orm_rows
        ]

    if not rows:
        log.info("No signal rows need actuals backfill")
        return 0

    # Filter to signals where 3+ trading days of data exist after signal date
    eligible_rows = []
    for row in rows:
        sig_date = row["signal_date"]
        future_trading_dates = [d for d in trading_dates if d > sig_date]
        if len(future_trading_dates) >= 3:
            eligible_rows.append(row)

    if not eligible_rows:
        log.info("No signals have 3 trading days of data yet")
        return 0

    signal_dates = sorted({r["signal_date"] for r in eligible_rows})
    tickers = list({r["ticker"] for r in eligible_rows})

    # Load prices with a generous forward buffer
    price_start = min(signal_dates) - timedelta(days=5)
    price_end = as_of
    prices = load_prices(tickers=tickers, start=price_start, end=price_end)
    market = load_market(start=price_start, end=price_end)

    if prices.empty:
        log.warning("No prices available for actuals backfill")
        return 0

    prices["date"] = pd.to_datetime(prices["date"])
    prices = prices.sort_values(["ticker", "date"]).reset_index(drop=True)

    # Build NSE20 benchmark returns
    if not market.empty:
        market["date"] = pd.to_datetime(market["date"])
        market = market.sort_values("date")
        market["bench_3d"] = market["nse_20"].pct_change(3)
        bench_map = market.set_index("date")["bench_3d"].to_dict()
    else:
        bench_map = {}

    updated = 0
    now = datetime.utcnow()

    with session_scope() as s:
        for row in eligible_rows:
            sig_date = pd.Timestamp(row["signal_date"])
            tkr_prices = prices[prices["ticker"] == row["ticker"]].copy()
            if tkr_prices.empty:
                continue

            # Find the t0 and t+3 closes
            future = tkr_prices[tkr_prices["date"] > sig_date].sort_values("date")
            t0_rows = tkr_prices[tkr_prices["date"] == sig_date]
            if t0_rows.empty or len(future) < 3:
                continue

            close_t0 = t0_rows.iloc[0]["close"]
            close_t3 = future.iloc[2]["close"]  # 3rd trading day after signal
            t3_date = future.iloc[2]["date"]

            actual_ret_3d = float(close_t3 / close_t0 - 1)

            # Benchmark: 3d return ending at t+3
            # FIXED: Skip rows with missing benchmark data instead of silently assuming 0
            _b = bench_map.get(t3_date)
            if _b is None or _b != _b:  # NaN check
                log.debug("Skipping: no benchmark data for %s on %s", row["ticker"], t3_date)
                continue
            bench_ret = float(_b)
            actual_alpha = actual_ret_3d - bench_ret - BASE_COST_FRACTION

            pred_edge = row["predicted_edge_bps"]
            pred_frac = pred_edge / 10_000 if pred_edge is not None else None
            pred_error = float(actual_alpha - pred_frac) if pred_frac is not None else None

            def _f(v):
                """Convert to Python float; return None for NaN/inf."""
                if v is None:
                    return None
                v = float(v)
                return None if (v != v or v == float("inf") or v == float("-inf")) else v

            s.query(SignalHistory).filter(SignalHistory.id == row["id"]).update({
                "actual_return_3d": _f(actual_ret_3d),
                "actual_net_alpha_3d": _f(actual_alpha),
                "prediction_error": _f(pred_error),
                "outcome_filled_at": now,
            })
            updated += 1

    log.info("Backfilled actuals for %d signal rows", updated)
    return updated


def get_ledger(
    start: date | None = None,
    end: date | None = None,
    approved_only: bool = False,
) -> pd.DataFrame:
    """Load signal_history into a DataFrame."""
    from nse.db.schema import get_session

    with get_session() as s:
        q = s.query(SignalHistory)
        if start:
            q = q.filter(SignalHistory.signal_date >= start)
        if end:
            q = q.filter(SignalHistory.signal_date <= end)
        if approved_only:
            q = q.filter(SignalHistory.approved.is_(True))
        rows = q.all()

    if not rows:
        return pd.DataFrame()

    records = [
        {c.name: getattr(r, c.name) for c in SignalHistory.__table__.columns}
        for r in rows
    ]
    df = pd.DataFrame(records)
    df["signal_date"] = pd.to_datetime(df["signal_date"])
    return df.sort_values(["signal_date", "rank"]).reset_index(drop=True)
