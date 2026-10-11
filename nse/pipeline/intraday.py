"""
Intraday signal pipeline.

Runs during the NSE trading session (09:00 – 15:30 EAT) using the latest
snapshots from live_stock_data.  Produces a ranked signal based on:

  1. Daily model scores (from the production ranker, using today's features
     built earlier by nse-pipeline or recomputed on demand)
  2. Intraday overlay: real-time order book, foreign flow, session momentum

Output: ranked DataFrame + optional JSON report in reports/output/.

Usage
-----
    nse-intraday                     # score with current session data
    nse-intraday --date 2026-04-21   # score a specific past session
    nse-intraday --no-daily-score    # use only intraday signals, no stored model score

Intraday signal logic
---------------------
For each tradable ticker the intraday score is computed as:

    intraday_score = base_score  (from production model, range 0-1)
                   + ORDER_BOOK_WEIGHT  * bid_ask_imbalance       (range -1 to +1)
                   + FOREIGN_FLOW_WEIGHT * foreign_net_flow_sign  (range -1 to +1)
                   + MOMENTUM_WEIGHT   * session_return_sign      (range -1 to +1)

Then re-normalised to [0, 1] cross-sectionally.

Tuning constants are in config.INTRADAY (added below).
"""

from __future__ import annotations

import json
import logging
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from rich.console import Console
from rich.table import Table

from nse.config import REPORTS_DIR, SEED_UNIVERSE
from nse.db import init_db
from nse.features.intraday_features import compute_intraday_features
from nse.ingest.live_prices import load_intraday
from nse.ingest.prices_live import load_prices
from nse.models.ranker import NSERanker

log = logging.getLogger(__name__)
console = Console()

# Weights for the intraday overlay on top of the base model score.
# All three inputs are normalised to [-1, +1] before weighting.
_ORDER_BOOK_WEIGHT = 0.15
_FOREIGN_FLOW_WEIGHT = 0.20
_MOMENTUM_WEIGHT = 0.10


def run_intraday_pipeline(
    as_of: date | None = None,
    model_label: str = "production",
    use_daily_score: bool = True,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Score the current session and return a ranked DataFrame.

    Parameters
    ----------
    as_of           : session date (default: today)
    model_label     : saved model to load for base scores
    use_daily_score : if True, blend intraday overlay with stored model score;
                      if False, rank purely on intraday signals
    verbose         : print rich table to stdout

    Returns
    -------
    DataFrame ranked by intraday_score descending.
    """
    as_of = as_of or date.today()
    init_db()

    if verbose:
        console.rule(f"[bold cyan]NSE Intraday Pipeline — {as_of}")

    # Load intraday snapshots for today
    intraday = load_intraday(tickers=list(SEED_UNIVERSE), start=as_of, end=as_of)
    if intraday.empty:
        console.print(f"[yellow]No intraday snapshots found for {as_of}. Run nse-ingest first.[/yellow]")
        return pd.DataFrame()

    snapshot_time = intraday["scraped_at"].max()
    n_snapshots = len(intraday)
    if verbose:
        console.print(
            f"[dim]Loaded {n_snapshots} snapshots "
            f"(latest: {snapshot_time.strftime('%H:%M:%S')} EAT)[/dim]"
        )

    # Compute intraday features from today's snapshots
    prices_daily = load_prices(tickers=list(SEED_UNIVERSE), start=as_of, end=as_of)
    intraday_feat = compute_intraday_features(intraday, prices_daily)
    if intraday_feat.empty:
        console.print("[yellow]Could not compute intraday features.[/yellow]")
        return pd.DataFrame()

    # Base model score (optional)
    base_scores: dict[str, float] = {}
    if use_daily_score:
        try:
            from nse.features.pipeline import load_features
            today_feat = load_features(
                start=as_of, end=as_of, tickers=list(SEED_UNIVERSE), tradable_only=False
            )
            if not today_feat.empty:
                ranker = NSERanker.load(model_label)
                scores_df = ranker.predict(today_feat)
                base_scores = dict(zip(scores_df["ticker"], scores_df["score"]))
        except Exception as exc:
            log.warning("Could not load base model scores: %s — using intraday signals only", exc)

    # Build intraday score for each ticker
    rows = []
    for _, feat in intraday_feat.iterrows():
        ticker = feat["ticker"]
        base = base_scores.get(ticker, 0.5)

        # Normalise each overlay component to [-1, +1]
        ob_signal = _safe_clip(feat.get("bid_ask_imbalance"))
        ff_signal = _sign_clip(feat.get("foreign_net_flow_pct"))
        mom_signal = _sign_clip(feat.get("session_return_pct"))

        overlay = (
            _ORDER_BOOK_WEIGHT * (ob_signal or 0)
            + _FOREIGN_FLOW_WEIGHT * (ff_signal or 0)
            + _MOMENTUM_WEIGHT * (mom_signal or 0)
        )
        raw_score = base + overlay

        rows.append({
            "ticker": ticker,
            "base_model_score": round(base, 4),
            "bid_ask_imbalance": round(feat.get("bid_ask_imbalance") or 0, 4),
            "foreign_net_flow_pct": round(feat.get("foreign_net_flow_pct") or 0, 2),
            "session_return_pct": round(feat.get("session_return_pct") or 0, 2),
            "morning_return_pct": round(feat.get("morning_return_pct") or 0, 2),
            "spread_bps": round(feat.get("spread_bps") or 0, 1),
            "wap_vs_close_bps": round(feat.get("wap_vs_close_bps") or 0, 1),
            "foreign_participation_pct": round(feat.get("foreign_participation_pct") or 0, 2),
            "volume_morning_pct": round(feat.get("volume_morning_pct") or 0, 3),
            "session_snapshots": int(feat.get("session_snapshot_count") or 0),
            "_raw_score": raw_score,
        })

    if not rows:
        return pd.DataFrame()

    result = pd.DataFrame(rows)

    # Normalise raw_score to [0, 1]
    lo, hi = result["_raw_score"].min(), result["_raw_score"].max()
    if hi > lo:
        result["intraday_score"] = (result["_raw_score"] - lo) / (hi - lo)
    else:
        result["intraday_score"] = 0.5
    result = result.drop(columns=["_raw_score"])
    result = result.sort_values("intraday_score", ascending=False).reset_index(drop=True)
    result.insert(0, "rank", range(1, len(result) + 1))

    if verbose:
        _print_intraday_table(result, as_of, snapshot_time)

    _save_intraday_report(result, as_of)
    return result


# ---------------------------------------------------------------------------
# Display and persistence
# ---------------------------------------------------------------------------

def _print_intraday_table(
    result: pd.DataFrame, as_of: date, snapshot_time: pd.Timestamp
) -> None:
    console.print(
        f"\n[bold cyan]INTRADAY SIGNAL — {as_of} "
        f"(as of {snapshot_time.strftime('%H:%M')} EAT)[/bold cyan]\n"
    )

    table = Table(show_header=True, header_style="bold cyan", border_style="dim")
    table.add_column("#", width=3)
    table.add_column("Ticker", width=7)
    table.add_column("Score", width=7)
    table.add_column("Base", width=7)
    table.add_column("OB Imbal", width=9)
    table.add_column("Fgn Flow%", width=10)
    table.add_column("Sess Ret%", width=10)
    table.add_column("Spread bps", width=11)
    table.add_column("WAP vs Cls", width=11)

    for _, row in result.iterrows():
        score = row["intraday_score"]
        score_color = "green" if score > 0.6 else ("yellow" if score > 0.4 else "red")

        ff = row["foreign_net_flow_pct"]
        ff_color = "green" if ff > 0 else "red"

        sr = row["session_return_pct"]
        sr_color = "green" if sr > 0 else "red"

        table.add_row(
            str(int(row["rank"])),
            row["ticker"],
            f"[{score_color}]{score:.3f}[/{score_color}]",
            f"{row['base_model_score']:.3f}",
            f"{row['bid_ask_imbalance']:+.3f}",
            f"[{ff_color}]{ff:+.1f}%[/{ff_color}]",
            f"[{sr_color}]{sr:+.2f}%[/{sr_color}]",
            f"{row['spread_bps']:.0f}",
            f"{row['wap_vs_close_bps']:+.0f}",
        )

    console.print(table)


def _save_intraday_report(result: pd.DataFrame, as_of: date) -> Path:
    path = REPORTS_DIR / f"intraday_{as_of.isoformat()}.json"
    path.write_text(
        json.dumps(
            {
                "date": as_of.isoformat(),
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "signals": result.to_dict(orient="records"),
            },
            indent=2,
            default=str,
        )
    )
    return path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_clip(v, lo: float = -1.0, hi: float = 1.0) -> float | None:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    return float(np.clip(v, lo, hi))


def _sign_clip(v, scale: float = 5.0) -> float | None:
    """Soft-sign normalisation: maps ±scale% to approximately ±1."""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    return float(np.clip(v / scale, -1.0, 1.0))
