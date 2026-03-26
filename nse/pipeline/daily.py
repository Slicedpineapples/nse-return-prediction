"""
Daily batch pipeline.

End-of-day / morning workflow (from spec section 9):
  1. Ingest latest NSE prices, turnover, deals, index values.
  2. Ingest new NSE announcements and news.
  3. Run LLM extraction on new text only.
  4. Generate features_daily.
  5. Score all tradable names.
  6. Apply hard risk filters.
  7. Produce ranked trade sheet.
  8. Human reviews and approves.  (This module stops here; approval is external.)
  9. Log outcomes for monitoring.

Output: a ranked signal report saved to reports/output/ and printed to stdout.
"""

from __future__ import annotations

import json
import logging
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
from rich.console import Console
from rich.table import Table

from nse.config import REPORTS_DIR, RISK, SEED_UNIVERSE
from nse.db import init_db
from nse.features.pipeline import load_features, run_feature_pipeline
from nse.ingest.announcements_live import ingest_announcements_from_db, load_events
from nse.ingest.macro import ingest_macro
from nse.ingest.market_live import ingest_market_from_stocksdata
from nse.ingest.prices_live import ingest_prices_from_stocksdata, load_prices
from nse.llm.extractor import generate_trade_explanation, process_batch
from nse.models.ranker import NSERanker
from nse.monitoring.ledger import backfill_actuals, save_signals
from nse.risk.rules import RiskEngine, compute_position_limits

log = logging.getLogger(__name__)
console = Console()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_daily_pipeline(
    as_of: date | None = None,
    model_label: str = "production",
    skip_ingest: bool = False,
    skip_llm: bool = False,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Run the full daily pipeline for a given date.

    Parameters
    ----------
    as_of       : Signal date. Defaults to today.
    model_label : Which saved model to load ('production' or a slice ID).
    skip_ingest : Skip data download (useful if already ingested today).
    skip_llm    : Skip LLM extraction batch (API cost saving).
    verbose     : Print rich output to console.

    Returns
    -------
    Ranked signal DataFrame (approved names only, highest score first).
    """
    as_of = as_of or date.today()
    init_db()

    if verbose:
        console.rule(f"[bold blue]NSE Daily Pipeline — {as_of}")

    # Backfill outcomes for signals from ~3 days ago
    try:
        n_backfilled = backfill_actuals(as_of=as_of)
        if verbose and n_backfilled:
            console.print(f"[dim]Backfilled actuals for {n_backfilled} past signals[/dim]")
    except Exception as exc:
        log.warning("Actuals backfill failed (non-fatal): %s", exc)

    # Step 1: Ingest prices, market stats, macro
    if not skip_ingest:
        _step("1/7 Ingesting prices", verbose)
        ingest_prices_from_stocksdata(as_of, as_of)
        ingest_market_from_stocksdata(as_of, as_of)
        ingest_macro(as_of, as_of)

    # Step 2: Ingest announcements
    if not skip_ingest:
        _step("2/7 Ingesting announcements", verbose)
        ingest_announcements_from_db(as_of - timedelta(days=7), as_of)

    # Step 3: LLM extraction
    if not skip_llm:
        _step("3/7 Running LLM extraction", verbose)
        n_processed = process_batch(batch_size=50)
        if verbose:
            console.print(f"   Processed {n_processed} events with LLM")

    # Step 4: Feature pipeline
    _step("4/7 Building features", verbose)
    features = run_feature_pipeline(
        start=as_of,
        end=as_of,
        tickers=SEED_UNIVERSE,
    )

    if features.empty:
        console.print("[red]No features generated for today. Check data ingestion.[/red]")
        return pd.DataFrame()

    # Step 5: Score all tradable names
    _step("5/7 Scoring with model", verbose)
    try:
        ranker = NSERanker.load(model_label)
    except FileNotFoundError:
        log.warning("No saved model found for '%s'. Run nse-train first.", model_label)
        console.print(f"[yellow]No model '{model_label}' found. Run `nse-train` first.[/yellow]")
        return pd.DataFrame()

    today_features = features[features["date"] == pd.Timestamp(as_of)]
    scores = ranker.predict(today_features)

    # Step 6: Apply risk filters
    _step("6/7 Applying risk filters", verbose)
    prices_60d = load_prices(
        tickers=SEED_UNIVERSE,
        start=as_of - timedelta(days=90),
        end=as_of,
    )

    engine = RiskEngine()
    approved = engine.filter_signals(
        scores=scores,
        features=features,
        prices=prices_60d,
        as_of=as_of,
    )

    if approved.empty:
        console.print("[yellow]Risk engine approved zero names today.[/yellow]")
        return pd.DataFrame()

    # Step 7: Build signal report with explanations
    _step("7/7 Building signal report", verbose)
    report = _build_signal_report(
        approved=approved,
        features=today_features,
        events=load_events(
            tickers=list(approved["ticker"]),
            start=as_of - timedelta(days=5),
            end=as_of,
            llm_only=True,
        ),
        prices=prices_60d,
        as_of=as_of,
        skip_llm=skip_llm,
    )

    if verbose:
        _print_signal_table(report, as_of)

    # Save report
    out_path = _save_report(report, as_of)
    if verbose:
        console.print(f"\n[green]Report saved: {out_path}[/green]")

    # Persist signals to prediction ledger
    try:
        ledger_df = report.copy()
        ledger_df["approved"] = ledger_df["recommended_action"] == "LONG"
        ledger_df["predicted_edge_bps"] = ledger_df["expected_edge_after_costs_bps"]
        save_signals(signal_date=as_of, ranked=ledger_df, model_label=model_label)
    except Exception as exc:
        log.warning("Signal ledger save failed (non-fatal): %s", exc)

    return report


# ---------------------------------------------------------------------------
# Report building
# ---------------------------------------------------------------------------

def _build_signal_report(
    approved: pd.DataFrame,
    features: pd.DataFrame,
    events: pd.DataFrame,
    prices: pd.DataFrame,
    as_of: date,
    skip_llm: bool = False,
) -> pd.DataFrame:
    """Build the ranked signal report with explanations and cost estimates."""
    from nse.backtest.costs import estimate_cost_bps

    rows = []
    for _, sig in approved.iterrows():
        ticker = sig["ticker"]
        score = sig["score"]
        top_prob = sig.get("top_bucket_prob", None)

        # Feature values for explanation
        tkr_feat = features[features["ticker"] == ticker]
        feat_dict = tkr_feat.iloc[0].to_dict() if not tkr_feat.empty else {}

        # Recent LLM summaries
        tkr_events = events[events["ticker"] == ticker] if not events.empty else pd.DataFrame()
        event_summaries = []
        if not tkr_events.empty:
            for _, ev in tkr_events.head(3).iterrows():
                if ev.get("llm_summary"):
                    event_summaries.append(
                        f"[{ev.get('event_type','?')}] {ev['llm_summary']}"
                    )

        # Cost estimate (base scenario)
        recent_prices = prices[prices["ticker"] == ticker].tail(20)
        adv_20d = recent_prices["value_traded"].mean() if not recent_prices.empty else 0
        cost_bps = estimate_cost_bps(adv_20d * 0.05, adv_20d, scenario="base")

        # Expected net edge (very rough: model's score as proxy for signal strength)
        expected_edge_bps = (score - 0.5) * 1000 - cost_bps  # illustrative

        # Explanation
        if not skip_llm and score > RISK["min_score_threshold"]:
            explanation = generate_trade_explanation(ticker, score, feat_dict, event_summaries)
        else:
            explanation = f"Score {score:.3f}. Events: {'; '.join(event_summaries) or 'None'}"

        rows.append(
            {
                "rank": len(rows) + 1,
                "ticker": ticker,
                "score": round(score, 4),
                "top_bucket_prob": round(top_prob, 3) if top_prob and not pd.isna(top_prob) else None,
                "expected_edge_after_costs_bps": round(expected_edge_bps, 1),
                "cost_bps_estimate": round(cost_bps, 1),
                "recommended_action": "LONG" if score >= RISK["min_score_threshold"] else "NO TRADE",
                "ret_3d": feat_dict.get("ret_3d"),
                "vol_20d": feat_dict.get("vol_20d"),
                "llm_sentiment": feat_dict.get("llm_weighted_sentiment_3d"),
                "llm_materiality": feat_dict.get("llm_max_materiality_today"),
                "recent_events": " | ".join(event_summaries[:2]),
                "explanation": explanation,
            }
        )

    return pd.DataFrame(rows)


def _print_signal_table(report: pd.DataFrame, as_of: date) -> None:
    console.print(f"\n[bold green]DAILY SIGNAL REPORT — {as_of}[/bold green]\n")

    table = Table(show_header=True, header_style="bold cyan", border_style="dim")
    table.add_column("Rank", width=5)
    table.add_column("Ticker", width=8)
    table.add_column("Score", width=8)
    table.add_column("Edge (bps)", width=10)
    table.add_column("Action", width=10)
    table.add_column("3d Ret", width=8)
    table.add_column("Vol 20d", width=8)
    table.add_column("LLM Sent.", width=10)

    for _, row in report.iterrows():
        edge = row["expected_edge_after_costs_bps"]
        edge_color = "green" if edge > 0 else "red"
        table.add_row(
            str(row["rank"]),
            str(row["ticker"]),
            f"{row['score']:.3f}",
            f"[{edge_color}]{edge:.0f}[/{edge_color}]",
            f"[bold]{row['recommended_action']}[/bold]",
            f"{row['ret_3d']:.1%}" if row.get("ret_3d") else "N/A",
            f"{row['vol_20d']:.3f}" if row.get("vol_20d") else "N/A",
            f"{row['llm_sentiment']:.2f}" if row.get("llm_sentiment") else "N/A",
        )

    console.print(table)

    console.print("\n[bold]Explanations[/bold]")
    for _, row in report.iterrows():
        console.print(f"\n[cyan]{row['ticker']}[/cyan]: {row['explanation']}")


def _save_report(report: pd.DataFrame, as_of: date) -> Path:
    path = REPORTS_DIR / f"signal_{as_of.isoformat()}.csv"
    report.to_csv(path, index=False)

    # Also save JSON for downstream use
    json_path = REPORTS_DIR / f"signal_{as_of.isoformat()}.json"
    json_path.write_text(report.to_json(orient="records", indent=2))

    return path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _step(msg: str, verbose: bool) -> None:
    if verbose:
        console.print(f"[blue]{msg}[/blue]")
    log.info(msg)
