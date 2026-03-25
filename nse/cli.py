"""
CLI entry points.

Commands:
  nse-ingest    Download and store NSE prices, market stats, announcements, macro.
  nse-features  Build features_daily for a date range.
  nse-train     Walk-forward training; produces production model.
  nse-backtest  Run backtest on historical features + scores.
  nse-signal    Run daily pipeline and output signal report.
  nse-pipeline  Full end-to-end daily batch (ingest → signal).
"""

from __future__ import annotations

import logging
import sys
from datetime import date, datetime

import click
from rich.console import Console
from rich.logging import RichHandler

from nse.db import init_db

console = Console()


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(message)s",
        handlers=[RichHandler(console=console, rich_tracebacks=True)],
    )


def _parse_date(ctx, param, value: str) -> date:
    if value is None:
        return date.today()
    try:
        return datetime.strptime(value, "%Y-%m-%d").date()
    except ValueError:
        raise click.BadParameter(f"Expected YYYY-MM-DD, got: {value}")


# ---------------------------------------------------------------------------
# ingest
# ---------------------------------------------------------------------------

@click.command()
@click.option("--start", default=None, help="Start date YYYY-MM-DD (default: 60 days ago)")
@click.option("--end", default=None, help="End date YYYY-MM-DD (default: today)")
@click.option("--force", is_flag=True, help="Re-download even if cached")
@click.option("--verbose", "-v", is_flag=True)
def ingest(start: str | None, end: str | None, force: bool, verbose: bool) -> None:
    """Download and store NSE prices, market stats, announcements, and macro data."""
    _setup_logging(verbose)
    init_db()

    from datetime import timedelta

    end_date = _parse_date(None, None, end)
    start_date = _parse_date(None, None, start) if start else end_date - timedelta(days=60)

    console.print(f"Ingesting data from {start_date} to {end_date}")

    from nse.ingest.prices_live import ingest_prices_from_stocksdata
    from nse.ingest.market_live import ingest_market_from_stocksdata
    from nse.ingest.announcements_live import ingest_announcements_from_db
    from nse.ingest.macro import ingest_macro

    n = ingest_prices_from_stocksdata(start_date, end_date, force=force)
    console.print(f"[green]Prices: {n} rows[/green]")
    n = ingest_market_from_stocksdata(start_date, end_date, force=force)
    console.print(f"[green]Market: {n} rows[/green]")
    n = ingest_announcements_from_db(start_date, end_date, force=force)
    console.print(f"[green]Announcements: {n} rows[/green]")
    n = ingest_macro(start_date, end_date, force=force)
    console.print(f"[green]Macro: {n} rows[/green]")


# ---------------------------------------------------------------------------
# features
# ---------------------------------------------------------------------------

@click.command()
@click.option("--start", required=True, help="Start date YYYY-MM-DD")
@click.option("--end", default=None, help="End date YYYY-MM-DD (default: today)")
@click.option("--force", is_flag=True)
@click.option("--verbose", "-v", is_flag=True)
def features(start: str, end: str | None, force: bool, verbose: bool) -> None:
    """Build features_daily for a date range."""
    _setup_logging(verbose)
    init_db()

    from nse.features.pipeline import run_feature_pipeline
    from nse.labels.returns import compute_and_store_labels

    start_date = _parse_date(None, None, start)
    end_date = _parse_date(None, None, end)

    console.print(f"Building features {start_date} to {end_date}")
    df = run_feature_pipeline(start_date, end_date, force=force)
    console.print(f"[green]Generated {len(df)} feature rows[/green]")

    console.print("Computing labels...")
    label_df = compute_and_store_labels(start_date, end_date)
    console.print(f"[green]Stored {len(label_df)} label rows[/green]")


# ---------------------------------------------------------------------------
# train
# ---------------------------------------------------------------------------

@click.command()
@click.option("--n-months", default=12, help="Walk-forward months to evaluate")
@click.option("--prod-only", is_flag=True, help="Skip walk-forward; train production model only")
@click.option("--verbose", "-v", is_flag=True)
def train(n_months: int, prod_only: bool, verbose: bool) -> None:
    """Walk-forward training; saves model to models/saved/."""
    _setup_logging(verbose)
    init_db()

    from nse.models.walk_forward import run_walk_forward, train_production_model
    from nse.features.pipeline import load_features

    console.print("Loading features from DB...")
    feat = load_features(tradable_only=False)
    console.print(f"  {len(feat)} rows loaded")

    if not prod_only:
        console.print(f"Running {n_months}-month walk-forward evaluation...")
        results = run_walk_forward(features=feat, n_months=n_months)
        _print_walk_forward_summary(results)

    console.print("Training production model...")
    ranker = train_production_model(features=feat)
    console.print(f"[green]Production model saved.[/green]")

    # Feature importance
    fi = ranker.feature_importance().head(15)
    console.print("\nTop 15 features:")
    console.print(fi.to_string(index=False))


# ---------------------------------------------------------------------------
# backtest
# ---------------------------------------------------------------------------

@click.command()
@click.option("--start", required=True, help="Backtest start date YYYY-MM-DD")
@click.option("--end", default=None, help="Backtest end date YYYY-MM-DD")
@click.option("--model", "model_label", default="production", help="Model label to use")
@click.option("--scenario", default="base",
              type=click.Choice(["optimistic", "base", "stressed"]))
@click.option("--verbose", "-v", is_flag=True)
def backtest(
    start: str, end: str | None, model_label: str, scenario: str, verbose: bool
) -> None:
    """Run backtest on historical data and print performance metrics."""
    _setup_logging(verbose)
    init_db()

    from nse.backtest.engine import BacktestConfig, run_backtest
    from nse.features.pipeline import load_features
    from nse.ingest.prices_live import load_prices
    from nse.ingest.market_live import load_market
    from nse.models.ranker import NSERanker
    from nse.config import SEED_UNIVERSE

    start_date = _parse_date(None, None, start)
    end_date = _parse_date(None, None, end)

    console.print(f"Loading model '{model_label}'...")
    try:
        ranker = NSERanker.load(model_label)
    except FileNotFoundError:
        console.print(f"[red]Model '{model_label}' not found. Run nse-train first.[/red]")
        sys.exit(1)

    console.print("Loading features and prices...")
    feat = load_features(start=start_date, end=end_date)
    prices = load_prices(tickers=SEED_UNIVERSE, start=start_date, end=end_date)
    market = load_market(start=start_date, end=end_date)

    if feat.empty or prices.empty:
        console.print("[red]No data available for the specified period.[/red]")
        sys.exit(1)

    console.print("Scoring...")
    scores = ranker.predict(feat)

    config = BacktestConfig(scenario=scenario)
    console.print(f"Running backtest (scenario={scenario})...")
    result = run_backtest(scores, prices, market, config)

    console.print(result.summary())


# ---------------------------------------------------------------------------
# signal
# ---------------------------------------------------------------------------

@click.command()
@click.option("--date", "as_of_str", default=None, help="Signal date YYYY-MM-DD (default: today)")
@click.option("--model", "model_label", default="production")
@click.option("--skip-ingest", is_flag=True)
@click.option("--skip-llm", is_flag=True, help="Skip LLM extraction (saves API cost)")
@click.option("--verbose", "-v", is_flag=True, default=True)
def signal(
    as_of_str: str | None,
    model_label: str,
    skip_ingest: bool,
    skip_llm: bool,
    verbose: bool,
) -> None:
    """Score today's universe and output ranked signal report."""
    _setup_logging(verbose)
    init_db()

    from nse.pipeline.daily import run_daily_pipeline

    as_of = _parse_date(None, None, as_of_str)
    run_daily_pipeline(
        as_of=as_of,
        model_label=model_label,
        skip_ingest=skip_ingest,
        skip_llm=skip_llm,
        verbose=verbose,
    )


# ---------------------------------------------------------------------------
# pipeline (full end-to-end)
# ---------------------------------------------------------------------------

@click.command()
@click.option("--date", "as_of_str", default=None)
@click.option("--model", "model_label", default="production")
@click.option("--skip-ingest", is_flag=True, help="Skip data ingestion (use existing DB data)")
@click.option("--skip-llm", is_flag=True, help="Skip LLM extraction (saves API cost)")
@click.option("--verbose", "-v", is_flag=True, default=True)
def pipeline(
    as_of_str: str | None,
    model_label: str,
    skip_ingest: bool,
    skip_llm: bool,
    verbose: bool,
) -> None:
    """Full daily batch: ingest → LLM → features → score → risk → report."""
    _setup_logging(verbose)
    init_db()

    from nse.pipeline.daily import run_daily_pipeline

    as_of = _parse_date(None, None, as_of_str)
    run_daily_pipeline(
        as_of=as_of,
        model_label=model_label,
        skip_ingest=skip_ingest,
        skip_llm=skip_llm,
        verbose=verbose,
    )


# ---------------------------------------------------------------------------
# Walk-forward summary helper
# ---------------------------------------------------------------------------

def _print_walk_forward_summary(results: list) -> None:
    import numpy as np

    if not results:
        console.print("[yellow]No walk-forward results[/yellow]")
        return

    console.print("\n[bold]Walk-Forward Results[/bold]")
    console.print(f"{'Period':<12} {'val_IC':>8} {'test_IC':>9} {'test_hit':>10} {'n_train':>8}")
    console.print("-" * 52)
    for r in results:
        console.print(
            f"{r.slice_id:<12}"
            f" {r.val_ic:>8.3f}"
            f" {r.test_ic:>9.3f}"
            f" {r.test_hit_rate:>9.1%}"
            f" {r.n_train:>8,}"
        )

    val_ics = [r.val_ic for r in results if not np.isnan(r.val_ic)]
    test_ics = [r.test_ic for r in results if not np.isnan(r.test_ic)]
    if val_ics:
        console.print(
            f"\n[green]Mean val IC: {np.mean(val_ics):.3f} | "
            f"Mean test IC: {np.mean(test_ics):.3f}[/green]"
        )
