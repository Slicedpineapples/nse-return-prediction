"""
CLI entry points.

Commands:
  nse-ingest    Download and store NSE prices, market stats, announcements, macro.
  nse-features  Build features_daily for a date range.
  nse-train     Walk-forward training; produces production model.
  nse-backtest  Run backtest on historical features + scores.
  nse-signal    Run daily pipeline and output signal report.
  nse-pipeline  Full end-to-end daily batch (ingest → signal).
  nse-monitor   IC/hit-rate monitoring and feature drift report.
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
@click.option("--recency-half-life", "recency_half_life", default=None, type=int,
              help="Exponential sample weight half-life in days (e.g. 90). Default: no weighting.")
@click.option("--verbose", "-v", is_flag=True)
def train(n_months: int, prod_only: bool, recency_half_life: int | None, verbose: bool) -> None:
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

    if recency_half_life:
        console.print(f"Training production model (recency half-life={recency_half_life}d)...")
    else:
        console.print("Training production model...")
    ranker = train_production_model(features=feat, recency_half_life_days=recency_half_life)
    console.print("[green]Production model saved.[/green]")

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
# monitor
# ---------------------------------------------------------------------------

@click.command()
@click.option("--start", default=None, help="Start date YYYY-MM-DD (default: 1 year ago)")
@click.option("--end", default=None, help="End date YYYY-MM-DD (default: today)")
@click.option("--window", default=20, help="Rolling IC window in trading days (default: 20)")
@click.option("--backfill", is_flag=True, help="Run actuals backfill before reporting")
@click.option("--drift", is_flag=True, help="Also run PSI feature drift check")
@click.option("--verbose", "-v", is_flag=True)
def monitor(
    start: str | None,
    end: str | None,
    window: int,
    backfill: bool,
    drift: bool,
    verbose: bool,
) -> None:
    """Live IC monitoring, hit rate, and optional drift check."""
    _setup_logging(verbose)
    init_db()

    from datetime import timedelta

    from nse.monitoring.ledger import backfill_actuals
    from nse.monitoring.metrics import get_summary_stats

    end_date = _parse_date(None, None, end)
    start_date = _parse_date(None, None, start) if start else end_date - timedelta(days=365)

    if backfill:
        console.print("[blue]Backfilling actuals...[/blue]")
        n = backfill_actuals(as_of=end_date)
        console.print(f"  Updated {n} rows")

    console.print(f"[blue]Computing signal metrics {start_date} → {end_date}[/blue]")
    stats = get_summary_stats(start=start_date, end=end_date, window=window)

    if stats["status"] == "no_data":
        console.print("[yellow]No signal history found. Run nse-pipeline first.[/yellow]")
        return

    if stats["status"] == "insufficient_data":
        console.print(
            f"[yellow]Insufficient data: {stats['n_with_outcomes']} outcomes "
            f"out of {stats['n_signals']} signals.[/yellow]"
        )
        return

    # Summary table
    from rich.table import Table

    table = Table(title=f"Signal Performance  {start_date} → {end_date}", border_style="dim")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", justify="right")

    def _fmt(v, fmt=".4f"):
        return f"{v:{fmt}}" if v is not None else "N/A"

    table.add_row("Signals (total)", str(stats["n_signals"]))
    table.add_row("Outcomes filled", str(stats["n_with_outcomes"]))
    ic_color = "green" if (stats["ic_overall"] or 0) > 0.05 else "red"
    table.add_row("IC (overall)", f"[{ic_color}]{_fmt(stats['ic_overall'])}[/{ic_color}]")
    table.add_row("IC p-value", _fmt(stats["ic_pvalue"]))
    recent_color = "yellow" if stats.get("signal_degraded") else "green"
    table.add_row(f"IC (last {window}d)", f"[{recent_color}]{_fmt(stats['ic_recent'])}[/{recent_color}]")
    table.add_row("Hit rate (approved)", _fmt(stats["hit_rate"], ".2%") if stats["hit_rate"] else "N/A")
    table.add_row("Mean alpha (bps)", _fmt(stats["mean_alpha_bps"], ".1f"))
    table.add_row("Mean pred error (bps)", _fmt(stats["mean_pred_error_bps"], ".1f") if stats["mean_pred_error_bps"] is not None else "N/A")

    console.print(table)

    if stats.get("signal_degraded"):
        console.print(
            "[bold red]⚠  Recent IC is <50% of historical — consider retraining.[/bold red]"
        )

    # Score calibration
    cal = stats.get("calibration")
    if cal is not None and not cal.empty:
        console.print("\n[bold]Score Calibration (quintiles)[/bold]")
        console.print(cal.to_string(index=False))

    # Drift check
    if drift:
        console.print("\n[blue]Checking feature drift (PSI)...[/blue]")
        from nse.monitoring.drift import check_feature_drift
        drift_df = check_feature_drift()
        if drift_df.empty:
            console.print("[yellow]Not enough data for drift check[/yellow]")
        else:
            from rich.table import Table as RTable
            dtable = RTable(title="Feature Drift (PSI)", border_style="dim")
            dtable.add_column("Feature", style="cyan")
            dtable.add_column("PSI", justify="right")
            dtable.add_column("Status")
            for _, row in drift_df.iterrows():
                status_color = {"stable": "green", "warning": "yellow", "alert": "bold red"}.get(
                    row["status"], "white"
                )
                dtable.add_row(
                    row["feature"],
                    f"{row['psi']:.4f}" if row["psi"] is not None else "N/A",
                    f"[{status_color}]{row['status']}[/{status_color}]",
                )
            console.print(dtable)


# ---------------------------------------------------------------------------
# intraday
# ---------------------------------------------------------------------------

@click.command()
@click.option("--date", "as_of_str", default=None, help="Session date YYYY-MM-DD (default: today)")
@click.option("--model", "model_label", default="production")
@click.option("--no-daily-score", "no_daily_score", is_flag=True,
              help="Rank purely on intraday signals; skip base model score")
@click.option("--ingest", is_flag=True,
              help="Pull latest snapshots from live_stock_data before scoring")
@click.option("--verbose", "-v", is_flag=True, default=True)
def intraday(
    as_of_str: str | None,
    model_label: str,
    no_daily_score: bool,
    ingest: bool,
    verbose: bool,
) -> None:
    """Score the current session using intraday order book and flow data."""
    _setup_logging(verbose)
    init_db()

    from nse.pipeline.intraday import run_intraday_pipeline

    as_of = _parse_date(None, None, as_of_str)

    if ingest:
        from nse.ingest.live_prices import ingest_live_prices
        console.print(f"[blue]Ingesting live snapshots for {as_of}...[/blue]")
        n_daily, n_intra = ingest_live_prices(as_of, as_of, intraday=True)
        console.print(f"[green]Ingested: {n_daily} EOD rows, {n_intra} intraday snapshots[/green]")

    run_intraday_pipeline(
        as_of=as_of,
        model_label=model_label,
        use_daily_score=not no_daily_score,
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
