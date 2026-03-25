"""
Backtest performance metrics.

All metrics computed from the equity curve (daily portfolio returns series).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_metrics(
    returns: pd.Series,
    benchmark_returns: pd.Series | None = None,
    periods_per_year: int = 252,
) -> dict:
    """
    Compute comprehensive backtest performance metrics.

    Parameters
    ----------
    returns          : daily net portfolio returns (fraction, not pct)
    benchmark_returns: daily benchmark returns (NSE 20)
    periods_per_year : trading days per year (252 for NSE)

    Returns
    -------
    Dict of metric name → value.
    """
    rets = returns.dropna()
    if rets.empty:
        return {}

    equity = (1 + rets).cumprod()
    total_return = equity.iloc[-1] - 1
    ann_return = (1 + total_return) ** (periods_per_year / len(rets)) - 1

    vol = rets.std() * np.sqrt(periods_per_year)
    sharpe = ann_return / (vol + 1e-10)

    # Drawdown
    running_max = equity.cummax()
    drawdown = (equity - running_max) / running_max
    max_drawdown = drawdown.min()

    # Hit rate
    hit_rate = (rets > 0).mean()
    avg_gain = rets[rets > 0].mean() if (rets > 0).any() else 0.0
    avg_loss = rets[rets < 0].mean() if (rets < 0).any() else 0.0
    gain_loss_ratio = abs(avg_gain / (avg_loss + 1e-10))

    # Calmar
    calmar = ann_return / (abs(max_drawdown) + 1e-10)

    metrics = {
        "total_return": total_return,
        "annualised_return": ann_return,
        "annualised_volatility": vol,
        "sharpe_ratio": sharpe,
        "calmar_ratio": calmar,
        "max_drawdown": max_drawdown,
        "hit_rate": hit_rate,
        "avg_gain": avg_gain,
        "avg_loss": avg_loss,
        "gain_loss_ratio": gain_loss_ratio,
        "n_days": len(rets),
    }

    # Benchmark-relative
    if benchmark_returns is not None:
        bench = benchmark_returns.reindex(rets.index).dropna()
        if not bench.empty:
            active_rets = rets.reindex(bench.index) - bench
            tracking_error = active_rets.std() * np.sqrt(periods_per_year)
            information_ratio = active_rets.mean() * periods_per_year / (tracking_error + 1e-10)
            bench_total = (1 + bench).prod() - 1
            metrics["benchmark_total_return"] = bench_total
            metrics["excess_return"] = total_return - bench_total
            metrics["tracking_error"] = tracking_error
            metrics["information_ratio"] = information_ratio

    return metrics


def pct_trades_surviving_stressed_costs(
    trades: pd.DataFrame,
    stressed_cost_bps: float = 600,
) -> float:
    """
    What fraction of simulated trades remain profitable under stressed costs?
    """
    if trades.empty or "gross_return" not in trades.columns:
        return np.nan
    gross_bps = trades["gross_return"] * 10_000
    return (gross_bps > stressed_cost_bps).mean()


def format_metrics(metrics: dict) -> str:
    """Format metrics dict as readable text table."""
    lines = [f"{'Metric':<35} {'Value':>15}"]
    lines.append("-" * 52)
    fmt_map = {
        "total_return": "{:.1%}",
        "annualised_return": "{:.1%}",
        "annualised_volatility": "{:.1%}",
        "sharpe_ratio": "{:.2f}",
        "calmar_ratio": "{:.2f}",
        "max_drawdown": "{:.1%}",
        "hit_rate": "{:.1%}",
        "avg_gain": "{:.3%}",
        "avg_loss": "{:.3%}",
        "gain_loss_ratio": "{:.2f}",
        "information_ratio": "{:.2f}",
        "tracking_error": "{:.1%}",
        "excess_return": "{:.1%}",
    }
    for k, v in metrics.items():
        if v is None or (isinstance(v, float) and np.isnan(v)):
            formatted = "N/A"
        elif k in fmt_map:
            try:
                formatted = fmt_map[k].format(v)
            except Exception:
                formatted = str(v)
        elif isinstance(v, int):
            formatted = str(v)
        else:
            formatted = f"{v:.4f}"
        lines.append(f"{k:<35} {formatted:>15}")
    return "\n".join(lines)
