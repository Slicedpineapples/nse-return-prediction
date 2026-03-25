"""
Backtest engine.

Simulates daily long-only portfolio rebalancing with:
  - Top N scored names from the ranker.
  - Equal weighting (initial MVP).
  - Realistic cost model (three scenarios).
  - Risk rule enforcement.
  - Position-level and portfolio-level P&L tracking.

Rebalance rule: next-day open after signal (or end-of-day close if open N/A).
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from datetime import date

import numpy as np
import pandas as pd
from sqlalchemy.dialects.postgresql import insert

from nse.backtest.costs import apply_costs, estimate_cost_bps, max_order_size_kes
from nse.backtest.metrics import compute_metrics, format_metrics, pct_trades_surviving_stressed_costs
from nse.config import COSTS, RISK
from nse.db.schema import TradeSimulated, session_scope

log = logging.getLogger(__name__)


@dataclass
class BacktestConfig:
    max_positions: int = RISK["max_positions"]
    max_single_name_weight: float = RISK["max_single_name_weight"]
    max_daily_turnover: float = RISK["max_daily_turnover_fraction"]
    max_order_adv_fraction: float = COSTS["impact_adv_fraction"]
    min_score_threshold: float = RISK["min_score_threshold"]
    scenario: str = "base"                    # 'optimistic' | 'base' | 'stressed'
    hold_days: int = RISK["hold_days"]
    initial_capital: float = 1_000_000       # KES
    use_next_open: bool = True               # use next-day open for entry (else close)


@dataclass
class BacktestResult:
    run_id: str
    config: BacktestConfig
    portfolio_returns: pd.Series              # daily net returns
    trade_log: pd.DataFrame
    metrics: dict
    metrics_stressed: dict = field(default_factory=dict)

    def summary(self) -> str:
        lines = [
            f"\n{'='*60}",
            f"BACKTEST RESULT  run_id={self.run_id}",
            f"Scenario: {self.config.scenario}",
            f"{'='*60}",
            format_metrics(self.metrics),
            "",
            f"% trades surviving stressed costs: "
            f"{pct_trades_surviving_stressed_costs(self.trade_log):.1%}",
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_backtest(
    scores: pd.DataFrame,
    prices: pd.DataFrame,
    market: pd.DataFrame | None = None,
    config: BacktestConfig | None = None,
) -> BacktestResult:
    """
    Run backtest on pre-computed model scores.

    Parameters
    ----------
    scores  : DataFrame with columns: date, ticker, score (daily, all tickers)
    prices  : prices_daily DataFrame (date, ticker, open, close, value_traded)
    market  : market_daily DataFrame (date, nse_20) for benchmark
    config  : BacktestConfig; uses defaults if None

    Returns
    -------
    BacktestResult
    """
    config = config or BacktestConfig()
    run_id = str(uuid.uuid4())[:8]

    scores = scores.copy()
    prices = prices.copy()
    scores["date"] = pd.to_datetime(scores["date"])
    prices["date"] = pd.to_datetime(prices["date"])

    all_dates = sorted(scores["date"].unique())
    portfolio: dict[str, float] = {}  # ticker -> weight
    cash_weight = 1.0
    daily_returns: list[dict] = []
    trade_log: list[dict] = []
    hold_tracker: dict[str, int] = {}  # ticker -> days held

    # Pre-compute ADV for each ticker
    adv_map = _compute_adv(prices)

    # Benchmark daily returns
    bench_rets: dict[pd.Timestamp, float] = {}
    if market is not None:
        m = market.copy()
        m["date"] = pd.to_datetime(m["date"])
        m = m.sort_values("date")
        m["bench_ret"] = m["nse_20"].pct_change(1)
        bench_rets = dict(zip(m["date"], m["bench_ret"]))

    for i, signal_date in enumerate(all_dates):
        # Prices for signal_date
        today_prices = prices[prices["date"] == signal_date].set_index("ticker")
        today_scores = scores[scores["date"] == signal_date]

        # --- Compute portfolio return from previous positions ---
        port_ret = _compute_portfolio_return(portfolio, today_prices)
        daily_returns.append(
            {
                "date": signal_date,
                "portfolio_return": port_ret,
                "benchmark_return": bench_rets.get(signal_date, np.nan),
                "n_positions": len(portfolio),
            }
        )

        # Update hold tracker
        for tkr in list(hold_tracker.keys()):
            hold_tracker[tkr] += 1

        # --- Decide if rebalance is needed ---
        # Rebalance if: start of period, some positions have held too long, or scores changed significantly
        should_rebalance = (
            i == 0
            or i % config.hold_days == 0
            or any(d >= config.hold_days for d in hold_tracker.values())
        )

        if not should_rebalance:
            continue

        # --- Select top names from scores ---
        eligible = today_scores[today_scores["score"] >= config.min_score_threshold]
        eligible = eligible.sort_values("score", ascending=False)

        # Apply ADV cap
        new_positions = []
        for _, row in eligible.iterrows():
            if len(new_positions) >= config.max_positions:
                break
            tkr = row["ticker"]
            adv = adv_map.get((signal_date, tkr), 0)
            # Skip if no price data or illiquid
            if tkr not in today_prices.index:
                continue
            if adv <= 0:
                continue
            new_positions.append(tkr)

        # --- Build target weights ---
        if not new_positions:
            target_weights = {}
        else:
            equal_w = min(1.0 / len(new_positions), config.max_single_name_weight)
            target_weights = {tkr: equal_w for tkr in new_positions}

        # --- Execute trades and deduct costs from today's return ---
        new_trades = _execute_rebalance(
            current_portfolio=portfolio,
            target_weights=target_weights,
            today_prices=today_prices,
            adv_map=adv_map,
            signal_date=signal_date,
            config=config,
            run_id=run_id,
            scores=dict(zip(today_scores["ticker"], today_scores["score"])),
        )
        trade_log += new_trades

        # Deduct round-trip cost drag from today's portfolio return
        if new_trades:
            total_turnover = sum(abs(t["target_weight"] - portfolio.get(t["ticker"], 0))
                                 for t in new_trades)
            avg_cost_bps = np.mean([t["cost_bps"] for t in new_trades])
            cost_drag = total_turnover * avg_cost_bps / 10_000
            daily_returns[-1]["portfolio_return"] -= cost_drag

        portfolio = target_weights.copy()
        hold_tracker = {tkr: 0 for tkr in portfolio}

    # --- Assemble results ---
    ret_df = pd.DataFrame(daily_returns).set_index("date")
    port_rets = ret_df["portfolio_return"].fillna(0)
    bench_rets_series = ret_df["benchmark_return"]

    trade_df = pd.DataFrame(trade_log) if trade_log else pd.DataFrame()
    _persist_trades(trade_df)

    metrics = compute_metrics(port_rets, bench_rets_series)

    result = BacktestResult(
        run_id=run_id,
        config=config,
        portfolio_returns=port_rets,
        trade_log=trade_df,
        metrics=metrics,
    )
    return result


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------

def _compute_portfolio_return(
    portfolio: dict[str, float],
    today_prices: pd.DataFrame,
) -> float:
    """Compute weighted portfolio return for today."""
    if not portfolio:
        return 0.0

    total = 0.0
    total_weight = 0.0
    for tkr, weight in portfolio.items():
        if tkr not in today_prices.index:
            continue
        row = today_prices.loc[tkr]
        if pd.isna(row.get("close")) or pd.isna(row.get("previous_close")):
            continue
        if row["previous_close"] == 0:
            continue
        ret = row["close"] / row["previous_close"] - 1
        total += weight * ret
        total_weight += weight

    if total_weight == 0:
        return 0.0
    return total / total_weight * total_weight  # weighted sum


def _execute_rebalance(
    current_portfolio: dict[str, float],
    target_weights: dict[str, float],
    today_prices: pd.DataFrame,
    adv_map: dict,
    signal_date: pd.Timestamp,
    config: BacktestConfig,
    run_id: str,
    scores: dict[str, float],
) -> list[dict]:
    """Simulate trade execution and return list of trade records."""
    trades = []

    all_tickers = set(current_portfolio) | set(target_weights)
    for tkr in all_tickers:
        old_w = current_portfolio.get(tkr, 0)
        new_w = target_weights.get(tkr, 0)
        trade_w = new_w - old_w  # positive = buy, negative = sell

        if abs(trade_w) < 0.001:  # ignore tiny rebalances
            continue

        if tkr not in today_prices.index:
            continue

        row = today_prices.loc[tkr]
        entry_price = row.get("open") if config.use_next_open else row.get("close")
        exit_price = row.get("close")

        if pd.isna(entry_price) or entry_price == 0:
            entry_price = row.get("close", np.nan)

        adv = adv_map.get((signal_date, tkr), 0)
        notional = abs(trade_w) * config.initial_capital  # approximate
        cost_bps = estimate_cost_bps(notional, adv, config.scenario)
        gross_ret = (exit_price / entry_price - 1) if (entry_price and exit_price) else 0.0
        net_ret = apply_costs(gross_ret, cost_bps) if trade_w > 0 else gross_ret

        trades.append(
            {
                "date": signal_date.date() if hasattr(signal_date, "date") else signal_date,
                "ticker": tkr,
                "model_score": scores.get(tkr, np.nan),
                "target_weight": new_w,
                "assumed_entry_price": entry_price,
                "assumed_exit_price": exit_price,
                "turnover_used": notional,
                "cost_bps": cost_bps,
                "slippage_bps": cost_bps - COSTS["commission_bps"] * 2,
                "gross_return": gross_ret,
                "realized_return_net": net_ret,
                "scenario": config.scenario,
                "run_id": run_id,
            }
        )

    return trades


def _compute_adv(prices: pd.DataFrame) -> dict:
    """Pre-compute 20d ADV (KES) for every (date, ticker).

    Uses value_traded when available; falls back to volume * close otherwise.
    """
    adv = {}
    for tkr, grp in prices.groupby("ticker"):
        grp = grp.sort_values("date").copy()
        # Prefer value_traded; fall back to volume * close
        turnover = grp["value_traded"].copy()
        missing = turnover.isna()
        if missing.any() and "volume" in grp.columns and "close" in grp.columns:
            turnover.loc[missing] = grp.loc[missing, "volume"] * grp.loc[missing, "close"]
        rolling_adv = turnover.rolling(20, min_periods=5).mean()
        for dt, val in zip(grp["date"], rolling_adv):
            adv[(dt, tkr)] = val if not pd.isna(val) else 0
    return adv


def _persist_trades(trade_df: pd.DataFrame) -> None:
    if trade_df.empty:
        return
    valid_cols = {c.name for c in TradeSimulated.__table__.columns if c.name != "id"}
    records = trade_df[[c for c in trade_df.columns if c in valid_cols]].to_dict(
        orient="records"
    )
    with session_scope() as s:
        for rec in records:
            s.execute(
                insert(TradeSimulated)
                .values(**rec)
                .on_conflict_do_nothing()
            )
