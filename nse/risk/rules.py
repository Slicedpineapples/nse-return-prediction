"""
Rule-based risk engine.

The risk layer is entirely separate from the model and runs AFTER scoring.
It enforces hard constraints that cannot be overridden by model output.

Rules applied (in order)
------------------------
1. Universe filter: only approved tickers.
2. Tradability filter: daily liquidity check must pass.
3. Max positions: no more than config.max_positions names.
4. Max single-name weight: cap at max_single_name_weight.
5. Max order size: no more than max_order_adv_fraction of 20d ADV.
6. Event blackout: skip ambiguous or very recent large-impact announcements.
7. Score threshold: model score must exceed min_score_threshold.
8. Drawdown kill switch: if portfolio is in breach of max_drawdown_stop.
9. Data quality: skip if features are flagged as low quality.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, timedelta

import numpy as np
import pandas as pd

from nse.config import RISK, SEED_UNIVERSE
from nse.features.liquidity_features import compute_tradability

log = logging.getLogger(__name__)

APPROVED_UNIVERSE = set(SEED_UNIVERSE)


@dataclass
class RiskCheckResult:
    ticker: str
    approved: bool
    reasons_blocked: list[str] = field(default_factory=list)


@dataclass
class PortfolioRiskState:
    """Mutable state tracked by the risk engine across rebalances."""
    peak_value: float = 1.0
    current_value: float = 1.0
    kill_switch_triggered: bool = False
    drawdown_stop: float = RISK["max_portfolio_drawdown_stop"]

    def update(self, daily_return: float) -> None:
        self.current_value *= (1 + daily_return)
        self.peak_value = max(self.peak_value, self.current_value)

    @property
    def current_drawdown(self) -> float:
        return (self.current_value - self.peak_value) / self.peak_value

    def check_kill_switch(self) -> bool:
        if self.current_drawdown <= -self.drawdown_stop:
            self.kill_switch_triggered = True
            log.warning(
                "KILL SWITCH triggered: drawdown %.1f%% exceeds limit %.1f%%",
                self.current_drawdown * 100, self.drawdown_stop * 100,
            )
        return self.kill_switch_triggered


class RiskEngine:
    """
    Applies all hard risk rules to a ranked signal list.

    Usage
    -----
        engine = RiskEngine()
        approved = engine.filter_signals(scores_df, features_df, prices_df, as_of)
    """

    def __init__(self) -> None:
        self.portfolio_state = PortfolioRiskState()

    def filter_signals(
        self,
        scores: pd.DataFrame,
        features: pd.DataFrame,
        prices: pd.DataFrame,
        as_of: date,
    ) -> pd.DataFrame:
        """
        Filter and rank the signal list according to all risk rules.

        Parameters
        ----------
        scores   : DataFrame with columns: date, ticker, score
        features : features_daily for today
        prices   : prices_daily for last 60 days
        as_of    : signal date

        Returns
        -------
        Filtered DataFrame with approved tickers, sorted by score descending.
        Also includes a 'risk_notes' column explaining any adjustments.
        """
        if self.portfolio_state.check_kill_switch():
            log.warning("Kill switch active: no trades today")
            return pd.DataFrame(columns=scores.columns)

        today_scores = scores[scores["date"] == pd.Timestamp(as_of)].copy()
        if today_scores.empty:
            return today_scores

        today_features = features[features["date"] == pd.Timestamp(as_of)]

        results = []
        for _, row in today_scores.iterrows():
            check = self._check_ticker(
                ticker=row["ticker"],
                score=row["score"],
                today_features=today_features,
                prices=prices,
                as_of=as_of,
            )
            results.append(check)

        approved_tickers = {r.ticker for r in results if r.approved}
        blocked_notes = {r.ticker: r.reasons_blocked for r in results if not r.approved}

        if blocked_tickers := set(today_scores["ticker"]) - approved_tickers:
            log.info(
                "Risk engine blocked %d tickers: %s",
                len(blocked_tickers),
                {t: blocked_notes.get(t, []) for t in list(blocked_tickers)[:5]},
            )

        approved_df = today_scores[today_scores["ticker"].isin(approved_tickers)].copy()
        approved_df = approved_df.sort_values("score", ascending=False)

        # Apply position cap
        approved_df = approved_df.head(RISK["max_positions"])

        # Apply weight cap (for downstream portfolio construction)
        approved_df["target_weight"] = min(
            1.0 / max(len(approved_df), 1),
            RISK["max_single_name_weight"],
        )

        return approved_df.reset_index(drop=True)

    def _check_ticker(
        self,
        ticker: str,
        score: float,
        today_features: pd.DataFrame,
        prices: pd.DataFrame,
        as_of: date,
    ) -> RiskCheckResult:
        blocked = []

        # 1. Universe filter
        if ticker not in APPROVED_UNIVERSE:
            blocked.append(f"not in approved universe")

        # 2. Score threshold
        if score < RISK["min_score_threshold"]:
            blocked.append(f"score {score:.3f} below threshold {RISK['min_score_threshold']}")

        # 3. Tradability
        if not prices.empty:
            tradable = compute_tradability(prices, as_of)
            if ticker in tradable.index and not tradable[ticker]:
                blocked.append("fails daily tradability filter")

        # 4. Data quality
        if not today_features.empty:
            tkr_feat = today_features[today_features["ticker"] == ticker]
            if not tkr_feat.empty:
                row = tkr_feat.iloc[0]
                if not row.get("dq_features_ok", True):
                    blocked.append("feature data quality flag")
                if row.get("dq_tradable") is False:
                    blocked.append("dq_tradable flag False")
                # Stale price check
                if row.get("stale_price_flag", 0) == 1:
                    blocked.append("stale price flag (likely thin/halted)")
                # Large move on low volume
                if row.get("large_move_low_vol_flag", 0) == 1:
                    blocked.append("large move on low volume (microstructure concern)")

        # 5. Event blackout
        if not today_features.empty:
            tkr_feat = today_features[today_features["ticker"] == ticker]
            if not tkr_feat.empty:
                row = tkr_feat.iloc[0]
                # Block if there is a very recent ambiguous high-materiality event
                mat = row.get("llm_max_materiality_today", 0) or 0
                if mat > 0.8:
                    horizon = row.get("llm_horizon", "3d")
                    if horizon == "1d":
                        blocked.append(
                            f"high-materiality event today (mat={mat:.2f}), horizon=1d: "
                            "market access uncertain"
                        )

        return RiskCheckResult(
            ticker=ticker,
            approved=len(blocked) == 0,
            reasons_blocked=blocked,
        )

    def update_portfolio_value(self, daily_return: float) -> None:
        """Call each day with the realised portfolio return to track drawdown."""
        self.portfolio_state.update(daily_return)


def compute_position_limits(
    tickers: list[str],
    prices: pd.DataFrame,
    as_of: date,
    portfolio_aum: float,
) -> dict[str, float]:
    """
    Compute max order size in KES for each ticker (5% of 20d ADV).
    """
    prices = prices.copy()
    prices["date"] = pd.to_datetime(prices["date"])
    cutoff = pd.Timestamp(as_of)

    limits = {}
    for tkr, grp in prices.groupby("ticker"):
        if tkr not in tickers:
            continue
        recent = grp[grp["date"] <= cutoff].tail(20)
        adv = recent["value_traded"].mean()
        if pd.isna(adv) or adv <= 0:
            limits[tkr] = 0.0
        else:
            limits[tkr] = min(
                adv * RISK["max_daily_turnover_fraction"],
                portfolio_aum * RISK["max_single_name_weight"],
            )

    return limits
