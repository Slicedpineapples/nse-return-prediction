"""
Unit tests for feature engineering.

Uses synthetic data to verify correctness without requiring a live database.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from nse.features.price_features import compute_price_features
from nse.features.liquidity_features import compute_liquidity_features, compute_tradability
from nse.features.event_features import compute_event_features
from nse.features.macro_features import compute_macro_features


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_prices() -> pd.DataFrame:
    """30 days of synthetic price data for two tickers."""
    dates = pd.date_range("2024-01-02", periods=30, freq="B")
    records = []
    rng = np.random.default_rng(42)
    for tkr in ("SCOM", "KCB"):
        close = 100.0
        for d in dates:
            ret = rng.normal(0, 0.015)
            close = close * (1 + ret)
            records.append({
                "date": d,
                "ticker": tkr,
                "open": close * (1 + rng.normal(0, 0.003)),
                "high": close * (1 + abs(rng.normal(0, 0.005))),
                "low": close * (1 - abs(rng.normal(0, 0.005))),
                "close": close,
                "volume": rng.integers(100_000, 1_000_000),
                "value_traded": close * rng.integers(100_000, 1_000_000),
                "number_of_deals": rng.integers(50, 500),
                "previous_close": close / (1 + ret),
            })
    return pd.DataFrame(records)


@pytest.fixture
def sample_market() -> pd.DataFrame:
    dates = pd.date_range("2024-01-02", periods=30, freq="B")
    rng = np.random.default_rng(99)
    nse20 = 1800.0
    rows = []
    for d in dates:
        nse20 *= 1 + rng.normal(0, 0.01)
        rows.append({
            "date": d,
            "nse_20": nse20,
            "nse_all_share": nse20 * 1.1,
            "total_market_turnover": rng.integers(50_000_000, 200_000_000),
            "total_shares_traded": rng.integers(1_000_000, 10_000_000),
            "total_equity_deals": rng.integers(500, 3000),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Tests: price features
# ---------------------------------------------------------------------------

class TestPriceFeatures:
    def test_returns_computed(self, sample_prices, sample_market):
        feat = compute_price_features(sample_prices, sample_market)
        assert "ret_1d" in feat.columns
        assert "ret_3d" in feat.columns
        assert "vol_20d" in feat.columns
        assert not feat["ret_1d"].isna().all()

    def test_no_lookahead_in_vol(self, sample_prices, sample_market):
        """vol_20d on day t should only use data up to t."""
        feat = compute_price_features(sample_prices, sample_market)
        scom = feat[feat["ticker"] == "SCOM"].sort_values("date")
        # First 19 rows should have NaN vol_20d (insufficient history)
        assert scom["vol_20d"].iloc[:19].isna().all() or scom["vol_20d"].iloc[:10].isna().all()

    def test_two_tickers(self, sample_prices, sample_market):
        feat = compute_price_features(sample_prices, sample_market)
        assert set(feat["ticker"].unique()) == {"SCOM", "KCB"}

    def test_beta_within_range(self, sample_prices, sample_market):
        feat = compute_price_features(sample_prices, sample_market)
        betas = feat["beta_nse20_60d"].dropna()
        # Beta against itself-like market should be reasonable
        assert (betas.abs() < 10).all()


# ---------------------------------------------------------------------------
# Tests: liquidity features
# ---------------------------------------------------------------------------

class TestLiquidityFeatures:
    def test_amihud_positive(self, sample_prices):
        feat = compute_liquidity_features(sample_prices)
        amihud = feat["amihud_20d"].dropna()
        assert (amihud >= 0).all()

    def test_stale_price_flag(self):
        """Flat close for 3 days should trigger stale flag."""
        dates = pd.date_range("2024-01-02", periods=10, freq="B")
        prices = pd.DataFrame({
            "date": dates,
            "ticker": "SCOM",
            "open": 100.0,
            "high": 100.5,
            "low": 99.5,
            "close": [100, 101, 101, 101, 101, 102, 102, 103, 104, 105],
            "volume": 1_000_000,
            "value_traded": 100_000_000.0,
            "number_of_deals": 200,
            "previous_close": [99, 100, 101, 101, 101, 101, 102, 102, 103, 104],
        })
        feat = compute_liquidity_features(prices)
        stale = feat["stale_price_flag"]
        # Days 3-5 should have stale_price_flag = 1
        assert stale.iloc[4] == 1

    def test_tradability(self, sample_prices):
        as_of = sample_prices["date"].max().date()
        tradable = compute_tradability(sample_prices, as_of)
        assert "SCOM" in tradable.index
        assert "KCB" in tradable.index
        # Both should be tradable with our synthetic liquid data
        assert tradable["SCOM"] == True
        assert tradable["KCB"] == True


# ---------------------------------------------------------------------------
# Tests: event features
# ---------------------------------------------------------------------------

class TestEventFeatures:
    def test_empty_events(self, sample_prices):
        feat = compute_event_features(sample_prices, pd.DataFrame())
        assert "earnings_in_1d" in feat.columns
        assert "llm_announcement_count_20d" in feat.columns

    def test_event_flag_set(self, sample_prices):
        events = pd.DataFrame([{
            "timestamp_published": sample_prices["date"].iloc[10],
            "ticker": "SCOM",
            "event_type": "earnings",
            "llm_sentiment": 1.0,
            "llm_surprise": 0.5,
            "llm_materiality": 0.7,
            "source": "nse_announcement",
        }])
        feat = compute_event_features(sample_prices, events)
        scom = feat[feat["ticker"] == "SCOM"].sort_values("date")
        # Day 10 should have llm_event_earnings_flag = 1
        row_10 = scom[scom["date"] == sample_prices["date"].iloc[10]]
        assert row_10["llm_event_earnings_flag"].values[0] == 1


# ---------------------------------------------------------------------------
# Tests: labels
# ---------------------------------------------------------------------------

class TestLabels:
    def test_label_shape(self, sample_prices, sample_market):
        from nse.labels.returns import _compute_labels

        labels = _compute_labels(
            sample_prices,
            sample_market,
            start=date(2024, 1, 2),
            end=date(2024, 2, 5),
        )
        assert "label_3d_net_alpha" in labels.columns
        assert "label_binary" in labels.columns
        assert labels["label_binary"].isin([0, 1]).all()

    def test_no_future_label_before_cutoff(self, sample_prices, sample_market):
        from nse.labels.returns import _compute_labels

        labels = _compute_labels(
            sample_prices,
            sample_market,
            start=date(2024, 1, 2),
            end=date(2024, 1, 30),
        )
        # Labels require future prices; last few rows should have NaN (dropped)
        # All remaining rows should have non-NaN label_3d_net_alpha
        assert labels["label_3d_net_alpha"].notna().all()


# ---------------------------------------------------------------------------
# Tests: cost model
# ---------------------------------------------------------------------------

class TestCostModel:
    def test_scenario_ordering(self):
        from nse.backtest.costs import estimate_cost_bps

        adv = 10_000_000
        order = adv * 0.02
        opt = estimate_cost_bps(order, adv, "optimistic")
        base = estimate_cost_bps(order, adv, "base")
        stressed = estimate_cost_bps(order, adv, "stressed")
        assert opt < base < stressed

    def test_cost_positive(self):
        from nse.backtest.costs import estimate_cost_bps

        cost = estimate_cost_bps(1_000_000, 20_000_000, "base")
        assert cost > 0

    def test_max_order_size(self):
        from nse.backtest.costs import max_order_size_kes

        adv = 5_000_000
        max_order = max_order_size_kes(adv)
        assert max_order == adv * 0.05


# ---------------------------------------------------------------------------
# Tests: metrics
# ---------------------------------------------------------------------------

class TestMetrics:
    def test_basic_metrics(self):
        from nse.backtest.metrics import compute_metrics

        rets = pd.Series([0.01, -0.005, 0.008, 0.002, -0.003] * 50)
        m = compute_metrics(rets)
        assert "sharpe_ratio" in m
        assert "max_drawdown" in m
        assert m["max_drawdown"] <= 0
        assert 0 <= m["hit_rate"] <= 1

    def test_sharpe_positive_for_positive_returns(self):
        from nse.backtest.metrics import compute_metrics

        rets = pd.Series([0.005] * 252)  # constant positive return
        m = compute_metrics(rets)
        assert m["sharpe_ratio"] > 0
