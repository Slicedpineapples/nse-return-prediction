"""
SQLAlchemy ORM schema for the core NSE tables.

Tables
------
prices_daily       – one row per ticker per day
market_daily       – one row per market day (indices, aggregate stats)
issuer_events      – one row per announcement / corporate event
macro_daily        – one row per day (KES/USD, T-bills, CBK rate, inflation)
features_daily     – one row per ticker per day after feature engineering
trades_simulated   – one row per simulated order fill
signal_history     – prediction ledger: one row per ticker per signal date
"""

from __future__ import annotations

from contextlib import contextmanager

from sqlalchemy import (
    Boolean,
    Column,
    Date,
    DateTime,
    Float,
    Integer,
    String,
    Text,
    UniqueConstraint,
    create_engine,
)
from sqlalchemy.orm import DeclarativeBase, Session

from nse.config import get_db_url


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

_db_url = get_db_url()
_engine_kwargs: dict = {"echo": False}
if _db_url.startswith("sqlite"):
    _engine_kwargs["connect_args"] = {"check_same_thread": False}
else:
    # PostgreSQL: connection pool tuned for a single-process daily batch
    _engine_kwargs["pool_size"] = 5
    _engine_kwargs["max_overflow"] = 10
    _engine_kwargs["pool_pre_ping"] = True

engine = create_engine(_db_url, **_engine_kwargs)


# ---------------------------------------------------------------------------
# Base
# ---------------------------------------------------------------------------

class Base(DeclarativeBase):
    pass


# ---------------------------------------------------------------------------
# 1. prices_daily
# ---------------------------------------------------------------------------

class PriceDaily(Base):
    __tablename__ = "prices_daily"
    __table_args__ = (UniqueConstraint("date", "ticker", name="uq_price_date_ticker"),)

    id = Column(Integer, primary_key=True, autoincrement=True)
    date = Column(Date, nullable=False, index=True)
    ticker = Column(String(16), nullable=False, index=True)
    open = Column(Float)
    high = Column(Float)
    low = Column(Float)
    close = Column(Float, nullable=False)
    volume = Column(Float)                     # shares traded
    value_traded = Column(Float)               # KES
    number_of_deals = Column(Integer)
    previous_close = Column(Float)
    corporate_action_adjustment_factor = Column(Float, default=1.0)
    tradable_flag = Column(Boolean, default=True)


# ---------------------------------------------------------------------------
# 2. market_daily
# ---------------------------------------------------------------------------

class MarketDaily(Base):
    __tablename__ = "market_daily"
    __table_args__ = (UniqueConstraint("date", name="uq_market_date"),)

    id = Column(Integer, primary_key=True, autoincrement=True)
    date = Column(Date, nullable=False, index=True)
    nse_20 = Column(Float)
    nse_all_share = Column(Float)
    sector_banking = Column(Float)
    sector_commercial = Column(Float)
    sector_investment = Column(Float)
    sector_manufacturing = Column(Float)
    sector_energy = Column(Float)
    sector_agricultural = Column(Float)
    sector_insurance = Column(Float)
    sector_telecom = Column(Float)
    total_market_turnover = Column(Float)      # KES
    total_shares_traded = Column(Float)
    total_equity_deals = Column(Integer)


# ---------------------------------------------------------------------------
# 3. issuer_events
# ---------------------------------------------------------------------------

class IssuerEvent(Base):
    __tablename__ = "issuer_events"

    id = Column(Integer, primary_key=True, autoincrement=True)
    event_id = Column(String(64), unique=True, nullable=False, index=True)
    timestamp_published = Column(DateTime, nullable=False, index=True)
    ticker = Column(String(16), nullable=False, index=True)
    source = Column(String(64))                # 'nse_announcement' | 'corporate_action'
    raw_text = Column(Text)
    event_type = Column(String(32))            # earnings | dividend | agm | ...
    event_subtype = Column(String(64))
    filing_period = Column(String(32))         # 'H1 2024' | 'FY 2023' | ...
    attachment_url = Column(String(512))

    # LLM-derived fields
    llm_sentiment = Column(Float)              # -2 to +2
    llm_surprise = Column(Float)              # -2 to +2
    llm_horizon = Column(String(16))          # '1d' | '3d' | '1m' | 'structural'
    llm_confidence = Column(Float)            # 0-1
    llm_materiality = Column(Float)           # 0-1
    llm_persistence = Column(String(16))      # 'temporary' | 'persistent'
    llm_summary = Column(Text)
    llm_key_numbers = Column(Text)            # JSON string
    llm_risks_mentioned = Column(Text)        # JSON string
    llm_processed_at = Column(DateTime)

    # Derived score (combination of materiality + surprise)
    materiality_score = Column(Float)


# ---------------------------------------------------------------------------
# 4. macro_daily
# ---------------------------------------------------------------------------

class MacroDaily(Base):
    __tablename__ = "macro_daily"
    __table_args__ = (UniqueConstraint("date", name="uq_macro_date"),)

    id = Column(Integer, primary_key=True, autoincrement=True)
    date = Column(Date, nullable=False, index=True)
    kes_usd = Column(Float)                   # KES per USD
    short_rate = Column(Float)                # overnight / short-end rate %
    tbill_91d = Column(Float)                 # 91-day T-bill yield %
    tbill_182d = Column(Float)
    tbill_364d = Column(Float)
    policy_rate = Column(Float)               # CBK rate %
    inflation_latest = Column(Float)          # latest CPI YoY %
    macro_event_flag = Column(Integer, default=0)  # 1 if MPC / budget day


# ---------------------------------------------------------------------------
# 5. features_daily
# ---------------------------------------------------------------------------

class FeaturesDaily(Base):
    """
    One row per ticker per day.  All numeric columns from feature engineering
    plus label columns.  New features are added as migrations; existing rows
    get NULL for the new column until regenerated.
    """
    __tablename__ = "features_daily"
    __table_args__ = (UniqueConstraint("date", "ticker", name="uq_feat_date_ticker"),)

    id = Column(Integer, primary_key=True, autoincrement=True)
    date = Column(Date, nullable=False, index=True)
    ticker = Column(String(16), nullable=False, index=True)

    # --- Price/volume features ---
    ret_1d = Column(Float)
    ret_3d = Column(Float)
    ret_5d = Column(Float)
    ret_20d = Column(Float)
    overnight_gap = Column(Float)
    vol_5d = Column(Float)
    vol_20d = Column(Float)
    volume_zscore_20d = Column(Float)
    value_traded_zscore_20d = Column(Float)
    deals_zscore_20d = Column(Float)
    intraday_range_pct = Column(Float)
    momentum_reversal_5d = Column(Float)
    beta_nse20_60d = Column(Float)
    corr_sector_60d = Column(Float)

    # --- Liquidity/microstructure features ---
    median_value_20d = Column(Float)
    median_value_60d = Column(Float)
    zero_vol_fraction_20d = Column(Float)
    amihud_20d = Column(Float)
    gap_frequency_20d = Column(Float)
    stale_price_flag = Column(Integer)
    large_move_low_vol_flag = Column(Integer)
    days_since_normal_liquidity = Column(Integer)
    price_band_proximity_pct = Column(Float)

    # --- Event/calendar features ---
    earnings_in_1d = Column(Integer)
    earnings_in_3d = Column(Integer)
    earnings_in_5d = Column(Integer)
    dividend_flag = Column(Integer)
    agm_flag = Column(Integer)
    investor_briefing_flag = Column(Integer)
    days_since_last_announcement = Column(Integer)
    month_of_year = Column(Integer)
    reporting_season_flag = Column(Integer)
    corporate_action_window_flag = Column(Integer)

    # --- Macro features ---
    kes_usd_ret_1d = Column(Float)
    kes_usd_ret_5d = Column(Float)
    tbill_spread_chg_1d = Column(Float)
    policy_rate_chg_flag = Column(Integer)
    inflation_surprise_proxy = Column(Float)
    market_turnover_regime = Column(Float)
    sector_regime = Column(Float)

    # --- LLM-derived event features ---
    llm_max_materiality_today = Column(Float)
    llm_weighted_sentiment_3d = Column(Float)
    llm_weighted_surprise_5d = Column(Float)
    llm_announcement_count_20d = Column(Integer)
    llm_event_earnings_flag = Column(Integer)
    llm_event_dividend_flag = Column(Integer)
    llm_event_agm_flag = Column(Integer)
    llm_event_mgmt_flag = Column(Integer)
    llm_event_regulatory_flag = Column(Integer)
    llm_event_financing_flag = Column(Integer)

    # --- Labels ---
    label_1d_net_alpha = Column(Float)
    label_3d_net_alpha = Column(Float)
    label_5d_net_alpha = Column(Float)
    label_binary = Column(Integer)
    label_top_bucket = Column(Integer)

    # --- Data-quality flags ---
    dq_price_ok = Column(Boolean, default=True)
    dq_features_ok = Column(Boolean, default=True)
    dq_labels_ok = Column(Boolean, default=False)  # False until labels materialise
    dq_tradable = Column(Boolean, default=True)


# ---------------------------------------------------------------------------
# 6. trades_simulated
# ---------------------------------------------------------------------------

class TradeSimulated(Base):
    __tablename__ = "trades_simulated"

    id = Column(Integer, primary_key=True, autoincrement=True)
    date = Column(Date, nullable=False, index=True)
    ticker = Column(String(16), nullable=False)
    model_score = Column(Float)
    target_weight = Column(Float)
    assumed_entry_price = Column(Float)
    assumed_exit_price = Column(Float)
    turnover_used = Column(Float)              # KES notional
    cost_bps = Column(Float)
    slippage_bps = Column(Float)
    gross_return = Column(Float)
    realized_return_net = Column(Float)
    scenario = Column(String(16))             # 'optimistic'|'base'|'stressed'
    run_id = Column(String(64), index=True)   # backtest run identifier


# ---------------------------------------------------------------------------
# 7. signal_history  (prediction ledger)
# ---------------------------------------------------------------------------

class SignalHistory(Base):
    """
    One row per ticker per signal date.  Populated by the daily pipeline
    when signals are generated; outcome columns filled in ~3 days later
    by the backfill job in nse.monitoring.ledger.
    """
    __tablename__ = "signal_history"
    __table_args__ = (UniqueConstraint("signal_date", "ticker", name="uq_signal_date_ticker"),)

    id = Column(Integer, primary_key=True, autoincrement=True)
    signal_date = Column(Date, nullable=False, index=True)
    ticker = Column(String(16), nullable=False, index=True)

    # --- Prediction ---
    rank = Column(Integer)                     # rank on the day (1 = top)
    score = Column(Float)                      # raw model score
    predicted_edge_bps = Column(Float)         # expected alpha in bps
    approved = Column(Boolean, default=False)  # passed human/liquidity filter?
    model_label = Column(String(64))           # e.g. 'lgbm_v3'

    # --- Outcome (filled ~3 trading days later) ---
    actual_return_3d = Column(Float)           # close[t+3]/close[t] - 1
    actual_net_alpha_3d = Column(Float)        # actual return - benchmark - costs
    prediction_error = Column(Float)           # actual_net_alpha_3d - predicted_edge_bps/10000
    outcome_filled_at = Column(DateTime)       # when actuals were written

    created_at = Column(DateTime)              # when the signal was generated


# ---------------------------------------------------------------------------
# Session helpers
# ---------------------------------------------------------------------------

def get_session() -> Session:
    return Session(engine)


@contextmanager
def session_scope():
    session = get_session()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def init_db() -> None:
    """Create all tables if they do not exist."""
    Base.metadata.create_all(engine)
