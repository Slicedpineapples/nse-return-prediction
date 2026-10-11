"""Central configuration: universe, cost model, risk limits, paths."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
CACHE_DIR = DATA_DIR / "cache"
MODELS_DIR = ROOT / "models" / "saved"
REPORTS_DIR = ROOT / "reports" / "output"
LOGS_DIR = ROOT / "logs"
DB_PATH = DATA_DIR / "nse.db"  # fallback for local SQLite


def get_db_url() -> str:
    """
    Build the database URL from environment variables.

    If DB_HOST is set, returns a PostgreSQL URL.
    Otherwise falls back to the local SQLite file (useful for offline dev/tests).
    """
    host = os.getenv("DB_HOST")
    if host:
        port = os.getenv("DB_PORT", "5432")
        name = os.getenv("DB_NAME", "nse")
        user = os.getenv("DB_USER", "postgres")
        password = os.getenv("DB_PASS", "")
        sslmode = os.getenv("DB_SSLMODE", "prefer")
        return (
            f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{name}"
            f"?sslmode={sslmode}"
        )
    return f"sqlite:///{DB_PATH}"

for _d in (RAW_DIR, PROCESSED_DIR, CACHE_DIR, MODELS_DIR, REPORTS_DIR, LOGS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Universe
# ---------------------------------------------------------------------------

# Seed universe: 12 of the most consistently traded NSE equities.
# Expand after validating data coverage.
SEED_UNIVERSE: list[str] = [
    "SCOM",   # Safaricom
    "EQTY",   # Equity Group
    "KCB",    # KCB Group
    "COOP",   # Co-operative Bank
    "BAT",    # BAT Kenya
    "ABSA",   # ABSA Bank Kenya
    "SCBK",   # Standard Chartered Kenya
    "JUB",    # Jubilee Holdings
    "EABL",   # East African Breweries
    "BAMB",   # Bamburi Cement
    "NMG",    # Nation Media Group
    "BRIT",   # Britam Holdings
]

# Liquidity thresholds for the daily tradability filter
LIQUIDITY = {
    "lookback_days": 60,
    "min_median_value_traded_kes": 1_000_000,   # KES 1 M median daily turnover
    "min_session_fraction": 0.80,               # traded on ≥80% of sessions
    "max_spread_proxy_pct": 0.10,               # 10% intraday range cap as spread proxy (NSE stocks commonly >5%)
    "stale_price_max_days": 3,                  # max consecutive flat-close days
}

# ---------------------------------------------------------------------------
# Cost model (basis points unless noted)
# ---------------------------------------------------------------------------

COSTS = {
    # Verified from live broker statement: brokerage + statutory charges = 184 bps one-way
    # (brokerage KES 1,554.80 + statutory KES 648.31 on KES 119,600 order = 1.842%)
    "commission_bps": 184,

    # Slippage by liquidity bucket (bps, one-way)
    "slippage_liquid_bps": 30,
    "slippage_mid_bps": 60,
    "slippage_thin_bps": 120,

    # Market impact cap
    "impact_adv_fraction": 0.05,     # never trade >5% of 20d ADV

    # Total round-trip cost scenarios (bps)
    # Floor is ~368 bps (2 × 184 bps verified fees, zero slippage).
    # Optimistic adds minimal slippage on a liquid name; base adds mid slippage;
    # stressed adds thin-book slippage + adverse price movement.
    "scenario_optimistic_bps": 400,   # 368 fees + ~30 bps slippage
    "scenario_base_bps": 500,         # 368 fees + ~130 bps slippage
    "scenario_stressed_bps": 700,     # 368 fees + ~330 bps slippage + impact
}

# ---------------------------------------------------------------------------
# Portfolio / risk limits
# ---------------------------------------------------------------------------

RISK = {
    "max_positions": 5,
    "max_single_name_weight": 0.20,          # 20%
    "max_daily_turnover_fraction": 0.40,     # 40% of portfolio AUM per day
    "max_order_adv_fraction": 0.05,          # 5% of 20d ADV per order
    "max_portfolio_drawdown_stop": 0.15,     # 15% drawdown → kill switch
    "min_score_threshold": 0.55,             # model score must exceed to trade
    "event_blackout_days": 1,                # skip if announcement < 1 day old and ambiguous
    "hold_days": 3,                          # target holding period
}

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

MODEL = {
    "primary_label": "label_3d_net_alpha",
    "secondary_labels": ["label_1d_net_alpha", "label_5d_net_alpha",
                         "label_binary", "label_top_bucket"],
    "train_start": "2019-01-01",
    "validate_start": "2023-01-01",
    "test_start": "2024-01-01",
    "retrain_frequency": "monthly",           # 'monthly' | 'quarterly'
    "window_type": "expanding",               # 'expanding' | 'rolling'
    "rolling_window_years": 3,
}

# ---------------------------------------------------------------------------
# LLM
# ---------------------------------------------------------------------------

LLM = {
    # Provider: "anthropic" (default) or "ollama"
    "provider": os.getenv("LLM_PROVIDER", "anthropic"),

    # Anthropic settings
    "model": os.getenv("ANTHROPIC_MODEL", "claude-opus-4-6"),
    "max_tokens": 1024,
    "temperature": 0.0,
    "api_key_env": "ANTHROPIC_API_KEY",

    # Ollama settings
    "ollama_model": os.getenv("OLLAMA_MODEL", "llama3.2"),
    "ollama_base_url": os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
}

# ---------------------------------------------------------------------------
# NSE data sources
# ---------------------------------------------------------------------------

NSE = {
    "base_url": "https://www.nse.co.ke",
    "prices_url": "https://www.nse.co.ke/market-statistics/",
    "announcements_url": "https://www.nse.co.ke/listed-companies/company-announcements/",
    "corporate_actions_url": "https://www.nse.co.ke/listed-companies/corporate-actions/",
    "market_stats_url": "https://www.nse.co.ke/market-statistics/",
    "request_timeout": 30,
    "request_delay_seconds": 2,   # polite crawl delay
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_source_db_url() -> str:
    """
    URL for the live nse scraper database (DB_NAME2 env var).

    This is the TimescaleDB instance that holds stocksdata.
    Requires PostgreSQL — raises if DB_HOST is not set.
    """
    host = os.getenv("DB_HOST")
    if not host:
        raise EnvironmentError("DB_HOST not set — source DB requires PostgreSQL")
    port = os.getenv("DB_PORT", "5432")
    name = os.getenv("DB_NAME2", "nse")
    user = os.getenv("DB_USER", "postgres")
    password = os.getenv("DB_PASS", "")
    sslmode = os.getenv("DB_SSLMODE", "prefer")
    return (
        f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{name}"
        f"?sslmode={sslmode}"
    )


def get_anthropic_key() -> str:
    key = os.getenv(LLM["api_key_env"])
    if not key:
        raise EnvironmentError(
            f"Missing environment variable {LLM['api_key_env']}. "
            "Set it in your .env file or shell."
        )
    return key
