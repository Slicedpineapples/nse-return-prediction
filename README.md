# ml.nse — NSE Daily Alpha Ranking System

A daily batch pipeline that ranks a small universe of liquid Nairobi Securities Exchange (NSE) equities by expected net return after costs over the next 1–3 trading days.

**MVP scope:** long-only, top 5 names, human approval required, no autonomous execution.

---

## How it works

The system answers one question each trading day: *which 5 stocks have the best expected return after costs over the next 3 days?*

It has four layers:

**1. Data** — OHLCV prices are read directly from the live `nse` PostgreSQL database (`stocksdata` TimescaleDB hypertable, scraped every ~10 seconds). Announcements and macro data are ingested each morning.

**2. Features** — ~50 numeric features are computed per ticker per day: price/volume signals, liquidity/microstructure flags, event/calendar markers, and macro inputs. Company announcements are sent through Claude to extract structured fields (sentiment, materiality, surprise) which become additional numeric features.

**3. Model** — A LightGBM ranker trained on `label_3d_net_alpha = future_3d_return − benchmark_return − round_trip_cost` ranks tickers cross-sectionally each day. Training uses strict walk-forward validation (train 2019–2022, validate 2023, test 2024) with monthly expanding-window retraining.

**4. Risk + output** — A rule-based risk engine filters the ranked list (liquidity, score threshold, stale-price flag, drawdown kill-switch). The approved names are written to a ranked signal report for human review.

### Daily flow

```
After NSE close / next morning
  │
  ├─ 1. Aggregate daily OHLCV from stocksdata (nse DB)
  ├─ 2. Ingest new company announcements
  ├─ 3. LLM extraction on new filings (Claude API)
  ├─ 4. Build features_daily
  ├─ 5. Score all tradable names with saved model
  ├─ 6. Risk engine filters candidates
  └─ 7. Write signal_YYYY-MM-DD.csv  →  human review
```

---

## Databases

The system uses two PostgreSQL databases on the same host:

| Env var | Database | Purpose |
|---|---|---|
| `DB_NAME` | `ml_nse` | ML pipeline tables: `prices_daily`, `features_daily`, model outputs |
| `DB_NAME2` | `nse` | Live scraper database: `stocksdata` (TimescaleDB), `nse_historical_daily_data` |

`nse-ingest` reads from `nse` and writes aggregated daily OHLCV into `ml_nse`.

### Data sources

| Data | Source |
|---|---|
| Daily prices (OHLCV) | `stocksdata` hypertable in `nse` DB — aggregated to daily bars |
| Market stats (NSE 20, NASI, sector indices) | NSE website scraping |
| Company announcements & corporate actions | NSE website scraping |
| KES/USD exchange rate | Yahoo Finance (`KESUSD=X`) via yfinance |
| T-bill yields (91d, 182d, 364d) | CBK website scraping |
| LLM event extraction | Claude API (`claude-opus-4-6`) — skip with `--skip-llm` |

---

## Setup

### Requirements

- Python 3.11+
- Access to the `nse` and `ml_nse` PostgreSQL databases
- An Anthropic API key (for LLM extraction; skippable with `--skip-llm`)

### Install

```bash
git clone <repo>
cd ml.nse

python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

### Environment variables

```bash
cp .env.example .env
```

Edit `.env`:

```env
# Required for LLM extraction
ANTHROPIC_API_KEY=sk-ant-...

# PostgreSQL — ML pipeline database
DB_HOST=192.168.0.116
DB_PORT=5432
DB_NAME=ml_nse
DB_USER=postgres
DB_PASS=secret
DB_SSLMODE=disable

# Live scraper database (source of price data)
DB_NAME2=nse
```

### Initialise the ML database

```bash
python scripts/init_db.py
```

This creates all tables in `ml_nse`. The `nse` database already exists and is managed separately.

---

## Shell completion & man page

### Tab completion

**Zsh** (add to `~/.zshrc` before `compinit`):

```zsh
fpath=(/path/to/ml.nse/scripts/completions $fpath)
autoload -Uz compinit && compinit
```

Or source directly for the current session:

```zsh
source ml.nse/scripts/completions/_nse
```

**Bash** (add to `~/.bashrc` or `~/.bash_profile`):

```bash
source /path/to/ml.nse/scripts/completions/nse_completions.bash
```

Both scripts complete all flags for every `nse-*` command, including
`--scenario (optimistic|base|stressed)` choices.

### Man page

Install locally:

```bash
# view without installing
man docs/nse.1

# install to your local man path
mkdir -p ~/.local/share/man/man1
cp docs/nse.1 ~/.local/share/man/man1/
man nse
```

---

## Running the pipeline

### 1. Backfill historical data (once)

```bash
# Aggregate prices from stocksdata + ingest market stats, announcements, macro
nse-ingest --start 2019-01-01 --end 2024-12-31

# Build features and labels for that period
nse-features --start 2019-01-01 --end 2024-12-31
```

### 2. Train the model

```bash
# Walk-forward evaluation across monthly windows, then save production model
nse-train

# Skip walk-forward, train production model only (faster)
nse-train --prod-only
```

Saves model to `models/saved/ranker_production.pkl`.

### 3. Validate on recent history

```bash
nse-backtest --start 2024-01-01 --scenario base
nse-backtest --start 2024-01-01 --scenario stressed
```

### 4. Daily production run

```bash
# Full pipeline: ingest → LLM → features → score → risk → report
nse-pipeline

# Skip LLM to save API cost (reuses yesterday's extractions)
nse-pipeline --skip-llm

# Run for a specific past date
nse-pipeline --date 2024-11-15
```

Output is printed to the terminal as a ranked table and saved to:

```
reports/output/signal_YYYY-MM-DD.csv
reports/output/signal_YYYY-MM-DD.json
```

### Individual commands

```bash
nse-ingest   --start 2025-03-23 --end 2025-03-23   # ingest one day
nse-features --start 2025-03-23                     # features for one day
nse-signal   --skip-ingest                          # score only, no re-ingest
```

---

## Project structure

```
ml.nse/
├── nse/
│   ├── config.py              # Universe, cost params, risk limits, DB URLs
│   ├── cli.py                 # Click CLI entry points
│   ├── db/
│   │   └── schema.py          # 6 SQLAlchemy tables (ml_nse)
│   ├── ingest/
│   │   ├── prices_live.py     # Aggregate daily OHLCV from stocksdata (nse DB)
│   │   ├── prices.py          # NSE website scraper (legacy fallback)
│   │   ├── market.py          # NSE market stats & indices
│   │   ├── announcements.py   # Company announcements & corporate actions
│   │   └── macro.py           # KES/USD, T-bills, CBK rate, inflation
│   ├── features/
│   │   ├── price_features.py      # Returns, volatility, beta, z-scores
│   │   ├── liquidity_features.py  # Amihud, stale-price, ADV, tradability
│   │   ├── event_features.py      # Event flags, LLM aggregations
│   │   ├── macro_features.py      # FX changes, T-bill spreads, regimes
│   │   └── pipeline.py            # Orchestrates all feature groups → DB
│   ├── labels/
│   │   └── returns.py         # Net alpha labels (1d / 3d / 5d)
│   ├── models/
│   │   ├── ranker.py          # LightGBM regressor + classifier
│   │   └── walk_forward.py    # Strict temporal cross-validation
│   ├── backtest/
│   │   ├── engine.py          # Daily long-only simulation
│   │   ├── costs.py           # Three-scenario cost model
│   │   └── metrics.py         # Sharpe, drawdown, IC, hit rate
│   ├── llm/
│   │   └── extractor.py       # Claude API: event classification & extraction
│   ├── risk/
│   │   └── rules.py           # Hard filters: liquidity, kill switch, blackout
│   └── pipeline/
│       └── daily.py           # End-to-end daily batch orchestration
├── scripts/
│   └── init_db.py             # Create all tables in ml_nse
├── tests/
│   └── test_features.py       # 16 unit tests
├── data/
│   └── raw/                   # Cached downloads and manual drop-ins
├── models/saved/              # Serialised model artefacts
├── reports/output/            # Daily signal CSV/JSON reports
├── .env.example
└── pyproject.toml
```

---

## Universe

12 seed tickers (configurable in `nse/config.py`):

`SCOM` `EQTY` `KCB` `COOP` `BAT` `ABSA` `SCBK` `JUB` `EABL` `BAMB` `NMG` `BRIT`

A ticker is only eligible on a given day if it passes the daily tradability filter: median value traded over last 60 days > KES 1M, traded on ≥ 80% of sessions, and today's intraday range is not extreme.

---

## Risk limits

| Limit | Value |
|---|---|
| Max positions | 5 |
| Max single-name weight | 20% |
| Max daily portfolio turnover | 40% |
| Max order size | 5% of 20d ADV |
| Portfolio drawdown kill-switch | 15% |
| Min model score to trade | 0.55 |

---

## Cost model

Round-trip cost scenarios used in backtesting and label construction:

| Scenario | Round-trip bps |
|---|---|
| Optimistic | 300 |
| Base | 400 |
| Stressed | 600 |

Costs include broker commission, NSE/CDS levies, stamp duty, and slippage scaled by order size relative to ADV.
