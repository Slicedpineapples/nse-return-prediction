# Pipeline operations

`ml.nse` is a daily batch pipeline for ranking a small universe of Nairobi Securities Exchange equities by expected short-horizon alpha after benchmark adjustment and trading costs. It also supports intraday scoring during the trading session using live order book and foreign flow data.

The system is advisory only. It generates ranked long candidates, applies hard risk filters, writes reports, and logs outcomes for monitoring. It does not place orders.

## Current state

- Universe: 12 seed NSE tickers from `nse/config.py`
- Holding horizon: 3 trading days (daily model); same-session (intraday model)
- Primary label: `label_3d_net_alpha`
- Model: LightGBM regressor with optional top-bucket classifier
- Pipeline output: CSV and JSON reports in `reports/output/`
- Monitoring: signal ledger backfill, IC, hit rate, calibration, optional PSI drift
- LLM support: Anthropic by default, optional Ollama config paths exist in code
- Intraday signals: order book imbalance, foreign flows, session momentum from `live_stock_data`

## End-to-end flow

```text
Daily batch (nse-pipeline):
1. Ingest daily prices, market data, announcements, and macro data
2. Run LLM extraction on new issuer events unless --skip-llm is used
3. Build daily features (price, liquidity, event, macro, intraday microstructure)
4. Compute labels for historical feature rows
5. Train or load the saved model
6. Score the current day and apply hard risk rules
7. Save signal reports and write the ranked names to signal_history
8. Backfill realised 3-day outcomes once enough future trading days exist
9. Monitor IC, hit rate, calibration, and drift

Intraday (nse-intraday):
1. Load live_stock_data snapshots for today from DB_NAME2
2. Compute order book, foreign flow, and session momentum features
3. Blend with base model score and re-rank
4. Output ranked intraday signal table
```

## Data and storage

The app supports two runtime modes:

- PostgreSQL mode: if `DB_HOST` is set, `ml.nse` connects to the configured PostgreSQL database for pipeline tables.
- SQLite fallback: if `DB_HOST` is not set, it uses `data/nse.db` for local development.

When PostgreSQL mode is used, the project expects two logical databases on the same host:

| Env var | Typical DB | Purpose |
|---|---|---|
| `DB_NAME` | `ml_nse` | Pipeline tables: `prices_daily`, `features_daily`, `signal_history`, etc. |
| `DB_NAME2` | `nse` | Source market database: `live_stock_data`, `stocksdata`, `corporate_actions`, etc. |

### Source table routing

Prices are ingested from three source tables depending on date:

| Date range | Source table | Fields available |
|---|---|---|
| before 2025-08-01 | `nse_historical_daily_data` | OHLCV only |
| 2025-08-01 – 2026-03-30 | `stocksdata` (TimescaleDB) | OHLCV, avg_price |
| 2026-03-31 onwards | `live_stock_data` | OHLCV + order book, foreign flows, WAP, market cap, trade count |

Corporate actions are read from `corporate_actions` in `DB_NAME2` and upserted into `issuer_events`.

### Core tables (ml_nse)

| Table | Description |
|---|---|
| `prices_daily` | One row per ticker per day. Includes new live-data fields (foreign flows, order book snapshot, WAP, market cap) for dates ≥ 2026-03-31 |
| `prices_intraday` | Raw snapshots from `live_stock_data` (~10s cadence). Used for intraday features and the intraday pipeline |
| `market_daily` | Market index and aggregate stats per day |
| `issuer_events` | Corporate announcements with LLM-derived sentiment, materiality, and surprise |
| `macro_daily` | KES/USD, T-bill rates, CBK rate, inflation |
| `features_daily` | Engineered features per ticker per day, including intraday microstructure columns |
| `trades_simulated` | Simulated trade fills from backtesting |
| `signal_history` | Prediction ledger: model scores and realised outcomes |

## Features, labels, and risk

Feature generation combines:

- **Price and volume**: returns (1d, 3d, 5d, 20d), volatility, momentum, beta, overnight gap
- **Liquidity and microstructure**: Amihud illiquidity, zero-volume fraction, price band proximity, stale price flag
- **Order book** *(dates ≥ 2026-03-31)*: bid-ask imbalance, order book depth ratio, spread in bps, WAP vs close
- **Foreign flows** *(dates ≥ 2026-03-31)*: net foreign flow %, foreign participation %
- **Intraday session** *(dates ≥ 2026-03-31)*: session return, morning/afternoon returns, volume profile
- **Event and calendar**: earnings flags, dividend/AGM flags, reporting season, days since last announcement
- **Macro**: KES/USD return, T-bill spread, policy rate change, inflation surprise
- **LLM-derived**: sentiment, surprise, materiality scores from issuer event text

Important current behaviour:

- `earnings_in_1d/3d/5d` are past-window flags (not forward-looking) to avoid leakage.
- Training filters on `dq_labels_ok` only. Prediction (`nse-pipeline`) filters on `dq_tradable`. Walk-forward evaluation uses all rows with valid labels.
- Signal outcome backfilling waits for 3 future trading days.
- Intraday features are computed automatically when `prices_intraday` data is available for the requested date range.

Risk rules currently enforce:

- maximum 5 approved names
- maximum 20% single-name weight
- maximum 40% daily turnover fraction
- maximum 5% of ADV per order
- 15% portfolio drawdown kill switch
- minimum model score of `0.55`

## Cost model

Configured round-trip cost scenarios in `nse/config.py`:

| Scenario | Bps | Notes |
|---|---:|---|
| `optimistic` | 400 | Verified fees (368 bps) + minimal slippage |
| `base` | 500 | Verified fees + mid-book slippage |
| `stressed` | 700 | Verified fees + thin-book slippage + impact |

Verified one-way cost: brokerage + statutory charges = **184 bps** (confirmed from live broker statements). Round-trip floor is ~368 bps before any slippage. The previous 300 bps optimistic scenario was below actual fees and has been corrected.

## Installation

Requirements:

- Python 3.11+
- access to the source market database if you want live ingestion
- an Anthropic API key if you want Anthropic-backed LLM extraction

Install:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Create a `.env` file in the repo root:

```env
ANTHROPIC_API_KEY=sk-ant-...

DB_HOST=127.0.0.1
DB_PORT=5432
DB_NAME=ml_nse
DB_NAME2=nse
DB_USER=postgres
DB_PASS=secret
DB_SSLMODE=prefer

# Optional LLM overrides
LLM_PROVIDER=anthropic
ANTHROPIC_MODEL=claude-opus-4-6
# OLLAMA_MODEL=llama3.2
# OLLAMA_BASE_URL=http://localhost:11434
```

For a fresh database, initialize the current schema:

```bash
python scripts/init_db.py
# Mark the fresh schema as current (do not run the additive migration twice).
alembic stamp head
```

## CLI commands

| Command | Description |
|---|---|
| `nse-ingest` | Download prices, market stats, announcements, macro |
| `nse-features` | Build features_daily for a date range |
| `nse-train` | Walk-forward evaluation + production model training |
| `nse-backtest` | Simulate portfolio returns on historical data |
| `nse-signal` | Score today and output signal report |
| `nse-pipeline` | Full daily batch (ingest → features → score → report) |
| `nse-monitor` | IC, hit rate, calibration, and drift monitoring |
| `nse-intraday` | Score the current session using live order book data |

### Typical bootstrap

```bash
nse-ingest --start 2019-01-01 --end 2024-12-31
nse-features --start 2019-01-01 --end 2024-12-31
nse-train
```

### Training options

```bash
# Walk-forward (12 months) plus production training
nse-train

# Production model only, skip walk-forward
nse-train --prod-only

# Recency-weighted: downweights data older than ~3 months
nse-train --recency-half-life 90
```

`nse-train` runs walk-forward evaluation for the requested number of months, then trains a production model. A champion/challenger check compares the new model against the saved `production` model on the last 60 days of labelled data. The challenger is deployed only if its IC exceeds the champion's.

### Daily run

```bash
nse-pipeline
nse-pipeline --skip-llm
nse-pipeline --date 2026-04-18

# Equivalent entry point
nse-signal --skip-ingest
```

Daily pipeline behaviour:

- backfills old signals first when possible
- ingests same-day prices (routed to `live_stock_data` for recent dates), market, macro, and recent announcements unless `--skip-ingest`
- builds same-day features including intraday microstructure when available
- loads the production model
- scores tradable names and applies risk filters
- saves CSV and JSON reports to `reports/output/`
- writes approved names and predicted edge to `signal_history`

### Intraday run

```bash
# Score with snapshots already in DB
nse-intraday

# Pull latest snapshots from live_stock_data first, then score
nse-intraday --ingest

# Pure intraday signals — skip base model score
nse-intraday --no-daily-score

# Score a past session
nse-intraday --date 2026-04-17
```

Intraday scoring blends:
- **Base model score** (production ranker trained on daily features)
- **Order book imbalance** — `(bid_qty − offer_qty) / (bid_qty + offer_qty)` (weight ±0.15)
- **Foreign net flow** — `(foreign_buy − foreign_sell) / turnover` (weight ±0.20)
- **Session momentum** — return from open to latest price (weight ±0.10)

Output is saved to `reports/output/intraday_YYYY-MM-DD.json`.

### Backtesting and monitoring

```bash
nse-backtest --start 2024-01-01 --scenario base
nse-backtest --start 2024-01-01 --scenario stressed

nse-monitor --start 2026-01-01 --backfill
nse-monitor --start 2026-01-01 --window 20 --drift
python -m scripts.diagnose_signals
```

`nse-monitor` reports:

- total signals and outcomes filled
- overall IC and IC p-value
- recent rolling IC (warns if < 50% of historical — retrain signal)
- approved-name hit rate
- mean alpha and prediction error
- score calibration table (quintiles)
- optional PSI feature drift report

## Reports and artifacts

| Path | Description |
|---|---|
| `models/saved/ranker_production.pkl` | Current production model |
| `models/saved/ranker_production_prev.pkl` | Previous champion (archived on challenger win) |
| `reports/output/signal_YYYY-MM-DD.csv` | Daily ranked signal (CSV) |
| `reports/output/signal_YYYY-MM-DD.json` | Daily ranked signal (JSON) |
| `reports/output/intraday_YYYY-MM-DD.json` | Intraday signal (JSON) |

## Project layout

```text
ml.nse/
├── nse/
│   ├── cli.py
│   ├── config.py
│   ├── db/schema.py
│   ├── ingest/
│   │   ├── prices_live.py       # daily price ingest (routing logic)
│   │   ├── live_prices.py       # live_stock_data ingest + prices_intraday
│   │   ├── market_live.py
│   │   ├── announcements_live.py  # corporate_actions from DB_NAME2
│   │   └── macro.py
│   ├── features/
│   │   ├── pipeline.py
│   │   ├── price_features.py
│   │   ├── liquidity_features.py
│   │   ├── event_features.py
│   │   ├── macro_features.py
│   │   └── intraday_features.py   # order book, foreign flows, session features
│   ├── labels/returns.py
│   ├── models/
│   │   ├── ranker.py
│   │   └── walk_forward.py
│   ├── backtest/
│   ├── monitoring/
│   ├── risk/
│   ├── llm/
│   └── pipeline/
│       ├── daily.py
│       └── intraday.py            # intraday signal pipeline
├── alembic/
│   └── versions/
│       └── 001_add_live_data_fields.py
├── scripts/init_db.py
├── scripts/diagnose_signals.py
├── docs/nse.1
├── tests/
├── models/saved/
└── reports/output/
```

## Known limitations

- The primary label subtracts the base 500 bps round-trip cost on a 3-trading-day horizon. This suppresses modelled alpha on a market with thin liquidity.
- `nse/labels/returns.py` falls back to a zero benchmark if `nse_20` data is missing during label construction; monitoring backfill skips rows with missing benchmark data instead of imputing zero.
- `nse-signal` and `nse-pipeline` are separate entry points but execute the same daily pipeline.
- `amihud_20d` is only populated from 2025-08-15 onward. Earlier rows have `dq_features_ok=False` but train correctly since `dq_features_ok` does not gate training.
- Intraday features (`bid_ask_imbalance`, foreign flows, etc.) are only available for dates ≥ 2026-03-31 when `live_stock_data` coverage begins. Earlier rows have NULL for these columns.
- The intraday overlay weights (order book 0.15, foreign flow 0.20, momentum 0.10) are heuristic. No intraday-specific model has been trained yet; a separate intraday model with same-session labels is a planned next step.

## Man page and shell completion

```bash
man ./docs/nse.1
```

Shell completion files:

- Zsh: `scripts/completions/_nse`
- Bash: `scripts/completions/nse_completions.bash`

For an existing pre-intraday database, back it up and run `alembic upgrade head` instead of stamping it. SQLite supports local schema development; source ingestion requires PostgreSQL access.
