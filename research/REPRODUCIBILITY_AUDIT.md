> Historical audit from 2026-10-04. For current instructions and results, see REPRODUCING.md and RESULTS_SUMMARY.md.

# Reproducibility Audit

Project: `ml.nse`

Manuscript: "Cross-Sectional Return Prediction in a Small, Illiquid Equity Market: Machine Learning Evidence from the Nairobi Securities Exchange"

Audit started: 2026-10-04

## Phase 0 State

Repository path: `<repository-root>`

Current branch: `main`

Current commit:

```text
f3d0b39cf7b449ab34f8138bef6a66f19368c4a0
```

The repository was not clean at audit start. Because there were uncommitted and
untracked user changes, no research branch was created.

Observed dirty state:

```text
 M README.md
 M docs/nse.1
 M nse/cli.py
 M nse/config.py
 M nse/db/schema.py
 M nse/features/event_features.py
 M nse/features/pipeline.py
 M nse/ingest/prices_live.py
 M nse/models/ranker.py
 M nse/monitoring/ledger.py
 M nse/pipeline/daily.py
 M pyproject.toml
?? ROOT_CAUSE_ANALYSIS.md
?? docs/interpreting_results.md
?? nse/features/intraday_features.py
?? nse/ingest/live_prices.py
?? nse/pipeline/intraday.py
?? scripts/diagnose_signals.py
```

## Environment

System Python:

```text
python: command not found
python3: Python 3.14.5
```

Virtual environment:

```text
.venv/bin/python: Python 3.14.5
pip: 26.0.1
```

Operating system:

```text
macOS 26.5.2, Darwin 25.5.0, arm64
```

Important installed package versions observed:

```text
numpy==2.4.3
pandas==3.0.1
scipy==1.17.1
scikit-learn==1.8.0
lightgbm==4.6.0
SQLAlchemy==2.0.48
psycopg2-binary==2.9.11
```

## Data Access Status

The project is configured for PostgreSQL mode through `.env`. The database URL
resolves to a network PostgreSQL host rather than the local SQLite fallback.

Initial direct connection testing using `psycopg2.connect(..., connect_timeout=5)` failed:

```text
connection to server at "<private-db-host>", port 5432 failed: timeout expired
```

A later retry succeeded intermittently. Lightweight count queries returned:

```text
features_daily:    57,373 rows, 2006-09-11 to 2026-06-12
prices_daily:      57,703 rows, 2006-09-11 to 2026-07-14
market_daily:         225 rows, 2025-08-04 to 2026-07-14
macro_daily:        1,886 rows, 2019-01-01 to 2026-03-24
issuer_events:         58 rows, 2025-08-07 to 2026-03-23
trades_simulated:   8,824 rows, 2025-08-04 to 2026-04-21
```

However, larger read/export operations remained unstable. A narrowed
`COPY` export for `features_daily` from 2019-01-01 through 2024-12-31 stalled
after 13,540 data rows, at 2023-06-15. The incomplete file was preserved as:

```text
research/outputs/reproduction/features_daily_snapshot.partial.csv
```

The local SQLite fallback at `data/nse.db` exists but is empty:

```text
features_daily: 0 rows
prices_daily: 0 rows
market_daily: 0 rows
macro_daily: 0 rows
issuer_events: 0 rows
```

Monthly chunk export succeeded for 2019-01 through 2024-12 using
`research/scripts/reproduce_2024_walk_forward.py`, producing a local snapshot:

```text
research/outputs/reproduction/features_daily_snapshot.csv
```

The earlier incomplete export remains marked as:

```text
research/outputs/reproduction/features_daily_snapshot.partial.csv
```

## Existing Artifacts

Observed local artifacts:

```text
models/saved/ranker_2024-01.pkl
models/saved/ranker_2024-02.pkl
models/saved/ranker_2024-03.pkl
models/saved/ranker_2024-04.pkl
models/saved/ranker_2024-05.pkl
models/saved/ranker_2024-06.pkl
models/saved/ranker_2024-07.pkl
models/saved/ranker_2024-08.pkl
models/saved/ranker_2024-09.pkl
models/saved/ranker_2024-10.pkl
models/saved/ranker_2024-11.pkl
models/saved/ranker_2024-12.pkl
models/saved/ranker_production.pkl
reports/NSE_Alpha_Ranking_System_Technical_Report.md
reports/NSE_Alpha_Ranking_System_Technical_Report.tex
```

These artifacts are not sufficient by themselves to reproduce the reported
results because the underlying feature rows, labels, predictions, and daily IC
series are not present in the local SQLite database.

## Code Map

Initial repository inspection identified these components:

| Area | Files |
|---|---|
| Configuration | `nse/config.py` |
| Database schema | `nse/db/schema.py` |
| Feature pipeline | `nse/features/pipeline.py` |
| Price features | `nse/features/price_features.py` |
| Liquidity features | `nse/features/liquidity_features.py` |
| Event/LLM features | `nse/features/event_features.py` |
| Macro features | `nse/features/macro_features.py` |
| Labels | `nse/labels/returns.py` |
| Ranker | `nse/models/ranker.py` |
| Walk-forward evaluation | `nse/models/walk_forward.py` |
| Backtesting | `nse/backtest/engine.py`, `nse/backtest/costs.py`, `nse/backtest/metrics.py` |
| CLI | `nse/cli.py` |

## Preliminary Methodological Findings

These findings are from code inspection only. They must be confirmed against
data and experiment outputs once the database is reachable.

1. `run_walk_forward()` saves monthly models to `models/saved/ranker_YYYY-MM.pkl`.
   It should not be called directly for research reproduction unless output paths
   are redirected, because doing so can overwrite existing artifacts.

2. `NSERanker.fit()` fits `StandardScaler` inside each training slice, which is
   correct with respect to global scaler leakage.

3. `_prepare_xy()` fills missing values using `X.median()` on whichever slice is
   being prepared. For validation/test prediction, this means missing-value
   medians are computed from the evaluation slice itself. This may use
   contemporaneous evaluation-slice distributional information. A publication
   pipeline should fit imputation values on training data only and reuse them
   for validation/test.

4. The label definition in `nse/labels/returns.py` is:

   ```text
   label_Nd_net_alpha = future_Nd_return - future_benchmark_return - BASE_COST_FRACTION
   ```

   where `BASE_COST_FRACTION = COSTS["scenario_base_bps"] / 10000`.

5. Current `nse/config.py` defines cost scenarios as:

   ```text
   optimistic: 400 bps
   base:       500 bps
   stressed:   700 bps
   ```

   Therefore current labels use 500 bps, not 400 bps. This differs from the
   current manuscript draft and must be reconciled before publication.

6. If the benchmark is unavailable, label generation falls back to a zero
   benchmark. In that case, `label_3d_net_alpha` is actually net forward return,
   not benchmark-adjusted alpha.

7. If a constant cost is subtracted from every security in a cross-section, the
   cross-sectional rank is unchanged. Such a cost affects hit-rate and economic
   profitability but not Spearman cross-sectional IC.

8. Macro feature construction uses the date attached to `macro_daily` and does
   not currently encode economic release timestamps. Publication analysis must
   audit whether `kes_usd`, T-bill, policy-rate, and inflation observations were
   historically available at the feature date.

9. Event feature code currently uses past publication windows for earnings/event
   flags, which is directionally appropriate. Corporate-action timing still
   requires audit against actual publication/ex-date availability.

10. The seed universe is a fixed 12-security list in `nse/config.py`. The
    repository does not yet document whether this universe was selected using
    information available at the beginning of 2019 or using present-day
    liquidity/survivorship information.

## Reproduction Status

The originally reported manuscript result:

```text
mean 2024 OOS IC approximately 0.091
positive monthly IC = 11/12
```

was not exactly reproduced under the current code/data snapshot. The generated
LightGBM reproduction gives:

```text
mean validation IC:   0.0952
mean monthly test IC: 0.0852
positive months:     11/12
mean top-1 hit rate:  17.0%
```

Therefore the manuscript should replace `0.0911` with the reproducible
`0.0852` monthly IC unless a separate archived experiment artifact explains the
older number.
