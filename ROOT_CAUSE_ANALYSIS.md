# NSE Signal System Root Cause Analysis

## Summary

The main unresolved issue is still the same: the production label subtracts the base 500 bps round-trip cost from a 3-trading-day alpha target. On NSE data, that cost assumption is large relative to the short-horizon return distribution, so the target can be dominated by costs instead of cross-sectional signal.

Several implementation issues have been fixed in code since the original diagnosis. Those fixes improve data integrity and monitoring accuracy, but they do not by themselves solve the label-design problem.

## What is fixed in the current codebase

### 1. Event leakage was reduced

File: `nse/features/event_features.py`

Current behavior:

- `earnings_in_1d`, `earnings_in_3d`, and `earnings_in_5d` still exist as columns
- they now look backward over past event windows instead of checking future scheduled events

Important clarification:

- the earlier write-up that said these columns were removed or renamed is no longer accurate
- the schema still contains those original column names
- the leak fix is semantic, not a schema rename

### 2. Signal outcome backfilling now uses trading-day logic

File: `nse/monitoring/ledger.py`

Current behavior:

- backfill waits until at least 3 future trading days exist after the signal date
- the backfill job uses `market_daily` to derive trading dates
- rows with missing benchmark data are skipped instead of silently treated as zero benchmark return

This is a material improvement over the old calendar-day approach.

### 3. Model training and scoring now respect data-quality flags

File: `nse/models/ranker.py`

Current behavior:

- training filters on `dq_labels_ok`
- validation filters on `dq_labels_ok`
- the daily pipeline filters prediction inputs on `dq_tradable`
- the ranker itself does not gate prediction on `dq_features_ok`

This prevents the model from fitting or scoring obviously bad rows.

### 4. A diagnostics script exists for operational checks

File: `scripts/diagnose_signals.py`

It reports:

- signal coverage
- trading calendar coverage
- recent backfill eligibility
- recent data-quality flag rates
- current label distribution

## What remains unresolved

### Primary issue: cost model versus 3-day horizon

Current config:

```python
COSTS["scenario_base_bps"] = 500
RISK["hold_days"] = 3
MODEL["primary_label"] = "label_3d_net_alpha"
```

Current label formula in `nse/labels/returns.py`:

```python
label_3d_net_alpha = future_3d_return - future_3d_benchmark_return - BASE_COST_FRACTION
```

Why this is still a problem:

- a 3-day NSE stock move is often small in absolute terms
- subtracting 500 bps from every training target can compress most rows into unattractive or uniformly negative labels
- that makes it harder for the ranker to learn useful cross-sectional separation

### Secondary inconsistency: benchmark fallback differs by subsystem

There is still one notable mismatch across the codebase:

- `nse/monitoring/ledger.py` skips backfills when benchmark data is missing
- `nse/labels/returns.py` still falls back to zero benchmark returns if `nse_20` data is unavailable during label construction

That behavior is safer than before in monitoring, but the training-label path is not yet fully aligned with the monitoring path.

## Recommended next changes

### Option A: reduce the training cost assumption for the 3-day label

Fastest change:

- lower `scenario_base_bps`
- or introduce horizon-specific label costs instead of one shared base scenario

This is the smallest code change and the quickest way to test whether the signal improves.

### Option B: move the strategy to a longer holding horizon

More structural change:

- extend labels beyond 3 trading days
- align `MODEL["primary_label"]` and `RISK["hold_days"]`

This gives price moves more room to dominate execution costs.

### Option C: train on gross alpha and apply costs downstream

Most invasive option:

- remove cost subtraction from the training label
- keep cost handling in backtest, portfolio construction, and approval logic

This separates forecasting from execution assumptions, but it changes how live accept/reject thresholds should be interpreted.

## Operational status as of April 18, 2026

Implemented:

- event feature leakage reduction
- trading-day-aware outcome backfill
- benchmark-missing skip in monitoring backfill
- data-quality filters in model fit and predict
- diagnostics script for signal coverage and label inspection

Not yet implemented:

- cost-model redesign
- longer-horizon labels
- full alignment between training-label benchmark handling and monitoring benchmark handling

## Suggested validation workflow

```bash
python -m scripts.diagnose_signals
nse-monitor --start 2026-01-01 --backfill
nse-train
nse-backtest --start 2024-01-01 --scenario base
```

After any cost-model or label-horizon change, retrain and compare:

- walk-forward IC
- recent monitoring IC
- hit rate on approved names
- label distribution from `scripts/diagnose_signals.py`
