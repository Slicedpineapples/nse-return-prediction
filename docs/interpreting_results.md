# Interpreting Results

## Daily signal report (`nse-pipeline`)

The daily report ranks stocks from highest expected alpha to lowest. Columns that matter:

| Column | What it means |
|---|---|
| `score` | Model's prediction, normalised 0–1 within the day's universe. Higher = model expects better relative performance over the next 3 trading days. Not a probability. |
| `expected_edge_after_costs_bps` | Rough net-alpha estimate in basis points (score converted to alpha minus estimated round-trip costs). Positive = worth considering. Negative = model doesn't expect enough return to cover costs. |
| `recommended_action` | `LONG` if score ≥ 0.55 and all risk rules pass. `NO TRADE` otherwise. |
| `llm_sentiment` | Average LLM-derived sentiment from recent announcements (−2 to +2). Negative doesn't automatically mean avoid — context matters. |
| `recent_events` | The 1–2 most recent LLM-processed announcements for that ticker. |

**What to do with the output:** treat `LONG` as a shortlist, not a buy order. The system flags up to 5 names. Check the `explanation` field and recent events before acting. The model ranks relative attractiveness — it doesn't predict absolute returns.

---

## Walk-forward results (`nse-train`)

```
Period    val_IC   test_IC   test_hit   n_train
2024-01    0.090     0.011     22.7%    14,460
...
Mean val IC: 0.081 | Mean test IC: 0.076
```

| Metric | What it means | What's good |
|---|---|---|
| `val_IC` | Spearman rank correlation between model scores and actual returns on the validation period. Measures ranking quality. | > 0.05 is usable. > 0.10 is strong for a 12-stock universe. |
| `test_IC` | Same on the held-out test month (fully unseen). More honest than val_IC. | Positive and close to val_IC. Large gaps mean overfitting. |
| `test_hit` | % of days the top-ranked stock had a positive return. Noisy on a 12-stock universe — one bad week can zero it. | Ignore on a per-month basis. Only meaningful aggregated over 6+ months. |
| `n_train` | Training rows used. Should grow each slice (expanding window). | Confirms no data leak. |

**Key signal:** if `test_IC` is consistently positive across months, the model has genuine ranking skill. If it's close to zero or negative for several consecutive months, consider retraining with `--recency-half-life 90`.

---

## Backtest results (`nse-backtest`)

```
total_return              -14.3%
annualised_return         -40.5%
annualised_volatility      28.4%
sharpe_ratio               -1.43
hit_rate                   46.7%
avg_gain                   1.204%
avg_loss                  -1.658%
gain_loss_ratio             0.73
% trades surviving stressed costs: 2.1%
```

| Metric | What it means | Warning signs |
|---|---|---|
| `total_return` | Cumulative P&L over the period, net of estimated costs. | Negative even under `--scenario optimistic` means a signal problem, not a cost problem. |
| `sharpe_ratio` | Return per unit of risk. Industry standard: > 1 is good, > 0.5 is acceptable. | Negative means losses dominate. |
| `hit_rate` | % of trading days the portfolio made money. | Below 50% means the model is picking the wrong direction more often than right. |
| `gain_loss_ratio` | Average winning day / average losing day. Should be > 1 to be profitable even with hit_rate < 50%. | Below 1 with hit_rate < 50% = both problems at once. |
| `max_drawdown` | Worst peak-to-trough loss. | > 30% suggests the strategy has tail risk beyond normal volatility. |
| `% trades surviving stressed costs` | How many simulated trades would still be profitable under worst-case 600 bps round-trip costs. | Below 10% means the strategy is extremely cost-sensitive and fragile on NSE's thin liquidity. |

**Cost scenario comparison:**

Always run all three scenarios back to back:
```bash
nse-backtest --start 2026-01-01 --scenario optimistic
nse-backtest --start 2026-01-01 --scenario base
nse-backtest --start 2026-01-01 --scenario stressed
```

Verified round-trip fee floor is **~368 bps** (184 bps brokerage+statutory each way). Scenarios:

| Scenario | Bps | Realistic? |
|---|---:|---|
| `optimistic` | 400 | Yes — liquid name, low slippage |
| `base` | 500 | Yes — typical NSE execution |
| `stressed` | 700 | Worst case: wide spread + market impact |

If `optimistic` (400 bps) is negative → signal problem. If `optimistic` is positive but `base` is negative → liquidity/slippage problem. If all three are negative → both problems.

---

## Monitor output (`nse-monitor`)

```
Signals (total)       143
Outcomes filled        98
IC (overall)        0.081
IC p-value          0.012
IC (last 20d)       0.034
Hit rate (approved) 54.3%
Mean alpha (bps)    -12.4
```

| Metric | What it means | Action |
|---|---|---|
| `IC (overall)` | Spearman rank correlation between predicted scores and actual 3-day returns across all signals. | > 0.05 = model has signal. < 0 = model is anti-predictive. |
| `IC p-value` | Statistical significance of the IC. | < 0.05 means the IC is unlikely to be random noise. |
| `IC (last 20d)` | Same but only over the most recent 20 trading days. Highlighted yellow if < 50% of historical IC. | If this degrades significantly, retrain: `nse-train --recency-half-life 90`. |
| `Hit rate (approved)` | % of approved names that had positive returns 3 days later. | > 50% means the final shortlist is directionally right. |
| `Mean alpha (bps)` | Average realised net alpha (return minus benchmark minus costs). | Negative in a falling market doesn't mean the model is broken — compare to the NSE 20 index for the same period. |

**Calibration table:** shows whether high-scoring stocks actually outperform low-scoring ones. If quintile 5 (top scores) has higher mean alpha than quintile 1 (bottom scores), the model is well-calibrated. If the ordering is inverted, something has drifted.

---

## Intraday signal (`nse-intraday`)

```
#  Ticker  Score   Base   OB Imbal  Fgn Flow%  Sess Ret%  Spread bps
1  SCOM    1.000  0.657    -0.320    +58.4%      +0.50%        17
2  NMG     0.958  1.000    -0.983    +0.0%       -4.61%       175
```

| Column | What it means |
|---|---|
| `Score` (intraday) | Composite score blending the base model + order book + foreign flow + session momentum. Normalised 0–1. |
| `Base` | The production model's score from `nse-pipeline` run earlier today. This is the core signal. |
| `OB Imbal` | Order book imbalance: `(bid_qty − offer_qty) / (bid_qty + offer_qty)`. Positive = more buyers queued than sellers. Range −1 to +1. |
| `Fgn Flow%` | Net foreign buying as % of turnover so far today. Positive = foreigners are net buyers. This is a strong institutional signal on NSE. |
| `Sess Ret%` | Price return from today's open to the latest snapshot. Positive = stock is up on the day. |
| `Spread bps` | Current bid-ask spread in basis points. High spread (> 100 bps) means execution will be expensive and the score is less reliable. |
| `WAP vs Cls` | Latest price minus the weighted average price, in bps. Positive = stock is trading above its session average (buyers are aggressive). |

**Important:** intraday scores shift throughout the session as new snapshots arrive. A score at 10:00 EAT is not comparable to one at 14:30 EAT. Always note the "as of" timestamp in the header. The base model score is static; only the overlay components change intraday.

---

## Quick reference: when to act

| Situation | Interpretation | Action |
|---|---|---|
| High `score`, positive `edge`, `LONG` | Model and risk engine both agree | Review explanation + events, consider entry |
| High `score`, negative `edge` | Model likes it but costs eat the alpha | Only act if you have reason to expect lower execution cost |
| Low `IC (last 20d)` warning | Model signal has degraded recently | Retrain before relying on new signals |
| Positive `hit_rate`, negative `total_return` | Gains are too small relative to losses | Gain/loss ratio problem — not a signal problem |
| High `spread_bps` intraday | Wide market on that name today | Be cautious — actual execution cost will be high |
| Large positive `Fgn Flow%` | Foreigners accumulating | Strong confirmation signal, especially for large-caps |
