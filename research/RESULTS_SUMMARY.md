# Results Summary

Status: saved experiments supporting the accompanying preprint. The figures below summarize existing artifacts, not a new independent rerun.

## Reproduced Original IC

The original manuscript value of mean 2024 monthly OOS IC approximately
`0.0911` was not reproduced under the current code/data snapshot.

Current reproducible LightGBM walk-forward output:

```text
mean validation IC:     0.0952
mean monthly test IC:   0.0852
positive test months:   11/12
mean top-1 hit rate:    17.0%
```

Artifacts:

```text
research/outputs/reproduction/walk_forward_2024_summary.csv
research/outputs/reproduction/walk_forward_2024_daily_ic.csv
research/outputs/reproduction/walk_forward_2024_predictions.csv
research/outputs/reproduction/manifest.json
```

## Corrected IC

Not separately corrected for leakage yet. The current reproducible value is
`0.0852` mean monthly test IC for LightGBM.

## Baseline Results

Core ranking baselines were run for 2024 using the same test periods and label:

```text
Random Forest:      mean daily IC  0.1330, positive months 12/12
LightGBM:           mean daily IC  0.0846, positive months 11/12
1-day reversal:     mean daily IC  0.0625, positive months 8/12
3-day reversal:     mean daily IC  0.0563, positive months 9/12
Ridge regression:   mean daily IC  0.0195, positive months 8/12
Linear regression:  mean daily IC  0.0194, positive months 8/12
Random ranking:     mean daily IC -0.0059, positive months 6/12
5-day momentum:     mean daily IC -0.0494, positive months 3/12
20-day momentum:    mean daily IC -0.0520, positive months 4/12
```

Artifact:

```text
research/outputs/baselines/baseline_comparison_2024.csv
```

## Statistical Significance

LightGBM daily IC was evaluated with HAC/Newey-West standard errors using lag 3:

```text
n days:       254
mean IC:      0.0846
median IC:    0.0874
IC std:       0.3027
ICIR:         0.2794
95% CI:       [0.0437, 0.1254]
t-stat:       4.0585
p-value:      6.59e-05
IC > 0 days:  61.8%
```

Artifact:

```text
research/outputs/daily_ic/lightgbm_hac_inference_2024.csv
```

## Strongest Baseline

Random Forest was the strongest model in the baseline run by mean daily IC
(`0.1330`) and positive-month count (`12/12`).

## LightGBM Incremental Performance

LightGBM outperformed simple reversal/momentum and linear/Ridge baselines, but
did not outperform Random Forest in the current baseline run.

## Portfolio, costs, ablations and extended evaluation

Completed experiment outputs are in `outputs/phases_9_15/`:

- Top-1 overlapping three-day sleeves at 400 bps: cumulative return −94.2% (`portfolio_cost_sensitivity_2024.csv`).
- Returns/price-only ablation: mean daily IC 0.125; full features: 0.111 (`feature_ablation_summary_2024.csv`). This is a separate protocol from the original model comparison.
- LLM/event permutation group: zero measured IC drop (`permutation_importance_group_summary_2024.csv`).
- Extended retrospective OOS, 2025-01-02 to 2026-03-31: mean daily IC 0.0815; 11/15 positive months (`extended_oos_classification_2025_2026.csv`).
- Robustness and survivorship diagnostics: `robustness_summary_2024.csv` and `survivorship_audit.csv`. Universe-selection risk remains unresolved.

Runtime base labels use 500 bps. The paper highlights a 400 bps portfolio scenario; changing runtime costs does not rewrite frozen input labels.

## Major Limitations Identified So Far

- The original `0.0911` LightGBM IC claim was not reproduced; current output is
  `0.0852`.
- Random Forest outperformed LightGBM by mean daily IC in the baseline run.
- The local SQLite fallback is empty; reproduction depends on exported research
  snapshots under `research/outputs/reproduction`.
- Missing-value imputation appears to compute medians on validation/test slices;
  this should be corrected or justified before publication.
- Macro release timing, universe selection, and survivorship bias remain open
  audit items.
- Label terminology depends on whether benchmark data are available. If the
  benchmark falls back to zero, the label is net forward return, not alpha.

## Scientific claim

The defensible interim claim is:

> Under the current code/data snapshot, the LightGBM walk-forward model shows
> positive 2024 cross-sectional ranking information in the studied 12-security
> NSE universe, with mean daily IC 0.0846 and HAC/Newey-West p-value
> 6.6e-05. However, the original manuscript's 0.0911 monthly IC was not exactly
> reproduced, Random Forest produced a higher mean daily IC, and the tested
> portfolio rules fail after costs.
