# Claim Audit

This audit checks sensitive language against the completed experiment outputs in `research/outputs`.

## Sensitive Terms

- `significant`: Used only for statistical interpretation of the HAC/Newey--West LightGBM IC test or to distinguish economic significance from statistical ranking evidence.
- `robust`: Used only in the phrase `robustness` or to describe standard errors/robustness tests, not to claim robust alpha.
- `outperform`: Used for experimentally supported comparisons: Random Forest versus LightGBM in the 2024 model comparison; returns/price-only versus full feature set in the ablation protocol.
- `profitable`: Used only to deny demonstrated profitability under tested assumptions.
- `alpha`: Avoided as a substantive claim; retained only in the implementation variable name `label_3d_net_alpha` and in the phrase `alpha construction` as a limitation.
- `predictive`: Used for ranking predictability within the studied universe, not for profitability.
- `improvement`: Avoided as a central claim; richer features are not described as improvements.
- `superior`: Not used for unsupported model claims.

## Source-of-Truth Checks

- Random Forest mean daily IC `0.133` and positive months `12/12`: `research/outputs/baselines/baseline_comparison_2024.csv`.
- LightGBM model-comparison mean daily IC `0.085`: `research/outputs/baselines/baseline_comparison_2024.csv`.
- Returns/price-only ablation mean daily IC `0.125` and full feature set `0.111`: `research/outputs/phases_9_15/feature_ablation_summary_2024.csv`.
- LLM/event variables zero measured permutation IC drop: `research/outputs/phases_9_15/permutation_importance_group_summary_2024.csv`.
- 2025--2026 retrospective extended OOS mean daily IC `0.0815`, positive months `11/15`: `research/outputs/phases_9_15/extended_oos_classification_2025_2026.csv`.
- Top-1, 400 bps cumulative return about `-94.2%`: `research/outputs/phases_9_15/portfolio_cost_sensitivity_2024.csv`.
- Survivorship/universe-selection risk unresolved: `research/outputs/phases_9_15/survivorship_audit.csv`.

## Conclusion

The manuscript supports statistical cross-sectional ranking predictability within the studied 12-security universe. It does not claim deployable alpha, production-grade profitability, or incremental LLM predictive value.
