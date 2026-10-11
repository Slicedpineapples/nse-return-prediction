# Reproducing the research

Run all Python commands from the repository root with the virtual environment activated. Install with `pip install -e ".[dev,research]"`.

## Included artifacts

Derived predictions, daily/monthly metrics, cost sensitivity, ablations, inference and manifests are included under `research/outputs/`. These allow inspection of the paper’s reported numbers and regeneration of publication figures/tables without database access:

```bash
python research/publication/scripts/generate_assets.py
make -C research/publication paper
```

`research/publication/nse.pdf` is the exact accompanying preprint supplied for this repository. `paper.tex` is the available manuscript source; `make paper` writes a rebuilt `paper.pdf`. The legacy `generate_current_paper.py` writes an earlier draft and should not be run against the current manuscript.

## Retraining inputs

Raw database exports and engineered feature snapshots are kept local pending confirmation of redistribution rights. Full retraining requires:

- `research/outputs/reproduction/features_daily_snapshot.csv`: 18,240 rows, 2019-01-02–2024-12-31, 12 tickers.
- `research/outputs/phases_9_15/features_2025_2026_oos.csv`: 3,840 rows, 2025-01-02–2026-03-31.

Inputs need the `features_daily` schema, including dates, tickers, numeric features, labels and data-quality flags. The saved experiment manifest records the primary label and model configuration. It does not establish a checksum or complete point-in-time provenance of the original inputs.

If the first snapshot is absent, the reproduction script attempts a PostgreSQL export using `DB_HOST`, `DB_PORT`, `DB_NAME`, `DB_USER` and `DB_PASS` from your local `.env`. The extended experiment similarly requires a cached snapshot or database access. A fresh SQLite database contains no research data.

With authorized inputs available, run in order:

```bash
python -m research.scripts.reproduce_2024_walk_forward
python -m research.scripts.analyze_2024_core
python -m research.scripts.run_phases_9_15
python research/publication/scripts/generate_assets.py
make -C research/publication paper
```

These commands replace research outputs. Preserve a copy of the published outputs before rerunning. They isolate research artifacts from production model storage; operational `nse-train` can overwrite saved monthly/production models and is not the paper reproduction entry point.

## Reproduction limits

Dependencies use lower bounds rather than an exact environment lock. The observed environment from the 2026-10-04 audit was Python 3.14.5 on macOS arm64, with NumPy 2.4.3, pandas 3.0.1, SciPy 1.17.1, scikit-learn 1.8.0, LightGBM 4.6.0, SQLAlchemy 2.0.48 and psycopg2-binary 2.9.11. This records the historical environment rather than a tested dependency lock; numerical equality across environments is not guaranteed. Saved snapshot labels may differ from labels regenerated with current cost configuration. The original monthly IC claim of 0.0911 was not reproduced; the saved reproduction yields 0.0852.

The selected universe, missing benchmark history, slice-dependent imputation, macro release timing and overlapping-label split boundaries require further methodological review. Rebuilding tables is artifact reproduction, not independent validation of the underlying data or research design.

## Validation and methodological limits

Release checks passed: 16 existing feature tests, Python source compilation, intraday CLI help, publication asset generation and LaTeX compilation. These packaging checks do not independently validate the research design. Full model retraining was not repeated for the release.

The historical export initially stalled; monthly chunk exports subsequently produced the complete local 2019–2024 snapshot. Partial exports must not be used as experiment inputs. Original inputs lack recorded checksums and complete point-in-time provenance.

Specific audit concerns:

- The scaler is fitted on training rows, but the operational ranker's imputation uses medians from whichever slice it receives. Validation/test medians therefore use evaluation-slice information; training-only imputation needs review.
- Macro observation dates do not encode release timestamps. Corporate-action publication and ex-date availability also need review.
- Overlapping forward labels can cross temporal split boundaries; split purging needs review.
- The fixed 12-security seed universe does not establish historical selection or eliminate survivorship bias.
- Label generation substitutes zero for missing benchmark returns, while monitoring skips outcomes without benchmark data. Alpha terminology must reflect that difference.

A constant cost deducted from every security on a date does not change cross-sectional Spearman ranks. It changes positive-label hit rates and economic returns. Negative labels alone do not establish a loss of ranking information.

Claims are restricted to statistical ranking predictability within the studied universe. The 2025–2026 evaluation is retrospective, and richer features or LLM/event variables have no demonstrated incremental predictive value in the completed experiments. Statistical significance refers to the reported HAC/Newey–West IC inference; it does not establish deployable alpha or profitability. See [results and supporting artifacts](RESULTS_SUMMARY.md) for the evidence behind each numerical claim.
