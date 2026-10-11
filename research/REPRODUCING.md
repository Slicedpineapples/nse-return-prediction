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

Dependencies use lower bounds rather than an exact environment lock. Historical observed package versions are in `REPRODUCIBILITY_AUDIT.md`; numerical equality across environments is not guaranteed. Saved snapshot labels may differ from labels regenerated with current cost configuration. The original monthly IC claim of 0.0911 was not reproduced; the saved reproduction yields 0.0852.

The selected universe, missing benchmark history, slice-dependent imputation, macro release timing and overlapping-label split boundaries require further methodological review. Rebuilding tables is artifact reproduction, not independent validation of the underlying data or research design.
