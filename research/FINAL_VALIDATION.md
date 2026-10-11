# Validation status

The repository includes the supplied preprint, manuscript source, saved experiment outputs and scripts for rebuilding tables and figures. `PUBLICATION_CHECKLIST.md` records the earlier manuscript-build checks; it is not an independent methodological certification.

Current packaging checks: existing automated feature tests pass, Python sources compile, intraday CLI help loads, and publication assets regenerate from included derived outputs. Full model retraining has not been repeated for this release.

Remaining limits:

- Raw and engineered input snapshots are not publicly distributed; full retraining needs authorized data access.
- Dependency versions are not fully locked and original inputs lack a recorded checksum.
- Survivorship/universe selection, macro release timing, slice-dependent imputation and overlapping-label split boundaries remain audit concerns.
- Benchmark gaps constrain alpha terminology.
- The extended evaluation is retrospective, not a pristine live holdout.
- Positive IC does not establish profitability; cost-sensitive portfolio results are negative.

The paper PDF `publication/nse.pdf` is copied byte-for-byte from the supplied research paper. The available source can be rebuilt separately to `publication/paper.pdf`.
