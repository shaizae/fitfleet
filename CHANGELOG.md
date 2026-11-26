# Changelog

All notable changes to fitfleet. Versions follow [semantic versioning](https://semver.org/); while the
version is 0.x, a minor release may contain breaking changes, and they are listed under **Removed** or
**Changed**.

## 0.3.0 - unreleased

### Added
- `postgres_set()`: read the features and the target with two SQL queries from PostgreSQL
  (`pip install "fitfleet[postgres]"`). A `docker-compose.yml` starts a throwaway database for the live test.
- Additional metrics in every report and in `print(result)`, computed from the out-of-sample predictions
  (reported only, not used for ranking):
  - classification: balanced accuracy, macro precision / recall / F1, MCC, Cohen's kappa, and, with
    `predict_proba`, ROC AUC, average precision, Brier score (binary), one-vs-rest ROC AUC (multiclass)
    and log loss.
  - regression: explained variance, median absolute error, max error, MAPE (when no target is 0) and
    mean residual (bias).
  - `result.extra_metrics` and `report.extra_metrics_table()` return them.
- `result.fold_metrics()`: the ranking metrics on each k-fold validation fold. The per-model PDF shows them
  in a new "Cross-Validation Analysis" table with mean and standard deviation.
- The reports say how many rows and folds the metrics were computed on ("evaluated on"), and the
  experiment PDF's leaderboard has a fit-time column.

### Changed
- `save_pdf_report()` (experiment and per-model) no longer writes a separate `execution-log_<id>_<time>.pdf`
  next to the report. The execution log is still in the report as its "Execution Summary" section; use
  `run.save_pdf_log` to write it as its own file.
- `fitfleet.utils` is now a package split by topic. `from fitfleet.utils import ...` works as before.

### Removed
- The deprecated aliases from before 0.2: `run.fetchers` (use `run.features`), `fetcher_selection()`
  (use `feature_selection()`) and `brut_force_features()` (use `brute_force_features()`).

## 0.2.0 - 2026-11-18

First release on PyPI: `Classification` / `Regression` runs in a shared-memory process pool, grid / random /
halving search, forward / backward feature selection, mixed data types, `FeatureResearch`, experiment and
per-model PDF reports, execution tracking, the `fitfleet` command line and `SavedModel`.
