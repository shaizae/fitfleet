# Changelog

All notable changes to fitfleet. Versions follow [semantic versioning](https://semver.org/); while the
version is 0.x, a minor release may contain breaking changes, and they are listed under **Removed** or
**Changed**.

## 0.3.0 - unreleased

### Added
- Optional groups: `set(..., groups=...)` (a NumPy array or Series, one value per row) keeps every group on one
  side of each split - `k_folds` uses `StratifiedGroupKFold` (classification, falling back to `GroupKFold`)
  or `GroupKFold` (regression); `train_test_split`, halving rounds and `feature_selection` pick whole groups;
  new `leave_one_group_out()`, and `run.groups` returns the group of every row. Without groups nothing changes.
- HTML reports with the same content as the PDFs, as one self-contained file (plots embedded):
  `save_report(folder, report_format)` on the experiment report, on every result and on `FeatureResearch`,
  with the new `ReportFormat` enum: `ReportFormat.pdf` (default), `.html` or `.dual` (both files, from one
  rendering of the plots). `save_results(folder, report_format)` passes it on, and `run.save_html_log`
  writes the execution log as HTML. In a config file, `[output] report_format = "pdf" | "html" | "dual"`
  (default `"pdf"`).
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
- The experiment and per-model reports no longer write a separate `execution-log_<id>_<time>.pdf`
  next to the report. The execution log is still in the report as its "Execution Summary" section; use
  `run.save_pdf_log` to write it as its own file.
- `fitfleet.utils` is now a package split by topic. `from fitfleet.utils import ...` works as before.

### Removed
- `save_pdf_report()` on the experiment report, on every result and on `FeatureResearch`. Use
  `save_report(folder)`: it writes the same PDF (`ReportFormat.pdf` is the default) and returns a list of
  paths, so `x.save_pdf_report(folder)` becomes `x.save_report(folder)[0]`. The `FeatureResearch` stage is
  now named "Feature research report".
- The report arguments. `save_report(folder, report_format)` takes nothing else and reads the rest from
  the instance: `FeatureResearch(X, y, model=...)` (the model every importance method and the report
  train when none is passed), `research.correlation_threshold` / `.missing_threshold` / `.leakage_threshold`
  (0.9 / 0.5 / 0.98) and `report.top` (20 leaderboard rows). The feature research report always compares
  with random noise columns (the old `probe=True` default).
- The deprecated aliases from before 0.2: `run.fetchers` (use `run.features`), `fetcher_selection()`
  (use `feature_selection()`) and `brut_force_features()` (use `brute_force_features()`).

## 0.2.0 - 2026-11-18

First release on PyPI: `Classification` / `Regression` runs in a shared-memory process pool, grid / random /
halving search, forward / backward feature selection, mixed data types, `FeatureResearch`, experiment and
per-model PDF reports, execution tracking, the `fitfleet` command line and `SavedModel`.
