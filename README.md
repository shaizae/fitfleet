# fitfleet

A scikit-learn package for running many tests in multiprocessing with minimal garbage creation:
every model x feature subset x hyper-parameter combination, scored on the same splits, best ones refitted.

## Install

```
pip install fitfleet         # from PyPI
pip install -e .            # the library, from a clone
pip install -e ".[dev]"     # + test tools
pip install "fitfleet[postgres]"  # + postgres_set() (psycopg)
```

```python
import fitfleet
from fitfleet import Classification, FilteringCriteriaClassification, ReportFormat

with Classification() as run:
    run.set(X, y, [RandomForestClassifier(), SVC()])
    run.features.standard_scaler()
    run.k_folds(5)
    best = run.filter_by(FilteringCriteriaClassification.accuracy)
```

## Mixed data: categories, dates, text and missing values

Features don't have to be numbers. Text and pandas `category` columns, yes/no columns, dates, free text and
missing values are handled without any preparation:

```python
with Classification() as run:
    run.set(X, y, [LogisticRegression(), RandomForestClassifier(), HistGradientBoostingClassifier()],
            ordinal={"size": ["S", "M", "L"]})   # optional: kinds are detected, declare to override
    run.features.standard_scaler()              # applied to number columns only
    run.k_folds(5)
```

| Column kind | Detected from | How models get it |
|---|---|---|
| numeric, boolean | numbers, `bool` | as is (yes/no as 0/1) |
| categorical | text with few words, pandas `category` | one-hot for linear / distance models (target encoding above `max_onehot_categories`, default 40, categories); integer codes for tree models; native categories for HistGradientBoosting |
| ordinal | declared `ordinal={col: [lowest, ..., highest]}`, ordered pandas `category` | its position in the order, as a number |
| datetime | datetime dtype, or declared `datetime=[...]` | year, month, day, weekday, day of year, hour, days since 1970 |
| text | strings of 3+ words on average with many distinct values, or declared `text=[...]` | 30 numbers per column (character TF-IDF + SVD) |
| missing values | NaN / None | kept for models that accept NaN (trees, HistGradientBoosting, LightGBM, XGBoost); otherwise filled with the median plus a "was missing" column |

- Every encoder, imputer and scaler is fitted on the training rows of each split only. The category list and
  the text embedding are learned from all rows before splitting; neither looks at the target.
- Feature subsets (brute force, forward / backward selection, `feature_selection`) choose *original* columns:
  a categorical or text column is kept or dropped as a whole. Purely numeric tables without gaps behave as before.
- Text columns where almost every row has its own value (ids, keys) are left out with a warning
  (`drop_ids=False`, or declaring the column, keeps them). Date columns trigger a warning about shuffled folds.
- `run.categorical_encoding = "onehot" | "ordinal" | "target"` forces one encoding for every model. To compare
  encodings on the same folds, search them like a hyper-parameter:
  `run.set_hyper_parameters({"categorical_encoding": ["onehot", "target"]})`.
- A saved model takes raw rows: text, unseen categories, new dates and missing values are encoded the same way.
- polars DataFrames are accepted wherever pandas ones are.
- In a config file, the optional `[columns]` section sets `categorical`, `numeric`, `ordinal`, `datetime`, `text`,
  `drop`, `drop_ids`, `encoding` and `max_onehot` (see `fitfleet init`).

## Data from PostgreSQL

`postgres_set()` works like `set()`, but reads the features and the target with two SQL queries.
Both must return the same rows in the same order (give them the same `ORDER BY`); the target query
returns one column. Keyword arguments such as `categorical=` go to `set()`.

```python
with Classification() as run:
    run.postgres_set(
        "postgresql://user:password@localhost:5432/shop",  # or {"host": ..., "dbname": ..., "user": ..., "password": ...}
        features_query="SELECT age, city, spend FROM customers ORDER BY id",
        target_query="SELECT churned FROM customers ORDER BY id",
        models=[RandomForestClassifier(), LogisticRegression()],
    )
    run.k_folds()
```

## Groups: rows that belong together

When several rows come from the same patient, user, store or session, testing on rows whose "siblings"
were trained on gives optimistic scores. Pass `groups=` to `set()` (optional; without it nothing changes)
and a group is never split between training and testing:

```python
with Classification() as run:
    run.set(X, y, models, groups=df["patient_id"])   # a NumPy array or Series, one value per row
    run.k_folds(5)               # StratifiedGroupKFold (classification), GroupKFold (regression)
    run.train_test_split(0.8)    # 80% of the groups train, 20% test
    run.leave_one_group_out()    # one fold per group
```

- Classification falls back to `GroupKFold`, with a warning, when a class appears in fewer groups than folds.
- `n_splits` can't be more than the number of groups; `leave_one_out()` is refused (use `leave_one_group_out()`).
- `run.groups` returns the group of every row (the labels you passed) as a NumPy array, or `None` without groups.
- Halving rounds and `feature_selection`'s 10% hold-out pick whole groups too.

## Command line

```
fitfleet init experiment.toml          # example config with every option explained
fitfleet run experiment.toml --check   # validate the config and the data, no training
fitfleet run experiment.toml           # train every configuration
fitfleet predict runs/<date_time>/1_<Model>/model.joblib new.csv -o predictions.csv --proba
fitfleet log runs/<date_time>          # what passed, failed or was skipped, where and why
```

`python -m fitfleet ...` works too. A config names the data file, target column and task, the models (any
scikit-learn class name, or an import path like `xgboost.XGBClassifier`), a hyper-parameter grid, preprocessing,
optional feature selection and the validation. Each run writes a new `runs/<date_time>/` folder with
`leaderboard.csv` (every configuration, best first; ties go to fewer features), a copy of the config, and the
top `keep` models, each in its own folder with `model.joblib` and a report. The reports are PDFs by default;
`[output] report_format = "html"` writes them as HTML files instead, and `"dual"` writes both.

## Execution tracking

Every stage of a run is recorded: what ran, where, when it started and finished, whether it PASSED,
FAILED, was SKIPPED or CANCELLED and, for a failure, a plain-language reason, the exception type, the
location (`file.py:line in function`, inside the worker process for training tasks) and the full traceback.
`fitfleet run` prints one line per stage and an execution summary at the end (`-v` adds tracebacks), and
the run folder gets `execution_<id>.log` (developer log), `.jsonl` (live JSON events), `.json` and
`_summary.txt`; the experiment report gets an Execution Summary section.

```
fitfleet log runs/<date_time>                       # summary + stage tree
fitfleet log runs/<date_time> --status fail         # failed stages with reason and location
fitfleet log runs/<date_time> --expand "Train and validate" --depth 2
fitfleet log runs/<date_time> --details "Load data" # everything about one stage, with its traceback
```

Filters: `--status`, `--stage`, `--module`, `--level`, `--since` / `--until`, `--error-type`,
`--execution-id`; `--json` for scripts. In Python, a run has four log properties, each
with the summary and the stage tree: `run.execution_log` (text), `run.save_pdf_log`, `run.save_html_log`
and `run.save_json_log` (written into `log_dir`, set in the constructor:
`Classification(log_dir="reports")`). `save_report()` adds the log to the
report as its Execution Summary section; use `run.save_pdf_log` or `run.save_html_log` for a separate log file.
See [docs/execution_tracking.md](https://github.com/shaizae/fitfleet/blob/master/docs/execution_tracking.md).

## Search smarter than every combination

The default tries every model x feature subset x hyper-parameter combination (grid). For large grids:

```python
run.search("random", n_iter=200)                  # 200 combinations drawn at random
run.search("halving", factor=3)                   # successive halving (below)
run.search("halving", n_iter=500, min_rows=200)   # halving on a random sample of the grid
run.forward_selection(max_features=10, tol=0.002) # greedy feature subsets instead of brute force
run.backward_elimination(min_features=3)
run.k_folds(5)
run.search_log                                    # what each round trained
```

- **halving**: every combination is trained on a small share of the rows, the best 1/`factor` go on to
  `factor` x more rows, and the last round, on every row, is an ordinary k-fold / train-test run. Only the
  last round's results are returned, so they compare fairly. Not available with leave-one-out.
- **forward / backward**: for each model x hyper-parameter combination, add (remove) the one feature that
  helps most, until a step gains no more than `tol`. About n²/2 subsets instead of 2ⁿ, so it works with many
  columns. Every subset trained is in the results. Uses the default grid search.
  `tol` only decides when to stop: the last step's tries (one feature more, which did not gain more than
  `tol`) are in the results too, and one of them can still top the leaderboard by a tiny, chance margin.
  To get the subset the search settled on, take the best of the step before (`run.search_log` lists the
  steps).
- Rounds are compared by `select_by` (default accuracy / r2): `run.search("halving", select_by=...)`.

In a config file: `[search] strategy = "halving"` (with `n_iter`, `factor`, `min_rows`), and
`[feature_selection] method = "forward"` or `"backward"` (with `max_features`, `min_features`, `tol`).
`select_by` from `[run]` is used for both.

## Experiment report

After a run, summarize every configuration it trained (classification or regression):

```python
run.k_folds(5)
report = run.experiment_report()          # ranked by accuracy / r2; or experiment_report("f1"), ("mae"), ...
print(report)                             # overview, best configuration, per-model summary, top 10
report.leaderboard                        # DataFrame: every configuration, best first
report.summary                            # DataFrame: per model - configurations, best / mean / std / worst, ...
report.best                               # the best configuration's row
report.save_csv("out")                    # out/leaderboard.csv + out/summary.csv
report.save_report("out")                 # out/experiment_report_<date>.pdf (ReportFormat.pdf is the default)
report.save_report("out", ReportFormat.html)  # out/experiment_report_<date>.html: the same report, one file
report.save_report("out", ReportFormat.dual)  # both files, from one rendering of the plots
report.roc_curves()                       # classification: {model: (fpr, tpr, auc)}, best config per model
report.plot_roc_curves()
```

`fitfleet run` writes `summary.csv` and the experiment report into the run folder too.

### Plots and the PDF / HTML reports

`save_report` draws every plot that applies to the run and puts all of them in the report. The sections are:

- an executive summary, the configuration and a dataset overview
- the model ranking and the spread of scores over configurations
- **classification**: metrics against the majority-class baseline, a confusion matrix per model (counts and
  row-normalized), per-class precision / recall / F1, ROC and precision-recall curves per model and per
  class, and a calibration diagram (binary)
- **regression**: R² / MAE / RMSE, actual vs predicted, residuals vs predicted, the residual distribution and
  a normal Q-Q plot, error by target range, and the error distribution across models
- per-fold scores (k-fold), hyper-parameter sensitivity and two-parameter heatmaps, and score against the
  number of features
- runtime per configuration and score against cost (with the Pareto front)
- feature importance for refitted models

Each plot comes with an explanation and the values it shows. A last section lists every plot that was
skipped (not applicable, or not measured, such as memory) or failed.

**HTML reports.** Every report (experiment, per-model, feature research) has a
`save_report(folder, report_format)` method; `report_format` is a `ReportFormat`
(`from fitfleet import ReportFormat`): `ReportFormat.pdf` (the default),
`ReportFormat.html` or `ReportFormat.dual` (both files). The HTML is the same report: the same sections and numbering, tables, plots,
explanations, appendix and execution log. Both formats are built from the same content, so they always
match. The HTML is one self-contained file: the plots are embedded, nothing is loaded from the network, so
you can open it in any browser, attach it to an email or archive it. It adds a table of contents, and when
printed it starts a new page wherever the PDF does. `ReportFormat.dual` draws the plots only once for
both files; the strings `"pdf"`, `"html"` and `"dual"` work too. `result.save_results(folder, ReportFormat.dual)`
saves the model with both reports. `save_report` returns the paths it wrote, PDF first.

**Plotting can never stop an experiment.** Each plot is drawn in its own error boundary
(SUCCESS / SKIPPED / FAILED, logged on the `fitfleet.visualization` logger). `fitfleet run` saves the
models first and writes the reports on a background thread. A failing plot, PDF or HTML report is logged as
a warning, and the run still finishes with its leaderboard and models.

```python
viz = report.visualize()                  # every plot, without a PDF; never raises
print(viz.summary())                      # "21 generated, 3 skipped, 0 failed"
viz.save_images("out/plots")              # one PNG per generated plot
viz.failed, viz.skipped                   # records with the reason / traceback
best[0].save_report("out")                # per-model PDF report (confusion, ROC, residuals, importance, ...)
best[0].save_report("out", ReportFormat.html)  # the same per-model report as HTML
```

The research behind the choice of plots, the architecture, and how to add a plot type are in
[docs/visualization.md](https://github.com/shaizae/fitfleet/blob/master/docs/visualization.md).

## Feature research

A stand-alone class (no `Classification` / `Regression` run needed) to understand the features before
choosing them. The task is guessed from `y` (or pass `task="classification"` / `"regression"`).

```python
from fitfleet import FeatureResearch

research = FeatureResearch(X, y, cv=5, model=model)  # model: optional, default a random forest
research.missing_report()               # missing values per column
research.constant_features(0.99)        # constant / quasi-constant columns
research.cardinality()                  # distinct values; flags id-like columns
research.distribution_summary()         # skew, kurtosis, % IQR outliers
research.univariate_scores()            # F-test (+ Pearson / Spearman for regression)
research.mutual_information()           # also catches non-linear relations
research.single_feature_score()         # CV score of a model on each feature alone
research.leakage_check(0.98)            # features that alone score suspiciously high
research.correlation_matrix()           # feature x feature (Spearman by default)
research.high_correlation_pairs(0.9)    # `drop` = the less informative of each pair
research.vif()                          # multicollinearity
research.model_importance()             # feature_importances_ / |coef_|
research.permutation_importance()       # score drop on held-out folds, any model
research.random_probe()                 # does each feature beat pure-noise columns?
research.report()                       # one row per feature, every score + mean rank
X = X.drop(columns=research.recommend(probe=True).index)   # constant / id / missing / redundant / noise
research.save_report("reports")         # all of the above in one PDF, with charts
research.save_report("reports", ReportFormat.html)  # the same report as one HTML file (.dual: both)
```

`save_report` writes `feature_research_<date>.pdf` (`ReportFormat.html`: `.html`): overview and target balance, columns to drop and
possible leakage, the feature ranking, a permutation-importance chart (with the best noise column), data
quality, correlated pairs, VIF and a correlation heatmap. It trains the instance's `model` and reads its
thresholds from the instance: `research.correlation_threshold` (0.9), `research.missing_threshold` (0.5)
and `research.leakage_threshold` (0.98). The charts are also available on their own:
`research.plot_importance()`, `research.plot_correlation()`.

A feature is called `noise` only when it beats the random noise columns on neither permutation importance
nor mutual information: permutation importance alone under-rates correlated features (shuffling one
costs nothing while its twin is still there).

The data is kept like in a run: `research.features` is a `Features` object and `research.targets` a
`Target` (label-encoded for classification, `research.targets.classes` maps back). A step applied there,
e.g. `research.features.standard_scaler()`, is used by every method after it.

Every method returns a DataFrame or Series. The statistical methods use the numeric columns only, with
missing values filled by the median (`research.skipped` lists the others). The default model is a random
forest; pass any scikit-learn estimator instead.

## Sound when training is done

Pass `sound=` to hear when a run has finished. The sound plays once, when the object goes out of
scope (the end of its `with` block, or when it is garbage-collected), and only if at least one
training run finished.

```python
from fitfleet import Classification, FilteringCriteriaClassification, Sound

with Classification(sound=Sound.success) as run:   # or sound="success"
    run.set(X, y, models)
    run.k_folds(5)
    best = run.filter_by(FilteringCriteriaClassification.accuracy)
# the success sound plays here

Classification.play_sound(Sound.chime)              # play one any time (a static method)
```

| `Sound`   | What it sounds like                                              |
|-----------|------------------------------------------------------------------|
| `none`    | silent (the default)                                             |
| `ping`    | one soft bell note                                               |
| `double`  | two rising notes, a happy "ding-ding"                            |
| `chime`   | three bell notes ringing into a major chord                      |
| `success` | quick rising arpeggio ending on a bright chord: "it worked!"     |
| `sparkle` | a fast twinkling run up high, playful                            |
| `alarm`   | repeated high/low tone, hard to miss                             |
| `bell`    | the terminal bell only                                           |

The sounds are generated in code (soft bell-like tones), so there are no audio files or extra
dependencies. They play with `winsound` on Windows, `afplay` on macOS and `paplay` / `aplay` on Linux;
if none of these is available, the terminal bell rings instead. Playing a sound never raises an error.

In a config file: `[output] sound = "success"`. The older `ping = true` still works (same as
`sound = "ping"`).

## Use a trained model

```python
path = best[0].save_model("models")       # a .joblib file in models/
model = fitfleet.load(path)                # only load files you trust
model.predict(new_df)                     # picks the training columns by name; returns the original labels
model.predict_proba(new_df)               # DataFrame, one column per class
model.metrics, model.params, model.feature_columns, model.class_names
```

The saved model includes the run's scaling, polynomial features and feature subset, so pass raw rows.
Models saved before 0.2 (a bare estimator) still load.

Renamed in 0.2: `fetchers` -> `features`, `fetcher_selection()` -> `feature_selection()`,
`brut_force_features()` -> `brute_force_features()`. The old names were removed. Imports moved from
`external_classes` / `utils` / `results` into the `fitfleet` package.

## Usage notes

- `train_test_split(ratio)`: `ratio` is the fraction of rows used for **training** (0.8 = 80% train).
- Scalers and polynomial features (`pipeline.features.standard_scaler()` etc.) are refitted on the
  training rows of every split, so test rows never leak into preprocessing.
- Classification uses stratified splits. All models in a run are scored on the same folds.
- k-folds shuffles the rows, so without a seed each run uses different folds and scores can change a
  little from run to run. Set `run.random_state = 0` (or `random_state` in `[run]`) for repeatable results.
- Results carry metrics and predictions only. `filter_by()` (or `refit(results)`) fits the chosen
  models; the fitted model is a scikit-learn `Pipeline` that takes raw rows.
- The worker pool is reused across runs on the same object. Use `with Classification() as c:` or
  call `c.close()` to stop the workers.

## Tests

```
pytest                              # unit tests (load tests are skipped)
pytest --run-load -m load -s        # CPU load tests, a few minutes, prints a CPU timeline
python -m benchmarks.bench          # timing / memory benchmark
```

The live PostgreSQL test runs only when `FITFLEET_TEST_POSTGRES_URL` is set. `docker-compose.yml` starts a
throwaway database for it and can run the test in Docker:

```
docker compose run --rm --build tests   # database + tests/test_postgres.py, all in Docker
docker compose down
```

or only the database, with the tests from your own environment (install `.[postgres]` first):

```
docker compose up -d --wait
FITFLEET_TEST_POSTGRES_URL=postgresql://postgres:pw@localhost:55432/postgres pytest tests/test_postgres.py
docker compose down
```


## Changes

What changed in each version, including breaking changes: [CHANGELOG.md](https://github.com/shaizae/fitfleet/blob/master/CHANGELOG.md).

## License

MIT - see [LICENSE](https://github.com/shaizae/fitfleet/blob/master/LICENSE).
