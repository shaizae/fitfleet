from __future__ import annotations

from abc import ABC, abstractmethod
import datetime
import os
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import TYPE_CHECKING, Self

import numpy as np

from fitfleet.model import SavedModel
from fitfleet.utils import ValidationType, create_path, in_enum, is_fitted

if TYPE_CHECKING:
    import pandas as pd
    from sklearn.base import BaseEstimator

# What each report-only metric means, printed next to its value in the PDFs.
METRIC_INFO = {
    "balanced_accuracy": "mean recall over the classes; not inflated by a large class",
    "macro_precision": "precision averaged over classes, each class weighted equally",
    "macro_recall": "recall averaged over classes, each class weighted equally",
    "macro_f1": "F1 averaged over classes, each class weighted equally (small classes count as much)",
    "mcc": "Matthews correlation, -1..1; 0 = no better than chance, robust to imbalance",
    "cohen_kappa": "agreement with the truth beyond chance; 0 = chance, 1 = perfect",
    "roc_auc": "ROC AUC for the positive (second) class; 0.5 = chance",
    "roc_auc_ovr": "ROC AUC one-vs-rest, averaged over classes (macro); 0.5 = chance",
    "average_precision": "area under the precision-recall curve of the positive class",
    "log_loss": "cross-entropy of the predicted probabilities (lower is better)",
    "brier": "mean squared error of the positive-class probability (lower is better)",
    "explained_variance": "share of the target's variance explained (like R², ignores a constant bias)",
    "median_ae": "median absolute error: the typical error, insensitive to outliers (target units)",
    "max_error": "largest absolute error of any row (target units)",
    "mape": "mean absolute percentage error, as a fraction (0.1 = 10%); left out if a target is 0",
    "mean_residual": "mean of actual - predicted: > 0 under-predicts, < 0 over-predicts on average",
}


class BaseResults(ABC):
    """Out-of-sample predictions and metrics of one model configuration.

    Besides the model, a result records `features_index` (the column indexes this run
    was trained on), `feature_columns` (the names of the raw input columns) and
    `class_names` (the original class labels, classification only), the last two set
    by refit(), and `candidate_id` (which task of its batch produced this result).
    Workers also record `fit_time` (seconds to fit and predict every split of this
    configuration, on one worker core) and, for k-fold runs, `fold_sizes` (rows per
    fold, in the order the pooled predictions are stored); both stay None otherwise.
    """

    def __init__(self, model: BaseEstimator) -> None:
        """Wrap `model` (an estimator or Pipeline) with no test data or predictions yet."""
        self.model = model
        self._features_test = None
        self._target_test = None
        self._pred_test = None
        self._validation = ""
        self.features_index = None
        self.feature_columns = None
        self.schema = None  # ColumnSchema of a mixed table (set by refit), so a saved model can encode new rows
        self.categorical_encoding = None  # set when the encoding was searched as a hyper-parameter
        self.class_names = None
        self.candidate_id = None
        self.fit_time: float | None = None
        self.fold_sizes: np.ndarray | None = None

    @property
    def name(self) -> str:
        """The model's class name; for an sklearn Pipeline, that of its final estimator."""
        model = self.model
        if hasattr(model, "steps"):
            model = model.steps[-1][1]
        return model.__class__.__name__

    @property
    def validation(self) -> ValidationType | str:
        """How the result was validated ("" until set)."""
        return self._validation

    @validation.setter
    def validation(self, validation: ValidationType | str) -> None:
        """Set the validation type; raises ValueError if it is not a ValidationType value."""
        if not in_enum(ValidationType, validation):
            raise ValueError(f"{validation} is not valid")
        self._validation = validation

    @property
    def features_test(self) -> np.ndarray | None:
        """The test feature rows, or None when not set (evaluate() clears them)."""
        if self._features_test is None:
            return None
        return self._features_test


    @property
    def target_test(self) -> np.ndarray | None:
        """The true targets of the test rows, or None when not set."""
        if self._target_test is None:
            return None
        return self._target_test

    def set_test(self, features_test: np.ndarray, target_test: np.ndarray) -> None:
        """Store the test features and targets for predict()."""
        self._features_test = features_test
        self._target_test = target_test

    def append_results(self, result: BaseResults) -> Self:
        """Concatenate `result`'s test features, targets and predictions onto this one's.

        Used to pool the folds of one configuration; the model is replaced by `result`'s.
        Returns self.
        """
        if result.features_test is not None:
            if self.features_test is None:
                self._features_test = result.features_test.copy()
            else:
                self._features_test = np.concatenate(
                    [self.features_test, result.features_test],
                    axis=0,
                )

        if result.target_test is not None:
            if self.target_test is None:
                self._target_test = result.target_test.copy()
            else:
                self._target_test = np.concatenate(
                    [self.target_test, result.target_test],
                    axis=0,
                )

        if result._pred_test is not None:
            if self._pred_test is None:
                self._pred_test = result._pred_test.copy()
            else:
                self._pred_test = np.concatenate(
                    [self._pred_test, result._pred_test],
                    axis=0,
                )

        self.model = result.model
        return self

    @property
    def params(self) -> dict:
        """The model's own parameters (of the final step when the model is a Pipeline), plus
        "categorical_encoding" when the encoding was one of the searched hyper-parameters."""
        model = self.model.steps[-1][1] if hasattr(self.model, "steps") else self.model
        params = model.get_params(deep=False) if hasattr(model, "get_params") else {}
        encoding = getattr(self, "categorical_encoding", None)
        if encoding is not None:
            params = {**params, "categorical_encoding": encoding}
        return params

    @property
    @abstractmethod
    def metrics(self) -> dict[str, float]:
        """Every metric of this result, by name."""
        ...

    @property
    def extra_metrics(self) -> dict[str, float]:
        """Metrics shown in the reports but not used to rank or filter, from the stored predictions.

        {} without predictions. A metric that does not apply (e.g. ROC AUC without predict_proba,
        MAPE when a target is 0) or cannot be computed is left out rather than estimated."""
        if self._target_test is None or self._pred_test is None or len(self._target_test) == 0:
            return {}
        out = {}
        for name, compute in self._extra_metric_functions():
            try:
                value = float(compute())
            except Exception:  # noqa: BLE001 - one inapplicable metric must not cost the others
                continue
            if np.isfinite(value):
                out[name] = value
        return out

    def _extra_metric_functions(self) -> list[tuple[str, Callable[[], float]]]:
        """Hook: (name, zero-argument function) for each extra metric of this task."""
        return []

    def fold_metrics(self) -> pd.DataFrame | None:
        """The ranking metrics on each validation fold, one row per fold (k-fold results only).

        The pooled predictions are stored fold by fold (`fold_sizes`); None when the folds are not
        recorded (train / test split), or are single rows (leave-one-out)."""
        import pandas as pd
        sizes, y, pred = self.fold_sizes, self._target_test, self._pred_test
        if sizes is None or y is None or pred is None:
            return None
        sizes = np.asarray(sizes, dtype=int)
        if len(sizes) < 2 or sizes.sum() != len(y) or sizes.min() < 2:
            return None
        proba = getattr(self, "_predict_score", None)
        bounds = np.concatenate([[0], np.cumsum(sizes)])
        rows = []
        for a, b in zip(bounds[:-1], bounds[1:]):
            part = type(self)(self.model)
            part.evaluate(y[a:b], pred[a:b], None if proba is None else proba[a:b])
            rows.append({"rows": int(b - a), **part.metrics})
        return pd.DataFrame(rows, index=pd.RangeIndex(1, len(rows) + 1, name="fold"))

    def evaluation_rows(self) -> str:
        """How many rows the metrics were computed on, and how they were split into folds."""
        n = 0 if self._target_test is None else len(self._target_test)
        sizes = self.fold_sizes
        if sizes is None or len(sizes) < 2:
            return f"{n} rows"
        low, high = int(np.min(sizes)), int(np.max(sizes))
        each = f"{low}" if low == high else f"{low}-{high}"
        return f"{n} rows in {len(sizes)} folds ({each} rows each)"

    def to_model(self) -> SavedModel:
        """The fitted model with its column names, labels and scores, ready to save or predict."""
        if not is_fitted(self.model):
            raise ValueError(f"{self.name} is not fitted: results from a run hold only the model's "
                             f"parameters. Use filter_by() or refit() on the pipeline before saving.")
        from fitfleet import __version__
        return SavedModel(
            estimator=self.model,
            task="classification" if getattr(self, "is_classification", False) else "regression",
            feature_columns=None if self.feature_columns is None else list(self.feature_columns),
            class_names=None if self.class_names is None else list(self.class_names),
            metrics={k: float(v) for k, v in self.metrics.items()},
            params={k: v for k, v in self.params.items() if isinstance(v, (int, float, str, bool, type(None)))},
            validation=str(self.validation),
            fitfleet_version=__version__,
            schema=getattr(self, "schema", None),
        )

    def save_model(self, path: str) -> Path:
        """Save to a new file in the folder `path`; load it with fitfleet.load(file)."""
        return self.to_model().save(path)

    def save_results(self, path: str) -> None:
        """Save the model and the PDF report into a new timestamped subfolder of `path`.

        The model is saved first; a failure while writing the report is logged and does
        not affect the saved model."""
        from fitfleet.visualization.core import safe_call
        name = f"{self.name}_{datetime.datetime.now():%Y-%m-%d_%H-%M-%S}"
        folder = os.path.join(path, name)

        create_path(folder)

        self.save_model(folder)
        safe_call(self.save_pdf_report, folder, what=f"PDF report of {self.name}")

    def visualize(self):
        """Every plot that applies to this result, each generated in isolation.

        Returns a VisualizationReport (one SUCCESS / SKIPPED / FAILED record per plot); never raises."""
        from fitfleet.visualization import VisualizationContext, VisualizationManager
        return VisualizationManager().run_for(lambda: VisualizationContext.from_result(self))

    # ExecutionTracker of the run that produced this result (set by refit / filter_by); the per-model
    # PDF adds its execution report. Not pickled: a saved result does not carry a live tracker.
    _execution = None

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        state.pop("_execution", None)
        return state

    def _build_pdf_report(self, filename: str, title: str) -> Path:
        """The per-model PDF: metrics, then every applicable plot, then the visualization appendix."""
        import pandas as pd
        from fitfleet.visualization.pdf import PdfReport
        visualizations = self.visualize()
        self.last_visualizations = visualizations.without_images()  # statuses only; images go to the PDF
        pdf = PdfReport(title, f"Validation: {self.validation or 'n/a'}  ·  generated by fitfleet",
                        landscape_pages=False)
        pdf.key_values("Executive Summary", [("model", self.name), ("validation", str(self.validation) or "n/a"),
                                             ("evaluated on", self.evaluation_rows())]
                       + [(k, f"{v:.4f}") for k, v in self.metrics.items()]
                       + ([("fit + predict time", f"{self.fit_time:.3f} s")] if self.fit_time is not None else []))
        extra = self.extra_metrics
        if extra:
            pdf.heading("Executive Summary", "Additional metrics (reported only, not used for ranking)")
            pdf.key_values("Executive Summary", [(k, f"{v:.4f}  ({METRIC_INFO[k]})" if k in METRIC_INFO
                                                  else f"{v:.4f}") for k, v in extra.items()])
        params = {k: v for k, v in self.params.items()}
        if params:
            pdf.key_values("Experiment Configuration", [(k, repr(v)) for k, v in sorted(params.items())])
        self._report_details(pdf)
        folds = self.fold_metrics()
        if folds is not None:
            section = "Cross-Validation Analysis"
            pdf.paragraph(section, "Each metric computed on every validation fold separately. The pooled values "
                                   "in the summary use every row at once, so they are not exactly the mean of "
                                   "the folds. Folds share training rows: the spread understates the true variance.")
            stats = folds.drop(columns=["rows"]).agg(["mean", "std"]).rename(index={"std": "sd"})
            pdf.table(section, pd.concat([folds.astype({"rows": object}), stats]).rename_axis("fold"))
        from fitfleet.results.experiment import _add_execution
        _add_execution(pdf, self._execution)  # the run's log, as the "Execution Summary" section
        return pdf.build(filename, visualizations)

    def _report_details(self, pdf) -> None:
        """Hook: add task-specific text to the per-model PDF."""

    @abstractmethod
    def predict(self) -> None:
        """Predict features_test with self.model, then evaluate()."""
        ...

    @abstractmethod
    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray, proba: np.ndarray | None = None) -> None:
        """Store out-of-sample predictions and compute the metrics from them."""
        ...

    @abstractmethod
    def report_lines(self) -> Iterator[str]:
        """Yield the lines of the text report ("" marks a blank line)."""
        ...

    @abstractmethod
    def save_pdf_report(self, filename: str) -> None:
        """Write a PDF report into the folder `filename` (with the run's execution log as a section)."""
        ...

    @abstractmethod
    def __str__(self) -> str:
        """The text report."""
        ...
