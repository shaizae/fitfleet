from __future__ import annotations

from collections.abc import Iterator
from typing import TYPE_CHECKING, Literal

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score, classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.utils.multiclass import unique_labels

from fitfleet.results.base import BaseResults

if TYPE_CHECKING:
    from sklearn.base import BaseEstimator

class ClassificationResults(BaseResults):
    """Results of a classifier: accuracy and weighted precision / recall / F1.

    `is_classification` tells the workers to align the predict_proba columns across folds.
    """
    is_classification = True

    def __init__(self, model: BaseEstimator) -> None:
        """Wrap the classifier `model`; all metrics start at 0."""
        super().__init__(model)
        self._predict_score = None
        self._accuracy: float = 0
        self._precision: float = 0
        self._recall: float = 0
        self._f1: float = 0

    @property
    def metrics(self) -> dict[str, float]:
        """Accuracy, precision, recall and F1, by name."""
        return {"accuracy": self.accuracy, "precision": self.precision, "recall": self.recall, "f1": self.f1}

    @property
    def accuracy(self) -> float:
        """Fraction of test rows predicted correctly."""
        return self._accuracy

    @property
    def precision(self) -> float:
        """Precision averaged over the classes, weighted by their support."""
        return self._precision

    @property
    def recall(self) -> float:
        """Recall averaged over the classes, weighted by their support."""
        return self._recall

    @property
    def f1(self) -> float:
        """F1 score averaged over the classes, weighted by their support."""
        return self._f1

    def _proba_classes(self) -> np.ndarray | None:
        """The class each predict_proba column belongs to, or None if that can't be told."""
        proba, y = self._predict_score, self._target_test
        if proba is None or y is None or np.ndim(proba) != 2:
            return None
        k = np.shape(proba)[1]
        try:
            classes = getattr(self.model, "classes_", None)  # a Pipeline delegates to its final step
        except Exception:  # noqa: BLE001 - unfitted models may raise anything here
            classes = None
        if classes is not None and len(classes) == k:
            return np.asarray(classes)
        seen = np.unique(y)
        if len(seen) == k:
            return seen
        if np.issubdtype(seen.dtype, np.integer) and seen.min() >= 0 and seen.max() < k:
            return np.arange(k)  # label-encoded targets: column i = class i (aligned by the workers)
        return None

    def _extra_metric_functions(self):
        """Imbalance-robust and chance-corrected scores; probability scores when predict_proba exists."""
        from sklearn.metrics import (average_precision_score, balanced_accuracy_score, brier_score_loss,
                                     cohen_kappa_score, log_loss, matthews_corrcoef, roc_auc_score)
        y, p = self._target_test, self._pred_test
        macro = {"average": "macro", "zero_division": 0}
        out = [("balanced_accuracy", lambda: balanced_accuracy_score(y, p)),
               ("macro_precision", lambda: precision_score(y, p, **macro)),
               ("macro_recall", lambda: recall_score(y, p, **macro)),
               ("macro_f1", lambda: f1_score(y, p, **macro)),
               ("mcc", lambda: matthews_corrcoef(y, p)),
               ("cohen_kappa", lambda: cohen_kappa_score(y, p))]
        classes = self._proba_classes()
        if classes is None:
            return out
        proba = np.asarray(self._predict_score, dtype=float)
        if len(classes) == 2:
            positive, score = np.asarray(y) == classes[1], proba[:, 1]
            out += [("roc_auc", lambda: roc_auc_score(positive, score)),
                    ("average_precision", lambda: average_precision_score(positive, score)),
                    ("brier", lambda: brier_score_loss(positive, score))]
        else:
            out.append(("roc_auc_ovr", lambda: roc_auc_score(y, proba, multi_class="ovr", average="macro",
                                                             labels=classes)))
        out.append(("log_loss", lambda: log_loss(y, proba, labels=classes)))
        return out

    def predict(self) -> None:
        """Predict features_test (with probabilities when the model has predict_proba), then evaluate()."""
        pred = self.model.predict(self.features_test)
        proba = self.model.predict_proba(self.features_test) if hasattr(self.model, "predict_proba") else None
        self.evaluate(self.target_test, pred, proba)

    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray, proba: np.ndarray | None = None) -> None:
        """Store the predictions and class probabilities `proba`, compute the metrics and drop features_test."""
        self._target_test = y_true
        self._pred_test = y_pred
        self._predict_score = proba
        self._accuracy = accuracy_score(y_true, y_pred)
        self._precision = precision_score(y_true, y_pred, average="weighted", zero_division=0)
        self._recall = recall_score(y_true, y_pred, average="weighted", zero_division=0)
        self._f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
        self._features_test = None

    def plot_confusion_matrix(self, labels: np.ndarray | list | None = None,
                              normalize: Literal["true", "pred", "all"] | None = None, show: bool = True) -> None:
        """
        normalize: None, 'true', 'pred', 'all'
        labels defaults to the classes in the order confusion_matrix uses (otherwise the axes would show 0..k-1).
        """
        if labels is None:
            labels = unique_labels(self.target_test, self._pred_test)
        cm = confusion_matrix(self.target_test, self._pred_test, labels=labels, normalize=normalize)

        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
        disp.plot(cmap="Blues", values_format=".2f" if normalize else "d")
        plt.title(f"Confusion Matrix - {self.name}")
        if show:
            plt.show()

    def plot_roc_curve(self, show: bool = True) -> None:
        """
        Works for binary classification.
        For multiclass, you’d need binarization (handled below).
        Binary curves pass pos_label (the second class) since the labels need not be 0/1 (e.g. "no"/"yes").
        Does nothing when there are no predicted probabilities.
        """
        if self._predict_score is None:
            return
        classes = np.unique(self.target_test)

        if len(classes) == 2:
            fpr, tpr, _ = roc_curve(self.target_test, self._predict_score[:, 1], pos_label=classes[1])
            roc_auc = auc(fpr, tpr)

            plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
            plt.plot([0, 1], [0, 1], linestyle="--")
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title(f"ROC Curve - {self.name}")
            plt.legend()
            if show:
                plt.show()
            return

        y_test_bin = label_binarize(self.target_test, classes=classes)

        plt.figure()

        for i in range(len(classes)):
            fpr, tpr, _ = roc_curve(y_test_bin[:, i], self._predict_score[:, i])
            roc_auc = auc(fpr, tpr)

            plt.plot(fpr, tpr, label=f"Class {classes[i]} (AUC={roc_auc:.2f})")

        plt.plot([0, 1], [0, 1], linestyle="--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"Multiclass ROC - {self.name}")
        plt.legend()
        if show:
            plt.show()

    def _report_title(self) -> str:
        """The title of this result's PDF / HTML report."""
        return f"Classification Report - {self.name}"

    def _report_details(self, pdf) -> None:
        """sklearn's per-class text report."""
        try:
            pdf.preformatted("Classification Analysis", self.report_matrix())
        except Exception as e:  # e.g. labels that sklearn cannot report; the plots still run
            pdf.paragraph("Classification Analysis", f"Per-class text report unavailable: {e}")

    def __str__(self) -> str:
        """The report lines followed by sklearn's classification report."""
        lines = list(self.report_lines())
        return "\n".join(lines) + "\n\n" + self.report_matrix()

    def report_lines(self) -> Iterator[str]:
        """Yield all report lines."""
        yield f"Classification Report - {self.name}"
        yield ""

        yield f"Accuracy : {self.accuracy:.4f}"
        yield f"Precision: {self.precision:.4f}"
        yield f"Recall   : {self.recall:.4f}"
        yield f"F1 Score : {self.f1:.4f}"
        yield f" validation type: {self.validation}"
        extra = self.extra_metrics
        if extra:
            yield ""
            yield "Additional metrics (not used for ranking)"
            for name, value in extra.items():
                yield f"  {name:<18}: {value:.4f}"

    def report_matrix(self) -> str:
        """sklearn's per-class classification report as text."""
        return classification_report(
            self.target_test,
            self._pred_test,
            zero_division=0,
            output_dict=False,
        )
