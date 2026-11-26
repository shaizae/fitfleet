"""Enums shared across the pipeline, and a version-independent membership check."""
from __future__ import annotations

from enum import Enum, StrEnum
from typing import Any


def in_enum(enum_type: type[Enum], value: Any) -> bool:
    """`value in enum_type` with Python 3.12+ semantics on every version: True for a
    member or a member's value, False for anything else (3.11 raises TypeError for non-members)."""
    try:
        enum_type(value)
    except (ValueError, TypeError):
        return False
    return True


class ValidationType(StrEnum):
    """How a model was validated."""
    train_test_split = "train_test_split"
    k_folds = "k_folds"
    leave_one_out = "leave_one_out"


class FilteringCriteriaClassification(StrEnum):
    """Metric used to rank and filter classification results."""
    accuracy = "accuracy"
    f1 = "f1"
    recall = "recall"
    precision = "precision"

    @property
    def maximize(self) -> bool:
        """Whether a higher value is better (true for every classification metric)."""
        return self == FilteringCriteriaClassification.accuracy or self == FilteringCriteriaClassification.f1 or self == FilteringCriteriaClassification.recall or self == FilteringCriteriaClassification.precision


class FilteringCriteriaRegression(StrEnum):
    """Metric used to rank and filter regression results."""
    mae = "mae"
    mse = "mse"
    rmse = "rmse"
    r2 = "r2"

    @property
    def maximize(self) -> bool:
        """Whether a higher value is better (only r2; the error metrics are minimized)."""
        return self == FilteringCriteriaRegression.r2


class ReportFormat(StrEnum):
    """File format of a saved report (save_report): a PDF, one self-contained HTML file with the
    same content, or dual = both, built from one rendering of the plots."""
    pdf = "pdf"
    html = "html"
    dual = "dual"

    @property
    def files(self) -> tuple[str, ...]:
        """The file extensions this format writes, PDF first: ("pdf",), ("html",) or ("pdf", "html")."""
        return ("pdf", "html") if self is ReportFormat.dual else (self.value,)

    @property
    def label(self) -> str:
        """"PDF", "HTML" or "PDF + HTML", for log and stage names."""
        return " + ".join(f.upper() for f in self.files)

    @classmethod
    def parse(cls, value: ReportFormat | str) -> ReportFormat:
        """A member, or its name in any case ("pdf", "HTML", "dual"); ValueError for anything else."""
        try:
            return cls(value.lower() if isinstance(value, str) else value)
        except ValueError:
            raise ValueError(f'report format must be ReportFormat.pdf, .html or .dual '
                             f'("pdf", "html" or "dual"), not {value!r}') from None
