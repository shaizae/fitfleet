"""Helpers for estimators and the files written next to them."""
from __future__ import annotations

from pathlib import Path
from typing import Any


def is_fitted(model: Any) -> bool:
    """Whether `model` has been fitted, per sklearn's check_is_fitted.

    Objects that aren't sklearn estimators are assumed fitted; the caller knows.
    """
    from sklearn.exceptions import NotFittedError
    from sklearn.utils.validation import check_is_fitted
    try:
        check_is_fitted(model)
        return True
    except NotFittedError:
        return False
    except TypeError:
        return True


def create_path(path: str) -> Path:
    """Create the folder `path`, including any missing parents, and return it as a Path."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p
