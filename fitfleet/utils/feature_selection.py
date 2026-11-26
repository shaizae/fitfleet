"""Helpers behind BaseML's feature selection: k-best columns, NaN filling and brute force."""
from __future__ import annotations

import warnings
from itertools import combinations
from typing import TYPE_CHECKING, Iterable

import numpy as np

if TYPE_CHECKING:
    from fitfleet.schema import ColumnSchema


def select_k_best_columns(algorithm, features: np.ndarray, target: np.ndarray, ks, schema: ColumnSchema) -> list:
    """Top-k *original* columns for every k in `ks`. A column's score is the best score of its matrix
    columns (a text column takes several). Ties keep the column order."""
    scores = algorithm(features, target)
    scores = np.asarray(scores[0] if isinstance(scores, tuple) else scores, dtype=float)
    scores = np.nan_to_num(scores, nan=-np.inf)
    per_column = np.array([scores[schema.matrix_index([i])].max() for i in range(len(schema.columns))])
    order = np.argsort(-per_column, kind="stable")
    out, seen = [], set()
    for k in ks:
        key = tuple(sorted(int(i) for i in order[:min(k, len(order))]))
        if key not in seen:
            seen.add(key)
            out.append(np.asarray(key))
    return out


def fill_missing(X: np.ndarray) -> np.ndarray:
    """NaN replaced by the column median (0 for an all-missing column); for scoring only."""
    if not np.isnan(X).any():
        return X
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        medians = np.nan_to_num(np.nanmedian(X, axis=0))
    return np.where(np.isnan(X), medians, X)


def brute_force(vec: np.ndarray) -> Iterable[tuple]:
    """Every non-empty combination of `vec`, smallest first."""
    for i in range(1, len(vec) + 1):
        for combo in combinations(vec, i):
            yield combo
