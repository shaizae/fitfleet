import warnings

import numpy as np
from sklearn.model_selection import GroupKFold, KFold, StratifiedGroupKFold, StratifiedKFold

from fitfleet.base import BaseML
from fitfleet.results.classification import ClassificationResults
from fitfleet.utils import FilteringCriteriaClassification


class Classification(BaseML):
    results_type = ClassificationResults
    criteria_type = FilteringCriteriaClassification
    default_criteria = FilteringCriteriaClassification.accuracy
    _encode_target = True

    def _stratify_labels(self, y: np.ndarray):
        return y

    def _make_splitter(self, n_splits: int, shuffle: bool, seed, y: np.ndarray = None, groups: np.ndarray = None):
        """Stratified folds keep each class's share the same in every fold. They need
        at least n_splits rows per class; otherwise fall back to plain KFold.
        With groups: StratifiedGroupKFold, which needs every class in at least n_splits
        groups; otherwise GroupKFold. A group is never split between folds."""
        y = self.targets.data if y is None else y
        if groups is not None:
            return self._make_group_splitter(n_splits, shuffle, seed, y, groups)
        counts = np.bincount(np.asarray(y, dtype=np.int64))
        smallest_class = counts[counts > 0].min()
        if smallest_class >= n_splits:
            return StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=seed)
        warnings.warn(f"smallest class has {smallest_class} rows (< {n_splits} folds): using KFold, "
                      f"not StratifiedKFold", stacklevel=3)
        return KFold(n_splits=n_splits, shuffle=shuffle, random_state=seed)

    @staticmethod
    def _make_group_splitter(n_splits: int, shuffle: bool, seed, y: np.ndarray, groups: np.ndarray):
        y = np.asarray(y, dtype=np.int64)
        pairs = np.unique(np.stack([y, np.asarray(groups, dtype=np.int64)]), axis=1)  # distinct (class, group)
        groups_per_class = np.bincount(pairs[0])
        fewest = groups_per_class[groups_per_class > 0].min()
        if fewest >= n_splits:
            return StratifiedGroupKFold(n_splits=n_splits, shuffle=shuffle, random_state=seed)
        warnings.warn(f"a class appears in only {fewest} groups (< {n_splits} folds): using GroupKFold, "
                      f"not StratifiedGroupKFold", stacklevel=4)
        return GroupKFold(n_splits=n_splits, shuffle=shuffle, random_state=seed)
