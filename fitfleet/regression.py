from sklearn.model_selection import GroupKFold, KFold

from fitfleet.base import BaseML
from fitfleet.results.regression import RegressionResults
from fitfleet.utils import FilteringCriteriaRegression


class Regression(BaseML):
    results_type = RegressionResults
    criteria_type = FilteringCriteriaRegression
    default_criteria = FilteringCriteriaRegression.r2
    _encode_target = False  # keep target values numeric; label-encoding turned them into ranks

    def _make_splitter(self, n_splits: int, shuffle: bool, seed, y=None, groups=None):
        if groups is not None:  # a group is never split between folds
            return GroupKFold(n_splits=n_splits, shuffle=shuffle, random_state=seed)
        return KFold(n_splits=n_splits, shuffle=shuffle, random_state=seed)
