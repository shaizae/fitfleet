"""Optional groups: rows of one group are never split between training and testing."""
import warnings

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import load_iris, make_regression
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from fitfleet.classification import Classification
from fitfleet.regression import Regression

N_GROUPS = 15


@pytest.fixture
def iris_groups():
    data = load_iris()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target)
    rng = np.random.default_rng(0)
    groups = rng.integers(0, N_GROUPS, len(y))
    return X, y, groups


def _assert_groups_kept_together(folds: np.ndarray, groups: np.ndarray):
    for g in np.unique(groups):
        assert len(np.unique(folds[groups == g])) == 1, f"group {g} is split between folds"


def test_groups_are_optional(iris_groups):
    X, y, _ = iris_groups
    cls = Classification()
    cls.set(X, y, [LogisticRegression(max_iter=500)])
    assert cls._groups is None and cls.n_groups is None
    ids = cls._fold_ids(5, True, 0, cls.targets.data)
    assert len(np.unique(ids)) == 5


def test_groups_property_returns_the_labels(iris_groups):
    X, y, groups = iris_groups
    cls = Classification()
    cls.set(X, y, [])
    assert cls.groups is None
    labels = pd.Series([f"p{g}" for g in groups])
    cls.set(X, y, [], groups=labels)
    assert isinstance(cls.groups, np.ndarray)
    np.testing.assert_array_equal(cls.groups, labels.to_numpy())
    cls.groups[0] = "changed"  # a copy: the run's groups don't change
    assert cls.groups[0] == labels[0]
    cls.set(X, y, [], groups=labels.to_numpy())
    np.testing.assert_array_equal(cls.groups, labels.to_numpy())


def test_groups_must_be_an_array_or_series(iris_groups):
    X, y, groups = iris_groups
    cls = Classification()
    with pytest.raises(TypeError, match="numpy array or a Series"):
        cls.set(X.assign(patient=groups), y, [], groups="patient")
    with pytest.raises(TypeError, match="numpy array or a Series"):
        cls.set(X, y, [], groups=list(groups))


def test_groups_validation(iris_groups):
    X, y, groups = iris_groups
    cls = Classification()
    with pytest.raises(ValueError, match="one value per row"):
        cls.set(X, y, [], groups=groups[:-1])
    with pytest.raises(ValueError, match="missing"):
        cls.set(X, y, [], groups=np.where(np.arange(len(y)) == 3, np.nan, 1.0))
    with pytest.raises(ValueError, match="at least 2"):
        cls.set(X, y, [], groups=np.zeros(len(y)))


@pytest.mark.parametrize("shuffle", [True, False])
def test_classification_fold_ids_keep_groups_together(iris_groups, shuffle):
    X, y, groups = iris_groups
    cls = Classification()
    cls.set(X, y, [LogisticRegression(max_iter=500)], groups=groups)
    seed = 0 if shuffle else None
    ids = cls._fold_ids(5, shuffle, seed, cls.targets.data, cls._groups)
    assert len(np.unique(ids)) == 5
    _assert_groups_kept_together(ids, groups)


def test_classification_uses_stratified_group_kfold(iris_groups):
    X, y, groups = iris_groups
    cls = Classification()
    cls.set(X, y, [], groups=groups)
    assert type(cls._make_splitter(5, True, 0, cls.targets.data, cls._groups)).__name__ == "StratifiedGroupKFold"


def test_classification_falls_back_to_group_kfold():
    # class 1 lives in a single group: it can't be spread over 3 folds
    X = pd.DataFrame({"a": np.arange(60.0)})
    y = np.array([0] * 50 + [1] * 10)
    groups = np.concatenate([np.arange(50) % 10, np.full(10, 99)])
    cls = Classification()
    cls.set(X, y, [], groups=groups)
    with pytest.warns(UserWarning, match="GroupKFold"):
        splitter = cls._make_splitter(3, True, 0, cls.targets.data, cls._groups)
    assert type(splitter).__name__ == "GroupKFold"


def test_regression_uses_group_kfold():
    X, y = make_regression(n_samples=80, n_features=3, random_state=0)
    groups = np.arange(80) % 8
    reg = Regression()
    reg.set(pd.DataFrame(X), y, [LinearRegression()], groups=groups)
    assert type(reg._make_splitter(4, True, 0, y, reg._groups)).__name__ == "GroupKFold"
    ids = reg._fold_ids(4, True, 0, y, reg._groups)
    _assert_groups_kept_together(ids, groups)


def test_k_folds_rejects_more_folds_than_groups(iris_groups):
    X, y, _ = iris_groups
    cls = Classification()
    cls.set(X, y, [LogisticRegression(max_iter=500)], groups=np.arange(len(y)) % 3)
    with pytest.raises(ValueError, match="number of groups"):
        cls.k_folds(4)


def test_leave_one_group_out_fold_ids(iris_groups):
    X, y, groups = iris_groups
    cls = Classification()
    cls.set(X, y, [], groups=groups)
    ids = cls._fold_ids(cls.n_groups, False, None, cls.targets.data, cls._groups)
    np.testing.assert_array_equal(ids, cls._groups)


def test_leave_one_out_and_leave_one_group_out_need_the_right_setup(iris_groups):
    X, y, groups = iris_groups
    cls = Classification()
    cls.set(X, y, [], groups=groups)
    with pytest.raises(ValueError, match="leave_one_group_out"):
        cls.leave_one_out()
    cls.set(X, y, [])
    with pytest.raises(ValueError, match="needs groups"):
        cls.leave_one_group_out()


def test_train_test_split_keeps_groups_apart(iris_groups):
    X, y, groups = iris_groups
    cls = Classification()
    cls.random_state = 0
    cls.set(X, y, [], groups=groups)
    train, test = cls._split_rows(0.8, cls.targets.data, cls._groups)
    assert not set(groups[train]) & set(groups[test])
    assert len(train) + len(test) == len(y)


def test_halving_subsample_takes_whole_groups(iris_groups):
    X, y, groups = iris_groups
    cls = Classification()
    cls.set(X, y, [], groups=groups)
    rows = np.arange(len(y))
    picked = cls._subsample(rows, 40, np.random.default_rng(0), min_groups=5)
    assert len(picked) >= 40
    chosen = np.unique(groups[picked])
    assert len(chosen) >= 5
    # every row of a chosen group is picked
    assert set(picked) == set(np.flatnonzero(np.isin(groups, chosen)))


def test_feature_selection_holds_out_whole_groups(iris_groups):
    from sklearn.feature_selection import f_classif
    X, y, groups = iris_groups
    cls = Classification()
    cls.random_state = 0
    cls.set(X, y, [], groups=groups)
    cls.feature_selection(f_classif, [2])
    remaining = len(cls.targets.data)
    assert remaining < len(y)
    assert len(cls._groups) == remaining
    assert cls.n_groups < N_GROUPS
    assert set(np.unique(cls._groups)) == set(range(cls.n_groups))
    assert len(cls.groups) == remaining and set(cls.groups) <= set(groups)


def test_k_folds_run_with_groups(iris_groups):
    X, y, groups = iris_groups
    with Classification() as cls:
        cls.random_state = 0
        cls.process_limit(2)
        cls.set(X, y, [DecisionTreeClassifier(random_state=0)], groups=groups)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            results = cls.k_folds(5)
        assert len(results) == 1
        assert results[0].fold_sizes.sum() == len(y)


def test_leave_one_group_out_run(iris_groups):
    X, y, groups = iris_groups
    with Classification() as cls:
        cls.process_limit(2)
        cls.set(X, y, [DecisionTreeClassifier(random_state=0)], groups=groups)
        results = cls.leave_one_group_out()
        assert len(results[0].fold_sizes) == cls.n_groups


def test_train_test_split_and_halving_run_with_groups(iris_groups):
    X, y, groups = iris_groups
    with Classification() as cls:
        cls.random_state = 0
        cls.process_limit(2)
        cls.set(X, y, [DecisionTreeClassifier(random_state=0)], groups=groups)
        cls.set_hyper_parameters({"max_depth": [1, 2, 3, 4]})
        assert len(cls.train_test_split(0.8)) == 4
        cls.search("halving", factor=2)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            assert cls.k_folds(3)
