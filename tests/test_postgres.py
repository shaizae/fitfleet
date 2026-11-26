"""postgres_set(): features and target read from PostgreSQL, then handed to set().

The unit tests use a fake psycopg module; the live test runs only when FITFLEET_TEST_POSTGRES_URL
points at a database (e.g. postgresql://postgres:pw@localhost:5432/postgres)."""
import os
import sys
import types
import uuid
from decimal import Decimal

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from fitfleet.classification import Classification
from fitfleet.regression import Regression

FEATURE_ROWS = [(1.0, "a", Decimal("10.5")), (2.0, "b", Decimal("11.5")), (3.0, "a", Decimal("12.5")),
                (4.0, "b", Decimal("13.5")), (5.0, "a", Decimal("14.5")), (6.0, "b", Decimal("15.5"))]
FEATURE_COLUMNS = ["x", "city", "spend"]
TARGET_ROWS = [(0,), (1,), (0,), (1,), (0,), (1,)]


class _Column:
    def __init__(self, name):
        self.name = name


class FakeCursor:
    def __init__(self, results):
        self._results = results  # query -> (columns, rows)
        self.description = None
        self._rows = []
        self.executed = []

    def execute(self, query, params=None):
        self.executed.append((query, params))
        columns, rows = self._results[query]
        self.description = None if columns is None else [_Column(c) for c in columns]
        self._rows = rows

    def fetchall(self):
        return list(self._rows)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class FakeConnection:
    def __init__(self, results):
        self.cursor_obj = FakeCursor(results)
        self.closed = False

    def cursor(self):
        return self.cursor_obj

    def close(self):
        self.closed = True


@pytest.fixture
def fake_psycopg(monkeypatch):
    """Installs a fake psycopg; returns a dict recording connect calls and the query results to serve."""
    state = {"calls": [], "connections": [], "results": {
        "features": (FEATURE_COLUMNS, FEATURE_ROWS),
        "target": (["label"], TARGET_ROWS),
    }}

    def connect(*args, **kwargs):
        state["calls"].append((args, kwargs))
        conn = FakeConnection(state["results"])
        state["connections"].append(conn)
        return conn

    monkeypatch.setitem(sys.modules, "psycopg", types.SimpleNamespace(connect=connect))
    return state


def test_loads_like_set(fake_psycopg):
    run = Classification()
    run.postgres_set("postgresql://u:p@h/db", "features", "target", [DecisionTreeClassifier()])

    expected = Classification()
    X = pd.DataFrame(FEATURE_ROWS, columns=FEATURE_COLUMNS).astype({"spend": float})
    expected.set(X, pd.Series([r[0] for r in TARGET_ROWS]), [DecisionTreeClassifier()])

    assert list(run.features.columns) == list(expected.features.columns)
    np.testing.assert_array_equal(run.features.raw_data, expected.features.raw_data)
    np.testing.assert_array_equal(run.targets.data, expected.targets.data)
    assert len(run._train_data) == 1
    assert fake_psycopg["connections"][0].closed


def test_decimal_columns_become_numeric(fake_psycopg):
    run = Regression()
    fake_psycopg["results"]["target"] = (["y"], [(Decimal(str(v)),) for v in range(6)])
    run.postgres_set("postgresql://u:p@h/db", "features", "target", [DecisionTreeClassifier()])
    # spend was Decimal: numeric, not a category
    assert "spend" not in [run.features.schema.columns[i] for i in run.features.schema.categorical_index]


def test_url_and_dict_connections(fake_psycopg):
    Classification().postgres_set("postgresql://u:p@h/db", "features", "target", [LogisticRegression()])
    Classification().postgres_set({"host": "h", "dbname": "db", "user": "u", "password": "p"},
                                  "features", "target", [LogisticRegression()])
    assert fake_psycopg["calls"][0] == (("postgresql://u:p@h/db",), {})
    assert fake_psycopg["calls"][1] == ((), {"host": "h", "dbname": "db", "user": "u", "password": "p"})


def test_bad_connection_type(fake_psycopg):
    with pytest.raises(TypeError, match="URL string or a dict"):
        Classification().postgres_set(42, "features", "target", [LogisticRegression()])


def test_params_passed_to_both_queries(fake_psycopg):
    Classification().postgres_set("url", "features", "target", [LogisticRegression()], params={"n": 6})
    executed = fake_psycopg["connections"][0].cursor_obj.executed
    assert executed == [("features", {"n": 6}), ("target", {"n": 6})]


def test_set_kwargs_forwarded(fake_psycopg):
    run = Classification()
    run.postgres_set("url", "features", "target", [LogisticRegression()], categorical=["x"])
    categorical = [run.features.schema.columns[i] for i in run.features.schema.categorical_index]
    assert "x" in categorical


def test_target_with_two_columns(fake_psycopg):
    fake_psycopg["results"]["target"] = (["a", "b"], [(0, 1)] * 6)
    with pytest.raises(ValueError, match="exactly one column"):
        Classification().postgres_set("url", "features", "target", [LogisticRegression()])


def test_row_count_mismatch(fake_psycopg):
    fake_psycopg["results"]["target"] = (["label"], TARGET_ROWS[:4])
    with pytest.raises(ValueError, match="6 rows but target_query 4"):
        Classification().postgres_set("url", "features", "target", [LogisticRegression()])


def test_no_feature_rows(fake_psycopg):
    fake_psycopg["results"]["features"] = (FEATURE_COLUMNS, [])
    with pytest.raises(ValueError, match="no data"):
        Classification().postgres_set("url", "features", "target", [LogisticRegression()])


def test_not_a_select(fake_psycopg):
    fake_psycopg["results"]["features"] = (None, [])
    with pytest.raises(ValueError, match="SELECT"):
        Classification().postgres_set("url", "features", "target", [LogisticRegression()])
    assert fake_psycopg["connections"][0].closed


def test_missing_psycopg(monkeypatch):
    monkeypatch.setitem(sys.modules, "psycopg", None)  # import psycopg -> ImportError
    with pytest.raises(ImportError, match=r"fitfleet\[postgres\]"):
        Classification().postgres_set("url", "features", "target", [LogisticRegression()])


@pytest.mark.skipif(not os.environ.get("FITFLEET_TEST_POSTGRES_URL"), reason="FITFLEET_TEST_POSTGRES_URL not set")
def test_live_postgres():
    psycopg = pytest.importorskip("psycopg")
    url = os.environ["FITFLEET_TEST_POSTGRES_URL"]
    table = f"fitfleet_test_{uuid.uuid4().hex[:8]}"
    rng = np.random.default_rng(0)
    with psycopg.connect(url, autocommit=True) as conn:
        conn.execute(f"CREATE TABLE {table} (id int, x double precision, city text, spend numeric, label int)")
        with conn.cursor() as cur:
            cur.executemany(f"INSERT INTO {table} VALUES (%s, %s, %s, %s, %s)",
                            [(i, float(rng.normal()), "ab"[i % 2], Decimal(i), i % 2) for i in range(40)])
    try:
        with Classification() as run:
            run.postgres_set(url, f"SELECT x, city, spend FROM {table} ORDER BY id",
                             f"SELECT label FROM {table} ORDER BY id", [DecisionTreeClassifier()])
            assert run.features.raw_data.shape[0] == 40
            assert len(run.k_folds(n_splits=2)) == 1
    finally:
        with psycopg.connect(url, autocommit=True) as conn:
            conn.execute(f"DROP TABLE {table}")
