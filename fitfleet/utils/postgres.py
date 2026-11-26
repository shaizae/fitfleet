"""Read features and target from PostgreSQL into DataFrames (needs the `postgres` extra)."""
from __future__ import annotations

from decimal import Decimal

from pandas import DataFrame
from pandas import to_numeric as pd_to_numeric


def _connect(connection: str | dict):
    try:
        import psycopg
    except ImportError as e:
        raise ImportError("postgres_set needs psycopg: pip install fitfleet[postgres]") from e
    if isinstance(connection, str):
        return psycopg.connect(connection)
    if isinstance(connection, dict):
        return psycopg.connect(**connection)
    raise TypeError(f"connection must be a URL string or a dict of connect arguments, "
                    f"not {type(connection).__name__}")


def _query_frame(cursor, query: str, params) -> DataFrame:
    """The rows of one query as a DataFrame; Postgres numeric (Decimal) columns become floats."""
    cursor.execute(query, params)
    if cursor.description is None:
        raise ValueError(f"query returned no rows to read (is it a SELECT?): {query!r}")
    frame = DataFrame(cursor.fetchall(), columns=[column.name for column in cursor.description])
    for name in frame.columns:
        column = frame[name]
        if column.dtype == object and len(column.dropna()) and \
                all(isinstance(v, Decimal) for v in column.dropna()):
            frame[name] = pd_to_numeric(column)
    return frame


def read_postgres(connection: str | dict, features_query: str, target_query: str,
                  params) -> tuple[DataFrame, DataFrame]:
    """Run both queries on one connection, closed afterwards."""
    conn = _connect(connection)
    try:
        with conn.cursor() as cursor:
            features = _query_frame(cursor, features_query, params)
            target = _query_frame(cursor, target_query, params)
    finally:
        conn.close()
    return features, target
