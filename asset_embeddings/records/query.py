"""Read API over one or many records databases of the current schema, as DataFrames.

Every table is read through its view, so each row carries the ``runs`` fields (``run_no``, ``experiment``,
``task``, ``config_file``, ``git_commit``, ``run_status``) and, for a result, the config columns its class declares;
filters may name any of them. Every function accepts one path or a sequence of paths; rows from several files are
concatenated with a ``source_db`` column (the file stem). Files that are not at the current schema raise
[`RecordsSchemaError`][asset_embeddings.records.RecordsSchemaError].

Filters: a scalar is an equality, ``None`` is SQL NULL, a tuple or set is a membership test. On a config column that
holds a list (``features``), a list is compared as a whole, and a tuple of lists is a membership test.
"""

from __future__ import annotations

import json
import math
import os
import sqlite3
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Type, Union

import pandas as pd

from .base import Detail, Result, record_classes
from .eval import MPResult, RCResult, RidgeSplit, RVResult, SampleStep
from .grouped import GroupedRCResult, GroupedRidgeSplit, GroupedRVResult
from .pairwise import PairwiseResult
from .probe import ProbeResult
from .store import RUNS_TABLE, assert_schema, detail_class, view_statements
from .train import Epoch

DbPaths = Union[str, os.PathLike, Sequence[Union[str, os.PathLike]]]
TableRef = Union[str, Type[Detail]]

#: Task name (``runs.task``) to result class.
RESULT_TASKS: Dict[str, Type[Result]] = {
    "RV": RVResult,
    "RC": RCResult,
    "RC_temporal": RCResult,
    "MP": MPResult,
    "GROUPED_RV": GroupedRVResult,
    "GROUPED_RC": GroupedRCResult,
    "PROBE": ProbeResult,
    "PAIRWISE": PairwiseResult,
}

#: Config columns whose values are lists; decoded from JSON on read.
LIST_COLUMNS = ("features", "match_on")
_SOURCE_DB = "source_db"


def _paths(dbs: DbPaths) -> List[str]:
    if isinstance(dbs, (str, os.PathLike)):
        return [str(dbs)]
    return [str(p) for p in dbs]


def _connect(path: str) -> sqlite3.Connection:
    if not Path(path).is_file():
        raise FileNotFoundError(path)
    conn = sqlite3.connect(path)
    try:
        assert_schema(conn, path)
        _current_views(conn)
    except Exception:
        conn.close()
        raise
    return conn


def _current_views(conn: sqlite3.Connection) -> None:
    """Shadow the file's views that predate the current definitions with temporary ones on this connection.

    A file gets the current view definitions when a run writes to it (``RecordStore`` rebuilds them on open); a read
    never writes. A view whose stored definition differs from the code, or is missing, is therefore defined as a TEMP
    view, which SQLite resolves before the file's own: every read sees the current config columns, NULL where an
    older run's config lacks the key, and the file is left as it was.
    """
    tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    for table, cls in record_classes().items():
        if table not in tables:
            continue
        for name, sql in view_statements(cls).items():
            stored = conn.execute("SELECT sql FROM sqlite_master WHERE type='view' AND name=?", [name]).fetchone()
            if stored is None or stored[0] != sql:
                conn.execute(sql.replace("CREATE VIEW", "CREATE TEMP VIEW", 1))


def _exists(conn: sqlite3.Connection, name: str) -> bool:
    sql = "SELECT 1 FROM sqlite_master WHERE name=? UNION ALL SELECT 1 FROM sqlite_temp_master WHERE name=?"
    return conn.execute(sql, [name, name]).fetchone() is not None


def _columns(conn: sqlite3.Connection, name: str) -> List[str]:
    return [row[1] for row in conn.execute(f"PRAGMA table_info({name})")]


def _is_list_of_lists(value: Any) -> bool:
    return isinstance(value, (tuple, set, frozenset)) and all(isinstance(v, list) for v in value)


def _where(filters: Dict[str, Any], columns: Sequence[str]) -> Tuple[str, List[Any]]:
    """Translate the filters into a parameterized WHERE clause.

    Raises:
        KeyError: A filter names a column the view does not have; the message lists what it has.
    """
    clauses: List[str] = []
    params: List[Any] = []
    for column, value in filters.items():
        if column not in columns:
            raise KeyError(f"unknown filter column {column!r}; available columns: {sorted(columns) + [_SOURCE_DB]}")
        if column in LIST_COLUMNS:
            if isinstance(value, list):
                value = json.dumps(value, separators=(",", ":"))
            elif _is_list_of_lists(value):
                value = tuple(json.dumps(v, separators=(",", ":")) for v in value)
        if isinstance(value, bool):
            value = int(value)
        if value is None:
            clauses.append(f"{column} IS NULL")
        elif isinstance(value, (list, tuple, set, frozenset)):
            values = list(value)
            if not values:
                clauses.append("0")  # an empty membership matches nothing
                continue
            clauses.append(f"{column} IN ({', '.join('?' for _ in values)})")
            params.extend(values)
        else:
            clauses.append(f"{column} = ?")
            params.append(value)
    return (" WHERE " + " AND ".join(clauses)) if clauses else "", params


def _selected(path: str, source: Optional[Any]) -> bool:
    if source is None:
        return True
    stem = Path(path).stem
    if isinstance(source, (list, tuple, set, frozenset)):
        return stem in source
    return stem == source


def _concat(frames: Iterable[pd.DataFrame]) -> pd.DataFrame:
    frames = [f for f in frames if not f.empty]
    if len(frames) == 1:
        return frames[0]
    # All-NA columns keep their place in the result; drop them from the dtype decision only.
    return (
        pd.concat([f.dropna(axis=1, how="all") for f in frames], ignore_index=True).reindex(
            columns=list(dict.fromkeys(c for f in frames for c in f.columns))
        )
        if frames
        else pd.DataFrame()
    )


def _read(dbs: DbPaths, view: str, filters: Dict[str, Any], order: Sequence[str] = ()) -> pd.DataFrame:
    """Rows of ``view`` (a view or table name) across the files, filtered, with ``source_db``."""
    filters = dict(filters)
    source = filters.pop(_SOURCE_DB, None)
    frames = []
    for path in _paths(dbs):
        if not _selected(path, source):
            continue
        conn = _connect(path)
        try:
            if not _exists(conn, view):
                continue
            columns = _columns(conn, view)
            where, params = _where(filters, columns)
            order_by = [c for c in order if c in columns]
            sql = f"SELECT * FROM {view}{where}" + (f" ORDER BY {', '.join(order_by)}" if order_by else "")
            df = pd.read_sql(sql, conn, params=params)
        finally:
            conn.close()
        for column in LIST_COLUMNS:
            if column in df.columns:
                df[column] = [json.loads(v) if isinstance(v, str) else v for v in df[column]]
        if not df.empty:
            df[_SOURCE_DB] = Path(path).stem
            frames.append(df)
    return _concat(frames)


def _task_filter(task_or_table: TableRef, filters: Dict[str, Any]) -> Dict[str, Any]:
    """``filters`` plus ``task`` when a task name was given.

    Two tasks share a result table (``RC`` and ``RC_temporal`` both write ``eval_rc_results``), so a task name selects
    its own runs; a table name or a class reads every task of the table. An explicit ``task`` filter wins.
    """
    if isinstance(task_or_table, str) and task_or_table in RESULT_TASKS:
        return {"task": task_or_table, **filters}
    return filters


def result_class(task_or_table: TableRef) -> Type[Result]:
    """The result class of a task name (``RV``, ``GROUPED_RC``, ...), a table name or a class.

    Raises:
        KeyError: Neither a task nor a result table.
    """
    if isinstance(task_or_table, type):
        return task_or_table
    if task_or_table in RESULT_TASKS:
        return RESULT_TASKS[task_or_table]
    cls = detail_class(task_or_table)
    if not issubclass(cls, Result):
        raise KeyError(f"{task_or_table!r} is not a result table")
    return cls


def load_runs(dbs: DbPaths, **filters: Any) -> pd.DataFrame:
    """All ``runs`` rows of the given databases.

    Args:
        dbs: One database path or a sequence of them.
        **filters: Filters on ``runs`` columns; ``source_db`` selects files by stem.

    Returns:
        The matching rows with ``source_db``; empty when nothing matches.
    """
    return _read(dbs, RUNS_TABLE, filters, order=("run_no",))


def load_results(dbs: DbPaths, task_or_table: TableRef, **filters: Any) -> pd.DataFrame:
    """Result rows of a task, a result table or a result class, with the runs fields and config columns.

    Args:
        dbs: One database path or a sequence of them.
        task_or_table: A task name (a key of ``RESULT_TASKS``: ``RV``, ``RC``, ``RC_temporal``, ...), which reads only
            that task's runs, or a result table name or a result class, which read every task of the table.
        **filters: Filters on the row columns, the runs fields and the class's config columns.

    Returns:
        The matching rows across the selected files; files without the table contribute none.

    Raises:
        KeyError: An unknown task or table, or a filter on a column the view does not have.
    """
    cls = result_class(task_or_table)
    return _read(
        dbs, f"v_{cls.__table__}", _task_filter(task_or_table, filters), order=("run_no",) + tuple(cls.__key__)
    )


def load_summary(dbs: DbPaths, task_or_table: TableRef, **filters: Any) -> pd.DataFrame:
    """Per-run summaries of a result table: ``mean_value``, ``std_value``, ``n_periods`` per run and summary key.

    ``std_value`` is the sample standard deviation over the periods (NaN for a single period).

    Args:
        dbs: One database path or a sequence of them.
        task_or_table: As in [`load_results`][asset_embeddings.records.query.load_results].
        **filters: Filters on the summary keys, the runs fields and the config columns.

    Returns:
        One row per run and summary key.
    """
    cls = result_class(task_or_table)
    df = _read(
        dbs,
        f"v_{cls.__table__}_summary",
        _task_filter(task_or_table, filters),
        order=("run_no",) + tuple(cls.__summary_keys__),
    )
    if not df.empty:
        df["std_value"] = [math.sqrt(v) if v is not None and v == v else float("nan") for v in df["var_value"]]
    return df


def load_details(dbs: DbPaths, table: TableRef, **filters: Any) -> pd.DataFrame:
    """Rows of any record table (a name or a class) through its view, in key order.

    Args:
        dbs: One database path or a sequence of them.
        table: A table name or a `Detail` class.
        **filters: Filters on the row columns and the runs fields.

    Returns:
        The matching rows.
    """
    cls = table if isinstance(table, type) else detail_class(table)
    return _read(dbs, f"v_{cls.__table__}", filters, order=("run_no",) + tuple(cls.__key__))


def load_sample(dbs: DbPaths, **filters: Any) -> pd.DataFrame:
    """The sample funnel (``eval_sample``) in step order."""
    return load_details(dbs, SampleStep, **filters)


def load_splits(dbs: DbPaths, **filters: Any) -> pd.DataFrame:
    """The holdout splits (``eval_ridge_splits``)."""
    return load_details(dbs, RidgeSplit, **filters)


def load_grouped_splits(dbs: DbPaths, **filters: Any) -> pd.DataFrame:
    """The holdout splits of the grouped cells (``eval_grouped_ridge_splits``)."""
    return load_details(dbs, GroupedRidgeSplit, **filters)


def load_epochs(dbs: DbPaths, **filters: Any) -> pd.DataFrame:
    """The per-epoch losses (``run_epochs``)."""
    return load_details(dbs, Epoch, **filters)


def load_config(db: Union[str, os.PathLike], run_id: str) -> Dict[str, Any]:
    """The config snapshot of one run as a dict.

    Args:
        db: The database that holds the run.
        run_id: The run's UUID.

    Returns:
        The parsed ``config_content`` of the ``runs`` row.

    Raises:
        KeyError: The run is unknown or has no config snapshot.
    """
    conn = _connect(str(db))
    try:
        row = conn.execute(f"SELECT config_content FROM {RUNS_TABLE} WHERE run_id = ?", [run_id]).fetchone()
    finally:
        conn.close()
    if row is None or row[0] is None:
        raise KeyError(f"run_id {run_id!r} not found in {db}")
    return json.loads(row[0])
