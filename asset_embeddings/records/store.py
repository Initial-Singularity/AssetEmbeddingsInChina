"""SQLite persistence for ``Record`` objects (schema v7).

Holds the ``RecordStore`` (table creation, inserts, the ``runs`` lifecycle, the views) and the schema check every
reader and writer runs before touching a file. A current file carries ``PRAGMA user_version = 7``; any other file with
tables is refused.
"""

import logging
import sqlite3
from collections import OrderedDict
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Sequence, Type

from .base import SCHEMA_VERSION, Detail, Record, RecordsSchemaError, Result, RunRecord, record_classes

RUNS_TABLE = RunRecord.__table__
# Busy timeout (seconds) so that two runners writing to one database (a training and an evaluation runner in
# separate tmux sessions) wait for each other's short transactions instead of failing with "database is locked"
# (sqlite3's default is 5 s).
SQLITE_BUSY_TIMEOUT = 60.0

#: The ``runs`` columns every view carries: exposed name -> column of ``runs``.
RUN_VIEW_COLUMNS: Dict[str, str] = {
    "run_no": "run_no",
    "experiment": "experiment",
    "task": "task",
    "config_file": "config_file",
    "git_commit": "git_commit",
    "run_status": "status",
}


#: The checkout that reads each earlier schema, named in the refusal message.
EARLIER_SCHEMAS = {
    6: "commit ff9497ee (records v6, columns named quarter)",
    5: "the archived code of records v5 (before the evaluator refactor)",
}


def assert_schema(conn: sqlite3.Connection, db_path: str) -> None:
    """Raise unless ``conn`` is an empty records database or one at ``SCHEMA_VERSION``.

    Args:
        conn: An open connection to the file.
        db_path: The file's path, for the message.

    Raises:
        RecordsSchemaError: The file has tables but is not at ``SCHEMA_VERSION`` (an earlier records file, or not a
            records file at all).
    """
    version = conn.execute("PRAGMA user_version").fetchone()[0]
    if version == SCHEMA_VERSION:
        return
    tables = [row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")]
    if tables:
        raise RecordsSchemaError(
            f"{db_path} is not a schema v{SCHEMA_VERSION} records database (user_version {version}, tables "
            f"{sorted(tables)[:3]}); "
            + (
                f"a schema v{version} file is read by {EARLIER_SCHEMAS[version]}, not by this checkout"
                if version in EARLIER_SCHEMAS
                else "files of schema v5 and older are read by the archived code, not by this checkout"
            )
        )


def row_view_sql(cls: Type[Detail]) -> str:
    """``CREATE VIEW v_<table>``: the rows, the ``runs`` fields and the class's config columns."""
    runs = [f"r.{col} AS {name}" for name, col in RUN_VIEW_COLUMNS.items()]
    config = [f"json_extract(r.config_content, '{path}') AS {name}" for name, path in cls.__config_columns__.items()]
    select = ", ".join(["t.*"] + runs + config)
    return (
        f"CREATE VIEW v_{cls.__table__} AS SELECT {select} "
        f"FROM {cls.__table__} t LEFT JOIN {RUNS_TABLE} r ON r.run_id = t.run_id"
    )


def summary_view_sql(cls: Type[Result]) -> str:
    """``CREATE VIEW v_<table>_summary``: mean, sample variance and count of ``value`` per run and summary key.

    The variance is computed in two passes, from the deviations from the group mean (a window average), so it is
    never negative and stays accurate when the values are close to each other; the one-pass
    ``SUM(v * v) - SUM(v) ** 2 / n`` cancels to a tiny negative number for identical values. SQLite has no ``sqrt``
    in every build, so the view carries the variance; the query API adds ``std_value``.
    """
    group = ["run_id"] + list(cls.__summary_keys__) + ["metric"]
    keys = [f"t.{k}" for k in group]
    n = "COUNT(t.value)"
    deviation = "(t.value - t.group_mean)"
    var = f"CASE WHEN {n} > 1 THEN SUM({deviation} * {deviation}) / ({n} - 1) END"
    rows = f"(SELECT *, AVG(value) OVER (PARTITION BY {', '.join(group)}) AS group_mean FROM {cls.__table__})"
    runs = [f"r.{col} AS {name}" for name, col in RUN_VIEW_COLUMNS.items()]
    config = [f"json_extract(r.config_content, '{path}') AS {name}" for name, path in cls.__config_columns__.items()]
    select = ", ".join(
        keys + ["AVG(t.value) AS mean_value", f"{var} AS var_value", f"{n} AS n_periods"] + runs + config
    )
    return (
        f"CREATE VIEW v_{cls.__table__}_summary AS SELECT {select} "
        f"FROM {rows} t LEFT JOIN {RUNS_TABLE} r ON r.run_id = t.run_id GROUP BY {', '.join(keys)}"
    )


def view_statements(cls: Type[Record]) -> Dict[str, str]:
    """The views of a record class by name: the row view of a ``Detail``, plus the summary view of a ``Result``."""
    views: Dict[str, str] = {}
    if issubclass(cls, Detail):
        views[f"v_{cls.__table__}"] = row_view_sql(cls)
        if issubclass(cls, Result):
            views[f"v_{cls.__table__}_summary"] = summary_view_sql(cls)
    return views


def sync_views(conn: sqlite3.Connection, cls: Type[Record]) -> bool:
    """Create the class's views, or rebuild those whose stored definition differs from the current code.

    Views hold no data, so a file written by earlier code gets the current definitions on the next open; an
    unchanged view is left alone, so concurrent runners do not rewrite the schema on every open.

    Args:
        conn: An open connection; the caller commits.
        cls: A record class whose table exists in the file.

    Returns:
        Whether a view was created or rebuilt.
    """
    changed = False
    for name, sql in view_statements(cls).items():
        stored = conn.execute("SELECT sql FROM sqlite_master WHERE type='view' AND name=?", [name]).fetchone()
        if stored is not None and stored[0] == sql:
            continue
        conn.execute(f"DROP VIEW IF EXISTS {name}")
        conn.execute(sql)
        changed = True
    return changed


class RecordStore:
    """SQLite persistence layer for ``Record`` objects.

    Run lifecycle: ``begin_run`` (returns ``run_no``), then ``save`` / ``save_many`` of the run's rows, then
    ``finish_run``. A table is created from its class on first use, with its views; opening an existing file
    brings the views of its tables to the current definitions. Write methods raise on failure; the policy for a
    failed write lives in [`Run`][asset_embeddings.records.Run].

    Args:
        db_path: The database file, or ``:memory:``; parent directories are created.

    Raises:
        RecordsSchemaError: The file exists with tables and is not at the current records schema.
    """

    def __init__(self, db_path: str):
        self.db_path = db_path
        self.logger = logging.getLogger(__name__)
        self._conn: Optional[sqlite3.Connection] = None  # persistent for :memory:
        if db_path != ":memory:":
            p = Path(db_path)
            if p.parent != Path("."):
                p.parent.mkdir(parents=True, exist_ok=True)
            if p.exists() and p.stat().st_size > 0:
                conn = sqlite3.connect(db_path, timeout=SQLITE_BUSY_TIMEOUT)
                try:
                    assert_schema(conn, db_path)
                    self._sync_all_views(conn)
                finally:
                    conn.close()

    def _sync_all_views(self, conn: sqlite3.Connection) -> None:
        """Bring the views of every record table in the file to the current definitions, in one transaction."""
        tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        classes = [cls for name, cls in record_classes().items() if name in tables]
        if not any(
            conn.execute("SELECT sql FROM sqlite_master WHERE type='view' AND name=?", [name]).fetchone() != (sql,)
            for cls in classes
            for name, sql in view_statements(cls).items()
        ):
            return
        conn.execute("BEGIN IMMEDIATE")
        try:
            rebuilt = [cls.__table__ for cls in classes if sync_views(conn, cls)]
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        self.logger.debug(f"Rebuilt the views of {rebuilt}")

    @contextmanager
    def connection(self) -> Generator[sqlite3.Connection, None, None]:
        """Yield a connection: one persistent connection for ``:memory:``, a fresh one per call for a file.

        Yields:
            The open connection; a file connection is closed on exit.
        """
        if self.db_path == ":memory:":
            if self._conn is None:
                self._conn = sqlite3.connect(":memory:")
            yield self._conn
        else:
            conn = sqlite3.connect(self.db_path, timeout=SQLITE_BUSY_TIMEOUT)
            try:
                yield conn
            finally:
                conn.close()

    def begin_run(self, run: RunRecord) -> int:
        """Insert the ``runs`` row (status ``running``); a new file is stamped with ``SCHEMA_VERSION``.

        Args:
            run: The row to insert; its ``run_no`` is set from the assigned rowid.

        Returns:
            The assigned ``run_no``.
        """
        with self.connection() as conn:
            if conn.execute("PRAGMA user_version").fetchone()[0] != SCHEMA_VERSION:
                assert_schema(conn, self.db_path)
                conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
            self._ensure_table(conn, RunRecord)
            conn.execute(f"CREATE INDEX IF NOT EXISTS idx_runs_experiment_task ON {RUNS_TABLE}(experiment, task)")
            conn.execute(f"CREATE INDEX IF NOT EXISTS idx_runs_config_hash ON {RUNS_TABLE}(config_hash)")
            run.run_no = self._insert(conn, run, replace=False)
            conn.commit()
            self.logger.debug(f"Run {run.run_id} (#{run.run_no}) started ({run.task}, experiment={run.experiment})")
            return run.run_no

    def finish_run(
        self,
        run_no: int,
        status: str,
        error_message: Optional[str] = None,
        resources: Optional[Dict[str, Optional[float]]] = None,
    ) -> None:
        """Mark a run ``completed`` / ``error`` with ``finished_at``, ``duration_sec`` and its resource peaks.

        Args:
            run_no: Key of the ``runs`` row.
            status: ``completed`` or ``error``.
            error_message: The traceback when ``status`` is ``error``.
            resources: ``ResourceProbe.summary()``: ``peak_ram_mb``, ``peak_vram_mb``.

        Raises:
            KeyError: ``run_no`` has no ``runs`` row.
        """
        finished = datetime.now()
        with self.connection() as conn:
            row = conn.execute(f"SELECT started_at FROM {RUNS_TABLE} WHERE run_no = ?", [run_no]).fetchone()
            if row is None:
                raise KeyError(f"run_no {run_no!r} not found in {RUNS_TABLE}")
            duration: Optional[float]
            try:
                duration = max(0.0, (finished - datetime.fromisoformat(row[0])).total_seconds())
            except (TypeError, ValueError):
                duration = None
            sets = ["status = ?", "error_message = ?", "finished_at = ?", "duration_sec = ?"]
            values: List[Any] = [status, error_message, finished.isoformat(), duration]
            for key in ("peak_ram_mb", "peak_vram_mb"):
                if resources is not None and key in resources:
                    sets.append(f"{key} = ?")
                    values.append(resources[key])
            conn.execute(f"UPDATE {RUNS_TABLE} SET {', '.join(sets)} WHERE run_no = ?", [*values, run_no])
            conn.commit()
            self.logger.debug(f"Run #{run_no} finished: {status}")

    def save(self, record: Record, replace: bool = False) -> None:
        """Insert one record into its class's table; ``replace`` overwrites the row with the same natural key.

        Raises on failure (the caller decides the policy).
        """
        self.save_many([record], replace=replace)

    def save_many(self, records: Sequence[Record], replace: bool = False) -> None:
        """Insert several records, of one or several classes, in one transaction; nothing is kept on failure."""
        if not records:
            return
        with self.connection() as conn:
            try:
                for cls in OrderedDict.fromkeys(type(r) for r in records):
                    self._ensure_table(conn, cls)
                for record in records:
                    self._insert(conn, record, replace=replace)
                conn.commit()
            except Exception:
                conn.rollback()
                raise
            self.logger.debug(f"Saved {len(records)} record(s)")

    def _ensure_table(self, conn: sqlite3.Connection, cls: Type[Record]) -> None:
        """Create the class's table and its views when the table does not exist yet."""
        exists = conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", [cls.__table__]).fetchone()
        if exists:
            return
        conn.execute(cls.get_table_ddl())
        sync_views(conn, cls)
        self.logger.debug(f"Created table '{cls.__table__}'")

    @staticmethod
    def _insert(conn: sqlite3.Connection, record: Record, replace: bool) -> Optional[int]:
        """Insert one row; returns the rowid (``run_no`` for ``runs``)."""
        data = record.to_dict()
        columns = list(data)
        verb = "INSERT OR REPLACE" if replace else "INSERT"
        cursor = conn.execute(
            f"{verb} INTO {record.__table__} ({', '.join(columns)}) VALUES ({', '.join('?' for _ in columns)})",
            [data[c] for c in columns],
        )
        return cursor.lastrowid


def detail_class(name: str) -> Type[Detail]:
    """The ``Detail`` class of a table name.

    Raises:
        KeyError: No record class has that table.
    """
    classes = record_classes()
    if name not in classes or not issubclass(classes[name], Detail):
        raise KeyError(f"no record table {name!r}; known: {sorted(t for t in classes if t != RUNS_TABLE)}")
    return classes[name]
