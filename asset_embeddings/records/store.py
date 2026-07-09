"""SQLite persistence layer for Record objects.

Handles connection management, table creation/migration, and insert operations.
Separated from Record to maintain single responsibility.
"""

import logging
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, Generator, List, Optional

from .base import Record, SQLITE_TYPE_AFFINITY


class RecordStore:
    """SQLite persistence layer for Record objects."""

    def __init__(self, db_path: str, logger: Optional[logging.Logger] = None):
        self.db_path = db_path
        self.logger = logger or logging.getLogger(__name__)
        self._conn: Optional[sqlite3.Connection] = None  # persistent for :memory:
        if db_path != ":memory:":
            p = Path(db_path)
            if p.parent != Path("."):
                p.parent.mkdir(parents=True, exist_ok=True)

    @contextmanager
    def connection(self) -> Generator[sqlite3.Connection, None, None]:
        """Context manager for DB connection.

        For :memory: databases, reuses a single persistent connection.
        For file databases, opens and closes per-call.
        """
        if self.db_path == ":memory:":
            if self._conn is None:
                self._conn = sqlite3.connect(":memory:")
            yield self._conn
        else:
            conn = sqlite3.connect(self.db_path)
            try:
                yield conn
            finally:
                conn.close()

    def save(self, record: Record, table_name: str) -> bool:
        """Save a single record. Auto-creates/migrates table."""
        with self.connection() as conn:
            try:
                self._ensure_table(conn, record, table_name)
                record_dict = record.to_dict()
                self._add_missing_columns(conn, record_dict, table_name)
                self._insert_row(conn, record_dict, table_name)
                conn.commit()
                self.logger.debug(f"Saved 1 record to '{table_name}'")
                return True
            except Exception as e:
                self.logger.error(f"Failed to save record to '{table_name}': {e}")
                conn.rollback()
                return False

    def save_or_replace(self, record: Record, table_name: str, key: str = "run_id") -> bool:
        """Delete existing row with matching key, then insert new row."""
        with self.connection() as conn:
            try:
                self._ensure_table(conn, record, table_name)
                record_dict = record.to_dict()
                self._add_missing_columns(conn, record_dict, table_name)
                key_value = record_dict.get(key)
                if key_value is not None:
                    conn.execute(f"DELETE FROM {table_name} WHERE {key} = ?", [key_value])
                self._insert_row(conn, record_dict, table_name)
                conn.commit()
                return True
            except Exception as e:
                self.logger.error(f"Failed to save_or_replace in '{table_name}': {e}")
                conn.rollback()
                return False

    def save_many(self, records: List[Record], table_name: str) -> bool:
        """Save multiple records in a single transaction."""
        if not records:
            return True

        with self.connection() as conn:
            try:
                # Use first record to ensure table exists
                self._ensure_table(conn, records[0], table_name)

                for record in records:
                    record_dict = record.to_dict()
                    self._add_missing_columns(conn, record_dict, table_name)
                    self._insert_row(conn, record_dict, table_name)

                conn.commit()
                self.logger.debug(f"Saved {len(records)} records to '{table_name}'")
                return True
            except Exception as e:
                self.logger.error(f"Failed to save records to '{table_name}': {e}")
                conn.rollback()
                return False

    def _ensure_table(self, conn: sqlite3.Connection, record: Record, table_name: str):
        """Create table from record's schema if it doesn't exist."""
        existing = self._get_existing_columns(conn, table_name)
        if existing:
            # Table exists — add any new core fields
            schema = record.get_table_schema()
            for col_name, col_type in schema.items():
                if col_name not in existing:
                    conn.execute(f"ALTER TABLE {table_name} ADD COLUMN {col_name} {col_type}")
                    self.logger.debug(f"Added column '{col_name}' ({col_type}) to '{table_name}'")
        else:
            # Create new table
            schema = record.get_table_schema()
            cols = ", ".join(f"{name} {stype}" for name, stype in schema.items())
            conn.execute(f"CREATE TABLE {table_name} ({cols})")
            self.logger.debug(f"Created table '{table_name}' with {len(schema)} columns")

    def _add_missing_columns(self, conn: sqlite3.Connection, record_dict: Dict, table_name: str):
        """Add columns for any dynamic fields not yet in the table."""
        existing = self._get_existing_columns(conn, table_name)
        for col_name, value in record_dict.items():
            if col_name not in existing:
                sql_type = SQLITE_TYPE_AFFINITY.get(type(value), "TEXT")
                conn.execute(f"ALTER TABLE {table_name} ADD COLUMN {col_name} {sql_type}")
                self.logger.debug(f"Added dynamic column '{col_name}' ({sql_type}) to '{table_name}'")

    def _insert_row(self, conn: sqlite3.Connection, record_dict: Dict, table_name: str):
        """Insert a single record row."""
        columns = list(record_dict.keys())
        placeholders = ", ".join(["?"] * len(columns))
        col_names = ", ".join(columns)
        values = [record_dict[c] for c in columns]
        conn.execute(f"INSERT INTO {table_name} ({col_names}) VALUES ({placeholders})", values)

    @staticmethod
    def _get_existing_columns(conn: sqlite3.Connection, table_name: str) -> Dict[str, str]:
        """Get existing columns as {name: type}. Empty dict if table doesn't exist."""
        try:
            cursor = conn.execute(f"PRAGMA table_info({table_name})")
            return {row[1]: row[2] for row in cursor.fetchall()}
        except sqlite3.OperationalError:
            return {}
