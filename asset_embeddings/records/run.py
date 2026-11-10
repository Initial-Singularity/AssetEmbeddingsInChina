"""The ``Run`` object: one script invocation's identity, lifecycle and every write that carries its key.

A script's ``main()`` opens one ``Run`` from its parsed arguments, hands it to the trainer or evaluator, and
records everything through it; the ``runs`` row, the error file and the write-failure policy live here.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path, PurePath
from typing import Any, Callable, Dict, Optional, Sequence

from ..logger import error_dir, is_local_main_process, job_log_paths, module_name, write_error_block
from ..utils.memory_utils import ResourceProbe
from .base import Record, RecordsSchemaError, RunRecord, config_hash, generate_run_id
from .provenance import collect_provenance, derive_experiment
from .store import RecordStore
from .train import Epoch

__all__ = ["Run", "is_local_main_process", "module_name"]

logger = logging.getLogger(__name__)

_STORE_ERRORS = (sqlite3.Error, RecordsSchemaError, OSError)


class Run:
    """One script invocation: the only object a script touches for the ``runs`` table.

    A ``Run`` exists from the entry point and knows only what the command line gives it; the ``runs`` row is
    inserted by ``set_config``, once the config is loaded and the task is known. ``run_id`` (UUID) is the public
    identifier, exists from construction and links every row the run writes; ``run_no`` is the display sequence
    that insert assigns. Without ``--result_file`` (or on a non-main rank of a distributed launch)
    the store is ``None`` and every write is a no-op.

    Failure policy: ``set_config`` propagates, so a broken database stops the run before any computation;
    ``save*`` / ``log_epoch`` never raise (the first failure is a WARNING plus a block in the run's error file,
    later ones are counted); ``__exit__`` writes the error file for any exception, never suppresses it and never
    lets a records error replace it.

    Args:
        store: The open store, or ``None`` when writes go nowhere.
        script: Module name of the running script (``runs.script``).
        config_file: Config path as given on the CLI.
        experiment: Experiment family; derived from ``config_file`` when omitted.
    """

    def __init__(
        self,
        store: Optional[RecordStore],
        *,
        script: str,
        config_file: Optional[str] = None,
        experiment: Optional[str] = None,
    ) -> None:
        self.store = store
        self._db_path: Optional[str] = store.db_path if store is not None else None
        self.script = script
        self.config_file = config_file
        if experiment is not None:
            self.experiment = experiment
        else:
            self.experiment = derive_experiment(config_file) if config_file else ""
        self.run_id: str = generate_run_id()
        self.run_no: Optional[int] = None
        self.task: Optional[str] = None
        self.config: Optional[Dict[str, Any]] = None
        self.started_at = datetime.now().isoformat()
        self.logger = logger
        self.write_failures = 0
        self._stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        self._provenance: Dict[str, Any] = {}
        self._t0 = time.monotonic()
        self.resources = ResourceProbe()  # peak RSS of the process tree, peak VRAM once CUDA is up; summarized at exit

    @classmethod
    def from_args(cls, args: Any, *, script: str) -> "Run":
        """Build the run from parsed CLI arguments.

        ``--result_file`` opens the store, ``--config`` is the config file, ``--experiment`` the family. The
        store is ``None`` without ``--result_file`` and on a non-main rank.

        Args:
            args: The parsed arguments; missing attributes count as absent options.
            script: Module name of the running script.

        Returns:
            The new run, not yet entered.

        Raises:
            RecordsSchemaError: The result file is at an older schema version.
            OSError: The result file cannot be opened. Either failure happens before the ``with`` block, so
                it is reported here like any other failure of the run (one logged traceback, one error file).
        """
        run = cls(
            None,
            script=script,
            config_file=getattr(args, "config", None),
            experiment=getattr(args, "experiment", None),
        )
        result_file = getattr(args, "result_file", None)
        if result_file and is_local_main_process():
            run._db_path = str(result_file)
            try:
                run.store = RecordStore(result_file)
            except BaseException:
                exc_info = sys.exc_info()
                run._log_failure(exc_info)
                run._terminal_block(exc_info)
                raise
        return run

    @property
    def active(self) -> bool:
        """``True`` when writes reach a database."""
        return self.store is not None

    @property
    def error_file(self) -> Path:
        """The one file that holds everything that went wrong in this run (created on first write).

        ``logs/error/<experiment>/<task>_<run_id[:8]>_<stamp>.log``; before ``set_config`` the label is the
        script's stem, since the task is not known yet.
        """
        label = self.task or self.script.rsplit(".", 1)[-1] or "run"
        return error_dir() / _experiment_dir(self.experiment) / f"{label}_{self.run_id[:8]}_{self._stamp}.log"

    def __enter__(self) -> "Run":
        self._provenance = collect_provenance()
        self._t0 = time.monotonic()
        self.resources.start()
        return self

    def set_config(self, task: str, config: Dict[str, Any]) -> None:
        """Record what the run is about and insert the ``runs`` row.

        Args:
            task: ``runs.task`` (``RV``, ``pretrain_BERT``, ...).
            config: The effective config as a dict; stored as the JSON snapshot and hashed.

        Raises:
            sqlite3.Error: The insert failed; this propagates on purpose so the run fails before any computation.
        """
        self.task, self.config = task, config
        if self.store is None:
            return
        if not self._provenance:  # set_config outside a `with` block (library use, tests)
            self._provenance = collect_provenance()
        record = RunRecord(
            run_id=self.run_id,
            experiment=self.experiment,
            task=task,
            script=self.script,
            config_file=self.config_file,
            config_hash=config_hash(config),
            config_content=json.dumps(config, ensure_ascii=False),
            started_at=self.started_at,
            **self._provenance,
        )
        self.run_no = self.store.begin_run(record)
        self.logger.info(f"run {self.run_id} (#{self.run_no}) {self.config_file}")

    def __exit__(self, exc_type, exc, tb) -> bool:
        exc_info = (exc_type, exc, tb)
        if exc is not None:
            self._log_failure(exc_info)  # the one traceback in the job log and on the console, at once
        resources = self.resources.summary()
        if is_local_main_process():
            outcome = "completed" if exc is None else "failed"
            self.logger.info(
                f"run {self.run_id} {outcome} after {time.monotonic() - self._t0:.0f} s: {ResourceProbe.describe(resources)}"
            )
        if self.store is not None and self.run_no is not None:
            if exc is None:
                status, message = "completed", None
            else:
                status = "error"
                message = "".join(traceback.format_exception(exc_type, exc, tb))
            try:
                self.store.finish_run(self.run_no, status, message, resources=resources)
            except Exception as store_exc:  # noqa: BLE001 - a records error must never replace the exception that ended the run
                self._record_failure("finish_run", {"status": status}, store_exc, level=logging.ERROR)
        if exc is not None:
            # after any finish_run block, so the error file ends with the terminal traceback
            self._terminal_block(exc_info)
        if self.write_failures:
            self.logger.warning(
                f"Run {self.run_id}: {self.write_failures} record write(s) failed; details in {self.error_file}"
            )
        return False

    def save(self, record: Record, replace: bool = False) -> None:
        """Insert one row into its class's table; the run's ``run_id`` is filled in. Never raises.

        Args:
            record: The row.
            replace: Overwrite the row with the same natural key (a one-per-run row refreshed during the run).
        """
        self._write(record, lambda: self.store.save(self._keyed(record), replace=replace))

    def save_many(self, records: Sequence[Record]) -> None:
        """Insert several rows, of one or several tables, in one transaction. Never raises."""
        if not records:
            return
        self._write(records, lambda: self.store.save_many([self._keyed(r) for r in records]))

    def log_epoch(
        self,
        epoch: int,
        train_loss: Optional[float] = None,
        val_loss: Optional[float] = None,
        test_loss: Optional[float] = None,
    ) -> None:
        """Write one ``run_epochs`` row. Never raises.

        Args:
            epoch: 1-based epoch index.
            train_loss: Mean training loss of the epoch.
            val_loss: Mean validation loss; ``None`` when the validation split was not scored.
            test_loss: Mean test loss; ``None`` when the test split was not scored.
        """
        if self.store is None:
            return

        def as_float(x: Optional[float]) -> Optional[float]:
            return None if x is None else float(x)

        self.save(
            Epoch(
                epoch=int(epoch),
                train_loss=as_float(train_loss),
                val_loss=as_float(val_loss),
                test_loss=as_float(test_loss),
            )
        )

    def _keyed(self, record: Record) -> Record:
        """Fill ``run_id`` when the record class has it and the caller left it unset."""
        if "run_id" in record.get_core_field_names() and getattr(record, "run_id", None) is None:
            record.run_id = self.run_id
        return record

    def _write(self, payload: Any, op: Callable[[], None]) -> None:
        if self.store is None:
            return
        table = ", ".join(
            dict.fromkeys(r.__table__ for r in (payload if isinstance(payload, (list, tuple)) else [payload]))
        )
        if self.run_no is None:
            self.logger.warning(f"Run {self.run_id}: write to '{table}' before set_config() ignored")
            return
        try:
            op()
        except _STORE_ERRORS as exc:
            self._record_failure(table, payload, exc)

    def _log_failure(self, exc_info) -> None:
        self.logger.error(f"run {self.run_id} failed: {exc_info[1]!r}", exc_info=exc_info)

    def _terminal_block(self, exc_info) -> None:
        """The last block of the error file: header fields and the traceback of the exception that ended the run."""
        if not is_local_main_process():
            return  # one file per run: the main rank writes it; every rank has logged the traceback
        try:
            write_error_block(self.error_file, self._header_fields(), exc_info)
        except OSError:
            pass  # the error channel itself is best effort

    def _header_fields(self) -> Dict[str, str]:
        return {
            "run_id": self.run_id,
            "run_no": str(self.run_no) if self.run_no is not None else "-",
            "database": self._db_path or "-",
            "task": self.task or "-",
            "experiment": self.experiment or "-",
            "config": self.config_file or "-",
            "job log": ", ".join(job_log_paths()) or "-",
            "command": " ".join(sys.argv),
        }

    def _record_failure(self, table: str, payload: Any, exc: BaseException, level: int = logging.WARNING) -> None:
        self.write_failures += 1
        if self.write_failures == 1 or level >= logging.ERROR:
            self.logger.log(
                level, f"Run {self.run_id}: could not write to '{table}' ({exc}); the computation continues"
            )
        else:
            self.logger.debug(f"Run {self.run_id}: could not write to '{table}' ({exc})")
        try:
            fields = {
                **self._header_fields(),
                "table": table,
                "record": json.dumps(_dump(payload), ensure_ascii=False, default=str),
            }
            write_error_block(self.error_file, fields, (type(exc), exc, exc.__traceback__))
        except OSError:
            pass  # the error channel itself is best effort


def _experiment_dir(experiment: Optional[str]) -> Path:
    """``<experiment>`` as a directory under ``error_dir()``.

    Nested names stay (``holdout/main``), an anchor or ``..`` cannot leave the error directory, and nothing usable
    becomes ``_``.
    """
    parts = [p for p in PurePath(experiment or "").parts if p not in ("..", ".") and not PurePath(p).anchor]
    return Path(*parts) if parts else Path("_")


def _dump(payload: Any) -> Any:
    if isinstance(payload, Record):
        return payload.to_dict()
    if isinstance(payload, (list, tuple)):
        return [_dump(x) for x in payload]
    return payload
