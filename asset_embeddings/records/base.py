"""Base classes of the records system (schema v7).

``Record`` is the mechanism (typed fields, the table DDL); ``RunRecord`` is the ``runs`` row, the one home of a run's
config and status; ``Detail`` is any row owned by a run (``run_id`` link and a declared natural key); ``Result`` is a
headline row: ``metric`` / ``value`` plus the config columns and summary keys its views expose. Every concrete class
names its table in ``__table__`` and registers itself by it.
"""

import hashlib
import json
import uuid
from typing import Any, Dict, List, Optional, Tuple, Type, get_args, get_origin, get_type_hints

#: Schema version, stored as the SQLite ``user_version`` of every file and in ``runs.schema_version``.
SCHEMA_VERSION = 7

#: The values ``Result.metric`` takes: what each headline number is.
METRICS: Dict[str, str] = {
    "r2": "Cross-sectional ridge R² on every firm of the period (RV explanatory, RC predictive)",
    "r2_holdout": (
        "Koijen test R², 1 - Var(resid) / Var(y), on held-out firms, aggregated over the firm splits "
        "(holdout_split_aggregate; the mean in runs written before that field)"
    ),
    "mp_score": "Managed-portfolio score: normalized log-likelihood of the masked holdings on the test split",
    "auc": "Label probe: mean test-fold ROC AUC of a logistic regression of a binary firm label on the features",
    "fm_coef": "Pairwise similarity: Fama-MacBeth mean over the periods of a regression coefficient",
    "r2_temporal": (
        "Temporal RC: Koijen R² of the slope(s) fitted on earlier periods (slope_window; the previous period only "
        "in runs written before that field), applied to this period"
    ),
    "r2_temporal_aligned": (
        "Temporal RC as r2_temporal, this period's embedding Procrustes-aligned onto each earlier period's"
    ),
}


class RecordsSchemaError(RuntimeError):
    """Raised when a database file is not at schema version ``SCHEMA_VERSION``."""


def config_hash(config_dict: Dict[str, Any]) -> str:
    """SHA-1 of the canonical JSON serialization of a config dict, independent of key order.

    Args:
        config_dict: The config as a plain dict.

    Returns:
        The 40-character hex digest.
    """
    canonical = json.dumps(config_dict, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha1(canonical.encode("utf-8")).hexdigest()


SQLITE_TYPE_AFFINITY = {
    int: "INTEGER",
    float: "REAL",
    str: "TEXT",
    bool: "INTEGER",
    bytes: "BLOB",
}


def _resolve_sqlite_type(type_hint: Type) -> str:
    """Convert a Python type hint to a SQLite type string."""
    if type_hint is None:
        return "TEXT"
    origin = get_origin(type_hint)
    # Optional[X] / Union[X, None]: the first non-None member decides the affinity
    if origin is type(None) or str(origin) == "typing.Union":
        non_none = [a for a in get_args(type_hint) if a is not type(None)]
        return _resolve_sqlite_type(non_none[0]) if non_none else "TEXT"
    if origin is None:
        return SQLITE_TYPE_AFFINITY.get(type_hint, "TEXT")
    return "TEXT"


class RecordField:
    """Field descriptor of the ``Record`` classes: type check on assignment, a default and a doc string.

    Args:
        type_hint: The value type; taken from the class annotation when omitted.
        default: Value returned while the field is unset.
        required: Reject a record that leaves the field ``None``.
        doc: What the column holds.
        primary_key: Emit ``PRIMARY KEY`` on the column.
        nullable: Emit ``NOT NULL`` when false; defaults to ``not required``.
    """

    def __init__(
        self,
        type_hint: Optional[Type] = None,
        default: Any = None,
        required: bool = False,
        doc: str = "",
        primary_key: bool = False,
        nullable: Optional[bool] = None,
    ):
        self.type_hint = type_hint
        self.default = default
        self.required = required
        self.doc = doc
        self.primary_key = primary_key
        self.nullable = (not required) if nullable is None else nullable
        self.name: Optional[str] = None

    def __set_name__(self, owner, name):
        self.name = name
        if self.type_hint is None:
            hints = get_type_hints(owner, include_extras=True)
            self.type_hint = hints.get(name)

    def __get__(self, obj, objtype=None):
        if obj is None:
            return self
        return obj._field_values.get(self.name, self.default)

    def __set__(self, obj, value):
        if value is not None and self.type_hint is not None and not self._check_type(value, self.type_hint):
            raise TypeError(f"Field '{self.name}': expected {self.type_hint}, got {type(value).__name__}")
        obj._field_values[self.name] = value

    def _check_type(self, value: Any, type_hint: Type) -> bool:
        origin = get_origin(type_hint)
        if origin is type(None) or str(origin) == "typing.Union":
            return any(self._check_type(value, arg) for arg in get_args(type_hint))
        if origin is None:
            # an int is a valid float; a bool is not a number here
            if type_hint is float:
                return isinstance(value, (int, float)) and not isinstance(value, bool)
            if type_hint is int:
                return isinstance(value, int) and not isinstance(value, bool)
            return isinstance(value, type_hint)
        return isinstance(value, origin)


_REGISTRY: Dict[str, Type["Record"]] = {}


def record_classes() -> Dict[str, Type["Record"]]:
    """Every concrete record class by its table name."""
    return dict(_REGISTRY)


class Record:
    """Base of every row type: typed fields and the table DDL.

    A concrete class sets ``__table__`` (and registers under it). Keyword arguments must name fields; the
    table has exactly the declared columns. A ``Record`` never touches a database connection; that is
    [`RecordStore`][asset_embeddings.records.store.RecordStore]'s job.
    """

    #: The table of a concrete class; None on the abstract bases.
    __table__: Optional[str] = None
    #: SQL table constraints (``CHECK (...)``, ``FOREIGN KEY ...``) declared by this class, merged along the MRO.
    __table_constraints__: Tuple[str, ...] = ()

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        core_fields = {}
        constraints: List[str] = []
        for klass in reversed(cls.__mro__):
            for name, value in vars(klass).items():
                if isinstance(value, RecordField):
                    core_fields[name] = value
            for clause in vars(klass).get("__table_constraints__", ()):
                if clause not in constraints:
                    constraints.append(clause)
        cls._core_fields: Dict[str, RecordField] = core_fields
        cls._table_constraints: Tuple[str, ...] = tuple(constraints)
        table = vars(cls).get("__table__")
        if table is not None:
            _REGISTRY[table] = cls

    def __init__(self, **kwargs):
        if type(self).__table__ is None:
            raise TypeError(f"{type(self).__name__} declares no __table__; only concrete record classes are built")
        unknown = [k for k in kwargs if k not in self._core_fields]
        if unknown:
            raise TypeError(f"{type(self).__name__} has no field(s) {unknown}")
        self._field_values: Dict[str, Any] = {}
        for key, value in kwargs.items():
            setattr(self, key, value)
        for name, field in self._core_fields.items():
            if field.required and self._field_values.get(name) is None:
                raise ValueError(f"Required field '{name}' is not set")

    @classmethod
    def get_core_field_names(cls) -> List[str]:
        """Names of the fields in declaration order, ancestors first."""
        return list(cls._core_fields.keys())

    @classmethod
    def get_table_schema(cls) -> Dict[str, str]:
        """``{field_name: sqlite_type}`` (types only; ``get_table_ddl`` adds the constraints)."""
        return {name: _resolve_sqlite_type(field.type_hint) for name, field in cls._core_fields.items()}

    @classmethod
    def table_constraints(cls) -> Tuple[str, ...]:
        """Table-level constraint clauses: the declared ones along the MRO."""
        return cls._table_constraints

    @classmethod
    def get_table_ddl(cls) -> str:
        """``CREATE TABLE`` statement of ``__table__`` with keys, NOT NULL, UNIQUE, CHECK and FOREIGN KEY clauses."""
        cols = []
        for name, field in cls._core_fields.items():
            parts = [name, _resolve_sqlite_type(field.type_hint)]
            if field.primary_key:
                parts.append("PRIMARY KEY")
            if not field.nullable:
                parts.append("NOT NULL")
            cols.append(" ".join(parts))
        return f"CREATE TABLE {cls.__table__} ({', '.join(cols + list(cls.table_constraints()))})"

    def to_dict(self) -> Dict[str, Any]:
        """The set fields as a flat dict (booleans as integers); unset fields are left out."""
        result = {}
        for name in self._core_fields:
            value = getattr(self, name)
            if value is not None:
                result[name] = int(value) if isinstance(value, bool) else value
        return result


class RunRecord(Record):
    """One row per script invocation: the single home of config and status."""

    __table__ = "runs"
    __table_constraints__ = (
        "UNIQUE (run_id)",
        "CHECK (status IN ('running', 'completed', 'error'))",
        f"CHECK (schema_version = {SCHEMA_VERSION})",
    )

    run_no: int = RecordField(
        primary_key=True, doc="Display sequence of the file (SQLite rowid alias), assigned on insert"
    )
    run_id: str = RecordField(
        required=True, doc="UUID4 identifier: the link key of every other table, in logs and runs_query"
    )
    experiment: str = RecordField(doc="Experiment family, e.g. main, timesplit_robust, pt_strategy_ablation/no_pt_init")
    task: str = RecordField(
        required=True,
        doc="RV / RC / MP / pretrain_BERT / finetune_BERT / pretrain_W2V / finetune_W2V / fit_RS / GROUPED_RV / GROUPED_RC",
    )
    script: str = RecordField(doc="Module name, e.g. asset_embeddings.scripts.eval.rv")
    config_file: str = RecordField(doc="Config path as given on the CLI")
    config_hash: str = RecordField(doc="sha1 of json.dumps(config, sort_keys=True)")
    config_content: str = RecordField(doc="Full config JSON snapshot; the views read config columns from it")
    started_at: str = RecordField(required=True, doc="ISO 8601")
    finished_at: str = RecordField(doc="ISO 8601")
    duration_sec: float = RecordField(doc="finished_at - started_at in seconds")
    peak_ram_mb: float = RecordField(
        doc="Peak resident memory of the process tree during the run (MB), DataLoader workers included"
    )
    peak_vram_mb: float = RecordField(doc="Peak CUDA memory allocated during the run (MB); NULL when CUDA was not used")
    status: str = RecordField(default="running", nullable=False, doc="running / completed / error")
    error_message: str = RecordField(doc="Traceback when status=error")
    git_commit: str = RecordField(doc="git rev-parse HEAD at run time, None outside a repo")
    git_dirty: bool = RecordField(doc="Working tree had uncommitted changes")
    hostname: str = RecordField()
    package_version: str = RecordField(doc="importlib.metadata version of asset_embeddings")
    schema_version: int = RecordField(default=SCHEMA_VERSION, nullable=False)


class Detail(Record):
    """A row owned by one run: the ``run_id`` link and a natural key, unique within the run.

    ``__key__`` names the key columns besides ``run_id``; the DDL turns them into ``UNIQUE (run_id, ...)``.
    ``__config_columns__`` (name to JSON path in ``runs.config_content``) are exposed by the table's view.
    """

    __key__: Tuple[str, ...] = ()
    __config_columns__: Dict[str, str] = {}
    __table_constraints__ = ("FOREIGN KEY (run_id) REFERENCES runs(run_id)",)

    run_id: str = RecordField(nullable=False, doc="UUID4 of the owning run (filled by Run.save when unset)")

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        missing = [k for k in cls.__key__ if k not in cls._core_fields]
        if missing:
            raise TypeError(f"{cls.__name__}.__key__ names no field(s) {missing}")
        clash = set(cls.__config_columns__) & set(cls._core_fields)
        if clash:
            raise TypeError(f"{cls.__name__}: config columns {sorted(clash)} collide with fields")

    @classmethod
    def table_constraints(cls) -> Tuple[str, ...]:
        """The declared clauses plus ``UNIQUE (run_id, <key>)``."""
        return (f"UNIQUE ({', '.join(('run_id',) + tuple(cls.__key__))})",) + cls._table_constraints


class Result(Detail):
    """A headline row: ``metric`` / ``value`` at the natural grain of its table.

    ``metric`` takes a value of `METRICS`. ``__summary_keys__`` are the key columns besides the period that the
    summary view groups by (with ``run_id`` and ``metric``).
    """

    __summary_keys__: Tuple[str, ...] = ()

    metric: str = RecordField(
        required=True,
        doc="What value is: a key of METRICS (r2, r2_holdout, mp_score, auc, fm_coef, r2_temporal, r2_temporal_aligned)",
    )
    value: float = RecordField(required=True, doc="The headline number")


def generate_run_id() -> str:
    """Generate a UUID4 run id."""
    return str(uuid.uuid4())
