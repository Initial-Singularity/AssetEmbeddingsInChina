"""Base classes for the Record system.

Provides RecordField descriptor, Record/Result/Recording base classes,
SQLite type mapping, and generate_run_id helper.
"""

import uuid
from typing import Any, Dict, List, Optional, Type, get_type_hints, get_origin, get_args


# SQLite type affinity mapping
SQLITE_TYPE_AFFINITY = {
    int: "INTEGER",
    float: "REAL",
    str: "TEXT",
    bool: "INTEGER",
    bytes: "BLOB",
}


def _resolve_sqlite_type(type_hint: Type) -> str:
    """Convert a Python type hint to SQLite type string."""
    if type_hint is None:
        return "TEXT"

    origin = get_origin(type_hint)

    # Handle Optional[X] (Union[X, None])
    if origin is type(None) or str(origin) == "typing.Union":
        args = get_args(type_hint)
        non_none = [a for a in args if a is not type(None)]
        if non_none:
            return _resolve_sqlite_type(non_none[0])
        return "TEXT"

    # Plain type
    if origin is None:
        return SQLITE_TYPE_AFFINITY.get(type_hint, "TEXT")

    return "TEXT"


class RecordField:
    """Lightweight field descriptor for Record classes.

    Simpler than ConfigField — no constraints system, just type info and docs.
    """

    def __init__(
        self,
        type_hint: Optional[Type] = None,
        default: Any = None,
        required: bool = False,
        doc: str = "",
    ):
        self.type_hint = type_hint
        self.default = default
        self.required = required
        self.doc = doc
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
        if value is not None and self.type_hint is not None:
            if not self._check_type(value, self.type_hint):
                raise TypeError(f"Field '{self.name}': expected {self.type_hint}, " f"got {type(value).__name__}")
        obj._field_values[self.name] = value

    def _check_type(self, value: Any, type_hint: Type) -> bool:
        origin = get_origin(type_hint)

        # Handle Optional / Union
        if origin is type(None) or str(origin) == "typing.Union":
            args = get_args(type_hint)
            return any(self._check_type(value, arg) for arg in args)

        # Basic type
        if origin is None:
            return isinstance(value, type_hint)

        # Generic types — check origin
        return isinstance(value, origin)


class Record:
    """Base class for structured records with SQLite compatibility.

    Core fields are defined via RecordField descriptors (type-safe).
    Dynamic fields can be added at init time for extensibility.
    Record does NOT manage DB connections — that's RecordStore's job.
    """

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        # Walk MRO to collect all RecordField descriptors from all ancestors
        core_fields = {}
        for klass in reversed(cls.__mro__):
            for name, value in vars(klass).items():
                if isinstance(value, RecordField):
                    core_fields[name] = value
        cls._core_fields: Dict[str, RecordField] = core_fields

    def __init__(self, **kwargs):
        self._field_values: Dict[str, Any] = {}
        self._dynamic_fields: Dict[str, Any] = {}

        for key, value in kwargs.items():
            if key in self._core_fields:
                setattr(self, key, value)
            else:
                self._dynamic_fields[key] = value

        self._validate_required()

    def _validate_required(self):
        for name, field in self._core_fields.items():
            if field.required and self._field_values.get(name) is None:
                raise ValueError(f"Required field '{name}' is not set")

    @classmethod
    def get_core_field_names(cls) -> List[str]:
        return list(cls._core_fields.keys())

    @classmethod
    def get_table_schema(cls) -> Dict[str, str]:
        """Return {field_name: sqlite_type} for CREATE TABLE."""
        schema = {}
        for name, field in cls._core_fields.items():
            schema[name] = _resolve_sqlite_type(field.type_hint)
        return schema

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to flat dict. Skips None-valued core fields."""
        result = {}
        for name in self._core_fields:
            value = getattr(self, name)
            if value is not None:
                # Convert bool to int for SQLite compatibility
                if isinstance(value, bool):
                    result[name] = int(value)
                else:
                    result[name] = value
        result.update(self._dynamic_fields)
        return result

    def validate_sqlite_compatible(self) -> bool:
        """Check all field values are SQLite-compatible types."""
        data = self.to_dict()
        for key, value in data.items():
            if value is None:
                continue
            if not isinstance(value, (str, int, float, bool, bytes)):
                return False
        return True


class Result(Record):
    """Base for experiment outcome records."""

    run_id: str = RecordField(required=True, doc="UUID4 identifier")
    recorded_at: str = RecordField(required=True, doc="ISO 8601 timestamp")
    config_file: str = RecordField(doc="Path to config file used")
    config_content: str = RecordField(doc="Full config as JSON string")
    status: str = RecordField(default="completed", doc="completed / error")
    error_message: str = RecordField(doc="Traceback if status=error")


class Recording(Record):
    """Base for runtime diagnostic records."""

    run_id: str = RecordField(required=True, doc="UUID4, links to parent Result")
    recorded_at: str = RecordField(doc="ISO 8601 timestamp")
    config_file: str = RecordField(doc="Path to config file used")
    config_content: str = RecordField(doc="Full config as JSON string")
    status: str = RecordField(default="completed", doc="completed / error")
    error_message: str = RecordField(doc="Traceback if status=error")


def generate_run_id() -> str:
    """Generate a UUID4 run_id."""
    return str(uuid.uuid4())
