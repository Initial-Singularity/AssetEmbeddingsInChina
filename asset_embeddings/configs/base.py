"""The `Config` base class, the `ConfigField` descriptor and the metaclass that collects fields.

A config class declares its fields as `ConfigField` descriptors; the descriptor preprocesses and
checks each assignment, `Config` loads values from files, dicts, argparse and the environment, and
`validate()` runs the required-field, field-constraint and cross-field checks.
"""

import os
import copy
import argparse
from typing import Any, Dict, Union, Callable, Type, Self, ClassVar

import json
import yaml
import toml

from .constraints import (
    Constraint,
    NoConstraint,
    collect_cross_constraints,
)


# TOML has no null; these strings stand in for None in TOML files.
NONE_MARKERS = {"__none__", "__null__", "null", "None"}


def _convert_markers_to_none(obj: Any) -> Any:
    """Replace every None marker string in a loaded TOML structure with None, recursively."""
    if isinstance(obj, dict):
        return {k: _convert_markers_to_none(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_convert_markers_to_none(item) for item in obj]
    elif isinstance(obj, str) and obj in NONE_MARKERS:
        return None
    else:
        return obj


def _convert_none_to_markers(obj: Any) -> Any:
    """Replace every None in a structure with the `"__none__"` marker, recursively, for TOML output."""
    if isinstance(obj, dict):
        return {k: _convert_none_to_markers(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_convert_none_to_markers(item) for item in obj]
    elif obj is None:
        return "__none__"
    else:
        return obj


def read_configs(config_file: str) -> Dict:
    """Read a JSON, YAML or TOML config file into a dict, chosen by extension.

    In a TOML file the strings in `NONE_MARKERS` are read as None.

    Args:
        config_file: Path with extension `.json`, `.yaml`, `.yml` or `.toml`.

    Returns:
        The parsed content.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the extension is not one of the supported formats.
    """
    if not os.path.exists(config_file):
        raise FileNotFoundError(f"Config file `{config_file}` not found.")

    file_ext = os.path.splitext(config_file)[1].lower()

    with open(config_file, "r") as f:
        if file_ext == ".json":
            config_dict = json.load(f)
        elif file_ext == ".toml":
            config_dict = toml.load(f)
            config_dict = _convert_markers_to_none(config_dict)
        elif file_ext in [".yaml", ".yml"]:
            config_dict = yaml.safe_load(f)
        else:
            raise ValueError(f"Unsupported config file format: {file_ext}")

    return config_dict


def save_configs(config_file: str, config_dict: Dict) -> None:
    """Write a dict to a JSON, YAML or TOML file, chosen by extension.

    For TOML, None values are written as the `"__none__"` marker.

    Args:
        config_file: Destination path with extension `.json`, `.yaml`, `.yml` or `.toml`.
        config_dict: The values to write.

    Raises:
        ValueError: If the extension is not one of the supported formats.
    """
    file_ext = os.path.splitext(config_file)[1].lower()

    with open(config_file, "w") as f:
        if file_ext == ".json":
            json.dump(config_dict, f, indent=4)
        elif file_ext == ".toml":
            toml_dict = _convert_none_to_markers(config_dict)
            toml.dump(toml_dict, f)
        elif file_ext in [".yaml", ".yml"]:
            yaml.safe_dump(config_dict, f, default_flow_style=False)
        else:
            raise ValueError(f"Unsupported config file format: {file_ext}")


class ConfigField:
    """Descriptor that declares one field of a `Config` class.

    Assignment through the descriptor runs the preprocessor, the required check and the field
    constraint before the value is stored on the instance.

    Attributes:
        default: Value the field takes when nothing sets it.
        constraint: Field constraint checked on assignment and in `validate()`.
        preprocessor: Callable applied to a non-None value before it is checked.
        doc: The field's documentation.
        required: Whether None is rejected for this field.
        name: The attribute name, filled in by `__set_name__`.
    """

    def __init__(
        self,
        default: Any = None,
        constraint: Constraint = NoConstraint(),
        preprocessor: Callable[[Any], Any] = lambda x: x,
        doc: str = "",
        required: bool = False,
    ):
        """Declare a field.

        Args:
            default: Value the field takes when nothing sets it.
            constraint: Field constraint; the default `NoConstraint()` checks nothing.
            preprocessor: Callable applied to a non-None value before the checks, for example to
                normalize a path or split a comma-separated string.
            doc: The field's documentation.
            required: If True, assigning None raises and `validate()` rejects a None value.
        """
        self.default: Any = default
        self.constraint: Constraint = constraint
        self.preprocessor: Callable[[Any], Any] = preprocessor
        self.doc: str = doc
        self.required: bool = required
        # Only visible when a field is created outside a class body; __set_name__ overwrites it.
        self.name: str = "AnonymousField"

    def __set_name__(self, owner: Type, name: str):
        self.name = name

    def __get__(self, instance, owner):
        if instance is None:
            return self
        return instance._values.get(self.name, self.default)

    def __set__(self, instance, value):
        if self.preprocessor and value is not None:
            value = self.preprocessor(value)

        if self.required and value is None:
            raise ValueError(f"Field '{self.name}' is required")

        if self.constraint:
            self.constraint.check(self.name, value)

        instance._values[self.name] = value


class ConfigMeta(type):
    """Metaclass that collects the `ConfigField` descriptors and cross-constraints of a config class.

    Every class it creates carries `_config_fields` (name to descriptor, inherited fields
    included) and `_all_cross_constraints` (see `collect_cross_constraints`).
    """

    def __new__(mcs, name, bases, namespace):
        """Create the class and attach its field and cross-constraint tables.

        Args:
            name: Name of the class being created.
            bases: Its base classes.
            namespace: Its class body.

        Returns:
            The new class.
        """
        cls = super().__new__(mcs, name, bases, namespace)

        # Walk the MRO base-first so a field redeclared in a subclass replaces the inherited one.
        cls._config_fields = {}  # type: ignore[misc]
        for base in reversed(cls.__mro__):
            if hasattr(base, "__dict__"):
                for field_name, field_obj in base.__dict__.items():
                    if isinstance(field_obj, ConfigField):
                        cls._config_fields[field_name] = field_obj  # type: ignore[misc]

        cls._all_cross_constraints = collect_cross_constraints(cls)  # type: ignore[misc]

        return cls


class Config(metaclass=ConfigMeta):
    """Base class of every configuration class.

    Fields are `ConfigField` descriptors; values live in `_values` and are reachable as attributes
    or by key. Loading methods return `self` so they chain, later loads overriding earlier ones.
    `freeze()` makes the instance read-only and `validate()` runs all constraints.
    """

    _config_fields: ClassVar[Dict[str, ConfigField]] = {}

    def __init__(self, **initial_values: Any):
        """Create an instance from defaults, overridden by the given field values.

        Args:
            **initial_values: Field values to set, by field name.

        Raises:
            KeyError: When a key names no field of this class.
        """
        self._values: Dict[str, Any] = {}
        self._frozen = False
        unknown = [k for k in initial_values if k not in self._config_fields]
        if unknown:
            raise KeyError(f"{type(self).__name__} has no field(s) {unknown}")

        for name, field in self._config_fields.items():
            if name in initial_values:
                setattr(self, name, initial_values[name])
            else:
                default = field.default
                # A list/dict/set default is shared by every instance unless copied here, and
                # in-place mutation would then leak between them.
                if isinstance(default, (list, dict, set)):
                    default = copy.deepcopy(default)
                self._values[name] = default

    def __setattr__(self, name: str, value: Any):
        # Private names are the instance's own bookkeeping: they bypass the freeze check and the
        # descriptors, which is also what lets __init__ create _values and _frozen.
        if name.startswith("_") or name in ["_values", "_frozen"]:
            super().__setattr__(name, value)
            return

        if hasattr(self, "_frozen") and self._frozen:
            raise AttributeError(f"Cannot modify frozen config field '{name}'")

        if name not in self._config_fields:
            raise AttributeError(f"{type(self).__name__} has no field '{name}'")
        self._config_fields[name].__set__(self, value)

    def __getitem__(self, key: str) -> Any:
        if key in self._config_fields:
            return getattr(self, key)
        raise KeyError(f"Field '{key}' not found")

    def __setitem__(self, key: str, value: Any):
        if key in self._config_fields:
            setattr(self, key, value)
        else:
            raise KeyError(f"Field '{key}' not found")

    def __repr__(self) -> str:
        field_strs = []
        for name in self._config_fields:
            value = getattr(self, name)
            if value != self._config_fields[name].default:
                field_strs.append(f"{name}={value!r}")
        return f"{self.__class__.__name__}({', '.join(field_strs)})"

    def from_file(self, config_file: str) -> Self:
        """Load field values from a JSON, YAML or TOML file (chainable).

        Args:
            config_file: Path to the file.

        Returns:
            This instance.

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If the extension is not one of the supported formats.
        """
        configs = read_configs(config_file)
        return self.from_dict(configs)

    def from_dict(self, config_dict: Dict[str, Any]) -> Self:
        """Load field values from a dict (chainable).

        A key that names no field is refused before any value is applied: it is a typo or a retired key, and
        skipping it would leave the field it meant at its default.

        Args:
            config_dict: Values by field name.

        Returns:
            This instance.

        Raises:
            KeyError: When a key names no field of this class.
        """
        unknown = [k for k in config_dict if k not in self._config_fields]
        if unknown:
            raise KeyError(f"{type(self).__name__} has no field(s) {unknown}")
        for name in self._config_fields:
            if name in config_dict:
                setattr(self, name, config_dict[name])
        return self

    def from_kwargs(self, **kwargs: Any) -> Self:
        """Load field values from keyword arguments (chainable).

        Args:
            **kwargs: Values by field name.

        Returns:
            This instance.
        """
        self.from_dict(kwargs)
        return self

    def from_args(self, args: Union[argparse.Namespace, argparse.ArgumentParser]) -> Self:
        """Load field values from parsed command-line arguments (chainable).

        Only arguments named like a field and not None are applied, so an option the user did not
        pass leaves the value loaded earlier untouched.

        Args:
            args: A parsed namespace, or a parser on which `parse_args()` is called.

        Returns:
            This instance.
        """
        if isinstance(args, argparse.ArgumentParser):
            args = args.parse_args()

        for name in self._config_fields:
            if hasattr(args, name):
                value = getattr(args, name)
                if value is not None:
                    setattr(self, name, value)
        return self

    def from_env(self, prefix: str = "") -> Self:
        """Load field values from environment variables (chainable).

        The variable for a field is the prefix followed by the field name, upper-cased. Values
        arrive as strings; a field's preprocessor is the place to convert them.

        Args:
            prefix: Text prepended to each field name.

        Returns:
            This instance.
        """
        for name in self._config_fields:
            env_name = f"{prefix}{name}".upper()
            if env_name in os.environ:
                setattr(self, name, os.environ[env_name])
        return self

    def freeze(self) -> Self:
        """Make the instance read-only; field assignment then raises `AttributeError` (chainable).

        Returns:
            This instance.
        """
        self._frozen = True
        return self

    def unfreeze(self) -> Self:
        """Allow field assignment again (chainable).

        Returns:
            This instance.
        """
        self._frozen = False
        return self

    def copy(self) -> Self:
        """Return an unfrozen copy with the same field values.

        List, dict and set values are deep-copied. Fields that are None on the original are left
        at the new instance's default rather than assigned, because assigning None to a required
        field would raise.

        Returns:
            The new instance.
        """
        new_config = self.__class__()
        for name in self._config_fields:
            value = getattr(self, name)
            if value is not None:
                if isinstance(value, (list, dict, set)):
                    value = copy.deepcopy(value)
                setattr(new_config, name, value)
        return new_config

    def to_dict(self) -> Dict[str, Any]:
        """Return the current value of every field, by name."""
        return {name: getattr(self, name) for name in self._config_fields}

    def to_file(self, save_file: str) -> None:
        """Write the field values to a JSON, YAML or TOML file, chosen by extension.

        Args:
            save_file: Destination path.
        """
        save_configs(save_file, self.to_dict())

    def validate(self) -> Self:
        """Run every field constraint, then every cross-field constraint (chainable).

        Fail-fast: the first failure raises and later constraints do not run, so cross-field
        constraints only ever see field values that passed their own checks.

        Returns:
            This instance.

        Raises:
            ValueError: If a required field is None.
            ConstraintError: If a field value or a cross-field invariant fails.
        """
        for name, field in self._config_fields.items():
            value = getattr(self, name)
            if field.required and value is None:
                raise ValueError(f"Required field '{name}' is None")
            if field.constraint and value is not None:
                field.constraint.check(name, value)

        for cc in self._all_cross_constraints:
            cc.check(self)

        return self

    def get_field_info(self, field_name: str) -> Dict[str, Any]:
        """Return one field's declaration and current value.

        Args:
            field_name: Name of the field.

        Returns:
            Keys `name`, `default`, `required`, `doc`, `value` and `constraint`.

        Raises:
            ValueError: If no field has that name.
        """
        if field_name not in self._config_fields:
            raise ValueError(f"Field '{field_name}' not found")

        field = self._config_fields[field_name]
        return {
            "name": field_name,
            "default": field.default,
            "required": field.required,
            "doc": field.doc,
            "value": getattr(self, field_name),
            "constraint": field.constraint,
        }

    @classmethod
    def get_schema(cls) -> Dict[str, Any]:
        """Return the declaration of every field, by name.

        Returns:
            For each field the keys `default`, `required`, `doc` and `constraint` (its `repr`).
        """
        schema = {}
        for name, field in cls._config_fields.items():
            schema[name] = {
                "default": field.default,
                "required": field.required,
                "doc": field.doc,
                "constraint": repr(field.constraint),
            }
        return schema
