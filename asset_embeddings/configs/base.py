import os
import copy
import warnings
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


# TOML不原生支持None/null, 使用特殊字符串标记
NONE_MARKERS = {"__none__", "__null__", "null", "None"}


def _convert_markers_to_none(obj: Any) -> Any:
    """递归转换特殊None标记为Python的None值.

    TOML格式不支持null值，因此使用特殊字符串标记来表示None。
    此函数递归遍历配置字典，将所有None标记转换为Python的None。

    Args:
        obj: 待转换的对象（字典、列表或基本类型）

    Returns:
        转换后的对象，所有None标记替换为None

    Examples:
        >>> _convert_markers_to_none({"path": "__none__", "value": 42})
        {'path': None, 'value': 42}

        >>> _convert_markers_to_none(["__none__", "test", "__null__"])
        [None, 'test', None]
    """
    if isinstance(obj, dict):
        return {k: _convert_markers_to_none(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_convert_markers_to_none(item) for item in obj]
    elif isinstance(obj, str) and obj in NONE_MARKERS:
        return None
    else:
        return obj


def _convert_none_to_markers(obj: Any) -> Any:
    """递归转换Python的None值为TOML兼容的标记字符串.

    Args:
        obj: 待转换的对象（字典、列表或基本类型）

    Returns:
        转换后的对象，所有None替换为"__none__"标记

    Examples:
        >>> _convert_none_to_markers({"path": None, "value": 42})
        {'path': '__none__', 'value': 42}
    """
    if isinstance(obj, dict):
        return {k: _convert_none_to_markers(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_convert_none_to_markers(item) for item in obj]
    elif obj is None:
        return "__none__"
    else:
        return obj


def read_configs(config_file: str) -> Dict:
    """Reads configuration data from a file in various formats.

    Supports JSON, YAML, and TOML file formats, automatically detecting the format
    based on the file extension.

    Note:
        TOML does not natively support null/None values. Use special string markers
        "__none__", "__null__", "null", or "None" in TOML files to represent None.
        These will be automatically converted to Python None during loading.

    Args:
        config_file: Path to the configuration file. Must have one of the supported
                    extensions: .json, .yaml, .yml, or .toml.

    Returns:
        A dictionary containing the parsed configuration data.

    Raises:
        FileNotFoundError: If the specified config file does not exist.
        ValueError: If the file format is not supported.
        json.JSONDecodeError: If JSON file is malformed.
        yaml.YAMLError: If YAML file is malformed.
        toml.TomlDecodeError: If TOML file is malformed.

    Examples:
        >>> config_dict = read_configs("app_config.yaml")
        >>> print(config_dict["database"]["host"])
        localhost

        >>> config_dict = read_configs("settings.json")
        >>> print(config_dict["api_key"])
        abc123

        >>> # TOML with None markers
        >>> config_dict = read_configs("config.toml")
        >>> # embedding_path = "__none__" in TOML -> None in Python
        >>> print(config_dict["embedding_path"])
        None
    """
    if not os.path.exists(config_file):
        raise FileNotFoundError(f"Config file `{config_file}` not found.")

    file_ext = os.path.splitext(config_file)[1].lower()

    with open(config_file, "r") as f:
        if file_ext == ".json":
            config_dict = json.load(f)
        elif file_ext == ".toml":
            config_dict = toml.load(f)
            # TOML不支持null，转换特殊标记为None
            config_dict = _convert_markers_to_none(config_dict)
        elif file_ext in [".yaml", ".yml"]:
            config_dict = yaml.safe_load(f)
        else:
            raise ValueError(f"Unsupported config file format: {file_ext}")

    return config_dict


def save_configs(config_file: str, config_dict: Dict):
    """Saves configuration data to a file in the specified format.

    Automatically determines the output format based on the file extension and
    formats the data appropriately.

    Note:
        When saving to TOML format, Python None values are automatically converted
        to the string marker "__none__" since TOML does not support null values.

    Args:
        config_file: Path where the configuration file should be saved. The file
                    extension determines the format (.json, .yaml, .yml, .toml).
        config_dict: Dictionary containing the configuration data to save.

    Raises:
        ValueError: If the file format is not supported.
        PermissionError: If there are insufficient permissions to write the file.

    Examples:
        >>> config_data = {"database": {"host": "localhost", "port": 5432}}
        >>> save_configs("app_config.yaml", config_data)

        >>> save_configs("settings.json", {"debug": True, "log_level": "INFO"})

        >>> # Saving None values to TOML
        >>> save_configs("config.toml", {"embedding_path": None})
        >>> # Creates: embedding_path = "__none__"
    """
    file_ext = os.path.splitext(config_file)[1].lower()

    with open(config_file, "w") as f:
        if file_ext == ".json":
            json.dump(config_dict, f, indent=4)
        elif file_ext == ".toml":
            # TOML不支持null，转换None为特殊标记
            toml_dict = _convert_none_to_markers(config_dict)
            toml.dump(toml_dict, f)
        elif file_ext in [".yaml", ".yml"]:
            yaml.safe_dump(config_dict, f, default_flow_style=False)
        else:
            raise ValueError(f"Unsupported config file format: {file_ext}")


# ============================================================================
# 字段描述符 Field Descriptor
# ============================================================================


class ConfigField:
    """Descriptor for configuration fields with validation and preprocessing.

    This descriptor provides declarative field definitions with explicit constraint
    validation, preprocessing, and documentation support.

    Attributes:
        default: The default value for the field.
        constraint: Constraint object for validation, or None for no validation.
        preprocessor: Optional callable to transform values before validation.
        doc: Documentation string describing the field's purpose.
        required: Whether the field must have a non-None value.
        name: The attribute name (set automatically by __set_name__).
    """

    def __init__(
        self,
        default: Any = None,
        constraint: Constraint = NoConstraint(),
        preprocessor: Callable[[Any], Any] = lambda x: x,
        doc: str = "",
        required: bool = False,
    ):
        """Initializes a ConfigField with validation and processing options.

        Args:
            default: The default value for this field. Used when the field is
                    not explicitly set.
            constraint: Constraint for validation. Defaults to NoConstraint() (no
                       validation); pass an explicit constraint to validate this field.
            preprocessor: Optional callable that transforms the value before
                         validation. Useful for path normalization, string conversion, etc.
            doc: Documentation string describing the field's purpose and usage.
            required: If True, the field must be set to a non-None value.
                     Required fields will raise ValueError if left as None.
        """
        self.default: Any = default
        self.constraint: Constraint = constraint
        self.preprocessor: Callable[[Any], Any] = preprocessor
        self.doc: str = doc
        self.required: bool = required
        self.name: str = "AnonymousField"  # Will be set in __set_name__

    def __set_name__(self, owner: Type, name: str):
        """Called when the descriptor is assigned to a class attribute.

        Records the attribute name. Constraints are NOT inferred from type hints — each
        field declares its constraint explicitly via the ``constraint=`` argument (the
        default is ``NoConstraint()``, i.e. no validation).

        Args:
            owner: The class that owns this descriptor.
            name: The attribute name this descriptor is assigned to.
        """
        self.name = name

    def __get__(self, instance, owner):
        """Gets the field value from an instance.

        Args:
            instance: The instance to get the value from, or None for class access.
            owner: The class that owns this descriptor.

        Returns:
            The descriptor itself when accessed from class, or the field value
            when accessed from an instance.
        """
        if instance is None:
            return self
        return instance._values.get(self.name, self.default)

    def __set__(self, instance, value):
        """Sets the field value on an instance with validation.

        Applies preprocessing and validation before storing the value.

        Args:
            instance: The instance to set the value on.
            value: The value to set.

        Raises:
            ValueError: If the field is required but the value is None.
            ConstraintError: If the value fails constraint validation.
        """
        # Apply preprocessing if defined
        if self.preprocessor and value is not None:
            value = self.preprocessor(value)

        # Check required field constraint
        if self.required and value is None:
            raise ValueError(f"Field '{self.name}' is required")

        # Apply constraint validation if defined
        if self.constraint:
            self.constraint.check(self.name, value)

        instance._values[self.name] = value


# ============================================================================
# Config基类 Config Base Classes
# ============================================================================


class ConfigMeta(type):
    """Metaclass for Config classes that collects field information.

    This metaclass automatically discovers and collects all ConfigField descriptors
    from the class hierarchy, making them available for validation and introspection.
    It processes the method resolution order (MRO) to handle inheritance correctly.

    Attributes:
        _config_fields: Dictionary mapping field names to ConfigField descriptors.
    """

    def __new__(mcs, name, bases, namespace):
        """Creates a new Config class with collected field information.

        Args:
            name: Name of the class being created.
            bases: Base classes of the new class.
            namespace: Class namespace dictionary.

        Returns:
            The newly created class with _config_fields attribute populated.
        """
        cls = super().__new__(mcs, name, bases, namespace)

        # Collect all ConfigField descriptors from the class hierarchy
        cls._config_fields = {}  # type: ignore[misc]
        for base in reversed(cls.__mro__):
            if hasattr(base, "__dict__"):
                for field_name, field_obj in base.__dict__.items():
                    if isinstance(field_obj, ConfigField):
                        cls._config_fields[field_name] = field_obj  # type: ignore[misc]

        # Collect cross-constraints from reversed MRO (__dict__ only)
        cls._all_cross_constraints = collect_cross_constraints(cls)  # type: ignore[misc]

        return cls


class Config(metaclass=ConfigMeta):
    """Elegant configuration base class with declarative field definitions.

    This base class provides a powerful framework for creating type-safe,
    validated configuration classes with IDE-friendly attribute access.
    It supports automatic type inference, chainable configuration loading,
    and immutability through freezing.

    Features:
    - Declarative field definitions with explicit constraint validation
    - IDE-friendly attribute access with full autocomplete support
    - Chainable configuration loading from multiple sources
    - Immutability support through freezing
    - Dictionary-style access support
    - Comprehensive validation and error reporting

    Attributes:
        _config_fields: Dictionary of field names to ConfigField descriptors.
        _values: Internal storage for field values.
        _frozen: Boolean indicating whether the configuration is frozen.

    Examples:
        >>> class DatabaseConfig(Config):
        ...     host: str = ConfigField(default="localhost", doc="Database host")
        ...     port: int = ConfigField(default=5432, doc="Database port")
        ...     timeout: float = ConfigField(
        ...         default=30.0,
        ...         constraint=RangeConstraint(1.0, 300.0),
        ...         doc="Connection timeout in seconds"
        ...     )

        >>> config = DatabaseConfig()
        >>> config.from_file("db_config.yaml").from_env("DB_").freeze()
        >>> print(f"Connecting to {config.host}:{config.port}")
    """

    _config_fields: ClassVar[Dict[str, ConfigField]] = {}

    def __init__(self, **initial_values: Any):
        """Initializes a configuration instance with optional initial values.

        Args:
            **initial_values: Initial field values to set. Keys must match
                            defined configuration field names.

        Raises:
            KeyError: If any key in initial_values doesn't correspond to a
                     defined configuration field.
        """
        self._values: Dict[str, Any] = {}
        self._frozen = False

        # Initialize with default values and apply any provided initial values
        for name, field in self._config_fields.items():
            if name in initial_values:
                setattr(self, name, initial_values[name])
            else:
                default = field.default
                # Deep-copy mutable defaults so each instance owns its own copy; otherwise a
                # list/dict/set default (e.g. quarters, target_variants) is shared across all
                # instances and in-place mutation leaks between them.
                if isinstance(default, (list, dict, set)):
                    default = copy.deepcopy(default)
                self._values[name] = default

    def __setattr__(self, name: str, value: Any):
        """Sets attribute values with freeze protection and field validation.

        Args:
            name: Name of the attribute to set.
            value: Value to assign to the attribute.

        Raises:
            AttributeError: If attempting to modify a frozen configuration.
        """
        # Protect internal attributes
        if name.startswith("_") or name in ["_values", "_frozen"]:
            super().__setattr__(name, value)
            return

        # Check frozen state
        if hasattr(self, "_frozen") and self._frozen:
            raise AttributeError(f"Cannot modify frozen config field '{name}'")

        # Use field descriptor if available
        if name in self._config_fields:
            self._config_fields[name].__set__(self, value)
        else:
            super().__setattr__(name, value)

    def __getitem__(self, key: str) -> Any:
        """Provides dictionary-style access to configuration fields.

        Args:
            key: Name of the configuration field to retrieve.

        Returns:
            The value of the specified configuration field.

        Raises:
            KeyError: If the field name is not found in the configuration.
        """
        if key in self._config_fields:
            return getattr(self, key)
        raise KeyError(f"Field '{key}' not found")

    def __setitem__(self, key: str, value: Any):
        """Provides dictionary-style setting of configuration fields.

        Args:
            key: Name of the configuration field to set.
            value: Value to assign to the field.

        Raises:
            KeyError: If the field name is not found in the configuration.
        """
        if key in self._config_fields:
            setattr(self, key, value)
        else:
            raise KeyError(f"Field '{key}' not found")

    def __repr__(self) -> str:
        """Returns a string representation showing non-default field values.

        Returns:
            String representation in the format ClassName(field1=value1, field2=value2).
            Only shows fields that differ from their default values.
        """
        field_strs = []
        for name in self._config_fields:
            value = getattr(self, name)
            if value != self._config_fields[name].default:
                field_strs.append(f"{name}={value!r}")
        return f"{self.__class__.__name__}({', '.join(field_strs)})"

    # ========================================================================
    # 配置加载方法 Configuration Loading Methods - Chainable
    # ========================================================================

    def from_file(self, config_file: str) -> Self:
        """Loads configuration from a file (chainable).

        Supports JSON, YAML, and TOML formats. Only updates fields that exist
        in both the file and the configuration class.

        Args:
            config_file: Path to the configuration file.

        Returns:
            Self for method chaining.

        Raises:
            FileNotFoundError: If the configuration file doesn't exist.
            ValueError: If the file format is unsupported.

        Examples:
            >>> config = MyConfig().from_file("app.yaml").from_file("override.json")
        """
        configs = read_configs(config_file)
        return self.from_dict(configs)

    def from_dict(self, config_dict: Dict[str, Any]) -> Self:
        """Loads configuration from a dictionary (chainable).

        Only updates fields that exist in both the dictionary and the
        configuration class definition.

        Args:
            config_dict: Dictionary containing configuration values.

        Returns:
            Self for method chaining.

        Examples:
            >>> config_data = {"learning_rate": 0.001, "batch_size": 32}
            >>> config = MyConfig().from_dict(config_data)

        Note:
            Keys that do not match any field are ignored with a warning (likely typos —
            they would otherwise silently fall back to the default).
        """
        unknown = [k for k in config_dict if k not in self._config_fields]
        if unknown:
            warnings.warn(
                f"{type(self).__name__}.from_dict: ignoring unknown keys (possible typos): {unknown}",
                stacklevel=2,
            )
        for name in self._config_fields:
            if name in config_dict:
                setattr(self, name, config_dict[name])
        return self

    def from_kwargs(self, **kwargs: Any) -> Self:
        """Loads configuration from keyword arguments (chainable).

        Args:
            **kwargs: Configuration values as keyword arguments.

        Returns:
            Self for method chaining.

        Examples:
            >>> config = MyConfig().from_kwargs(learning_rate=0.001, epochs=100)
        """
        self.from_dict(kwargs)
        return self

    def from_args(self, args: Union[argparse.Namespace, argparse.ArgumentParser]) -> Self:
        """Loads configuration from command line arguments (chainable).

        Args:
            args: Either an ArgumentParser (will call parse_args()) or a
                 Namespace object from parse_args().

        Returns:
            Self for method chaining.

        Examples:
            >>> parser = argparse.ArgumentParser()
            >>> parser.add_argument('--learning-rate', type=float)
            >>> config = MyConfig().from_args(parser)
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
        """Loads configuration from environment variables (chainable).

        Environment variable names are constructed by combining the prefix
        with field names, converted to uppercase.

        Args:
            prefix: Prefix to prepend to field names when looking up environment
                   variables. Will be converted to uppercase.

        Returns:
            Self for method chaining.

        Examples:
            >>> # Looks for APP_LEARNING_RATE, APP_BATCH_SIZE, etc.
            >>> config = MyConfig().from_env("APP_")
        """
        for name in self._config_fields:
            env_name = f"{prefix}{name}".upper()
            if env_name in os.environ:
                setattr(self, name, os.environ[env_name])
        return self

    # ========================================================================
    # 实用方法 Utility Methods
    # ========================================================================

    def freeze(self) -> Self:
        """Freezes the configuration, making it immutable (chainable).

        After freezing, any attempt to modify field values will raise an
        AttributeError.

        Returns:
            Self for method chaining.

        Examples:
            >>> config = MyConfig().from_file("config.yaml").freeze()
            >>> config.learning_rate = 0.1  # Raises AttributeError
        """
        self._frozen = True
        return self

    def unfreeze(self) -> Self:
        """Unfreezes the configuration, allowing modifications (chainable).

        After unfreezing, field values can be modified again.

        Returns:
            Self for method chaining.
        """
        self._frozen = False
        return self

    def copy(self) -> Self:
        """Creates a copy of the configuration (never frozen, even if the original was).

        Mutable field values (list/dict/set) are deep-copied so the copy is independent of the
        original. Fields that are ``None`` on the original are left at the new instance's default
        — this skip is deliberate: setting a ``required`` field to ``None`` would trip
        required-validation, so copying a config with unset required fields must not do that.

        Returns:
            A new configuration instance with the same field values.

        Examples:
            >>> original = MyConfig().from_file("config.yaml").freeze()
            >>> editable_copy = original.copy()
            >>> editable_copy.learning_rate = 0.1  # Works fine
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
        """Converts the configuration to a dictionary.

        Returns:
            Dictionary containing all configuration field names and their
            current values.

        Examples:
            >>> config = MyConfig(learning_rate=0.001, epochs=100)
            >>> config_dict = config.to_dict()
            >>> print(config_dict)
            {'learning_rate': 0.001, 'epochs': 100, ...}
        """
        return {name: getattr(self, name) for name in self._config_fields}

    def to_file(self, save_file: str):
        """Saves the configuration to a file.

        The file format is determined by the file extension. Supports JSON, YAML, and TOML.

        Args:
            save_file: Path where the configuration should be saved.

        Examples:
            >>> config.to_file("saved_config.yaml")
            >>> config.to_file("backup_config.json")
        """
        save_configs(save_file, self.to_dict())

    def validate(self) -> Self:
        """Validates all configuration fields against their constraints.

        Performs comprehensive validation of all fields, checking both
        required field constraints and custom field constraints. Then
        runs cross-field constraints if all field constraints pass.

        Returns:
            Self for method chaining.

        Raises:
            ValueError: If any required field is None.
            ConstraintError: If any field value fails its constraint validation
                or if a cross-field constraint is violated.

        Examples:
            >>> config = MyConfig()
            >>> config.learning_rate = -0.1  # Invalid if constraint requires positive
            >>> config.validate()  # Raises ConstraintError
        """
        for name, field in self._config_fields.items():
            value = getattr(self, name)
            if field.required and value is None:
                raise ValueError(f"Required field '{name}' is None")
            if field.constraint and value is not None:
                field.constraint.check(name, value)

        # Cross-field validation (only if all field constraints passed)
        for cc in self._all_cross_constraints:
            cc.check(self)

        return self

    def get_field_info(self, field_name: str) -> Dict[str, Any]:
        """Retrieves comprehensive information about a configuration field.

        Args:
            field_name: Name of the field to get information about.

        Returns:
            Dictionary containing field metadata including name, default value,
            current value, constraints, documentation, and requirements.

        Raises:
            ValueError: If the field name is not found in the configuration.

        Examples:
            >>> info = config.get_field_info("learning_rate")
            >>> print(info["doc"])
            Learning rate for model training
            >>> print(info["constraint"])
            RangeConstraint
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
        """Retrieves the schema information for all fields in the configuration class.

        Returns:
            Dictionary mapping field names to their metadata, including defaults,
            constraints, documentation, and requirement status.

        Examples:
            >>> schema = MyConfig.get_schema()
            >>> for field_name, field_info in schema.items():
            ...     print(f"{field_name}: {field_info['doc']}")
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
