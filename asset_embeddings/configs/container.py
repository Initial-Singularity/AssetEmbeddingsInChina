import argparse
from abc import ABCMeta
from collections.abc import MutableMapping
from typing import Any, Dict, Union, Callable, Type, TypeVar, Optional, Self, get_origin

from .base import Config, read_configs, save_configs
from .constraints import CrossConstraint, collect_cross_constraints


T = TypeVar("T", bound="Config")


# ============================================================================
# 容器字段描述器 Container Field Descriptor
# ============================================================================


class ContainerFieldDescriptor:
    """Descriptor for typed container fields with auto-instantiation.

    This descriptor manages configuration fields in ConfigContainer subclasses,
    providing type checking, auto-instantiation, and factory function support.

    Attributes:
        name: Name of the field.
        field_type: Expected type of the field (Config or ConfigContainer subclass).
        default_factory: Optional callable to create default instances.
    """

    def __init__(self, name: str, field_type: Type, default_factory: Optional[Callable] = None):
        """Initialize the field descriptor.

        Args:
            name: Name of the field.
            field_type: Expected type for type checking.
            default_factory: Optional factory function for default values.
        """
        self.name = name
        self.field_type = field_type
        self.default_factory = default_factory
        self.private_name = f"_container_field_{name}"

    def __get__(self, obj, objtype=None):
        """Get the field value from the instance's _configs dict."""
        if obj is None:
            return self
        return obj._configs.get(self.name)

    def __set__(self, obj, value):
        """Set the field value with type checking."""
        if value is not None and not isinstance(value, (self.field_type, ConfigContainer, Config)):
            raise TypeError(f"Field '{self.name}' expects {self.field_type.__name__}, " f"got {type(value).__name__}")
        obj._configs[self.name] = value

    def __set_name__(self, owner, name):
        """Called when the descriptor is assigned to a class attribute."""
        self.name = name
        self.private_name = f"_container_field_{name}"


# ============================================================================
# 容器元类 Container Metaclass
# ============================================================================


class ConfigContainerMeta(ABCMeta):
    """Metaclass for ConfigContainer enabling type-annotated field declarations.

    This metaclass automatically processes type annotations in ConfigContainer
    subclasses, creating field descriptors for Config and ConfigContainer types.
    It enables IDE auto-completion and type checking while maintaining backward
    compatibility with dict-based initialization.

    Features:
    - Automatic field extraction from type annotations
    - Support for factory functions (callables as default values)
    - Type checking on field assignment
    - Inheritance support (merges parent and child annotations)
    - IDE auto-completion for typed fields

    Examples:
        >>> class MyContainer(ConfigContainer):
        ...     database: DatabaseConfig
        ...     api: ApiConfig = lambda: ApiConfig(port=8080)
        ...
        >>> container = MyContainer()
        >>> container.database  # ← IDE knows this is DatabaseConfig
    """

    def __new__(mcs, name, bases, namespace, **kwargs):
        """Create a new ConfigContainer class with field descriptors.

        Args:
            name: Name of the class being created.
            bases: Base classes.
            namespace: Class namespace dictionary.
            **kwargs: Additional keyword arguments.

        Returns:
            New class with field descriptors installed.
        """
        # Collect annotations from base classes
        all_annotations = {}
        for base in reversed(bases):
            if hasattr(base, "__annotations__"):
                all_annotations.update(base.__annotations__)

        # Add current class annotations
        if "__annotations__" in namespace:
            all_annotations.update(namespace["__annotations__"])

        # Store field definitions
        container_fields = {}

        # Process each annotated field
        for field_name, field_type in all_annotations.items():
            # Skip private fields
            if field_name.startswith("_"):
                continue

            # Get default value or factory
            default_value = namespace.get(field_name, None)

            # Determine if it's a Config/ConfigContainer type
            origin = get_origin(field_type)
            actual_type = origin if origin is not None else field_type

            # Check if it's a forward reference (string)
            if isinstance(field_type, str):
                # Forward reference - will be resolved later
                actual_type = field_type

            # Determine if default is a factory function
            default_factory = None
            if default_value is not None:
                # Check if it's a callable (factory) but not a class
                if callable(default_value) and not isinstance(default_value, type):
                    default_factory = default_value
                    # Remove from namespace to avoid issues
                    namespace.pop(field_name, None)
                elif isinstance(default_value, (Config, ConfigContainer)):
                    # It's an instance - we'll use a factory that returns a copy
                    def default_factory(v=default_value):
                        return v.copy() if hasattr(v, "copy") else v

                    namespace.pop(field_name, None)
            else:
                # No default - create auto-instantiation factory if it's a type
                if isinstance(actual_type, type) and issubclass(actual_type, (Config, ConfigContainer)):
                    default_factory = actual_type

            # Create field descriptor
            descriptor = ContainerFieldDescriptor(field_name, actual_type, default_factory)
            namespace[field_name] = descriptor

            # Store field metadata
            container_fields[field_name] = {"type": actual_type, "factory": default_factory}

        # Store field metadata in class
        namespace["_container_fields"] = container_fields

        # Create the class
        cls = super().__new__(mcs, name, bases, namespace, **kwargs)

        # Collect cross-constraints from reversed MRO (__dict__ only)
        cls._all_cross_constraints = collect_cross_constraints(cls)

        return cls


class ConfigContainer(MutableMapping, metaclass=ConfigContainerMeta):
    """Advanced configuration container supporting nesting, composition, and type safety.

    This container provides hierarchical organization of configuration objects with support for arbitrary nesting levels, type-safe access, and dynamic composition.
    It implements the MutableMapping protocol for dictionary-like operations while providing additional features for configuration management.

    Features:
    - Support for arbitrary levels of configuration nesting
    - Type-safe configuration access with generic type hints
    - Chainable operations for fluent API design
    - Dynamic configuration composition and merging
    - Namespace isolation for large applications
    - Nested path access with dot notation
    - Comprehensive validation across all contained configurations
    - Type-annotated field declarations with IDE auto-completion
    - Auto-instantiation of Config fields from type annotations

    Attributes:
        _configs: Internal dictionary storing configuration objects.
        _frozen: Boolean indicating whether the container is frozen.
        _namespace: Optional namespace string for organization.
        _container_fields: Metadata about type-annotated fields (class attribute).

    Examples:
        >>> # Traditional dict-based initialization (backward compatible)
        >>> container = ConfigContainer({
        ...     "database": DatabaseConfig(),
        ...     "api": ApiConfig()
        ... })

        >>> # New type-annotated subclass (IDE auto-completion)
        >>> class MyAppContainer(ConfigContainer):
        ...     database: DatabaseConfig
        ...     api: ApiConfig = lambda: ApiConfig(port=8080)
        ...
        >>> container = MyAppContainer()
        >>> container.database  # ← IDE knows type and provides auto-completion

        >>> # Load from file and access nested configurations
        >>> container.from_file("config.yaml")
        >>> db_config = container.get_config("database", DatabaseConfig)
        >>> learning_rate = container.get_nested("ml.training.learning_rate")
    """

    # Class attribute to store field metadata (populated by metaclass)
    _container_fields: Dict[str, Dict[str, Any]] = {}

    def __init__(self, configs: Optional[Dict[str, Union[Config, "ConfigContainer"]]] = None):
        """Initializes a configuration container with optional initial configurations.

        Supports both dict-based initialization (backward compatible) and auto-instantiation
        of type-annotated fields from the class definition.

        Args:
            configs: Optional dictionary mapping names to Config or ConfigContainer
                    objects. If None, auto-instantiates fields from type annotations.

        Examples:
            >>> # Dict-based (backward compatible)
            >>> container = ConfigContainer({"db": DatabaseConfig()})

            >>> # Auto-instantiation from annotations
            >>> class MyContainer(ConfigContainer):
            ...     database: DatabaseConfig
            >>> container = MyContainer()  # database auto-created
        """
        # Initialize internal state
        self._configs: Dict[str, Union[Config, "ConfigContainer"]] = {}
        self._frozen: bool = False
        self._namespace: str = ""
        self._instance_cross_constraints: list = []

        # Handle dict-based initialization (backward compatible)
        if configs is not None:
            self._configs.update(configs)

        # Auto-instantiate type-annotated fields not provided in configs
        if hasattr(self.__class__, "_container_fields"):
            for field_name, field_info in self.__class__._container_fields.items():
                # Only instantiate if not already provided
                if field_name not in self._configs:
                    factory = field_info.get("factory")
                    if factory is not None:
                        try:
                            # Call factory to create instance
                            instance = factory()
                        except Exception as e:
                            # Surface the real cause immediately. Silently leaving the
                            # field unset only resurfaces later as a cryptic
                            # "Config '<field>' not found" on first access (C-L5).
                            raise RuntimeError(
                                f"Failed to auto-instantiate container field "
                                f"'{field_name}' (type {field_info.get('type')!r}): {e}"
                            ) from e
                        self._configs[field_name] = instance

    # ========================================================================
    # 基础容器协议实现 Basic Container Protocol Implementation
    # ========================================================================

    def __getitem__(self, key: str) -> Union[Config, "ConfigContainer"]:
        """Gets a configuration object by name.

        Args:
            key: Name of the configuration to retrieve.

        Returns:
            The configuration object associated with the key.

        Raises:
            KeyError: If the key is not found in the container.
        """
        if key not in self._configs:
            raise KeyError(f"Config '{key}' not found")
        return self._configs[key]

    def __setitem__(self, key: str, value: Union[Config, "ConfigContainer"]):
        """Sets a configuration object in the container.

        Args:
            key: Name to associate with the configuration.
            value: Configuration object to store.

        Raises:
            ValueError: If attempting to modify a frozen container.
            TypeError: If the value is not a Config or ConfigContainer instance.
        """
        if self._frozen:
            raise ValueError("Cannot modify frozen ConfigContainer")
        if not isinstance(value, (Config, ConfigContainer)):
            raise TypeError(f"Value must be Config or ConfigContainer, got {type(value)}")
        self._configs[key] = value

    def __delitem__(self, key: str):
        """Removes a configuration object from the container.

        Args:
            key: Name of the configuration to remove.

        Raises:
            ValueError: If attempting to modify a frozen container.
            KeyError: If the key is not found in the container.
        """
        if self._frozen:
            raise ValueError("Cannot modify frozen ConfigContainer")
        del self._configs[key]

    def __iter__(self):
        """Returns an iterator over configuration names."""
        return iter(self._configs)

    def __len__(self):
        """Returns the number of configurations in the container."""
        return len(self._configs)

    def __contains__(self, key):
        """Checks if a configuration name exists in the container."""
        return key in self._configs

    # ========================================================================
    # 属性访问支持 Attribute Access Support
    # ========================================================================

    def __getattr__(self, name: str) -> Union[Config, "ConfigContainer"]:
        """Provides attribute-style access to configurations.

        Args:
            name: Name of the configuration to access.

        Returns:
            The configuration object associated with the name.

        Raises:
            AttributeError: If the configuration name is not found.
        """
        if name.startswith("_") or name in ["_configs", "_frozen", "_namespace"]:
            return super().__getattribute__(name)

        if name in self._configs:
            return self._configs[name]
        raise AttributeError(f"Config '{name}' not found")

    def __setattr__(self, name: str, value: Union[Config, "ConfigContainer"]):
        """Sets an attribute on the container.

        Args:
            name: Name of the attribute to set.
            value: Configuration object to assign.
        """
        if name.startswith("_") or name in ["_configs", "_frozen", "_namespace"]:
            super().__setattr__(name, value)
        else:
            self[name] = value

    def __repr__(self) -> str:
        """Returns a string representation of the container.

        Shows the number of configurations and their types for easy debugging.

        Returns:
            String representation showing container contents summary.
        """
        items = []
        for key, config in self._configs.items():
            if isinstance(config, Config):
                # Show count of non-default fields for Config objects
                non_default_count = len(
                    [
                        name
                        for name in config._config_fields
                        if getattr(config, name) != config._config_fields[name].default
                    ]
                )
                items.append(f"{key}={config.__class__.__name__}({non_default_count} configured)")
            else:
                items.append(f"{key}=ConfigContainer({len(config)} items)")

        return f"ConfigContainer({{{', '.join(items)}}})"

    # ========================================================================
    # 高级配置管理 Advanced Configuration Management
    # ========================================================================

    def add_cross_constraint(self, constraint: CrossConstraint) -> None:
        """Add a cross-constraint to this container instance.

        Args:
            constraint: CrossConstraint to add.
        """
        self._instance_cross_constraints.append(constraint)

    def remove_cross_constraint(self, name: str) -> bool:
        """Remove the first cross-constraint matching *name*.

        Args:
            name: Name of the constraint to remove.

        Returns:
            True if a constraint was found and removed, False otherwise.
        """
        for i, cc in enumerate(self._instance_cross_constraints):
            if cc.name == name:
                self._instance_cross_constraints.pop(i)
                return True
        return False

    def add_config(self, name: str, config: Union[Config, "ConfigContainer"]) -> Self:
        """Adds a configuration to the container (chainable).

        Args:
            name: Name to associate with the configuration.
            config: Configuration object to add.

        Returns:
            Self for method chaining.

        Examples:
            >>> container.add_config("database", DatabaseConfig())
            >>>          .add_config("api", ApiConfig())
        """
        self[name] = config
        return self

    def add_configs(self, **configs: Union[Config, "ConfigContainer"]) -> Self:
        """Adds multiple configurations at once (chainable).

        Args:
            **configs: Configuration objects as keyword arguments.

        Returns:
            Self for method chaining.
        """
        for name, config in configs.items():
            self[name] = config
        return self

    def remove_config(self, name: str) -> Self:
        """Removes a configuration from the container (chainable).

        Args:
            name: Name of the configuration to remove.

        Returns:
            Self for method chaining.

        Raises:
            KeyError: If the configuration name is not found.
        """
        del self[name]
        return self

    def get_config(self, name: str, config_type: Type[T] = None) -> T:
        """Retrieves a configuration with optional type checking.

        Provides type-safe access to configurations with runtime type validation.

        Args:
            name: Name of the configuration to retrieve.
            config_type: Expected type of the configuration for type safety.

        Returns:
            The configuration object, cast to the specified type if provided.

        Raises:
            KeyError: If the configuration name is not found.
            TypeError: If config_type is specified and the configuration doesn't match.
        """
        config = self[name]
        if config_type and not isinstance(config, config_type):
            raise TypeError(f"Expected {config_type.__name__}, got {type(config).__name__}")
        return config

    def has_config(self, name: str, config_type: Type[T] = None) -> bool:
        """Checks if a configuration exists and optionally matches a type.

        Args:
            name: Name of the configuration to check.
            config_type: Optional type to check against.

        Returns:
            True if the configuration exists (and matches the type if specified).
        """
        if name not in self:
            return False
        if config_type:
            return isinstance(self[name], config_type)
        return True

    # ========================================================================
    # 嵌套路径访问 Nested Path Access
    # ========================================================================

    def get_nested(self, path: str, separator: str = ".") -> Union[Config, "ConfigContainer", Any]:
        """Retrieves a value from nested configurations using dot notation.
        Supports accessing deeply nested configuration values using a simple path string with configurable separator.

        Args:
            path: Dot-separated path to the desired value (e.g., "ml.model.hidden_size").
            separator: String used to separate path components.

        Returns:
            The value at the specified path, which could be a Config, ConfigContainer,
            or any configuration field value.

        Raises:
            ValueError: If the path contains invalid segments.
            KeyError: If any segment in the path is not found.
            AttributeError: If trying to access a field that doesn't exist.
        """
        parts = path.split(separator)
        current = self

        for part in parts[:-1]:
            if not isinstance(current, ConfigContainer):
                raise ValueError(f"Cannot access '{part}' on non-container object")
            current = current[part]

        final_part = parts[-1]
        if isinstance(current, ConfigContainer):
            return current[final_part]
        elif isinstance(current, Config):
            return getattr(current, final_part)
        else:
            raise ValueError(f"Cannot access attribute on {type(current)}")

    def set_nested(self, path: str, value: Any, separator: str = ".") -> "ConfigContainer":
        """Sets a value in nested configurations using dot notation (chainable).

        Args:
            path: Dot-separated path to the value to set.
            value: Value to assign at the specified path.
            separator: String used to separate path components.

        Returns:
            Self for method chaining.

        Raises:
            ValueError: If the path contains invalid segments.
            KeyError: If any segment in the path is not found.
        """
        parts = path.split(separator)
        current = self

        for part in parts[:-1]:
            if not isinstance(current, ConfigContainer):
                raise ValueError(f"Cannot access '{part}' on non-container object")
            current = current[part]

        final_part = parts[-1]
        if isinstance(current, ConfigContainer):
            current[final_part] = value
        elif isinstance(current, Config):
            setattr(current, final_part, value)
        else:
            raise ValueError(f"Cannot set attribute on {type(current)}")

        return self

    def has_nested(self, path: str, separator: str = ".") -> bool:
        """Checks if a nested path exists in the configuration hierarchy.

        Args:
            path: Dot-separated path to check.
            separator: String used to separate path components.

        Returns:
            True if the path exists, False otherwise.
        """
        try:
            self.get_nested(path, separator)
            return True
        except (KeyError, AttributeError, ValueError):
            return False

    # ========================================================================
    # 配置加载和保存 Configuration Loading and Saving
    # ========================================================================

    def from_file(self, config_file: str, namespace: str = "") -> Self:
        """Loads configurations from a file with optional namespace support (chainable).

        Args:
            config_file: Path to the configuration file.
            namespace: Optional namespace prefix for configuration keys.

        Returns:
            Self for method chaining.
        """
        configs = read_configs(config_file)

        for name, config_obj in self._configs.items():
            config_key = f"{namespace}.{name}" if namespace else name
            # Both Config and ConfigContainer expose from_dict; polymorphic dispatch handles either.
            if config_key in configs:
                config_obj.from_dict(configs[config_key])
            elif name in configs:
                config_obj.from_dict(configs[name])

        return self

    def from_dict(self, config_dict: Dict[str, Any]) -> Self:
        """Loads configurations from a dictionary (chainable).

        Args:
            config_dict: Dictionary containing configuration data.

        Returns:
            Self for method chaining.
        """
        for name, config_obj in self._configs.items():
            if name in config_dict:
                config_obj.from_dict(config_dict[name])
        return self

    def from_args(self, args: Union[argparse.Namespace, argparse.ArgumentParser], prefix: str = "") -> Self:
        """Loads configurations from command line arguments (chainable).

        Uses prefixed argument names to route values to the appropriate configurations.

        Args:
            args: ArgumentParser or Namespace containing command line arguments.
            prefix: Prefix for filtering relevant arguments.

        Returns:
            Self for method chaining.
        """
        if isinstance(args, argparse.ArgumentParser):
            args = args.parse_args()

        args_dict = vars(args)

        for name, config_obj in self._configs.items():
            # Filter arguments by prefix
            filtered_args = {}
            config_prefix = f"{prefix}_{name}_" if prefix else f"{name}_"

            for arg_name, arg_value in args_dict.items():
                if arg_name.startswith(config_prefix):
                    # Remove prefix
                    clean_name = arg_name[len(config_prefix) :]
                    filtered_args[clean_name] = arg_value
                elif not prefix and arg_name in config_obj._config_fields:
                    # Direct matching when no prefix
                    filtered_args[arg_name] = arg_value

            if filtered_args:
                if isinstance(config_obj, ConfigContainer):
                    # Recursively handle nested containers
                    nested_namespace = argparse.Namespace(**filtered_args)
                    config_obj.from_args(nested_namespace, config_prefix.rstrip("_"))
                else:
                    config_obj.from_dict(filtered_args)

        return self

    def from_env(self, prefix: str = "") -> "ConfigContainer":
        """Loads configurations from environment variables (chainable).

        Args:
            prefix: Prefix for environment variable names.

        Returns:
            Self for method chaining.

        Examples:
            >>> # Environment: APP_DB_HOST=localhost, APP_API_PORT=8080
            >>> container.from_env("APP_")
        """
        for name, config_obj in self._configs.items():
            env_prefix = f"{prefix}_{name}_".upper() if prefix else f"{name}_".upper()
            # Both Config and ConfigContainer expose from_env; polymorphic dispatch handles either.
            config_obj.from_env(env_prefix.rstrip("_"))

        return self

    # ========================================================================
    # 配置组合和变换 Configuration Composition and Transformation
    # ========================================================================

    def merge(self, other: "ConfigContainer", strategy: str = "override") -> "ConfigContainer":
        """Merges another configuration container with conflict resolution strategies.

        Args:
            other: Another ConfigContainer to merge.
            strategy: Merge strategy - "override", "skip", or "error".
                     - "override": Replace duplicate configurations
                     - "skip": Keep existing configurations, ignore duplicates
                     - "error": Raise error on duplicate configuration names

        Returns:
            New ConfigContainer containing merged configurations.

        Raises:
            TypeError: If other is not a ConfigContainer.
            ValueError: If strategy is "error" and duplicate names are found,
                       or if strategy is not recognized.

        Examples:
            >>> base_config = ConfigContainer({"db": DatabaseConfig()})
            >>> override_config = ConfigContainer({"api": ApiConfig()})
            >>> merged = base_config.merge(override_config, strategy="override")
        """
        if not isinstance(other, ConfigContainer):
            raise TypeError("Can only merge with another ConfigContainer")

        result = self.copy()

        for name, config in other._configs.items():
            if name in result._configs:
                if strategy == "override":
                    result[name] = config.copy() if hasattr(config, "copy") else config
                elif strategy == "skip":
                    continue
                elif strategy == "error":
                    raise ValueError(f"Duplicate config name '{name}' found during merge")
                else:
                    raise ValueError(f"Unknown merge strategy: {strategy}")
            else:
                result[name] = config.copy() if hasattr(config, "copy") else config

        return result

    def subset(self, *config_names: str) -> "ConfigContainer":
        """Creates a new container with only the specified configurations.

        Args:
            *config_names: Names of configurations to include in the subset.

        Returns:
            New ConfigContainer containing only the specified configurations.

        Raises:
            KeyError: If any specified configuration name is not found.

        Examples:
            >>> subset = container.subset("database", "api")
            >>> # Contains only database and api configurations
        """
        subset_configs = {}
        for name in config_names:
            if name in self._configs:
                config = self._configs[name]
                subset_configs[name] = config.copy() if hasattr(config, "copy") else config
            else:
                raise KeyError(f"Config '{name}' not found")

        return ConfigContainer(subset_configs)

    def filter_by_type(self, config_type: Type) -> "ConfigContainer":
        """Creates a new container containing only configurations of the specified type.

        Args:
            config_type: Type to filter by.

        Returns:
            New ConfigContainer containing only configurations of the specified type.

        Examples:
            >>> model_configs = container.filter_by_type(ModelConfig)
            >>> # Contains only ModelConfig instances
        """
        filtered_configs = {name: config for name, config in self._configs.items() if isinstance(config, config_type)}
        return ConfigContainer(filtered_configs)

    # ========================================================================
    # 状态管理 State Management
    # ========================================================================

    def freeze(self) -> "ConfigContainer":
        """Freezes the container and all contained configurations (chainable).

        Returns:
            Self for method chaining.

        Examples:
            >>> container.from_file("config.yaml").freeze()
            >>> # Container and all configs are now immutable
        """
        self._frozen = True
        for config in self._configs.values():
            if hasattr(config, "freeze"):
                config.freeze()
        return self

    def copy(self, deep: bool = True) -> "ConfigContainer":
        """Creates a copy of the configuration container.

        Args:
            deep: If True, performs deep copying of all contained configurations.
                 If False, performs shallow copying (references to same config objects).

        Returns:
            New ConfigContainer with copied configurations.

        Examples:
            >>> # Deep copy for independent modifications
            >>> independent_copy = container.copy(deep=True)
            >>>
            >>> # Shallow copy for shared configurations
            >>> shared_copy = container.copy(deep=False)
        """
        if deep:
            copied_configs = {}
            for name, config in self._configs.items():
                if hasattr(config, "copy"):
                    copied_configs[name] = config.copy()
                else:
                    # For objects without copy method, try deep copy
                    import copy as copy_module

                    copied_configs[name] = copy_module.deepcopy(config)
        else:
            copied_configs = self._configs.copy()

        new_container = type(self)(copied_configs)
        new_container._namespace = self._namespace
        new_container._instance_cross_constraints = list(self._instance_cross_constraints)
        return new_container

    def validate(self) -> "ConfigContainer":
        """Validates all configurations in the container (chainable).

        Recursively validates child configs first. If all pass, runs
        container-level cross-constraints (class + instance).

        Returns:
            Self for method chaining.

        Raises:
            ValueError: If any child configuration fails validation.
            ConstraintError: If a container-level cross-constraint is violated.

        Examples:
            >>> container.from_file("config.yaml").validate().freeze()
        """
        for name, config in self._configs.items():
            if hasattr(config, "validate"):
                try:
                    config.validate()
                except Exception as e:
                    raise ValueError(f"Validation failed for config '{name}': {e}")

        # Container-level cross-constraints (class-level, then instance-level)
        for cc in self._all_cross_constraints:
            cc.check(self)
        for cc in self._instance_cross_constraints:
            cc.check(self)

        return self

    # ========================================================================
    # 信息导出 Information Export
    # ========================================================================

    def to_dict(self, flatten: bool = False, separator: str = ".") -> Dict[str, Any]:
        """Converts the container to a dictionary representation.

        Args:
            flatten: If True, flattens nested structures using dot notation.
            separator: Separator to use for flattened keys.

        Returns:
            Dictionary representation of all configurations.

        Examples:
            >>> # Nested structure
            >>> nested_dict = container.to_dict(flatten=False)
            >>> print(nested_dict["ml"]["model"]["hidden_size"])

            >>> # Flattened structure
            >>> flat_dict = container.to_dict(flatten=True)
            >>> print(flat_dict["ml.model.hidden_size"])
        """
        result = {}

        for name, config in self._configs.items():
            if hasattr(config, "to_dict"):
                config_dict = config.to_dict()
            elif isinstance(config, ConfigContainer):
                config_dict = config.to_dict(flatten=False)
            else:
                config_dict = config

            if flatten and isinstance(config_dict, dict):
                # Flatten nested dictionaries
                for key, value in config_dict.items():
                    flattened_key = f"{name}{separator}{key}"
                    result[flattened_key] = value
            else:
                result[name] = config_dict

        return result

    def get_summary(self) -> Dict[str, Any]:
        """Generates a comprehensive summary of the container contents.

        Returns:
            Dictionary containing statistics about configurations, types,
            nesting levels, and total field counts.

        Examples:
            >>> summary = container.get_summary()
            >>> print(f"Total configs: {summary['total_configs']}")
            >>> print(f"Config types: {summary['config_types']}")
        """
        summary = {"total_configs": len(self._configs), "config_types": {}, "nested_containers": 0, "total_fields": 0}

        for name, config in self._configs.items():
            config_type = type(config).__name__
            if config_type not in summary["config_types"]:
                summary["config_types"][config_type] = 0
            summary["config_types"][config_type] += 1

            if isinstance(config, ConfigContainer):
                summary["nested_containers"] += 1
                nested_summary = config.get_summary()
                summary["total_fields"] += nested_summary["total_fields"]
            elif hasattr(config, "_config_fields"):
                summary["total_fields"] += len(config._config_fields)

        return summary

    def to_file(self, save_file: str):
        """Saves all configurations to a file.

        Args:
            save_file: Path where the configuration file should be saved.

        Examples:
            >>> container.to_file("full_config.yaml")
        """
        config_dict = self.to_dict()
        save_configs(save_file, config_dict)

    # ========================================================================
    # 特殊方法和工具 Special Methods and Utilities
    # ========================================================================

    def keys(self):
        """Returns an iterator over configuration names."""
        return self._configs.keys()

    def values(self):
        """Returns an iterator over configuration objects."""
        return self._configs.values()

    def items(self):
        """Returns an iterator over (name, config) pairs."""
        return self._configs.items()


# ============================================================================
# 专门的配置工厂和构建器 (Specialized Configuration Factory and Builders)
# ============================================================================


class ConfigBuilder:
    """Fluent configuration builder for complex configuration hierarchies.

    This builder provides a fluent API for constructing complex configuration
    containers with nested structures. It supports method chaining and
    functional composition patterns for readable configuration setup.

    Attributes:
        _container: The ConfigContainer being built.

    Examples:
        >>> builder = ConfigBuilder()
        >>> container = (builder
        ...     .add("database", DatabaseConfig())
        ...     .add_from_class("api", ApiConfig, port=8080)
        ...     .nest("ml", lambda b: b
        ...         .add("model", ModelConfig())
        ...         .add("training", TrainingConfig())
        ...     )
        ...     .build()
        ... )
    """

    def __init__(self):
        """Initializes an empty configuration builder."""
        self._container = ConfigContainer()

    def add(self, name: str, config: Union[Config, ConfigContainer]) -> "ConfigBuilder":
        """Adds a configuration object to the builder (chainable).

        Args:
            name: Name to associate with the configuration.
            config: Configuration object to add.

        Returns:
            Self for method chaining.

        Examples:
            >>> builder.add("database", DatabaseConfig())
        """
        self._container[name] = config
        return self

    def add_from_class(self, name: str, config_class: Type[Config], **kwargs) -> "ConfigBuilder":
        """Creates and adds a configuration from a class with initial values (chainable).

        Args:
            name: Name to associate with the configuration.
            config_class: Configuration class to instantiate.
            **kwargs: Initial values to pass to the configuration constructor.

        Returns:
            Self for method chaining.

        Examples:
            >>> builder.add_from_class("api", ApiConfig, port=8080, debug=True)
        """
        config = config_class(**kwargs)
        self._container[name] = config
        return self

    def nest(self, name: str, builder_func: Callable[["ConfigBuilder"], "ConfigBuilder"]) -> "ConfigBuilder":
        """Creates a nested configuration container using a builder function (chainable).

        Args:
            name: Name to associate with the nested container.
            builder_func: Function that takes a ConfigBuilder and returns a ConfigBuilder
                         with the desired nested configuration structure.

        Returns:
            Self for method chaining.

        Examples:
            >>> builder.nest("ml", lambda b: (
            ...     b.add("model", ModelConfig())
            ...      .add("training", TrainingConfig())
            ... ))
        """
        nested_builder = ConfigBuilder()
        nested_builder = builder_func(nested_builder)
        self._container[name] = nested_builder.build()
        return self

    def build(self) -> ConfigContainer:
        """Builds and returns the final configuration container.

        Returns:
            A copy of the constructed ConfigContainer.

        Examples:
            >>> container = builder.build()
        """
        return self._container.copy()
