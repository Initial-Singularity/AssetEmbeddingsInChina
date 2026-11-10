"""`ConfigContainer`, a mapping of named `Config` objects (nestable), and `ConfigBuilder`.

A container holds child configs by name, reachable as items or attributes and through dotted
paths; a typed subclass declares its children as annotations and the metaclass turns each into a
descriptor with an auto-instantiation factory. Loading, freezing, copying and validation forward to
every child, and container-level cross-field constraints run after the children validate.
"""

import argparse
from abc import ABCMeta
from collections.abc import MutableMapping
from typing import Any, Dict, Union, Callable, Type, TypeVar, Optional, Self, get_origin

from .base import Config, read_configs, save_configs
from .constraints import CrossConstraint, collect_cross_constraints


T = TypeVar("T", bound="Config")


class ContainerFieldDescriptor:
    """Descriptor the metaclass installs for each annotated child of a `ConfigContainer` subclass.

    Reads come from the instance's `_configs` dict, so the descriptor and the mapping interface
    always agree.

    Attributes:
        name: The child's name.
        field_type: The annotated type.
        default_factory: Callable that builds the child when none is supplied, or None.
    """

    def __init__(self, name: str, field_type: Type, default_factory: Optional[Callable] = None):
        self.name = name
        self.field_type = field_type
        self.default_factory = default_factory
        self.private_name = f"_container_field_{name}"

    def __get__(self, obj, objtype=None):
        if obj is None:
            return self
        return obj._configs.get(self.name)

    def __set__(self, obj, value):
        if value is not None and not isinstance(value, (self.field_type, ConfigContainer, Config)):
            raise TypeError(f"Field '{self.name}' expects {self.field_type.__name__}, " f"got {type(value).__name__}")
        obj._configs[self.name] = value

    def __set_name__(self, owner, name):
        self.name = name
        self.private_name = f"_container_field_{name}"


class ConfigContainerMeta(ABCMeta):
    """Metaclass that turns the annotations of a `ConfigContainer` subclass into child fields.

    Each public annotation becomes a `ContainerFieldDescriptor`; a callable class attribute is
    its factory, a `Config`/`ConfigContainer` instance is copied per container, and an annotated
    class with no default is instantiated as-is. The class gets `_container_fields` (name to
    type and factory) and `_all_cross_constraints`.
    """

    def __new__(mcs, name, bases, namespace, **kwargs):
        """Create the class with its child descriptors installed.

        Args:
            name: Name of the class being created.
            bases: Its base classes.
            namespace: Its class body.
            **kwargs: Passed through to `ABCMeta.__new__`.

        Returns:
            The new class.
        """
        # Immediate bases first, so the class's own annotations override inherited ones.
        all_annotations = {}
        for base in reversed(bases):
            if hasattr(base, "__annotations__"):
                all_annotations.update(base.__annotations__)

        if "__annotations__" in namespace:
            all_annotations.update(namespace["__annotations__"])

        container_fields = {}

        for field_name, field_type in all_annotations.items():
            # Underscore names are the container's own state (_configs, _frozen, ...), not children.
            if field_name.startswith("_"):
                continue

            default_value = namespace.get(field_name, None)

            origin = get_origin(field_type)
            actual_type = origin if origin is not None else field_type

            # A string annotation (forward reference) stays a string; nothing resolves it later, so
            # such a field gets no auto-instantiation factory.
            if isinstance(field_type, str):
                actual_type = field_type

            default_factory = None
            if default_value is not None:
                if callable(default_value) and not isinstance(default_value, type):
                    default_factory = default_value
                    namespace.pop(field_name, None)
                elif isinstance(default_value, (Config, ConfigContainer)):
                    # Copy per container, so instances never share one default child.
                    def default_factory(v=default_value):
                        return v.copy() if hasattr(v, "copy") else v

                    namespace.pop(field_name, None)
            else:
                if isinstance(actual_type, type) and issubclass(actual_type, (Config, ConfigContainer)):
                    default_factory = actual_type

            # Installed in the namespace before the class is built so type.__new__ calls __set_name__.
            descriptor = ContainerFieldDescriptor(field_name, actual_type, default_factory)
            namespace[field_name] = descriptor

            container_fields[field_name] = {"type": actual_type, "factory": default_factory}

        namespace["_container_fields"] = container_fields

        cls = super().__new__(mcs, name, bases, namespace, **kwargs)

        cls._all_cross_constraints = collect_cross_constraints(cls)

        return cls


class ConfigContainer(MutableMapping, metaclass=ConfigContainerMeta):
    """A mutable mapping of named `Config` objects and nested containers.

    Children are reachable as items, as attributes and through dotted paths (`get_nested`). A
    subclass can declare its children as type annotations, which are created on construction
    unless supplied. Loading methods forward to each child and return `self`; `validate()` checks
    the children first and then the container's own cross-field constraints, class-level and
    instance-level.
    """

    _container_fields: Dict[str, Dict[str, Any]] = {}

    def __init__(self, configs: Optional[Dict[str, Union[Config, "ConfigContainer"]]] = None):
        """Create a container from a dict of children plus the annotated children not in it.

        Args:
            configs: Children by name; entries here take precedence over the annotated
                defaults.

        Raises:
            RuntimeError: If an annotated child's factory fails.
        """
        self._configs: Dict[str, Union[Config, "ConfigContainer"]] = {}
        self._frozen: bool = False
        self._namespace: str = ""
        self._instance_cross_constraints: list = []

        if configs is not None:
            self._configs.update(configs)

        if hasattr(self.__class__, "_container_fields"):
            for field_name, field_info in self.__class__._container_fields.items():
                if field_name not in self._configs:
                    factory = field_info.get("factory")
                    if factory is not None:
                        try:
                            instance = factory()
                        except Exception as e:
                            # Leaving the child unset would only resurface later as a cryptic
                            # "Config '<field>' not found" on first access.
                            raise RuntimeError(
                                f"Failed to auto-instantiate container field "
                                f"'{field_name}' (type {field_info.get('type')!r}): {e}"
                            ) from e
                        self._configs[field_name] = instance

    def __getitem__(self, key: str) -> Union[Config, "ConfigContainer"]:
        if key not in self._configs:
            raise KeyError(f"Config '{key}' not found")
        return self._configs[key]

    def __setitem__(self, key: str, value: Union[Config, "ConfigContainer"]):
        if self._frozen:
            raise ValueError("Cannot modify frozen ConfigContainer")
        if not isinstance(value, (Config, ConfigContainer)):
            raise TypeError(f"Value must be Config or ConfigContainer, got {type(value)}")
        self._configs[key] = value

    def __delitem__(self, key: str):
        if self._frozen:
            raise ValueError("Cannot modify frozen ConfigContainer")
        del self._configs[key]

    def __iter__(self):
        return iter(self._configs)

    def __len__(self):
        return len(self._configs)

    def __contains__(self, key):
        return key in self._configs

    def __getattr__(self, name: str) -> Union[Config, "ConfigContainer"]:
        # Private names take normal lookup so a missing one raises a plain AttributeError; that
        # also keeps _configs itself from being looked up inside _configs before __init__ sets it.
        if name.startswith("_") or name in ["_configs", "_frozen", "_namespace"]:
            return super().__getattribute__(name)

        if name in self._configs:
            return self._configs[name]
        raise AttributeError(f"Config '{name}' not found")

    def __setattr__(self, name: str, value: Union[Config, "ConfigContainer"]):
        # Public names route through __setitem__ so attribute assignment gets the same freeze
        # and type checks as item assignment.
        if name.startswith("_") or name in ["_configs", "_frozen", "_namespace"]:
            super().__setattr__(name, value)
        else:
            self[name] = value

    def __repr__(self) -> str:
        items = []
        for key, config in self._configs.items():
            if isinstance(config, Config):
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

    def add_cross_constraint(self, constraint: CrossConstraint) -> None:
        """Attach a cross-field constraint to this instance only; `validate()` runs it after the class-level ones.

        Args:
            constraint: The constraint to attach.
        """
        self._instance_cross_constraints.append(constraint)

    def remove_cross_constraint(self, name: str) -> bool:
        """Detach the first instance-level cross-field constraint with the given name.

        Args:
            name: The constraint's `name`.

        Returns:
            True if one was removed, False if none matched.
        """
        for i, cc in enumerate(self._instance_cross_constraints):
            if cc.name == name:
                self._instance_cross_constraints.pop(i)
                return True
        return False

    def add_config(self, name: str, config: Union[Config, "ConfigContainer"]) -> Self:
        """Add or replace one child (chainable).

        Args:
            name: The child's name.
            config: The child.

        Returns:
            This container.

        Raises:
            ValueError: If the container is frozen.
            TypeError: If the child is neither a `Config` nor a `ConfigContainer`.
        """
        self[name] = config
        return self

    def add_configs(self, **configs: Union[Config, "ConfigContainer"]) -> Self:
        """Add or replace several children (chainable).

        Args:
            **configs: Children by name.

        Returns:
            This container.
        """
        for name, config in configs.items():
            self[name] = config
        return self

    def remove_config(self, name: str) -> Self:
        """Remove one child (chainable).

        Args:
            name: The child's name.

        Returns:
            This container.

        Raises:
            ValueError: If the container is frozen.
            KeyError: If no child has that name.
        """
        del self[name]
        return self

    def get_config(self, name: str, config_type: Type[T] = None) -> T:
        """Return one child, optionally asserting its type.

        Args:
            name: The child's name.
            config_type: If given, the type the child must be an instance of.

        Returns:
            The child.

        Raises:
            KeyError: If no child has that name.
            TypeError: If `config_type` is given and the child is not an instance of it.
        """
        config = self[name]
        if config_type and not isinstance(config, config_type):
            raise TypeError(f"Expected {config_type.__name__}, got {type(config).__name__}")
        return config

    def has_config(self, name: str, config_type: Type[T] = None) -> bool:
        """Return whether a child of that name exists, and is of the given type if one is given.

        Args:
            name: The child's name.
            config_type: If given, the type the child must be an instance of.

        Returns:
            True if the child exists and matches.
        """
        if name not in self:
            return False
        if config_type:
            return isinstance(self[name], config_type)
        return True

    def get_nested(self, path: str, separator: str = ".") -> Union[Config, "ConfigContainer", Any]:
        """Return the child or field value at a dotted path.

        Every segment but the last must name a child container; the last names a child of that
        container or a field of that config.

        Args:
            path: Segments joined by the separator, such as `"ml.model.hidden_size"`.
            separator: The string that splits the path into segments.

        Returns:
            The child or field value at the path.

        Raises:
            ValueError: If an intermediate segment lands on something that is not a container.
            KeyError: If a segment names no child.
            AttributeError: If the last segment names no field of the config it lands on.
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
        """Assign the child or field value at a dotted path (chainable).

        Args:
            path: Segments joined by the separator; resolved as in `get_nested`.
            value: The child or field value to assign.
            separator: The string that splits the path into segments.

        Returns:
            This container.

        Raises:
            ValueError: If an intermediate segment lands on something that is not a container.
            KeyError: If an intermediate segment names no child.
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
        """Return whether `get_nested` would resolve the path.

        Args:
            path: Segments joined by the separator.
            separator: The string that splits the path into segments.

        Returns:
            True if the path resolves.
        """
        try:
            self.get_nested(path, separator)
            return True
        except (KeyError, AttributeError, ValueError):
            return False

    def from_file(self, config_file: str, namespace: str = "") -> Self:
        """Load each child from its section of a JSON, YAML or TOML file (chainable).

        A child named `db` reads the top-level key `db`, or `<namespace>.db` first when a
        namespace is given. Keys naming no child are ignored.

        Args:
            config_file: Path to the file.
            namespace: Prefix that selects the namespaced keys.

        Returns:
            This container.
        """
        configs = read_configs(config_file)

        for name, config_obj in self._configs.items():
            config_key = f"{namespace}.{name}" if namespace else name
            if config_key in configs:
                config_obj.from_dict(configs[config_key])
            elif name in configs:
                config_obj.from_dict(configs[name])

        return self

    def from_dict(self, config_dict: Dict[str, Any]) -> Self:
        """Load each child from the sub-dict under its name (chainable).

        Args:
            config_dict: Child dicts by child name.

        Returns:
            This container.

        Raises:
            KeyError: When a key names no child (a child refuses its own unknown keys).
        """
        unknown = [k for k in config_dict if k not in self._configs]
        if unknown:
            raise KeyError(f"{type(self).__name__} has no section(s) {unknown}")
        for name, config_obj in self._configs.items():
            if name in config_dict:
                config_obj.from_dict(config_dict[name])
        return self

    def from_args(self, args: Union[argparse.Namespace, argparse.ArgumentParser], prefix: str = "") -> Self:
        """Route command-line arguments to the children by name prefix (chainable).

        An argument `<prefix>_<child>_<field>` (or `<child>_<field>` with no prefix) reaches
        `<child>` as `<field>`; with no prefix an argument named exactly like one of a child
        config's fields reaches it too. A nested container recurses with its own prefix.

        Args:
            args: A parsed namespace, or a parser on which `parse_args()` is called.
            prefix: Leading segment of the argument names.

        Returns:
            This container.
        """
        if isinstance(args, argparse.ArgumentParser):
            args = args.parse_args()

        args_dict = vars(args)

        for name, config_obj in self._configs.items():
            filtered_args = {}
            config_prefix = f"{prefix}_{name}_" if prefix else f"{name}_"

            for arg_name, arg_value in args_dict.items():
                if arg_name.startswith(config_prefix):
                    clean_name = arg_name[len(config_prefix) :]
                    filtered_args[clean_name] = arg_value
                elif not prefix and arg_name in config_obj._config_fields:
                    filtered_args[arg_name] = arg_value

            if filtered_args:
                if isinstance(config_obj, ConfigContainer):
                    nested_namespace = argparse.Namespace(**filtered_args)
                    config_obj.from_args(nested_namespace, config_prefix.rstrip("_"))
                else:
                    config_obj.from_dict(filtered_args)

        return self

    def from_env(self, prefix: str = "") -> "ConfigContainer":
        """Load each child from environment variables under its own prefix (chainable).

        A child named `db` loads with the prefix `<prefix>_db` (upper-cased, trailing underscores
        stripped), which its `from_env` then completes with the field name.

        Args:
            prefix: Leading segment of the variable names.

        Returns:
            This container.
        """
        for name, config_obj in self._configs.items():
            env_prefix = f"{prefix}_{name}_".upper() if prefix else f"{name}_".upper()
            config_obj.from_env(env_prefix.rstrip("_"))

        return self

    def merge(self, other: "ConfigContainer", strategy: str = "override") -> "ConfigContainer":
        """Return a copy of this container with the other container's children added.

        Args:
            other: The container whose children are added.
            strategy: What to do with a name present in both: `"override"` takes the other's
                child, `"skip"` keeps this container's, `"error"` raises.

        Returns:
            The merged copy; neither input is modified.

        Raises:
            TypeError: If `other` is not a `ConfigContainer`.
            ValueError: If the strategy is `"error"` and a name clashes, or the strategy is unknown.
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
        """Return a plain `ConfigContainer` holding copies of the named children.

        Args:
            *config_names: Names of the children to keep.

        Returns:
            The new container.

        Raises:
            KeyError: If a name matches no child.
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
        """Return a plain `ConfigContainer` holding the children that are instances of a type.

        The children are shared with this container, not copied.

        Args:
            config_type: The type to keep.

        Returns:
            The new container.
        """
        filtered_configs = {name: config for name, config in self._configs.items() if isinstance(config, config_type)}
        return ConfigContainer(filtered_configs)

    def freeze(self) -> "ConfigContainer":
        """Freeze this container and every child; adding or removing children then raises (chainable).

        Returns:
            This container.
        """
        self._frozen = True
        for config in self._configs.values():
            if hasattr(config, "freeze"):
                config.freeze()
        return self

    def copy(self, deep: bool = True) -> "ConfigContainer":
        """Return an unfrozen container of the same class with the same children.

        Instance-level cross-field constraints are carried over.

        Args:
            deep: If True, each child is copied; if False, the children are shared.

        Returns:
            The new container.
        """
        if deep:
            copied_configs = {}
            for name, config in self._configs.items():
                if hasattr(config, "copy"):
                    copied_configs[name] = config.copy()
                else:
                    import copy as copy_module

                    copied_configs[name] = copy_module.deepcopy(config)
        else:
            copied_configs = self._configs.copy()

        new_container = type(self)(copied_configs)
        new_container._namespace = self._namespace
        new_container._instance_cross_constraints = list(self._instance_cross_constraints)
        return new_container

    def validate(self) -> "ConfigContainer":
        """Validate every child, then the container's cross-field constraints (chainable).

        Class-level constraints run before instance-level ones; the first failure raises.

        Returns:
            This container.

        Raises:
            ValueError: If a child fails validation; the message names the child.
            ConstraintError: If a container-level cross-field invariant fails.
        """
        for name, config in self._configs.items():
            if hasattr(config, "validate"):
                try:
                    config.validate()
                except Exception as e:
                    raise ValueError(f"Validation failed for config '{name}': {e}")

        for cc in self._all_cross_constraints:
            cc.check(self)
        for cc in self._instance_cross_constraints:
            cc.check(self)

        return self

    def to_dict(self, flatten: bool = False, separator: str = ".") -> Dict[str, Any]:
        """Return the children's field values as nested dicts, or as one level of joined keys.

        With `flatten`, each child's name and field name are joined into one key; a nested
        container's own dict is not flattened further.

        Args:
            flatten: Whether to join child and field names.
            separator: The string that joins them.

        Returns:
            The values, by child name or by joined key.
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
                for key, value in config_dict.items():
                    flattened_key = f"{name}{separator}{key}"
                    result[flattened_key] = value
            else:
                result[name] = config_dict

        return result

    def get_summary(self) -> Dict[str, Any]:
        """Count the children, their classes, the nested containers and the fields, recursively.

        Returns:
            Keys `total_configs`, `config_types` (count by class name), `nested_containers` and
            `total_fields`.
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

    def to_file(self, save_file: str) -> None:
        """Write the nested `to_dict()` to a JSON, YAML or TOML file, chosen by extension.

        Args:
            save_file: Destination path.
        """
        config_dict = self.to_dict()
        save_configs(save_file, config_dict)

    def keys(self):
        """Return a view of the child names."""
        return self._configs.keys()

    def values(self):
        """Return a view of the children."""
        return self._configs.values()

    def items(self):
        """Return a view of the (name, child) pairs."""
        return self._configs.items()


class ConfigBuilder:
    """Fluent builder that assembles a `ConfigContainer`, nested containers included.

    `build()` returns a copy, so the builder can go on being used afterwards.
    """

    def __init__(self):
        self._container = ConfigContainer()

    def add(self, name: str, config: Union[Config, ConfigContainer]) -> "ConfigBuilder":
        """Add a child (chainable).

        Args:
            name: The child's name.
            config: The child.

        Returns:
            This builder.
        """
        self._container[name] = config
        return self

    def add_from_class(self, name: str, config_class: Type[Config], **kwargs) -> "ConfigBuilder":
        """Instantiate a config class and add it as a child (chainable).

        Args:
            name: The child's name.
            config_class: The class to instantiate.
            **kwargs: Field values passed to the class.

        Returns:
            This builder.
        """
        config = config_class(**kwargs)
        self._container[name] = config
        return self

    def nest(self, name: str, builder_func: Callable[["ConfigBuilder"], "ConfigBuilder"]) -> "ConfigBuilder":
        """Add a nested container assembled by a function on a fresh builder (chainable).

        Args:
            name: The nested container's name.
            builder_func: Receives a new builder and returns the builder to build from.

        Returns:
            This builder.
        """
        nested_builder = ConfigBuilder()
        nested_builder = builder_func(nested_builder)
        self._container[name] = nested_builder.build()
        return self

    def build(self) -> ConfigContainer:
        """Return a copy of the assembled container."""
        return self._container.copy()
