"""Advanced configuration management framework with type safety and validation.

This module provides a comprehensive configuration management system built on top of the constraint validation framework. It supports declarative field definitions, explicit constraint validation, flexible data loading from multiple sources, and hierarchical configuration organization through containers.

Key features:
- Declarative configuration field definitions with automatic validation
- Type-safe configuration access with IDE support
- Chainable configuration loading from files, environment variables, and command line
- Hierarchical configuration containers with nested path access
- Immutable configuration support with freezing capabilities
- Multiple config file format support (JSON, YAML, TOML)
- Integration with argparse for command-line configuration
"""

from . import constraints, base, container, object_configs, runtime_configs
from .constraints import (
    BaseConstraint,
    Constraint,
    ConstraintError,
    LambdaConstraint,
    TypeConstraint,
    EqualityConstraint,
    RangeConstraint,
    LengthConstraint,
    ChoiceConstraint,
    ChoicesConstraint,
    IntersectionConstraints,
    UnionConstraints,
    NoConstraint,
    OptionalConstraint,
    CrossConstraint,
    LambdaCrossConstraint,
    MutualExclusionConstraint,
    MethodCrossConstraint,
    CrossIntersectionConstraints,
    CrossUnionConstraints,
    cross_validate,
)
from .base import Config, ConfigField
from .container import ConfigContainer
from .object_configs import (
    LoggerConfig,
    TokenizerConfig,
    OptimizerConfig,
    DatasetConfig,
    DataLoaderConfig,
    AcceleratorConfig,
    AssetBERTModelConfig,
)
from .runtime_configs import (
    AssetRSTrainConfig,
    AssetW2VTrainConfig,
    AssetBERTTrainConfig,
)

__all__ = [
    "constraints",
    "BaseConstraint",
    "Constraint",
    "ConstraintError",
    "LambdaConstraint",
    "TypeConstraint",
    "EqualityConstraint",
    "RangeConstraint",
    "LengthConstraint",
    "ChoiceConstraint",
    "ChoicesConstraint",
    "IntersectionConstraints",
    "UnionConstraints",
    "NoConstraint",
    "OptionalConstraint",
    "CrossConstraint",
    "LambdaCrossConstraint",
    "MutualExclusionConstraint",
    "MethodCrossConstraint",
    "CrossIntersectionConstraints",
    "CrossUnionConstraints",
    "cross_validate",
    "base",
    "container",
    "Config",
    "ConfigField",
    "ConfigContainer",
    "object_configs",
    "LoggerConfig",
    "TokenizerConfig",
    "OptimizerConfig",
    "DatasetConfig",
    "DataLoaderConfig",
    "AcceleratorConfig",
    "AssetBERTModelConfig",
    "runtime_configs",
    "AssetRSTrainConfig",
    "AssetW2VTrainConfig",
    "AssetBERTTrainConfig",
]
