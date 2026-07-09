# Configuration system

This document describes the AssetEmbeddings configuration system: the `Config` base class, the
`ConfigField` descriptor, and the constraint-validation framework.

## Contents

- [Overview](#overview)
- [The ConfigField descriptor](#the-configfield-descriptor)
- [The Config base class](#the-config-base-class)
- [Field constraints](#field-constraints)
- [Cross-field constraints](#cross-field-constraints)
- [Declaring constraints](#declaring-constraints)
- [Loading and saving](#loading-and-saving)
- [Freezing and copying](#freezing-and-copying)
- [Extending the system](#extending-the-system)

---

## Overview

The configuration system lives under `asset_embeddings/configs/` and provides:

- **Declarative field definitions** — define fields with the `ConfigField` descriptor.
- **Explicit constraint validation** — declare a field's constraint with `constraint=` (defaults to `NoConstraint()`, i.e. no validation).
- **Chained loading** — load from files, dicts, command-line arguments, and environment variables.
- **Field constraints** — a rich set of single-field constraint types that compose and extend.
- **Cross-field constraints** — declarative invariants across fields (`_cross_constraints` / `@cross_validate`), all evaluated in `validate()`.
- **Immutability** — freeze a config with `freeze()`.

### File layout

```
asset_embeddings/configs/
├── base.py            # Config base class and ConfigField descriptor
├── constraints.py     # constraint-validation framework
├── container.py       # ConfigContainer hierarchical config container
├── object_configs.py  # model / logger / tokenizer config classes
├── runtime_configs.py # training runtime config classes
└── __init__.py        # module exports
```

---

## The ConfigField descriptor

`ConfigField` is a Python descriptor for declaratively defining configuration fields.

### Basic usage

```python
from asset_embeddings.configs import Config, ConfigField
from asset_embeddings.configs.constraints import RangeConstraint

class TrainingConfig(Config):
    learning_rate: float = ConfigField(
        default=0.001,
        constraint=RangeConstraint(1e-6, 1.0),
        doc="Learning rate"
    )
    batch_size: int = ConfigField(
        default=32,
        required=True,
        doc="Batch size"
    )
    model_name: str = ConfigField(
        default="bert-base",
        preprocessor=lambda x: x.lower().strip(),
        doc="Model name (lower-cased automatically)"
    )
```

### ConfigField parameters

| Parameter | Type | Default | Description |
|------|------|--------|------|
| `default` | `Any` | `None` | Default value for the field |
| `constraint` | `Constraint` | `NoConstraint()` | Validation constraint |
| `preprocessor` | `Callable` | `lambda x: x` | Value-preprocessing function |
| `doc` | `str` | `""` | Field documentation |
| `required` | `bool` | `False` | Whether the field is required |

### Preprocessor examples

```python
# Path normalization
path: str = ConfigField(
    preprocessor=lambda x: os.path.normpath(x) if x else x,
    doc="File path"
)

# List conversion
tags: list = ConfigField(
    preprocessor=lambda x: x.split(",") if isinstance(x, str) else x,
    doc="Tag list; accepts a comma-separated string"
)
```

---

## The Config base class

`Config` is the base class for all config classes; the `ConfigMeta` metaclass collects field
information automatically.

### Creating a config instance

```python
# Option 1: direct instantiation (uses defaults)
config = TrainingConfig()

# Option 2: keyword arguments
config = TrainingConfig(learning_rate=0.01, batch_size=64)

# Option 3: chained loading
config = TrainingConfig().from_file("config.yaml").from_args(parser)
```

### Accessing values

```python
# Attribute access
lr = config.learning_rate

# Dict-style access
lr = config["learning_rate"]

# Export to a dict
config_dict = config.to_dict()
```

### Built-in methods

| Method | Description |
|------|------|
| `from_file(path)` | Load from a JSON/YAML/TOML file |
| `from_dict(dict)` | Load from a dict |
| `from_kwargs(**kw)` | Load from keyword arguments |
| `from_args(args)` | Load from an argparse namespace |
| `from_env(prefix)` | Load from environment variables |
| `to_dict()` | Export to a dict |
| `to_file(path)` | Save to a file |
| `freeze()` | Freeze the config, disallowing changes |
| `unfreeze()` | Unfreeze the config |
| `copy()` | Create a copy of the config |
| `validate()` | Validate all field and cross-field constraints; returns `self` |
| `get_field_info(name)` | Get field metadata |
| `get_schema()` | Get the config schema (classmethod) |

---

## Field constraints

The constraint system lives in `asset_embeddings/configs/constraints.py`. A field constraint
(`Constraint`) validates a single field's value, both on assignment and in `validate()`. Cross-field
constraints (`CrossConstraint`) are covered in the [next section](#cross-field-constraints).

Both share the `BaseConstraint` abstract base class, but they are **not interchangeable**: a
`Constraint` validates a single value, whereas a `CrossConstraint` validates an entire `Config` instance.

### Built-in constraints

| Class | Description | Example |
|--------|------|------|
| `TypeConstraint` | Type check | `TypeConstraint(int)` |
| `RangeConstraint` | Numeric range | `RangeConstraint(0, 100)` |
| `ChoiceConstraint` | Allowed choices | `ChoiceConstraint(["a", "b", "c"])` |
| `ChoicesConstraint` | Every element of a list/tuple is an allowed choice | `ChoicesConstraint(["a", "b", "c"])` |
| `EqualityConstraint` | Equality check | `EqualityConstraint(42)` |
| `LengthConstraint` | Length range (`min, max`; use `(N, N)` for an exact length) | `LengthConstraint(1, 10)` |
| `LambdaConstraint` | Custom function | `LambdaConstraint(lambda x: x > 0)` |
| `NoneConstraint` | Must be None | `NoneConstraint()` |
| `NoConstraint` | No constraint | `NoConstraint()` |

### Constraint containers

| Container | Description | Example |
|------|------|------|
| `OptionalConstraint` | Allows None | `OptionalConstraint(TypeConstraint(int))` |
| `InverseConstraint` | Negates a constraint | `InverseConstraint(TypeConstraint(str))` |
| `IntersectionConstraints` | All constraints must hold | `TypeConstraint(int) & RangeConstraint(0, 100)` |
| `UnionConstraints` | Any one constraint may hold | `TypeConstraint(str) \| TypeConstraint(int)` |

### Composing constraints

```python
from asset_embeddings.configs.constraints import *

# Combine with & (AND)
int_in_range = TypeConstraint(int) & RangeConstraint(0, 100)

# Combine with | (OR)
str_or_int = TypeConstraint(str) | TypeConstraint(int)

# Nested composition
complex_constraint = (
    (TypeConstraint(int) & RangeConstraint(0, 100)) |
    (TypeConstraint(str) & LengthConstraint(10))
)
```

### Custom constraints

```python
from asset_embeddings.configs.constraints import Constraint, ConstraintError

class EmailConstraint(Constraint):
    """Email-format validation constraint."""

    def validate(self, value: Any) -> bool:
        import re
        pattern = r'^[\w\.-]+@[\w\.-]+\.\w+$'
        return bool(re.match(pattern, value))

    def get_error_message(self, field_name: str, value: Any) -> str:
        return f"Field '{field_name}' must be a valid email address, got: {value}"

# Usage
email: str = ConfigField(
    constraint=EmailConstraint(),
    doc="Contact email"
)
```

### Advanced LambdaConstraint

```python
# Lambda constraint with a custom error message
positive = LambdaConstraint(
    func=lambda x: x > 0,
    error_message="Value must be positive"
)

# Dynamic error message
even_number = LambdaConstraint(
    func=lambda x: x % 2 == 0,
    error_message=lambda field, value: f"Field '{field}' must be even; {value} is odd"
)
```

---

## Cross-field constraints

A cross-field constraint (`CrossConstraint`) validates invariants between fields — for example,
"`persistent_workers` must be `False` when `num_workers=0`". These run only in `validate()`, not on
assignment, because fields may not all be set yet while a config is loading.

### Class hierarchy

```
BaseConstraint (ABC)           ← shared marker base
├── Constraint                 ← single-field constraint
│   ├── TypeConstraint, RangeConstraint, ...
│   ├── IntersectionConstraints (&)
│   └── UnionConstraints (|)
└── CrossConstraint            ← cross-field constraint
    ├── LambdaCrossConstraint
    ├── MutualExclusionConstraint
    ├── MethodCrossConstraint  ← generated by @cross_validate
    ├── CrossIntersectionConstraints (&)
    └── CrossUnionConstraints (|)
```

### Built-in cross-field constraints

| Class | Description | Example |
|----|------|------|
| `LambdaCrossConstraint` | General lambda constraint | `LambdaCrossConstraint(lambda c: c.x > c.y, error="...", name="...")` |
| `MutualExclusionConstraint` | At most one of the named fields is non-None | `MutualExclusionConstraint("mask_prob", "mask_indices")` |

`CrossConstraint` also supports the `&` (all hold) and `|` (any holds) operators.

### Two ways to declare

#### Option A: the `_cross_constraints` class attribute

```python
from asset_embeddings.configs import Config, ConfigField
from asset_embeddings.configs.constraints import LambdaCrossConstraint

class DataLoaderConfig(Config):
    num_workers: int = ConfigField(default=4)
    persistent_workers: bool = ConfigField(default=True)

    _cross_constraints = [
        LambdaCrossConstraint(
            lambda cfg: not (cfg.num_workers == 0 and cfg.persistent_workers),
            error="persistent_workers must be False when num_workers=0",
            name="persistent_workers_guard",
        ),
    ]
```

#### Option B: the `@cross_validate` decorator

```python
from asset_embeddings.configs.constraints import cross_validate

class MyConfig(Config):
    x: int = ConfigField(default=0)
    y: int = ConfigField(default=0)

    @cross_validate(error="x must be greater than y")
    def _check_x_gt_y(self):
        return self.x > self.y
```

The two styles can be mixed; the metaclass collects both the `_cross_constraints` list and the
`@cross_validate` methods into `_all_cross_constraints`.

### Validation flow

`validate()` is **fail-fast**:

1. **Field constraints** — check each `ConfigField`'s `required` flag and `constraint`.
2. **Cross-field constraints** — once all field constraints pass, run `_all_cross_constraints` in MRO order (parents first).
3. Any failure raises immediately; later constraints do not run.

```python
config = MyConfig().from_file("config.yaml")
config.validate()  # field constraints → cross-field constraints; returns self if all pass
```

`Preparer.set_config()` calls `validate()` internally, so any Preparer has a validated config before
it uses one.

### Inheritance

Cross-field constraints merge along the MRO, with parent constraints running first:

```python
class BaseConfig(Config):
    _cross_constraints = [constraint_A]

class ChildConfig(BaseConfig):
    _cross_constraints = [constraint_B]

# ChildConfig._all_cross_constraints == [constraint_A, constraint_B]
```

Under diamond inheritance, a common ancestor's constraints are not collected twice, because only each
class's own `__dict__` is read (not `getattr`).

### ConfigContainer integration

A `ConfigContainer`'s cross-field constraints run after all child configs have been validated
recursively. It supports `_cross_constraints` and `@cross_validate` too.

`ConfigContainer` also exposes an instance-level dynamic API:

```python
container.add_cross_constraint(my_constraint)        # add
container.remove_cross_constraint("constraint_name") # remove by name; returns bool
```

Instance-level constraints affect only the current instance, not other instances of the same class.
`validate()` runs class-level then instance-level constraints in order.

### Cross-field constraints already in the project

A representative (not exhaustive) sample of the cross-constraints the project's own config classes declare:

| Config class | Constraint name | Rule |
|-----------|--------|------|
| `DataLoaderConfig` | `persistent_workers_guard` | `persistent_workers` must be False when `num_workers=0` |
| `DatasetConfig` | `mask_mode_exclusion` | at most one of `mask_prob` / `mask_indices` is set |
| `AssetBERTTrainConfig` | `best_metric_requires_val_split` | `validation_split` is required when `best_metric` is a val/eval loss |
| `AssetBERTTrainConfig` | `early_stop_metric_requires_val_split` | same, for `early_stop_metric` |
| `OptimizerConfig` | `adafactor_scheduler_requires_optimizer` | `lr_scheduler_type="adafactor"` requires `optimizer_type="Adafactor"` |

---

## Declaring constraints

The configuration system **does not infer constraints from type hints**. Each field declares its
constraint explicitly via `constraint=`; when omitted, the default is `NoConstraint()` (no validation
at all). For type or value checks, use `TypeConstraint`, `ChoiceConstraint`, `OptionalConstraint`, and
so on explicitly.

### Recommended constraints by field type

| Field type | Recommended explicit constraint |
|----------|----------------|
| `int`, `str`, `float`, `bool` | `TypeConstraint(int)`, etc. |
| `Optional[T]` | `OptionalConstraint(TypeConstraint(T))` |
| `Literal["a", "b"]` | `ChoiceConstraint(["a", "b"])` |
| `List[T]` / `Dict[K, V]` / `Tuple[...]` | `TypeConstraint(list/dict/tuple)` checks the container type only; use `ChoicesConstraint([...])` to also check each list/tuple element against an allowed set |

### Example

```python
from typing import Optional, Literal, List
from asset_embeddings.configs.constraints import TypeConstraint, ChoiceConstraint, OptionalConstraint

class MyConfig(Config):
    count: int = ConfigField(default=0, constraint=TypeConstraint(int))
    name: Optional[str] = ConfigField(default=None, constraint=OptionalConstraint(TypeConstraint(str)))
    mode: Literal["train", "val", "test"] = ConfigField(
        default="train", constraint=ChoiceConstraint(["train", "val", "test"])
    )
    layers: List[int] = ConfigField(default=[64, 128, 256], constraint=TypeConstraint(list))
```

### Notes

- A field without `constraint=` is **not** type- or value-checked (its constraint defaults to `NoConstraint()`).
- `required=True` still applies independently: a required field left as `None` fails `validate()`, regardless of whether a constraint is declared.
- Generic containers (such as `List[int]`) check only the container type, not the element type.

---

## Loading and saving

### Supported file formats

| Format | Extension | Special handling |
|------|--------|----------|
| JSON | `.json` | native None support |
| YAML | `.yaml`, `.yml` | native None support |
| TOML | `.toml` | uses the `__none__` marker for None |

### TOML None handling

TOML has no native null/None, so a special string marker is used:

```toml
# Representing None in a TOML file
embedding_path = "__none__"
# or
embedding_path = "__null__"
```

Recognized markers: `__none__`, `__null__`, `null`, `None`.

### Chained-loading precedence

Later loads override earlier ones:

```python
config = (
    MyConfig()
    .from_file("base.yaml")        # base config
    .from_file("override.yaml")    # override some values
    .from_env("APP_")              # environment variables override again
    .from_args(parser)             # command-line arguments win
)
```

### CLI integration pattern

```python
import argparse
from asset_embeddings.configs import Config, ConfigField

class MyConfig(Config):
    input_path: str = ConfigField(required=True, doc="Input path")
    output_path: str = ConfigField(required=True, doc="Output path")
    batch_size: int = ConfigField(default=32, doc="Batch size")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", "-i", required=True)
    parser.add_argument("--output_path", "-o", required=True)
    parser.add_argument("--batch_size", "-b", type=int, default=32)
    parser.add_argument("--config", "-c", help="Config file path")

    args = parser.parse_args()

    config = MyConfig()
    if args.config:
        config = config.from_file(args.config)
    config = config.from_args(args)
    config.validate()
```

---

## Freezing and copying

### Freezing

```python
config = MyConfig().from_file("config.yaml")
config.freeze()  # freeze

# Attempting to modify raises AttributeError
try:
    config.learning_rate = 0.1
except AttributeError as e:
    print(f"Config is frozen: {e}")
```

### Unfreezing

```python
config.unfreeze()
config.learning_rate = 0.1  # now writable
config.freeze()  # freeze again
```

### Copying

```python
# Create an editable copy (even if the original is frozen)
original = MyConfig().from_file("config.yaml").freeze()
editable = original.copy()  # the copy is not frozen
editable.learning_rate = 0.1
```

---

## Extending the system

### Creating a project-specific config class

```python
from asset_embeddings.configs import Config, ConfigField
from asset_embeddings.configs.constraints import (
    RangeConstraint, ChoiceConstraint, LambdaCrossConstraint, cross_validate,
)

class MyProjectConfig(Config):
    """My project config."""

    # Required field
    data_path: str = ConfigField(
        required=True,
        doc="Dataset path"
    )

    # Field with a constraint
    epochs: int = ConfigField(
        default=100,
        constraint=RangeConstraint(1, 1000),
        doc="Number of training epochs"
    )

    # Choice field
    optimizer: str = ConfigField(
        default="adam",
        constraint=ChoiceConstraint(["adam", "sgd", "adamw"]),
        doc="Optimizer type"
    )

    learning_rate: float = ConfigField(
        default=0.001,
        constraint=RangeConstraint(1e-6, 1.0),
        doc="Learning rate"
    )

    # Cross-field constraint: declarative, no need to override validate()
    _cross_constraints = [
        LambdaCrossConstraint(
            lambda cfg: cfg.optimizer != "sgd" or cfg.learning_rate <= 0.1,
            error="SGD optimizer requires learning_rate <= 0.1",
            name="sgd_lr_guard",
        ),
    ]

    # Or the decorator style
    @cross_validate(error="SGD optimizer requires learning_rate <= 0.1")
    def _check_sgd_lr(self):
        return self.optimizer != "sgd" or self.learning_rate <= 0.1
```

### Config inheritance

```python
class BaseTrainConfig(Config):
    """Base training config."""
    seed: int = ConfigField(default=42, doc="Random seed")
    device: str = ConfigField(default="cuda", doc="Compute device")

class BERTTrainConfig(BaseTrainConfig):
    """BERT training config, inheriting the base."""
    hidden_size: int = ConfigField(default=768, doc="Hidden size")
    num_layers: int = ConfigField(default=12, doc="Number of Transformer layers")

# BERTTrainConfig inherits the seed and device fields automatically
```

### Nested configs

For complex configs, use `ConfigContainer` (see `asset_embeddings/configs/container.py`) or nest
manually:

```python
class DatasetConfig(Config):
    path: str = ConfigField(required=True)
    batch_size: int = ConfigField(default=32)

class ModelConfig(Config):
    hidden_size: int = ConfigField(default=256)

class FullConfig(Config):
    dataset: dict = ConfigField(default={})
    model: dict = ConfigField(default={})

    def get_dataset_config(self) -> DatasetConfig:
        return DatasetConfig().from_dict(self.dataset)

    def get_model_config(self) -> ModelConfig:
        return ModelConfig().from_dict(self.model)
```

---

## FAQ

### How are fields that aren't defined on the config handled?

`from_file()` and `from_dict()` load only the fields defined on the config class. Unknown keys are
ignored, but a `UserWarning` is emitted to surface typos rather than swallow them silently.

### Can I add fields dynamically at runtime?

No — the system is descriptor-based and does not support adding fields dynamically. If you need dynamic
configuration, store the extra data in a `dict`-typed field.

### What happens when constraint validation fails?

A `ConstraintError` is raised, with a detailed error message.

### How do I debug config-loading issues?

Use `config.to_dict()` to inspect all current field values, and `Config.get_schema()` to inspect the
field definitions.
