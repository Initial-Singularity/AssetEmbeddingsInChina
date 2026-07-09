# 配置系统

本文档介绍 AssetEmbeddings 的配置系统：`Config` 基类、`ConfigField` 描述符以及约束验证框架。

## 目录

- [概述](#概述)
- [ConfigField 描述符](#configfield-描述符)
- [Config 基类](#config-基类)
- [字段约束](#字段约束)
- [跨字段约束](#跨字段约束)
- [声明约束](#声明约束)
- [加载与保存](#加载与保存)
- [冻结与复制](#冻结与复制)
- [扩展系统](#扩展系统)

---

## 概述

配置系统位于 `asset_embeddings/configs/`，提供：

- **声明式字段定义**——用 `ConfigField` 描述符定义字段。
- **显式约束验证**——通过 `constraint=` 声明字段约束（默认 `NoConstraint()`，即不验证）。
- **链式加载**——从文件、字典、命令行参数与环境变量加载。
- **字段约束**——一套可组合、可扩展的丰富的单字段约束类型。
- **跨字段约束**——跨字段不变量的声明式表达（`_cross_constraints` / `@cross_validate`），全部在 `validate()` 中求值。
- **不可变**——用 `freeze()` 冻结配置。

### 文件结构

```
asset_embeddings/configs/
├── base.py            # Config 基类与 ConfigField 描述符
├── constraints.py     # 约束验证框架
├── container.py       # ConfigContainer 层级配置容器
├── object_configs.py  # 模型 / 日志 / 分词器配置类
├── runtime_configs.py # 训练运行时配置类
└── __init__.py        # 模块导出
```

---

## ConfigField 描述符

`ConfigField` 是一个 Python 描述符，用于声明式地定义配置字段。

### 基本用法

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

### ConfigField 参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `default` | `Any` | `None` | 字段默认值 |
| `constraint` | `Constraint` | `NoConstraint()` | 验证约束 |
| `preprocessor` | `Callable` | `lambda x: x` | 取值预处理函数 |
| `doc` | `str` | `""` | 字段文档 |
| `required` | `bool` | `False` | 字段是否必填 |

### 预处理器示例

```python
# 路径规范化
path: str = ConfigField(
    preprocessor=lambda x: os.path.normpath(x) if x else x,
    doc="File path"
)

# 列表转换
tags: list = ConfigField(
    preprocessor=lambda x: x.split(",") if isinstance(x, str) else x,
    doc="Tag list; accepts a comma-separated string"
)
```

---

## Config 基类

`Config` 是所有配置类的基类；`ConfigMeta` 元类自动收集字段信息。

### 创建配置实例

```python
# 方式 1：直接实例化（使用默认值）
config = TrainingConfig()

# 方式 2：关键字参数
config = TrainingConfig(learning_rate=0.01, batch_size=64)

# 方式 3：链式加载
config = TrainingConfig().from_file("config.yaml").from_args(parser)
```

### 访问取值

```python
# 属性访问
lr = config.learning_rate

# 字典式访问
lr = config["learning_rate"]

# 导出为字典
config_dict = config.to_dict()
```

### 内置方法

| 方法 | 说明 |
|------|------|
| `from_file(path)` | 从 JSON/YAML/TOML 文件加载 |
| `from_dict(dict)` | 从字典加载 |
| `from_kwargs(**kw)` | 从关键字参数加载 |
| `from_args(args)` | 从 argparse namespace 加载 |
| `from_env(prefix)` | 从环境变量加载 |
| `to_dict()` | 导出为字典 |
| `to_file(path)` | 保存到文件 |
| `freeze()` | 冻结配置，禁止更改 |
| `unfreeze()` | 解冻配置 |
| `copy()` | 创建配置的副本 |
| `validate()` | 验证全部字段约束与跨字段约束；返回 `self` |
| `get_field_info(name)` | 获取字段元数据 |
| `get_schema()` | 获取配置 schema（classmethod） |

---

## 字段约束

约束系统位于 `asset_embeddings/configs/constraints.py`。字段约束（`Constraint`）验证单个字段的取值，既在赋值时、也在 `validate()` 时进行。跨字段约束（`CrossConstraint`）见[下一节](#跨字段约束)。

两者共享 `BaseConstraint` 抽象基类，但**不可互换**：`Constraint` 验证单个取值，而 `CrossConstraint` 验证整个 `Config` 实例。

### 内置约束

| 类 | 说明 | 示例 |
|--------|------|------|
| `TypeConstraint` | 类型检查 | `TypeConstraint(int)` |
| `RangeConstraint` | 数值范围 | `RangeConstraint(0, 100)` |
| `ChoiceConstraint` | 允许的取值 | `ChoiceConstraint(["a", "b", "c"])` |
| `ChoicesConstraint` | 列表/元组的每个元素都是允许的取值 | `ChoicesConstraint(["a", "b", "c"])` |
| `EqualityConstraint` | 相等检查 | `EqualityConstraint(42)` |
| `LengthConstraint` | 长度范围（`min, max`；用 `(N, N)` 表示精确长度） | `LengthConstraint(1, 10)` |
| `LambdaConstraint` | 自定义函数 | `LambdaConstraint(lambda x: x > 0)` |
| `NoneConstraint` | 必须为 None | `NoneConstraint()` |
| `NoConstraint` | 无约束 | `NoConstraint()` |

### 约束容器

| 容器 | 说明 | 示例 |
|------|------|------|
| `OptionalConstraint` | 允许 None | `OptionalConstraint(TypeConstraint(int))` |
| `InverseConstraint` | 对约束取反 | `InverseConstraint(TypeConstraint(str))` |
| `IntersectionConstraints` | 所有约束都必须成立 | `TypeConstraint(int) & RangeConstraint(0, 100)` |
| `UnionConstraints` | 任一约束成立即可 | `TypeConstraint(str) \| TypeConstraint(int)` |

### 组合约束

```python
from asset_embeddings.configs.constraints import *

# 用 & 组合（AND）
int_in_range = TypeConstraint(int) & RangeConstraint(0, 100)

# 用 | 组合（OR）
str_or_int = TypeConstraint(str) | TypeConstraint(int)

# 嵌套组合
complex_constraint = (
    (TypeConstraint(int) & RangeConstraint(0, 100)) |
    (TypeConstraint(str) & LengthConstraint(10))
)
```

### 自定义约束

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

# 用法
email: str = ConfigField(
    constraint=EmailConstraint(),
    doc="Contact email"
)
```

### 进阶 LambdaConstraint

```python
# 带自定义错误信息的 lambda 约束
positive = LambdaConstraint(
    func=lambda x: x > 0,
    error_message="Value must be positive"
)

# 动态错误信息
even_number = LambdaConstraint(
    func=lambda x: x % 2 == 0,
    error_message=lambda field, value: f"Field '{field}' must be even; {value} is odd"
)
```

---

## 跨字段约束

跨字段约束（`CrossConstraint`）验证字段之间的不变量——例如，“当 `num_workers=0` 时 `persistent_workers` 必须为 `False`”。它们只在 `validate()` 中运行、而非赋值时，因为配置加载过程中字段可能尚未全部设置。

### 类层级

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

### 内置跨字段约束

| 类 | 说明 | 示例 |
|----|------|------|
| `LambdaCrossConstraint` | 通用 lambda 约束 | `LambdaCrossConstraint(lambda c: c.x > c.y, error="...", name="...")` |
| `MutualExclusionConstraint` | 指定字段中至多一个非 None | `MutualExclusionConstraint("mask_prob", "mask_indices")` |

`CrossConstraint` 同样支持 `&`（全部成立）与 `|`（任一成立）运算符。

### 两种声明方式

#### 方式 A：`_cross_constraints` 类属性

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

#### 方式 B：`@cross_validate` 装饰器

```python
from asset_embeddings.configs.constraints import cross_validate

class MyConfig(Config):
    x: int = ConfigField(default=0)
    y: int = ConfigField(default=0)

    @cross_validate(error="x must be greater than y")
    def _check_x_gt_y(self):
        return self.x > self.y
```

两种风格可混用；元类把 `_cross_constraints` 列表与 `@cross_validate` 方法一并收集到 `_all_cross_constraints`。

### 验证流程

`validate()` 是**快速失败**的：

1. **字段约束**——检查每个 `ConfigField` 的 `required` 标志与 `constraint`。
2. **跨字段约束**——待全部字段约束通过后，按 MRO 顺序（父类优先）运行 `_all_cross_constraints`。
3. 任何失败都立即抛出；后续约束不再运行。

```python
config = MyConfig().from_file("config.yaml")
config.validate()  # 字段约束 → 跨字段约束；全部通过则返回 self
```

`Preparer.set_config()` 内部会调用 `validate()`，故任何 Preparer 在使用配置之前，其配置都已被验证。

### 继承

跨字段约束沿 MRO 合并，父类约束先运行：

```python
class BaseConfig(Config):
    _cross_constraints = [constraint_A]

class ChildConfig(BaseConfig):
    _cross_constraints = [constraint_B]

# ChildConfig._all_cross_constraints == [constraint_A, constraint_B]
```

在菱形继承下，公共祖先的约束不会被收集两次，因为只读取每个类自身的 `__dict__`（而非 `getattr`）。

### 与 ConfigContainer 的集成

`ConfigContainer` 的跨字段约束在所有子配置被递归验证之后运行。它同样支持 `_cross_constraints` 与 `@cross_validate`。

`ConfigContainer` 还暴露一个实例级动态 API：

```python
container.add_cross_constraint(my_constraint)        # 添加
container.remove_cross_constraint("constraint_name") # 按名称移除；返回 bool
```

实例级约束只影响当前实例，不影响同类的其他实例。`validate()` 按顺序先运行类级、再运行实例级约束。

### 项目中已有的跨字段约束

项目自身配置类所声明的跨字段约束的一个代表性（非穷尽）样本：

| 配置类 | 约束名 | 规则 |
|-----------|--------|------|
| `DataLoaderConfig` | `persistent_workers_guard` | 当 `num_workers=0` 时 `persistent_workers` 必须为 False |
| `DatasetConfig` | `mask_mode_exclusion` | `mask_prob` / `mask_indices` 至多设置一个 |
| `AssetBERTTrainConfig` | `best_metric_requires_val_split` | 当 `best_metric` 是 val/eval loss 时需要 `validation_split` |
| `AssetBERTTrainConfig` | `early_stop_metric_requires_val_split` | 同上，针对 `early_stop_metric` |
| `OptimizerConfig` | `adafactor_scheduler_requires_optimizer` | `lr_scheduler_type="adafactor"` 需要 `optimizer_type="Adafactor"` |

---

## 声明约束

配置系统**不从类型提示推断约束**。每个字段都通过 `constraint=` 显式声明其约束；省略时默认为 `NoConstraint()`（完全不验证）。要做类型或取值检查，请显式使用 `TypeConstraint`、`ChoiceConstraint`、`OptionalConstraint` 等。

### 按字段类型推荐的约束

| 字段类型 | 推荐的显式约束 |
|----------|----------------|
| `int`、`str`、`float`、`bool` | `TypeConstraint(int)` 等 |
| `Optional[T]` | `OptionalConstraint(TypeConstraint(T))` |
| `Literal["a", "b"]` | `ChoiceConstraint(["a", "b"])` |
| `List[T]` / `Dict[K, V]` / `Tuple[...]` | `TypeConstraint(list/dict/tuple)` 只检查容器类型；用 `ChoicesConstraint([...])` 还可检查每个列表/元组元素是否属于允许集合 |

### 示例

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

### 说明

- 不带 `constraint=` 的字段**不会**做类型或取值检查（其约束默认为 `NoConstraint()`）。
- `required=True` 仍独立生效：必填字段若留为 `None` 会使 `validate()` 失败，无论是否声明了约束。
- 泛型容器（如 `List[int]`）只检查容器类型，不检查元素类型。

---

## 加载与保存

### 支持的文件格式

| 格式 | 扩展名 | 特殊处理 |
|------|--------|----------|
| JSON | `.json` | 原生支持 None |
| YAML | `.yaml`、`.yml` | 原生支持 None |
| TOML | `.toml` | 用 `__none__` 标记表示 None |

### TOML 的 None 处理

TOML 没有原生的 null/None，故用特殊字符串标记：

```toml
# 在 TOML 文件中表示 None
embedding_path = "__none__"
# 或
embedding_path = "__null__"
```

可识别的标记：`__none__`、`__null__`、`null`、`None`。

### 链式加载优先级

后加载的覆盖先加载的：

```python
config = (
    MyConfig()
    .from_file("base.yaml")        # 基础配置
    .from_file("override.yaml")    # 覆盖部分取值
    .from_env("APP_")              # 环境变量再次覆盖
    .from_args(parser)             # 命令行参数最终胜出
)
```

### CLI 集成模式

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

## 冻结与复制

### 冻结

```python
config = MyConfig().from_file("config.yaml")
config.freeze()  # 冻结

# 尝试修改会抛出 AttributeError
try:
    config.learning_rate = 0.1
except AttributeError as e:
    print(f"Config is frozen: {e}")
```

### 解冻

```python
config.unfreeze()
config.learning_rate = 0.1  # 现在可写
config.freeze()  # 再次冻结
```

### 复制

```python
# 创建可编辑副本（即便原配置已冻结）
original = MyConfig().from_file("config.yaml").freeze()
editable = original.copy()  # 副本未冻结
editable.learning_rate = 0.1
```

---

## 扩展系统

### 创建项目专属配置类

```python
from asset_embeddings.configs import Config, ConfigField
from asset_embeddings.configs.constraints import (
    RangeConstraint, ChoiceConstraint, LambdaCrossConstraint, cross_validate,
)

class MyProjectConfig(Config):
    """My project config."""

    # 必填字段
    data_path: str = ConfigField(
        required=True,
        doc="Dataset path"
    )

    # 带约束的字段
    epochs: int = ConfigField(
        default=100,
        constraint=RangeConstraint(1, 1000),
        doc="Number of training epochs"
    )

    # 选择字段
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

    # 跨字段约束：声明式，无需重写 validate()
    _cross_constraints = [
        LambdaCrossConstraint(
            lambda cfg: cfg.optimizer != "sgd" or cfg.learning_rate <= 0.1,
            error="SGD optimizer requires learning_rate <= 0.1",
            name="sgd_lr_guard",
        ),
    ]

    # 或装饰器风格
    @cross_validate(error="SGD optimizer requires learning_rate <= 0.1")
    def _check_sgd_lr(self):
        return self.optimizer != "sgd" or self.learning_rate <= 0.1
```

### 配置继承

```python
class BaseTrainConfig(Config):
    """Base training config."""
    seed: int = ConfigField(default=42, doc="Random seed")
    device: str = ConfigField(default="cuda", doc="Compute device")

class BERTTrainConfig(BaseTrainConfig):
    """BERT training config, inheriting the base."""
    hidden_size: int = ConfigField(default=768, doc="Hidden size")
    num_layers: int = ConfigField(default=12, doc="Number of Transformer layers")

# BERTTrainConfig 自动继承 seed 与 device 字段
```

### 嵌套配置

对于复杂配置，使用 `ConfigContainer`（见 `asset_embeddings/configs/container.py`）或手动嵌套：

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

## 常见问题

### 配置上未定义的字段如何处理？

`from_file()` 与 `from_dict()` 只加载配置类上定义的字段。未知键被忽略，但会发出 `UserWarning`，以便暴露拼写错误而非悄无声息地吞掉。

### 能否在运行时动态添加字段？

不能——系统基于描述符，不支持动态添加字段。若需要动态配置，把额外数据存进一个 `dict` 类型的字段。

### 约束验证失败会怎样？

抛出 `ConstraintError`，并附带详细错误信息。

### 如何调试配置加载问题？

用 `config.to_dict()` 查看当前全部字段取值，用 `Config.get_schema()` 查看字段定义。
