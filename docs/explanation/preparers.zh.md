# Preparers

本文档介绍 AssetEmbeddings 的 Preparer 模式，用于以配置驱动的方式创建组件。

## 目录

- [概述](#概述)
- [Preparer 基类](#preparer-基类)
- [Logger_Preparer](#logger_preparer)
- [Tokenizer_Preparer](#tokenizer_preparer)
- [Dataset_Preparer](#dataset_preparer)
- [DataLoader_Preparer](#dataloader_preparer)
- [模型 Preparer](#模型-preparer)
- [Optimizer_Preparer](#optimizer_preparer)
- [Accelerator_Preparer](#accelerator_preparer)
- [扩展系统](#扩展系统)

---

## 概述

Preparer 模式是项目统一的组件初始化方式。其特点：

- **配置驱动**——初始化参数由 Config 对象定义。
- **链式 API**——`set_config()` 与 `set_logger()` 返回 `self`，支持链式调用。
- **可复用**——同一 Preparer 实例换配置后可复用。
- **日志集成**——内置 logger，便于调试。

### Preparer 列表

| Preparer | 配置类 | 输出类型 |
|----------|--------------|-------------|
| `Logger_Preparer` | `LoggerConfig` | `logging.Logger` |
| `Tokenizer_Preparer` | `TokenizerConfig` | `PreTrainedTokenizerFast` |
| `Dataset_Preparer` | `DatasetConfig` | `AssetBERTMLMDataset` |
| `DataLoader_Preparer` | `DataLoaderConfig` | `DataLoader` |
| `AssetBERT_Preparer` | `AssetBERTModelConfig` | `BertForMaskedLM` |
| `Optimizer_Preparer` | `OptimizerConfig` | `Optimizer, LRScheduler` |
| `Accelerator_Preparer` | `AcceleratorConfig` | `accelerate.Accelerator` |

---

## Preparer 基类

所有 Preparer 都继承自 `Preparer` 基类：

```python
class Preparer:
    def __init__(self):
        self.logger = logging.getLogger("Preparer")
        self.config: Config

    def set_config(self, config: Config) -> Self:
        """Validate and set the config (configs are validated eagerly, here)."""
        config.validate()
        self.config = config
        return self

    def get_config(self) -> Config:
        """Get the current config."""
        return self.config

    def set_logger(self, logger: logging.Logger) -> Self:
        """Set the logger."""
        self.logger = logger
        return self

    def prepare(self):
        """Run initialization (implemented by subclasses)."""
        ...
```

### 使用模式

```python
# 基本模式
result = SomePreparer().set_config(config).prepare()

# 带 logger
result = SomePreparer().set_logger(logger).set_config(config).prepare()

# 复用 Preparer
preparer = SomePreparer()
result1 = preparer.set_config(config1).prepare()
result2 = preparer.set_config(config2).prepare()
```

---

## Logger_Preparer

创建配置驱动的 logger。完整文档见 [logging-system.md](./logging-system.md)。

### 配置类：LoggerConfig

```python
from asset_embeddings.configs import LoggerConfig

config = LoggerConfig(
    log_name="MyApp",
    log_file="logs/app.log",
    console_level="INFO",
    file_level="DEBUG",
    console_stream="tqdm",
    enable_colors=True,
    color_target="format"
)
```

### 使用示例

```python
from asset_embeddings.preparers import Logger_Preparer
from asset_embeddings.configs import LoggerConfig

logger = Logger_Preparer().set_config(
    LoggerConfig(
        log_name="Training",
        log_file="logs/train.log",
        console_level="DEBUG" if args.verbose else "INFO"
    )
).prepare()
```

---

## Tokenizer_Preparer

为股票词元创建 HuggingFace `PreTrainedTokenizerFast`。

### 配置类：TokenizerConfig

```python
from asset_embeddings.configs import TokenizerConfig

config = TokenizerConfig(
    # 词表来源：以下四选一
    pretrained_tokenizer_file="tokenizer/tokenizer.json",  # 预训练 tokenizer
    vocab_file="vocab.json",                               # 词表文件
    w2v_model="model/w2v.model",                          # Word2Vec 模型
    embedding_file="embeddings.csv",                       # 嵌入文件

    # 股票别名映射
    alias_file="data/stock_names.csv"
)
```

### 词表来源优先级

1. `pretrained_tokenizer_file`：直接加载预训练 tokenizer
2. `vocab_file`：从词表 JSON 文件构建
3. `w2v_model`：从 Word2Vec 模型提取词表
4. `embedding_file`：从嵌入 CSV 文件提取词表

### 使用示例

```python
from asset_embeddings.preparers import Tokenizer_Preparer
from asset_embeddings.configs import TokenizerConfig

tokenizer = Tokenizer_Preparer().set_logger(logger).set_config(
    TokenizerConfig(
        pretrained_tokenizer_file="tokenizer/bert_tokenizer.json",
        alias_file="data/stock_alias.csv"
    )
).prepare()

# 使用 tokenizer
tokens = tokenizer.tokenize("000001 000002 600000")
ids = tokenizer.convert_tokens_to_ids(tokens)
```

### 特殊词元

tokenizer 自动添加以下特殊词元：

| 词元 | 用途 |
|-------|---------|
| `[MASK]` | 遮蔽位置 |
| `[PAD]` | 填充 |
| `[UNK]` | 未知词元 |
| `[CLS]` | 序列起始（可选） |
| `[SEP]` | 序列分隔（可选） |

---

## Dataset_Preparer

创建 `AssetBERTMLMDataset` 数据集。

### 配置类：DatasetConfig

```python
from asset_embeddings.configs import DatasetConfig

config = DatasetConfig(
    data_path="data/processed/train",
    data_format="csv",              # csv/json
    max_length=512,
    mask_prob=0.15,
    mask_indices=None,              # 固定遮蔽位置（列表）
    include_proportion=True,
    cache_size=10000,
    num_repeats=1,

    # 数据列名
    id_key="InvestorID",
    portfolio_key="Portfolio",
    proportion1_key="Proportion1",
    proportion2_key="Proportion2"
)
```

### 使用示例

```python
from asset_embeddings.preparers import Dataset_Preparer
from asset_embeddings.configs import DatasetConfig

dataset = Dataset_Preparer().set_logger(logger).set_config(
    DatasetConfig(
        data_path="data/processed/2023Q1",
        mask_prob=0.15,
        include_proportion=True
    )
).prepare(tokenizer=tokenizer)

print(f"Dataset size: {len(dataset)}")
```

### 数据集返回格式

每个样本返回一个 dict：

```python
{
    "input_ids": torch.Tensor,       # [seq_len]
    "attention_mask": torch.Tensor,  # [seq_len]
    "labels": torch.Tensor,          # [seq_len]，-100 标记非遮蔽位置
    "proportions1": torch.Tensor,    # [seq_len]，include_proportion=False 时填零
    "proportions2": torch.Tensor     # [seq_len]，include_proportion=False 时填零
}
```

---

## DataLoader_Preparer

创建 PyTorch `DataLoader`。

### 配置类：DataLoaderConfig

```python
from asset_embeddings.configs import DataLoaderConfig

config = DataLoaderConfig(
    batch_size=32,
    shuffle=True,
    num_workers=4,
    persistent_workers=True,
    pin_memory=True,
    drop_last=True
)
```

### 使用示例

```python
from asset_embeddings.preparers import DataLoader_Preparer
from asset_embeddings.configs import DataLoaderConfig

train_loader = DataLoader_Preparer().set_logger(logger).set_config(
    DataLoaderConfig(
        batch_size=32,
        shuffle=True,
        num_workers=4
    )
).prepare(dataset=train_dataset)

# 验证集不打乱
val_loader = DataLoader_Preparer().set_logger(logger).set_config(
    DataLoaderConfig(
        batch_size=32,
        num_workers=4
    )
).prepare(dataset=val_dataset, shuffle_override=False)
```

---

## 模型 Preparer

### AssetBERT_Preparer

创建标准的 `BertForMaskedLM` 模型。

#### 配置类：AssetBERTModelConfig

```python
from asset_embeddings.configs import AssetBERTModelConfig

config = AssetBERTModelConfig(
    vocab_size=5000,
    hidden_size=256,
    num_hidden_layers=4,
    num_attention_heads=4,
    intermediate_size=1024,
    max_position_embeddings=512,
    type_vocab_size=2,

    # 预训练 / 初始化
    model_checkpoint="model/pretrained/bert.safetensors",
    w2v_model="model/w2v.model",
    embedding_file="embeddings.csv",

    # 冻结选项
    freeze_embedding=False,
    freeze_encoder=False
)
```

#### 使用示例

```python
from asset_embeddings.preparers import AssetBERT_Preparer

model = AssetBERT_Preparer().set_logger(logger).set_config(
    AssetBERTModelConfig(
        vocab_size=len(tokenizer),
        hidden_size=256,
        num_hidden_layers=4,
        model_checkpoint="model/pretrained/base.safetensors"
    )
).prepare(tokenizer=tokenizer)
```

### 模型初始化流程

1. 创建模型结构（依据配置参数）
2. 加载检查点（若提供 `model_checkpoint`）
3. 加载预训练嵌入（来自 `w2v_model` 或 `embedding_file`）
4. 应用冻结设置

---

## Optimizer_Preparer

创建优化器与学习率调度器。

### 配置类：OptimizerConfig

```python
from asset_embeddings.configs import OptimizerConfig

config = OptimizerConfig(
    # 优化器
    optimizer_type="AdamW",
    learning_rate=1e-4,
    optimizer_kwargs={"weight_decay": 0.01},

    # 学习率调度器
    lr_scheduler_type="cosine",
    lr_scheduler_warmup_steps=1000,
    lr_scheduler_train_steps=10000,
    lr_scheduler_num_cycles=1,
    lr_scheduler_power=1.0
)
```

### 支持的优化器

| 类型 | 说明 |
|------|-------------|
| `Adam` | 标准 Adam |
| `Adam8bit` | 8-bit 量化 Adam（bitsandbytes） |
| `PagedAdam8bit` | Paged 8-bit Adam |
| `AdamW` | 带权重衰减的 Adam |
| `AdamW8bit` | 8-bit AdamW |
| `PagedAdamW8bit` | Paged 8-bit AdamW |
| `Lion` | Lion 优化器 |
| `Lion8bit` | 8-bit Lion |
| `PagedLion8bit` | Paged 8-bit Lion |
| `SGDNesterov` | 带 Nesterov 动量的 SGD |
| `SGD8bit` | 8-bit SGD |
| `DAdaptation` | 自适应学习率 |
| `Adafactor` | 内存高效优化器 |

### 支持的调度器

| 类型 | 说明 |
|------|-------------|
| `constant` | 恒定学习率（含 warmup） |
| `linear` | 线性衰减 |
| `cosine` | 余弦衰减 |
| `cosine_with_restarts` | 带重启的余弦 |
| `polynomial` | 多项式衰减 |
| `adafactor` | Adafactor 专用调度器 |

### 使用示例

```python
from asset_embeddings.preparers import Optimizer_Preparer
from asset_embeddings.configs import OptimizerConfig

optimizer, lr_scheduler = Optimizer_Preparer().set_logger(logger).set_config(
    OptimizerConfig(
        optimizer_type="AdamW",
        learning_rate=1e-4,
        lr_scheduler_type="cosine",
        lr_scheduler_warmup_steps=500,
        lr_scheduler_train_steps=len(train_loader) * epochs
    )
).prepare(trainable_params=model.parameters())
```

### 分步创建

```python
preparer = Optimizer_Preparer().set_logger(logger).set_config(config)

# 只创建优化器
optimizer = preparer.prepare_optimizer(model.parameters())

# 只创建调度器（需要已有的优化器）
lr_scheduler = preparer.prepare_lr_scheduler(optimizer)
```

---

## Accelerator_Preparer

创建一个配置好的 HuggingFace [`Accelerator`](https://huggingface.co/docs/accelerate)。它设置编译缓存环境变量、TF32 标志，并构建 `TorchDynamoPlugin`（用于 `torch.compile`），随后以配置好的梯度累积与混合精度设置构造 `Accelerator`。它**不**调用 `accelerator.prepare(model, optimizer, ...)`——那留在 trainer 中。

### 配置类：AcceleratorConfig

```python
from asset_embeddings.configs import AcceleratorConfig

config = AcceleratorConfig(
    mixed_precision="auto",          # "no" / "fp16" / "bf16" / "auto"
    gradient_accumulation_steps=1,
    use_compile=False,               # 经 TorchDynamoPlugin 启用 torch.compile
    compile_backend="inductor",
    compile_mode="default",
    compile_fullgraph=False,
    compile_dynamic=False,
    compile_cache_dir=None,          # 给定时设置 TORCHINDUCTOR_CACHE_DIR
    allow_tf32=True,
)
```

### 使用示例

```python
from asset_embeddings.preparers import Accelerator_Preparer
from asset_embeddings.configs import AcceleratorConfig

accelerator = Accelerator_Preparer().set_logger(logger).set_config(
    AcceleratorConfig(mixed_precision="auto", use_compile=True)
).prepare()
```

---

## 扩展系统

### 创建自定义 Preparer

```python
from asset_embeddings.preparers import Preparer
from asset_embeddings.configs import Config, ConfigField

class MyComponentConfig(Config):
    param1: int = ConfigField(default=10)
    param2: str = ConfigField(default="default")

class MyComponent_Preparer(Preparer):
    def __init__(self):
        super().__init__()
        self.config: MyComponentConfig = MyComponentConfig()

    def prepare(self) -> MyComponent:
        self.logger.info(f"Preparing MyComponent with param1={self.config.param1}")

        component = MyComponent(
            param1=self.config.param1,
            param2=self.config.param2
        )

        self.logger.info("MyComponent prepared.")
        return component
```

### 带依赖的 Preparer

```python
class ComplexComponent_Preparer(Preparer):
    def __init__(self):
        super().__init__()
        self.config: ComplexConfig = ComplexConfig()

    def prepare(self, dependency1, dependency2) -> ComplexComponent:
        """
        Args:
            dependency1: the first dependency component
            dependency2: the second dependency component
        """
        self.logger.info("Preparing ComplexComponent...")

        component = ComplexComponent(
            dep1=dependency1,
            dep2=dependency2,
            **self.config.to_dict()
        )

        return component
```

### 组合多个 Preparer

```python
def setup_training_components(args):
    """Set up all components needed for training."""

    # Logger
    logger = Logger_Preparer().set_config(
        LoggerConfig(log_name="Training", console_level="DEBUG" if args.verbose else "INFO")
    ).prepare()

    # Tokenizer
    tokenizer = Tokenizer_Preparer().set_logger(logger).set_config(
        TokenizerConfig(pretrained_tokenizer_file=args.tokenizer)
    ).prepare()

    # Dataset
    dataset = Dataset_Preparer().set_logger(logger).set_config(
        DatasetConfig(data_path=args.data, mask_prob=0.15)
    ).prepare(tokenizer=tokenizer)

    # DataLoader
    dataloader = DataLoader_Preparer().set_logger(logger).set_config(
        DataLoaderConfig(batch_size=args.batch_size)
    ).prepare(dataset=dataset)

    # Model
    model = AssetBERT_Preparer().set_logger(logger).set_config(
        AssetBERTModelConfig(vocab_size=len(tokenizer), hidden_size=args.hidden_size)
    ).prepare(tokenizer=tokenizer)

    # Optimizer
    optimizer, scheduler = Optimizer_Preparer().set_logger(logger).set_config(
        OptimizerConfig(optimizer_type="AdamW", learning_rate=args.lr)
    ).prepare(trainable_params=model.parameters())

    return {
        "logger": logger,
        "tokenizer": tokenizer,
        "dataset": dataset,
        "dataloader": dataloader,
        "model": model,
        "optimizer": optimizer,
        "scheduler": scheduler
    }
```
