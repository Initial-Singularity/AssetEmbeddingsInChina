# Preparers

This document describes the AssetEmbeddings Preparer pattern, used to create components in a configuration-driven way.

## Contents

- [Overview](#overview)
- [The Preparer base class](#the-preparer-base-class)
- [Logger_Preparer](#logger_preparer)
- [Tokenizer_Preparer](#tokenizer_preparer)
- [Dataset_Preparer](#dataset_preparer)
- [DataLoader_Preparer](#dataloader_preparer)
- [Model Preparers](#model-preparers)
- [Optimizer_Preparer](#optimizer_preparer)
- [Accelerator_Preparer](#accelerator_preparer)
- [Extending the system](#extending-the-system)

---

## Overview

The Preparer pattern is the project's uniform approach to component initialization. It has the following characteristics:

- **Configuration-driven** — initialization parameters are defined via a Config object.
- **Chained API** — `set_config()` and `set_logger()` return `self`, supporting chained calls.
- **Reusable** — the same Preparer instance can be reused after swapping its config.
- **Logging integration** — a built-in logger eases debugging.

### Preparer list

| Preparer | Config class | Output type |
|----------|--------------|-------------|
| `Logger_Preparer` | `LoggerConfig` | `logging.Logger` |
| `Tokenizer_Preparer` | `TokenizerConfig` | `PreTrainedTokenizerFast` |
| `Dataset_Preparer` | `DatasetConfig` | `AssetBERTMLMDataset` |
| `DataLoader_Preparer` | `DataLoaderConfig` | `DataLoader` |
| `AssetBERT_Preparer` | `AssetBERTModelConfig` | `BertForMaskedLM` |
| `Optimizer_Preparer` | `OptimizerConfig` | `Optimizer, LRScheduler` |
| `Accelerator_Preparer` | `AcceleratorConfig` | `accelerate.Accelerator` |

---

## The Preparer base class

All Preparers inherit from the `Preparer` base class:

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

### Usage patterns

```python
# Basic pattern
result = SomePreparer().set_config(config).prepare()

# With a logger
result = SomePreparer().set_logger(logger).set_config(config).prepare()

# Reusing a Preparer
preparer = SomePreparer()
result1 = preparer.set_config(config1).prepare()
result2 = preparer.set_config(config2).prepare()
```

---

## Logger_Preparer

Creates a configuration-driven logger. See [logging-system.md](./logging-system.md) for full documentation.

### Config class: LoggerConfig

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

### Usage example

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

Creates a HuggingFace `PreTrainedTokenizerFast` for stock tokens.

### Config class: TokenizerConfig

```python
from asset_embeddings.configs import TokenizerConfig

config = TokenizerConfig(
    # Vocabulary source: choose one of the following
    pretrained_tokenizer_file="tokenizer/tokenizer.json",  # pretrained tokenizer
    vocab_file="vocab.json",                               # vocabulary file
    w2v_model="model/w2v.model",                          # Word2Vec model
    embedding_file="embeddings.csv",                       # embedding file

    # Stock alias mapping
    alias_file="data/stock_names.csv"
)
```

### Vocabulary source priority

1. `pretrained_tokenizer_file`: load a pretrained tokenizer directly
2. `vocab_file`: build from a vocabulary JSON file
3. `w2v_model`: extract the vocabulary from a Word2Vec model
4. `embedding_file`: extract the vocabulary from an embedding CSV file

### Usage example

```python
from asset_embeddings.preparers import Tokenizer_Preparer
from asset_embeddings.configs import TokenizerConfig

tokenizer = Tokenizer_Preparer().set_logger(logger).set_config(
    TokenizerConfig(
        pretrained_tokenizer_file="tokenizer/bert_tokenizer.json",
        alias_file="data/stock_alias.csv"
    )
).prepare()

# Using the tokenizer
tokens = tokenizer.tokenize("000001 000002 600000")
ids = tokenizer.convert_tokens_to_ids(tokens)
```

### Special tokens

The tokenizer automatically adds the following special tokens:

| Token | Purpose |
|-------|---------|
| `[MASK]` | Masked position |
| `[PAD]` | Padding |
| `[UNK]` | Unknown token |
| `[CLS]` | Sequence start (optional) |
| `[SEP]` | Sequence separator (optional) |

---

## Dataset_Preparer

Creates the `AssetBERTMLMDataset` dataset.

### Config class: DatasetConfig

```python
from asset_embeddings.configs import DatasetConfig

config = DatasetConfig(
    data_path="data/processed/train",
    data_format="csv",              # csv/json
    max_length=512,
    mask_prob=0.15,
    mask_indices=None,              # fixed mask positions (list)
    include_proportion=True,
    cache_size=10000,
    num_repeats=1,

    # Data column names
    id_key="InvestorID",
    portfolio_key="Portfolio",
    proportion1_key="Proportion1",
    proportion2_key="Proportion2"
)
```

### Usage example

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

### Dataset return format

Each sample returns a dict:

```python
{
    "input_ids": torch.Tensor,       # [seq_len]
    "attention_mask": torch.Tensor,  # [seq_len]
    "labels": torch.Tensor,          # [seq_len], -100 marks non-masked positions
    "proportions1": torch.Tensor,    # [seq_len], zero-filled when include_proportion=False
    "proportions2": torch.Tensor     # [seq_len], zero-filled when include_proportion=False
}
```

---

## DataLoader_Preparer

Creates a PyTorch `DataLoader`.

### Config class: DataLoaderConfig

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

### Usage example

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

# Do not shuffle the validation set
val_loader = DataLoader_Preparer().set_logger(logger).set_config(
    DataLoaderConfig(
        batch_size=32,
        num_workers=4
    )
).prepare(dataset=val_dataset, shuffle_override=False)
```

---

## Model Preparers

### AssetBERT_Preparer

Creates a standard `BertForMaskedLM` model.

#### Config class: AssetBERTModelConfig

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

    # Pretraining / initialization
    model_checkpoint="model/pretrained/bert.safetensors",
    w2v_model="model/w2v.model",
    embedding_file="embeddings.csv",

    # Freeze options
    freeze_embedding=False,
    freeze_encoder=False
)
```

#### Usage example

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

### Model initialization flow

1. Create the model structure (from the config parameters)
2. Load a checkpoint (if `model_checkpoint` is provided)
3. Load pretrained embeddings (from `w2v_model` or `embedding_file`)
4. Apply the freeze settings

---

## Optimizer_Preparer

Creates the optimizer and the learning-rate scheduler.

### Config class: OptimizerConfig

```python
from asset_embeddings.configs import OptimizerConfig

config = OptimizerConfig(
    # Optimizer
    optimizer_type="AdamW",
    learning_rate=1e-4,
    optimizer_kwargs={"weight_decay": 0.01},

    # Learning-rate scheduler
    lr_scheduler_type="cosine",
    lr_scheduler_warmup_steps=1000,
    lr_scheduler_train_steps=10000,
    lr_scheduler_num_cycles=1,
    lr_scheduler_power=1.0
)
```

### Supported optimizers

| Type | Description |
|------|-------------|
| `Adam` | Standard Adam |
| `Adam8bit` | 8-bit quantized Adam (bitsandbytes) |
| `PagedAdam8bit` | Paged 8-bit Adam |
| `AdamW` | Adam with weight decay |
| `AdamW8bit` | 8-bit AdamW |
| `PagedAdamW8bit` | Paged 8-bit AdamW |
| `Lion` | Lion optimizer |
| `Lion8bit` | 8-bit Lion |
| `PagedLion8bit` | Paged 8-bit Lion |
| `SGDNesterov` | SGD with Nesterov momentum |
| `SGD8bit` | 8-bit SGD |
| `DAdaptation` | Adaptive learning rate |
| `Adafactor` | Memory-efficient optimizer |

### Supported schedulers

| Type | Description |
|------|-------------|
| `constant` | Constant learning rate (with warmup) |
| `linear` | Linear decay |
| `cosine` | Cosine decay |
| `cosine_with_restarts` | Cosine with restarts |
| `polynomial` | Polynomial decay |
| `adafactor` | Adafactor-specific scheduler |

### Usage example

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

### Step-by-step creation

```python
preparer = Optimizer_Preparer().set_logger(logger).set_config(config)

# Create only the optimizer
optimizer = preparer.prepare_optimizer(model.parameters())

# Create only the scheduler (requires an existing optimizer)
lr_scheduler = preparer.prepare_lr_scheduler(optimizer)
```

---

## Accelerator_Preparer

Creates a configured HuggingFace [`Accelerator`](https://huggingface.co/docs/accelerate). It sets the
compile-cache environment variables, the TF32 flags, and builds a `TorchDynamoPlugin` (for
`torch.compile`), then constructs the `Accelerator` with the configured gradient-accumulation and
mixed-precision settings. It does **not** call `accelerator.prepare(model, optimizer, ...)` — that stays
in the trainer.

### Config class: AcceleratorConfig

```python
from asset_embeddings.configs import AcceleratorConfig

config = AcceleratorConfig(
    mixed_precision="auto",          # "no" / "fp16" / "bf16" / "auto"
    gradient_accumulation_steps=1,
    use_compile=False,               # enable torch.compile via TorchDynamoPlugin
    compile_backend="inductor",
    compile_mode="default",
    compile_fullgraph=False,
    compile_dynamic=False,
    compile_cache_dir=None,          # sets TORCHINDUCTOR_CACHE_DIR when given
    allow_tf32=True,
)
```

### Usage example

```python
from asset_embeddings.preparers import Accelerator_Preparer
from asset_embeddings.configs import AcceleratorConfig

accelerator = Accelerator_Preparer().set_logger(logger).set_config(
    AcceleratorConfig(mixed_precision="auto", use_compile=True)
).prepare()
```

---

## Extending the system

### Creating a custom Preparer

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

### A Preparer with dependencies

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

### Composing multiple Preparers

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
