import os
from numbers import Number
from typing import Union, List, Optional, Literal


from .constraints import (
    TypeConstraint,
    RangeConstraint,
    ChoiceConstraint,
    OptionalConstraint,
    LambdaCrossConstraint,
    MutualExclusionConstraint,
)
from .base import Config, ConfigField


def path_preprocessor(p):
    if p is None:
        return None
    if isinstance(p, list):
        return [os.path.normpath(x) for x in p]
    return os.path.normpath(p)


class LoggerConfig(Config):
    """Configuration for logging system with flexible output options.

    This configuration class manages all aspects of application logging including
    output destinations, formatting, log levels, and file rotation settings.
    It supports both console and file logging with independent level controls.

    Attributes:
        log_name: Name identifier for the logger instance.
        log_file: Path(s) to log file(s), can be a single string or list of strings.
        log_format: Format string for log messages using Python logging format.
        date_format: Format string for timestamps in log messages.
        console_level: Minimum log level for console output.
        console_stream: Type of console output stream to use.
        file_level: Minimum log level for file output.
        max_bytes: Maximum size of log files before rotation (None for unlimited).
        backup_count: Number of backup files to keep during rotation.
    """

    log_name: str = ConfigField(default="app", constraint=TypeConstraint(str), doc="日志名称")
    log_file: Union[str, List[str], None] = ConfigField(
        default="app.log", constraint=OptionalConstraint(TypeConstraint(str) | TypeConstraint(list)), doc="日志文件路径"
    )
    log_format: str = ConfigField(
        default="%(asctime)s [%(name)s][%(levelname)s]: %(message)s", constraint=TypeConstraint(str), doc="日志格式"
    )
    date_format: str = ConfigField(default="%Y-%m-%d %H:%M:%S", constraint=TypeConstraint(str), doc="日期格式")
    console_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = ConfigField(
        default="INFO",
        constraint=ChoiceConstraint(["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]),
        doc="控制台日志级别",
    )
    console_stream: Literal["tqdm", "std", "html"] = ConfigField(
        default="tqdm", constraint=ChoiceConstraint(["tqdm", "std", "html"]), doc="控制台输出流"
    )
    file_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = ConfigField(
        default="DEBUG",
        constraint=ChoiceConstraint(["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]),
        doc="文件日志级别",
    )
    max_bytes: int = ConfigField(
        default=10_000_000, constraint=OptionalConstraint(RangeConstraint(0, None)), doc="最大字节数，为None时不限制"
    )
    backup_count: int = ConfigField(
        default=5, constraint=OptionalConstraint(RangeConstraint(0, None)), doc="备份文件数量，为0时不备份"
    )

    # Color control settings
    enable_colors: bool = ConfigField(default=True, constraint=TypeConstraint(bool), doc="是否启用颜色输出")
    color_target: Literal["message", "levelname", "format"] = ConfigField(
        default="format",
        constraint=ChoiceConstraint(["message", "levelname", "format"]),
        doc="着色目标：message(仅消息), levelname(仅日志级别), format(格式化字符串所有组件)",
    )


class TokenizerConfig(Config):
    """Configuration for text tokenization and vocabulary management.

    This configuration manages tokenizer settings including pre-trained models,
    vocabulary files, and word embeddings. It supports various tokenization
    approaches from simple vocabulary-based to advanced pre-trained tokenizers.

    Attributes:
        w2v_model: Path to Word2Vec model file for embedding initialization.
        embedding_file: Path to pre-computed embedding file.
        vocab_file: Path to vocabulary file containing token mappings.
        pretrained_tokenizer_file: Path to pre-trained tokenizer model.
        alias_file: Path to file containing token aliases or synonyms.
    """

    w2v_model: str = ConfigField(
        default=None, constraint=OptionalConstraint(TypeConstraint(str)), doc="Word2Vec模型路径"
    )
    embedding_file: str = ConfigField(
        default=None, constraint=OptionalConstraint(TypeConstraint(str)), doc="词嵌入文件路径"
    )
    vocab_file: str = ConfigField(
        default=None, constraint=OptionalConstraint(TypeConstraint(str)), doc="词汇表文件路径"
    )
    pretrained_tokenizer_file: str = ConfigField(
        default=None, constraint=OptionalConstraint(TypeConstraint(str)), doc="预训练分词器文件路径"
    )
    alias_file: str = ConfigField(default=None, constraint=OptionalConstraint(TypeConstraint(str)), doc="别名文件路径")


class DatasetConfig(Config):
    """Configuration for dataset creation and preprocessing.

    This configuration manages dataset-related settings including data sources,
    field mappings, and preprocessing options. It focuses on "what the data is"
    rather than "how to load it in batches".

    Attributes:
        data_path: Path to data file or directory containing training data.
        data_format: Format of the input data files.
        id_key: Field name for unique identifier in the dataset.
        portfolio_key: Field name for portfolio data in the dataset.
        proportion1_key: Field name for shareholding proportion relative to total shares.
        proportion2_key: Field name for shareholding proportion relative to tradable A-shares.
        include_proportion: Whether to include proportion information in data.
        max_length: Maximum sequence length for truncation/padding.
        mask_prob: Probability of masking tokens for masked language modeling.
        mask_indices: Specific indices to mask (alternative to probability-based masking).
        cache_size: Number of data files to cache in memory for streaming.
        num_repeats: Number of times to repeat the dataset.
    """

    data_path: Union[str, List[str]] = ConfigField(
        default=None, required=True, preprocessor=path_preprocessor, doc="数据文件, 文件夹路径或数据文件路径列表"
    )
    data_format: Literal["csv", "json", "binary"] = ConfigField(
        default="csv", constraint=ChoiceConstraint(["csv", "json", "binary"]), doc="数据文件格式"
    )
    id_key: str = ConfigField(
        default="InvestorID", constraint=TypeConstraint(str), doc="ID字段名,用于区分不同Portfolio"
    )
    portfolio_key: str = ConfigField(default="Portfolio", constraint=TypeConstraint(str), doc="Portfolio字段名")
    proportion1_key: str = ConfigField(
        default="Proportion1",
        constraint=TypeConstraint(str),
        doc="持股比例(相对总股本)字段名,当`include_proportion`为True时使用",
    )
    proportion2_key: str = ConfigField(
        default="Proportion2",
        constraint=TypeConstraint(str),
        doc="持股比例(相对流通A股)字段名,当`include_proportion`为True时使用",
    )
    include_proportion: bool = ConfigField(default=False, constraint=TypeConstraint(bool), doc="是否包含比例信息")
    max_length: int = ConfigField(default=50, constraint=TypeConstraint(int), doc="最大序列长度")
    mask_prob: float = ConfigField(
        default=0.15,
        constraint=OptionalConstraint(RangeConstraint(0.0, 1.0)),
        doc="掩码概率，与`mask_indices`字段不兼容",
    )
    mask_indices: Optional[List[int]] = ConfigField(
        default=None, constraint=OptionalConstraint(TypeConstraint(list)), doc="指定掩码位置，与`mask_prob`字段不兼容"
    )
    num_repeats: int = ConfigField(default=1, constraint=TypeConstraint(int), doc="数据集重复倍数")
    cache_size: int = ConfigField(default=10, constraint=TypeConstraint(int), doc="数据缓存文件数（适配流式加载）")

    _cross_constraints = [
        MutualExclusionConstraint("mask_prob", "mask_indices", name="mask_mode_exclusion"),
    ]


class DataLoaderConfig(Config):
    """Configuration for PyTorch DataLoader.

    This configuration manages DataLoader-related settings focusing on
    "how to load data in batches" rather than dataset creation.

    Attributes:
        batch_size: Number of samples per batch for training/inference.
        shuffle: Whether to shuffle the dataset during loading.
        num_workers: Number of worker processes for data loading.
        pin_memory: Whether to pin memory for faster GPU transfer.
    """

    batch_size: int = ConfigField(default=32, constraint=TypeConstraint(int), doc="批次大小")
    shuffle: bool = ConfigField(default=True, constraint=TypeConstraint(bool), doc="是否打乱数据")
    num_workers: int = ConfigField(default=4, constraint=TypeConstraint(int), doc="工作进程数")
    persistent_workers: bool = ConfigField(
        default=True, constraint=TypeConstraint(bool), doc="是否保持工作进程存活以重用（PyTorch 1.7+）"
    )
    pin_memory: bool = ConfigField(
        default=True, constraint=TypeConstraint(bool), doc="是否将数据固定在内存中（适用于 GPU 加速）"
    )
    drop_last: bool = ConfigField(default=False, constraint=TypeConstraint(bool), doc="是否丢弃最后一个不完整的批次")

    _cross_constraints = [
        LambdaCrossConstraint(
            lambda cfg: not (cfg.num_workers == 0 and cfg.persistent_workers),
            error="persistent_workers must be False when num_workers=0",
            name="persistent_workers_guard",
        ),
    ]


class AcceleratorConfig(Config):
    """Configuration for HuggingFace Accelerator with torch.compile support.

    This configuration manages Accelerator initialization, mixed precision,
    gradient accumulation, and torch.compile settings. It is designed to be
    used with Accelerator_Preparer.

    Attributes:
        mixed_precision: Mixed precision training mode.
        gradient_accumulation_steps: Number of steps for gradient accumulation.
        accelerator_checkpoint: Path to HuggingFace Accelerator checkpoint.
        use_compile: Whether to enable torch.compile.
        compile_backend: torch.compile backend (inductor, eager, aot_eager).
        compile_mode: torch.compile optimization mode.
        compile_fullgraph: Whether to require full graph compilation (no graph breaks).
        compile_dynamic: Whether to enable dynamic shapes.
        compile_cache_dir: Persistent compile cache directory.
        allow_tf32: Whether to enable TF32 acceleration on Ampere/Ada GPUs.
    """

    mixed_precision: Literal["auto", "no", "fp16", "bf16", "fp8"] = ConfigField(
        default="no", constraint=ChoiceConstraint(["auto", "no", "fp16", "bf16", "fp8"]), doc="混合精度训练模式"
    )
    gradient_accumulation_steps: int = ConfigField(
        default=1, constraint=TypeConstraint(int) & RangeConstraint(1, None), doc="梯度累积步数"
    )
    accelerator_checkpoint: Optional[str] = ConfigField(
        default=None,
        constraint=OptionalConstraint(TypeConstraint(str)),
        preprocessor=path_preprocessor,
        doc="HF Accelerator检查点路径",
    )
    use_compile: bool = ConfigField(default=False, constraint=TypeConstraint(bool), doc="是否启用 torch.compile")
    compile_backend: str = ConfigField(
        default="inductor", constraint=ChoiceConstraint(["inductor", "eager", "aot_eager"]), doc="torch.compile 后端"
    )
    compile_mode: str = ConfigField(
        default="reduce-overhead",
        constraint=ChoiceConstraint(["default", "reduce-overhead", "max-autotune"]),
        doc="torch.compile 优化模式",
    )
    compile_fullgraph: bool = ConfigField(
        default=False, constraint=TypeConstraint(bool), doc="是否要求完整图编译（不允许 graph break）"
    )
    compile_dynamic: bool = ConfigField(
        default=False, constraint=TypeConstraint(bool), doc="是否启用动态 shape（batch size 变化时需开启）"
    )
    compile_cache_dir: Optional[str] = ConfigField(
        default=None,
        constraint=OptionalConstraint(TypeConstraint(str)),
        preprocessor=path_preprocessor,
        doc="torch.compile 持久化缓存目录，None 使用 PyTorch 默认",
    )
    allow_tf32: bool = ConfigField(
        default=True, constraint=TypeConstraint(bool), doc="是否启用 TF32 加速（Ampere/Ada 架构 GPU）"
    )


class OptimizerConfig(Config):
    """Configuration for model optimization including optimizer and learning rate scheduling.

    This configuration manages optimizer selection, learning rate scheduling, and related
    hyperparameters for training machine learning models. It supports various optimizers
    including 8-bit variants for memory efficiency and multiple learning rate schedules.

    Attributes:
        optimizer_type: Type of optimizer algorithm to use.
        optimizer_kwargs: Additional optimizer-specific parameters.
        learning_rate: Initial learning rate for training.
        lr_scheduler_type: Type of learning rate scheduling strategy.
        lr_scheduler_warmup_steps: Number of warmup steps for learning rate.
        lr_scheduler_train_steps: Total training steps for scheduling.
        lr_scheduler_num_cycles: Number of cycles for cosine scheduling.
        lr_scheduler_power: Power parameter for polynomial scheduling.
    """

    optimizer_type: Literal[
        "Adam",
        "Adam8bit",
        "PagedAdam8bit",
        "AdamW",
        "AdamW8bit",
        "PagedAdamW8bit",
        "Lion",
        "Lion8bit",
        "PagedLion8bit",
        "SGDNesterov",
        "SGD8bit",
        "DAdaptation",
        "Adafactor",
    ] = ConfigField(
        default="Adam",
        constraint=ChoiceConstraint(
            [
                "Adam",
                "Adam8bit",
                "PagedAdam8bit",
                "AdamW",
                "AdamW8bit",
                "PagedAdamW8bit",
                "Lion",
                "Lion8bit",
                "PagedLion8bit",
                "SGDNesterov",
                "SGD8bit",
                "DAdaptation",
                "Adafactor",
            ]
        ),
        doc="优化器类型",
        required=True,
    )
    optimizer_kwargs: Optional[dict] = ConfigField(
        default=None,
        constraint=OptionalConstraint(TypeConstraint(dict)),
        doc=r"可选优化器附加参数。未指定的则使用默认参数。默认为`null`",
    )
    learning_rate: float = ConfigField(default=1e-3, constraint=RangeConstraint(1e-5, 1e-1), doc="初始学习率")
    lr_scheduler_type: Literal["constant", "linear", "cosine", "cosine_with_restarts", "polynomial", "adafactor"] = (
        ConfigField(
            default="linear",
            constraint=ChoiceConstraint(
                ["constant", "linear", "cosine", "cosine_with_restarts", "polynomial", "adafactor"]
            ),
            doc="学习率调度器类型，支持`constant`, `linear`, `cosine`, `cosine_with_restarts`, `polynomial` 和`adafactor`",
        )
    )
    lr_scheduler_warmup_steps: int = ConfigField(
        default=0, constraint=TypeConstraint(int), doc="学习率预热步数，对除`adafactor`以外的所有类型生效"
    )
    lr_scheduler_train_steps: int = ConfigField(
        default=None,
        constraint=OptionalConstraint(TypeConstraint(int)),
        doc=r"总训练步数，设为`null`则自动获取： $\text{steps} = \text{len(dataloader)} \times \text{epoches}$",
    )
    lr_scheduler_num_cycles: float = ConfigField(
        default=None,
        constraint=OptionalConstraint(TypeConstraint(Number)),
        doc="适用于 `cosine` 与`cosine_with_restarts`调度器的周期数",
    )
    lr_scheduler_power: float = ConfigField(
        default=None, constraint=OptionalConstraint(TypeConstraint(Number)), doc="适用于 `polynomial` 调度器的幂指数"
    )

    _cross_constraints = [
        LambdaCrossConstraint(
            lambda cfg: cfg.lr_scheduler_type != "adafactor" or cfg.optimizer_type == "Adafactor",
            error="lr_scheduler_type='adafactor' requires optimizer_type='Adafactor'",
            name="adafactor_scheduler_requires_optimizer",
        ),
    ]


class AssetBERTModelConfig(Config):
    """Configuration for AssetBERT model architecture and initialization.

    This configuration defines the architecture and initialization parameters for
    BERT models adapted for financial asset sequences. It supports various embedding
    initialization strategies and model freezing options for transfer learning.

    Attributes:
        model_type: Type identifier for the model architecture.
        model_checkpoint: Path to pre-trained model checkpoint.
        w2v_model: Path to Word2Vec model for embedding initialization.
        embedding_file: Path to embedding file for initialization.
        vocab_size: Size of the vocabulary.
        hidden_size: Dimensionality of hidden layers.
        num_hidden_layers: Number of transformer layers.
        num_attention_heads: Number of attention heads per layer.
        intermediate_size: Size of feed-forward intermediate layer.
        max_position_embeddings: Maximum sequence length.
        type_vocab_size: Number of token types (usually 1 for single sequences).
        freeze_embedding: Whether to freeze embedding layer during training.
        freeze_encoder: Whether to freeze encoder layers during training.
    """

    model_type: str = ConfigField(default="BERT", constraint=ChoiceConstraint(["BERT"]))
    model_checkpoint: str = ConfigField(
        default=None,
        constraint=OptionalConstraint(TypeConstraint(str)),
        preprocessor=path_preprocessor,
        doc="预训练模型检查点",
    )
    w2v_model: str = ConfigField(
        default=None,
        constraint=OptionalConstraint(TypeConstraint(str)),
        preprocessor=path_preprocessor,
        doc="Word2Vec模型路径",
    )
    embedding_file: str = ConfigField(
        default=None,
        constraint=OptionalConstraint(TypeConstraint(str)),
        preprocessor=path_preprocessor,
        doc="嵌入文件路径",
    )
    vocab_size: int = ConfigField(default=5521, constraint=TypeConstraint(int), doc="词汇表大小", required=True)
    hidden_size: int = ConfigField(default=100, constraint=TypeConstraint(int), doc="隐藏层大小", required=True)
    num_hidden_layers: int = ConfigField(
        default=4, constraint=TypeConstraint(int), doc="Transformer层数", required=True
    )
    num_attention_heads: int = ConfigField(default=2, constraint=TypeConstraint(int), doc="注意力头数", required=True)
    intermediate_size: int = ConfigField(default=512, constraint=TypeConstraint(int), doc="中间层大小", required=True)
    max_position_embeddings: int = ConfigField(
        default=64, constraint=TypeConstraint(int), doc="最大位置嵌入(序列的最大长度)", required=True
    )
    type_vocab_size: int = ConfigField(default=1, constraint=TypeConstraint(int), doc="类型词汇表大小", required=True)
    freeze_embedding: bool = ConfigField(default=False, constraint=TypeConstraint(bool), doc="是否冻结嵌入层")
    freeze_encoder: bool = ConfigField(default=False, constraint=TypeConstraint(bool), doc="是否冻结编码器")
    embedding_model: str = ConfigField(
        default=None,
        constraint=OptionalConstraint(TypeConstraint(str)),
        doc="上游嵌入模型名称 (e.g., 'AssetBERT', 'AssetW2V', 'RS_Binary', 'Chars')",
    )
