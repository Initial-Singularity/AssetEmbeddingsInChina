"""Component configs shared by the training and evaluation scripts.

Holds the logger, tokenizer, dataset, dataloader, accelerator and optimizer configs and the AssetBERT and
SimplifiedMLM model configs that the preparers turn into runtime objects. Every field is documented through
its `ConfigField(doc=...)` string, which also renders into the CLI help and the config reference.
"""

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


def path_preprocessor(p: Union[str, List[str], None]) -> Union[str, List[str], None]:
    """Normalize a path field value, passing `None` through and mapping over lists.

    Args:
        p: A path, a list of paths, or `None`.

    Returns:
        The same shape with every path passed through `os.path.normpath`.
    """
    if p is None:
        return None
    if isinstance(p, list):
        return [os.path.normpath(x) for x in p]
    return os.path.normpath(p)


class LoggerConfig(Config):
    """The parameter object of `asset_embeddings.logger.configure_logging`.

    Describes the process-wide logging setup: the console channel (tqdm-safe terminal, plain stdout or notebook
    HTML) with its level and colors, and the log file(s) the process appends to. Loggers themselves are the
    standard module hierarchy (`logging.getLogger(__name__)`); this object never names one.
    """

    log_file: Union[str, List[str], None] = ConfigField(
        default=None,
        constraint=OptionalConstraint(TypeConstraint(str) | TypeConstraint(list)),
        doc="Path of the log file, or a list of paths; None keeps console output only",
    )
    log_format: str = ConfigField(
        default="%(asctime)s [%(short)s][%(levelname)s]: %(message)s",
        constraint=TypeConstraint(str),
        doc="Log record format string",
    )
    date_format: str = ConfigField(
        default="%Y-%m-%d %H:%M:%S", constraint=TypeConstraint(str), doc="Timestamp format string"
    )
    console_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = ConfigField(
        default="INFO",
        constraint=ChoiceConstraint(["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]),
        doc="Minimum level shown on the console",
    )
    console_stream: Literal["tqdm", "std", "html"] = ConfigField(
        default="tqdm",
        constraint=ChoiceConstraint(["tqdm", "std", "html"]),
        doc="Console output stream: tqdm, std or html",
    )
    file_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = ConfigField(
        default="DEBUG",
        constraint=ChoiceConstraint(["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]),
        doc="Minimum level written to the log file",
    )
    enable_colors: bool = ConfigField(
        default=True, constraint=TypeConstraint(bool), doc="Whether console output is colored"
    )
    color_target: Literal["message", "levelname", "format"] = ConfigField(
        default="format",
        constraint=ChoiceConstraint(["message", "levelname", "format"]),
        doc="What gets colored: message (message only), levelname (level name only) or format (every component of the format string)",
    )


class TokenizerConfig(Config):
    """Configuration for text tokenization and vocabulary management.

    Names the files a tokenizer can be built from: a Word2Vec model, a pre-computed embedding file, a plain
    vocabulary file, a pretrained tokenizer file or an alias file. Which of them applies depends on the
    tokenizer the preparer builds.
    """

    w2v_model: str = ConfigField(
        default=None, constraint=OptionalConstraint(TypeConstraint(str)), doc="Path of the Word2Vec model"
    )
    embedding_file: str = ConfigField(
        default=None, constraint=OptionalConstraint(TypeConstraint(str)), doc="Path of the embedding file"
    )
    vocab_file: str = ConfigField(
        default=None, constraint=OptionalConstraint(TypeConstraint(str)), doc="Path of the vocabulary file"
    )
    pretrained_tokenizer_file: str = ConfigField(
        default=None,
        constraint=OptionalConstraint(TypeConstraint(str)),
        doc="Path of the pretrained tokenizer file",
    )
    alias_file: str = ConfigField(
        default=None, constraint=OptionalConstraint(TypeConstraint(str)), doc="Path of the alias file"
    )


class DatasetConfig(Config):
    """Configuration for dataset creation and preprocessing.

    Covers what the data is: the data sources (portfolio CSV files or a corpus directory built by ``data.corpus``),
    the column names of the portfolio table and the masking options. How the data is loaded in batches belongs to `DataLoaderConfig`.

    Note:
        `mask_prob` and `mask_indices` are mutually exclusive; the cross constraint rejects a config that
        sets both.
    """

    data_path: Union[str, List[str]] = ConfigField(
        default=None,
        required=True,
        preprocessor=path_preprocessor,
        doc="Data file, directory, or list of data file paths",
    )
    data_format: Literal["csv"] = ConfigField(
        default="csv",
        constraint=ChoiceConstraint(["csv"]),
        doc="Format of the data files; only 'csv' (the datasets count records as csv lines)",
    )
    id_key: str = ConfigField(
        default="InvestorID",
        constraint=TypeConstraint(str),
        doc="Name of the ID column that distinguishes portfolios",
    )
    portfolio_key: str = ConfigField(
        default="Portfolio", constraint=TypeConstraint(str), doc="Name of the portfolio column"
    )
    proportion1_key: str = ConfigField(
        default="Proportion1",
        constraint=TypeConstraint(str),
        doc="Name of the holding-proportion column relative to total shares; used when `include_proportion` is True",
    )
    proportion2_key: str = ConfigField(
        default="Proportion2",
        constraint=TypeConstraint(str),
        doc="Name of the holding-proportion column relative to tradable A-shares; used when `include_proportion` is True",
    )
    include_proportion: bool = ConfigField(
        default=False, constraint=TypeConstraint(bool), doc="Whether holding proportions are included"
    )
    max_length: int = ConfigField(default=50, constraint=TypeConstraint(int), doc="Maximum sequence length")
    mask_prob: float = ConfigField(
        default=0.15,
        constraint=OptionalConstraint(RangeConstraint(0.0, 1.0)),
        doc="Masking probability; incompatible with `mask_indices`",
    )
    mask_indices: Optional[List[int]] = ConfigField(
        default=None,
        constraint=OptionalConstraint(TypeConstraint(list)),
        doc="Fixed positions to mask; incompatible with `mask_prob`",
    )
    num_repeats: int = ConfigField(
        default=1,
        constraint=TypeConstraint(int) & RangeConstraint(1, None),
        doc="Times the training split is replayed per epoch; the validation and test splits are not repeated",
    )

    _cross_constraints = [
        MutualExclusionConstraint("mask_prob", "mask_indices", name="mask_mode_exclusion"),
    ]


class DataLoaderConfig(Config):
    """Configuration for the PyTorch DataLoader.

    Covers how the data is loaded in batches (batch size, shuffling, worker processes, memory pinning), as
    opposed to what the data is, which `DatasetConfig` holds.

    Note:
        PyTorch rejects `persistent_workers=True` with `num_workers=0`; the cross constraint catches that
        combination up front.
    """

    batch_size: int = ConfigField(default=32, constraint=TypeConstraint(int), doc="Batch size")
    shuffle: bool = ConfigField(default=True, constraint=TypeConstraint(bool), doc="Whether the data is shuffled")
    num_workers: int = ConfigField(default=4, constraint=TypeConstraint(int), doc="Number of loader worker processes")
    persistent_workers: bool = ConfigField(
        default=True,
        constraint=TypeConstraint(bool),
        doc="Whether worker processes stay alive for reuse between epochs (PyTorch 1.7+)",
    )
    pin_memory: bool = ConfigField(
        default=True,
        constraint=TypeConstraint(bool),
        doc="Whether batches are pinned in memory for faster GPU transfer",
    )
    drop_last: bool = ConfigField(
        default=False, constraint=TypeConstraint(bool), doc="Whether the last incomplete batch is dropped"
    )

    _cross_constraints = [
        LambdaCrossConstraint(
            lambda cfg: not (cfg.num_workers == 0 and cfg.persistent_workers),
            error="persistent_workers must be False when num_workers=0",
            name="persistent_workers_guard",
        ),
    ]


class AcceleratorConfig(Config):
    """Configuration for the HuggingFace Accelerator, including torch.compile.

    Covers Accelerator initialization, mixed precision, gradient accumulation, checkpoint restore and the
    torch.compile and TF32 switches. Consumed by `Accelerator_Preparer`.
    """

    mixed_precision: Literal["auto", "no", "fp16", "bf16", "fp8"] = ConfigField(
        default="no",
        constraint=ChoiceConstraint(["auto", "no", "fp16", "bf16", "fp8"]),
        doc="Mixed-precision training mode",
    )
    gradient_accumulation_steps: int = ConfigField(
        default=1,
        constraint=TypeConstraint(int) & RangeConstraint(1, None),
        doc="Number of gradient accumulation steps",
    )
    accelerator_checkpoint: Optional[str] = ConfigField(
        default=None,
        constraint=OptionalConstraint(TypeConstraint(str)),
        preprocessor=path_preprocessor,
        doc="Path of the HF Accelerator checkpoint",
    )
    use_compile: bool = ConfigField(
        default=False, constraint=TypeConstraint(bool), doc="Whether torch.compile is enabled"
    )
    compile_backend: str = ConfigField(
        default="inductor",
        constraint=ChoiceConstraint(["inductor", "eager", "aot_eager"]),
        doc="torch.compile backend",
    )
    compile_mode: str = ConfigField(
        default="reduce-overhead",
        constraint=ChoiceConstraint(["default", "reduce-overhead", "max-autotune"]),
        doc="torch.compile optimization mode",
    )
    compile_fullgraph: bool = ConfigField(
        default=False,
        constraint=TypeConstraint(bool),
        doc="Whether full-graph compilation is required (no graph breaks)",
    )
    compile_dynamic: bool = ConfigField(
        default=False,
        constraint=TypeConstraint(bool),
        doc="Whether dynamic shapes are enabled (needed when the batch size varies)",
    )
    compile_cache_dir: Optional[str] = ConfigField(
        default=None,
        constraint=OptionalConstraint(TypeConstraint(str)),
        preprocessor=path_preprocessor,
        doc="Persistent torch.compile cache directory; None uses the PyTorch default",
    )
    allow_tf32: bool = ConfigField(
        default=True,
        constraint=TypeConstraint(bool),
        doc="Whether TF32 acceleration is enabled (Ampere/Ada GPUs)",
    )


class OptimizerConfig(Config):
    """Configuration for the optimizer and the learning-rate schedule.

    Selects the optimizer (including the 8-bit and paged variants) with its extra keyword arguments, the
    initial learning rate and the scheduler with its warmup, step count, cycle and power parameters.

    Note:
        The `adafactor` scheduler only works with the `Adafactor` optimizer; the cross constraint rejects
        any other pairing.
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
        doc="Optimizer type",
        required=True,
    )
    optimizer_kwargs: Optional[dict] = ConfigField(
        default=None,
        constraint=OptionalConstraint(TypeConstraint(dict)),
        doc="Optional extra optimizer keyword arguments; unspecified ones keep their defaults, default `null`",
    )
    learning_rate: float = ConfigField(
        default=1e-3, constraint=RangeConstraint(1e-5, 1e-1), doc="Initial learning rate"
    )
    lr_scheduler_type: Literal["constant", "linear", "cosine", "cosine_with_restarts", "polynomial", "adafactor"] = (
        ConfigField(
            default="linear",
            constraint=ChoiceConstraint(
                ["constant", "linear", "cosine", "cosine_with_restarts", "polynomial", "adafactor"]
            ),
            doc="Learning-rate scheduler type: `constant`, `linear`, `cosine`, `cosine_with_restarts`, `polynomial` or `adafactor`",
        )
    )
    lr_scheduler_warmup_steps: int = ConfigField(
        default=0,
        constraint=TypeConstraint(int),
        doc="Number of learning-rate warmup steps; applies to every scheduler type except `adafactor`",
    )
    lr_scheduler_train_steps: int = ConfigField(
        default=None,
        constraint=OptionalConstraint(TypeConstraint(int)),
        doc=r"Total training steps; `null` derives them as $\text{steps} = \text{len(dataloader)} \times \text{epochs}$",
    )
    lr_scheduler_num_cycles: float = ConfigField(
        default=None,
        constraint=OptionalConstraint(TypeConstraint(Number)),
        doc="Number of cycles for the `cosine` and `cosine_with_restarts` schedulers",
    )
    lr_scheduler_power: float = ConfigField(
        default=None,
        constraint=OptionalConstraint(TypeConstraint(Number)),
        doc="Power exponent for the `polynomial` scheduler",
    )

    _cross_constraints = [
        LambdaCrossConstraint(
            lambda cfg: cfg.lr_scheduler_type != "adafactor" or cfg.optimizer_type == "Adafactor",
            error="lr_scheduler_type='adafactor' requires optimizer_type='Adafactor'",
            name="adafactor_scheduler_requires_optimizer",
        ),
    ]


class AssetBERTModelConfig(Config):
    """Configuration for the AssetBERT model architecture and initialization.

    Defines the BERT architecture adapted to portfolio sequences (vocabulary, width, depth, heads, positions)
    and how its weights start: from a checkpoint, a Word2Vec model or an embedding file, with the embedding
    layer or the encoder optionally frozen for transfer learning.
    """

    model_type: str = ConfigField(default="BERT", constraint=ChoiceConstraint(["BERT"]), doc="Model type identifier")
    model_checkpoint: str = ConfigField(
        default=None,
        constraint=OptionalConstraint(TypeConstraint(str)),
        preprocessor=path_preprocessor,
        doc="Pretrained model checkpoint path",
    )
    w2v_model: str = ConfigField(
        default=None,
        constraint=OptionalConstraint(TypeConstraint(str)),
        preprocessor=path_preprocessor,
        doc="Path of the Word2Vec model",
    )
    embedding_file: str = ConfigField(
        default=None,
        constraint=OptionalConstraint(TypeConstraint(str)),
        preprocessor=path_preprocessor,
        doc="Path of the embedding file",
    )
    vocab_size: int = ConfigField(default=5521, constraint=TypeConstraint(int), doc="Vocabulary size", required=True)
    hidden_size: int = ConfigField(default=100, constraint=TypeConstraint(int), doc="Hidden layer size", required=True)
    num_hidden_layers: int = ConfigField(
        default=4, constraint=TypeConstraint(int), doc="Number of Transformer layers", required=True
    )
    num_attention_heads: int = ConfigField(
        default=2, constraint=TypeConstraint(int), doc="Number of attention heads", required=True
    )
    intermediate_size: int = ConfigField(
        default=512, constraint=TypeConstraint(int), doc="Feed-forward intermediate layer size", required=True
    )
    max_position_embeddings: int = ConfigField(
        default=64,
        constraint=TypeConstraint(int),
        doc="Maximum position embeddings (the maximum sequence length)",
        required=True,
    )
    type_vocab_size: int = ConfigField(
        default=1, constraint=TypeConstraint(int), doc="Token-type vocabulary size", required=True
    )
    freeze_embedding: bool = ConfigField(
        default=False, constraint=TypeConstraint(bool), doc="Whether the embedding layer is frozen"
    )
    freeze_encoder: bool = ConfigField(
        default=False, constraint=TypeConstraint(bool), doc="Whether the encoder is frozen"
    )
    embedding_model: str = ConfigField(
        default=None,
        constraint=OptionalConstraint(TypeConstraint(str)),
        doc="Name of the upstream embedding model (e.g., 'AssetBERT', 'AssetW2V', 'RS_Binary', 'Chars')",
    )


class SimplifiedMLMModelConfig(Config):
    """Configuration for the simplified masked-language-model architecture.

    Defines the lightweight MLM (an embedding layer and a prediction head, no Transformer encoder) used
    where full BERT training is too costly, together with its initialization from a checkpoint or an
    embedding file and the freezing switches for the embedding layer and the head.
    """

    model_type: str = ConfigField(
        default="SimplifiedMLM", constraint=ChoiceConstraint(["SimplifiedMLM"]), doc="Model type identifier"
    )
    model_checkpoint: str = ConfigField(
        default=None,
        constraint=OptionalConstraint(TypeConstraint(str)),
        preprocessor=path_preprocessor,
        doc="Pretrained model checkpoint path",
    )
    embedding_file: str = ConfigField(
        default=None,
        constraint=OptionalConstraint(TypeConstraint(str)),
        preprocessor=path_preprocessor,
        doc="Path of the embedding file",
    )
    vocab_size: int = ConfigField(default=5521, constraint=TypeConstraint(int), doc="Vocabulary size", required=True)
    hidden_size: int = ConfigField(default=100, constraint=TypeConstraint(int), doc="Hidden layer size", required=True)
    max_position_embeddings: int = ConfigField(
        default=64,
        constraint=TypeConstraint(int),
        doc="Maximum position embeddings (the maximum sequence length)",
        required=True,
    )
    type_vocab_size: int = ConfigField(
        default=1, constraint=TypeConstraint(int), doc="Token-type vocabulary size", required=True
    )
    freeze_embedding: bool = ConfigField(
        default=False, constraint=TypeConstraint(bool), doc="Whether the embedding layer is frozen"
    )
    freeze_head: bool = ConfigField(
        default=False, constraint=TypeConstraint(bool), doc="Whether the MLM prediction head is frozen"
    )
    embedding_model: str = ConfigField(
        default=None,
        constraint=OptionalConstraint(TypeConstraint(str)),
        doc="Name of the upstream embedding model (e.g., 'AssetBERT', 'AssetW2V', 'RS_Binary', 'Chars')",
    )
