"""Configs of the training runs: the RS and Word2Vec trainers, and the masked-LM loop of AssetBERT and MP.

`MLMLoopConfig` declares the epoch loop shared by every `BaseMLMTrainer`; `AssetBERTTrainConfig` (pretraining and
finetuning) and `MPTrainConfig` (both MP trainers) add their own fields. `RunMetadataConfig` is the `run` block of
the MP trainers. Every field is documented by its `ConfigField(doc=...)` string.
"""

from typing import List, Literal, Optional, Union

from .base import Config, ConfigField
from .constraints import (
    ChoiceConstraint,
    LambdaCrossConstraint,
    OptionalConstraint,
    RangeConstraint,
    TypeConstraint,
    UnionConstraints,
)
from .object_configs import path_preprocessor

#: The selection metrics, each with the split field that must be positive for its loss to exist.
METRIC_SPLITS = {"train_loss": None, "val_loss": "validation_split", "test_loss": "test_split"}


def split_exists(cfg: Config, metric: str) -> bool:
    """Whether ``cfg`` holds out the split ``metric`` reads (always true for ``train_loss``).

    Args:
        cfg: A loop config.
        metric: One of `METRIC_SPLITS`.

    Returns:
        True when the metric needs no split or its split field is positive.
    """
    split = METRIC_SPLITS[metric]
    return split is None or (getattr(cfg, split, None) or 0) > 0


class AssetRSTrainConfig(Config):
    """Config of the recommender-system (PCA) training run.

    Loaded by the RS trainer: the holdings matrix variant to build, the PCA that turns it into embeddings, the
    data columns to read and where the fitted model is saved.
    """

    model_type: Literal["RS_Binary", "RS_Ranks", "RS_Level0", "RS_LevelMin"] = ConfigField(
        constraint=ChoiceConstraint(["RS_Binary", "RS_Ranks", "RS_Level0", "RS_LevelMin"]),
        doc="Recommender-system variant to train.",
        required=True,
    )
    n_components: int = ConfigField(
        default=10,
        constraint=TypeConstraint(int),
        doc="Number of PCA components kept, i.e. the embedding dimension.",
        required=True,
    )
    whiten: bool = ConfigField(
        default=False, constraint=TypeConstraint(bool), doc="Whether to whiten the PCA components."
    )
    data_path: str = ConfigField(
        default=None,
        constraint=TypeConstraint(str),
        preprocessor=path_preprocessor,
        doc="Path to the training data: a single file or a directory.",
        required=True,
    )
    data_format: str = ConfigField(
        default="csv",
        constraint=ChoiceConstraint(["csv"]),
        doc="Format of the data files; only 'csv'.",
        required=True,
    )
    id_key: str = ConfigField(
        default="InvestorID", constraint=TypeConstraint(str), doc="Name of the investor ID column in the data files."
    )
    portfolio_key: str = ConfigField(
        default="Portfolio", constraint=TypeConstraint(str), doc="Name of the portfolio column in the data files."
    )
    proportion1_key: str = ConfigField(
        default="Proportion1",
        constraint=TypeConstraint(str),
        doc="Name of the first holding-proportion column in the data files.",
    )
    proportion2_key: str = ConfigField(
        default="Proportion2",
        constraint=TypeConstraint(str),
        doc="Name of the second holding-proportion column in the data files.",
    )
    save_folder: str = ConfigField(
        default=None,
        constraint=TypeConstraint(str),
        preprocessor=path_preprocessor,
        doc="Directory the model is saved to.",
        required=True,
    )
    save_name: str = ConfigField(
        default=None,
        constraint=TypeConstraint(str),
        doc="File name of the saved model, without extension.",
        required=True,
    )
    save_format: str = ConfigField(
        default=".pkl",
        constraint=ChoiceConstraint([".pkl"]),
        doc="Serialization format of the saved model.",
        required=True,
    )
    seed: int = ConfigField(default=42, constraint=TypeConstraint(int), doc="Random seed.")


class AssetW2VTrainConfig(Config):
    """Config of the Word2Vec training run.

    Loaded by the W2V trainer: the gensim hyperparameters, the portfolio data to read, and an optional
    `pretrained_model` to fine-tune instead of training from scratch.
    """

    pretrained_model: str = ConfigField(
        default=None,
        constraint=OptionalConstraint(TypeConstraint(str)),
        preprocessor=path_preprocessor,
        doc="Pretrained Word2Vec model to fine-tune: a .model file saved by an earlier run of this script.",
    )
    embedding_dim: int = ConfigField(
        default=100, constraint=TypeConstraint(int), doc="Dimension of the token embeddings.", required=True
    )
    epochs: int = ConfigField(
        default=100, constraint=TypeConstraint(int), doc="Number of training epochs.", required=True
    )
    window: int = ConfigField(default=5, constraint=TypeConstraint(int), doc="Context window size.", required=True)
    min_count: int = ConfigField(
        default=5,
        constraint=TypeConstraint(int),
        doc="Minimum token frequency; rarer tokens are dropped from the vocabulary.",
        required=True,
    )
    sg: int = ConfigField(
        default=1, constraint=TypeConstraint(int), doc="Model architecture: 1 for skip-gram, 0 for CBOW.", required=True
    )
    sample: float = ConfigField(
        default=1e-3,
        constraint=TypeConstraint(float),
        doc="Subsampling threshold for frequent tokens (gensim sample).",
        required=True,
    )
    negative_sample: int = ConfigField(
        default=5, constraint=TypeConstraint(int), doc="Number of negative samples per positive pair.", required=True
    )
    data_path: Union[str, List[str]] = ConfigField(
        default=None,
        constraint=OptionalConstraint(UnionConstraints(TypeConstraint(str), TypeConstraint(list))),
        preprocessor=path_preprocessor,
        doc="Path to the training data: a single file, a directory or a list of file paths.",
        required=True,
    )
    data_format: str = ConfigField(
        default="csv",
        constraint=ChoiceConstraint(["csv"]),
        doc="Format of the data files; only 'csv'.",
        required=True,
    )
    id_key: str = ConfigField(
        default="InvestorID", constraint=TypeConstraint(str), doc="Name of the ID column that distinguishes portfolios."
    )
    portfolio_key: str = ConfigField(
        default="Portfolio", constraint=TypeConstraint(str), doc="Name of the portfolio column."
    )
    workers: int = ConfigField(
        default=1,
        constraint=TypeConstraint(int),
        doc="Number of training threads; a fully reproducible run needs workers=1 to remove ordering jitter.",
        required=True,
    )
    save_folder: str = ConfigField(
        default=None,
        constraint=TypeConstraint(str),
        preprocessor=path_preprocessor,
        doc="Directory the model is saved to.",
        required=True,
    )
    save_name: str = ConfigField(
        default=None,
        constraint=TypeConstraint(str),
        doc="Base name of the saved model, written as <save_name>.model.",
        required=True,
    )
    seed: int = ConfigField(default=42, constraint=TypeConstraint(int), doc="Random seed.")


class MLMLoopConfig(Config):
    """The epoch loop of a masked-LM trainer (its `train` block).

    Epochs, clipping, logging, saving, the validation split, best-epoch selection and early stopping. Hardware
    settings (mixed precision, torch.compile, TF32) live in `AcceleratorConfig`.

    Note:
        A metric that reads a held-out split needs that split at config time: `val_loss` a positive
        `validation_split`, `test_loss` a positive `test_split` (which only `MPTrainConfig` has).
    """

    max_epochs: int = ConfigField(
        default=30,
        constraint=TypeConstraint(int) & RangeConstraint(1, None),
        doc="Maximum number of training epochs.",
        required=True,
    )
    clip_grad_norm: Optional[float] = ConfigField(
        default=None,
        constraint=OptionalConstraint(RangeConstraint(0, None)),
        doc="Maximum gradient norm for clipping; None disables norm clipping.",
    )
    clip_grad_value: Optional[float] = ConfigField(
        default=None,
        constraint=OptionalConstraint(RangeConstraint(0, None)),
        doc="Maximum absolute gradient value for clipping; None disables value clipping.",
    )
    detect_anomaly: bool = ConfigField(
        default=False,
        constraint=TypeConstraint(bool),
        doc="Whether to enable PyTorch autograd anomaly detection; for debugging only, it about doubles the step time.",
    )
    log_every_steps: int = ConfigField(
        default=300,
        constraint=TypeConstraint(int) & RangeConstraint(1, None),
        doc="Log the training loss every n steps.",
    )
    report_every_epochs: int = ConfigField(
        default=1,
        constraint=TypeConstraint(int) & RangeConstraint(1, None),
        doc="Log the epoch losses and write them to TensorBoard every n epochs.",
    )
    save_every_epochs: int = ConfigField(
        default=1,
        constraint=TypeConstraint(int) & RangeConstraint(1, None),
        doc="Save a checkpoint (and the embeddings) every n epochs.",
    )
    save_folder: str = ConfigField(
        default=None,
        constraint=TypeConstraint(str),
        preprocessor=path_preprocessor,
        doc="Directory the checkpoints are saved to.",
        required=True,
    )
    save_name: str = ConfigField(
        default=None,
        constraint=TypeConstraint(str),
        doc="Base name of the saved checkpoints and embeddings.",
        required=True,
    )
    save_format: Literal[".pt", ".safetensors"] = ConfigField(
        default=".safetensors",
        constraint=ChoiceConstraint([".pt", ".safetensors"]),
        doc="Serialization format of the saved model.",
    )
    seed: int = ConfigField(default=42, constraint=TypeConstraint(int), doc="Random seed of the run and the split.")
    validation_split: Optional[float] = ConfigField(
        default=None,
        constraint=OptionalConstraint(RangeConstraint(0.0, 1.0)),
        doc="Share of the data held out for validation, e.g. 0.1; None or 0 means no validation split.",
    )
    split_method: Literal["sequential", "random"] = ConfigField(
        default="random",
        constraint=ChoiceConstraint(["sequential", "random"]),
        doc="How the held-out splits are taken: 'sequential' as contiguous blocks, 'random' at random.",
    )
    best_metric: Literal["train_loss", "val_loss", "test_loss"] = ConfigField(
        default="train_loss",
        constraint=ChoiceConstraint(list(METRIC_SPLITS)),
        doc="Loss that selects the best epoch: 'train_loss', 'val_loss' or 'test_loss'.",
    )
    early_stop_enabled: bool = ConfigField(
        default=False, constraint=TypeConstraint(bool), doc="Whether early stopping is enabled."
    )
    early_stop_metric: Literal["train_loss", "val_loss", "test_loss"] = ConfigField(
        default="train_loss",
        constraint=ChoiceConstraint(list(METRIC_SPLITS)),
        doc="Loss watched by early stopping: 'train_loss', 'val_loss' or 'test_loss'.",
    )
    early_stop_patience: int = ConfigField(
        default=5,
        constraint=TypeConstraint(int) & RangeConstraint(1, None),
        doc="Epochs without improvement before training stops.",
    )
    early_stop_min_delta: float = ConfigField(
        default=0.0,
        constraint=TypeConstraint(float) & RangeConstraint(0.0, None),
        doc="Minimum improvement that counts for early stopping.",
    )
    early_stop_mode: Literal["min", "max"] = ConfigField(
        default="min",
        constraint=ChoiceConstraint(["min", "max"]),
        doc="Early stopping direction: 'min' when lower is better (a loss), 'max' when higher is better.",
    )
    early_stop_relative: bool = ConfigField(
        default=True,
        constraint=TypeConstraint(bool),
        doc="Whether early_stop_min_delta is a relative improvement (True) or an absolute one (False).",
    )

    _cross_constraints = [
        LambdaCrossConstraint(
            lambda cfg: split_exists(cfg, cfg.best_metric),
            error="best_metric 'val_loss' needs validation_split > 0, 'test_loss' needs test_split > 0",
            name="best_metric_split_exists",
        ),
        LambdaCrossConstraint(
            lambda cfg: split_exists(cfg, cfg.early_stop_metric),
            error="early_stop_metric 'val_loss' needs validation_split > 0, 'test_loss' needs test_split > 0",
            name="early_stop_metric_split_exists",
        ),
    ]


class AssetBERTTrainConfig(MLMLoopConfig):
    """The `train` block of AssetBERT pretraining and finetuning: the loop plus the output options.

    There is no test split: `validation_split` is the only held-out part.
    """

    calculate_contextualized_embeddings: bool = ConfigField(
        default=False,
        constraint=TypeConstraint(bool),
        doc="Whether to also compute contextualized embeddings after training.",
    )
    keep_best_only: bool = ConfigField(
        default=False,
        constraint=TypeConstraint(bool),
        doc=(
            "Whether to keep only the best epoch's checkpoint and embeddings after training, deleting those of the "
            "other epochs."
        ),
    )


class MPTrainConfig(MLMLoopConfig):
    """The `train` block of both MP trainers (AssetBERT and SimplifiedMLM heads): the loop plus the test split.

    The MP metric is computed on the test split after training.
    """

    test_split: Optional[float] = ConfigField(
        default=None,
        constraint=OptionalConstraint(RangeConstraint(0.0, 1.0)),
        doc="Share of the data held out for testing, e.g. 0.2; None or 0 means no test split.",
    )


class RunMetadataConfig(Config):
    """Run-level metadata of an MP evaluation (its `run` block).

    Names the period explicitly so it need not be parsed from file paths.
    """

    period: str = ConfigField(
        default=None, constraint=TypeConstraint(str), doc="Period label of the run, e.g. '2023Q2'.", required=True
    )
