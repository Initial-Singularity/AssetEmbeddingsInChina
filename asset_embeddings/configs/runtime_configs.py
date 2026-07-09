import os
from typing import Union, List, Optional, Literal

from .constraints import (
    TypeConstraint,
    RangeConstraint,
    ChoiceConstraint,
    UnionConstraints,
    OptionalConstraint,
    LambdaCrossConstraint,
)
from .base import Config, ConfigField


def path_preprocessor(p):
    if p is None:
        return None
    if isinstance(p, list):
        return [os.path.normpath(x) for x in p]
    return os.path.normpath(p)


class AssetRSTrainConfig(Config):
    """Configuration for Asset Recommender System training using PCA-based embeddings.

    This configuration manages the training of PCA-based recommendation systems for
    financial assets. It supports various recommendation strategies and provides
    options for dimensionality reduction and model persistence.

    Attributes:
        model_type: Type of recommendation system model to train.
        n_components: Number of PCA components for dimensionality reduction.
        whiten: Whether to apply whitening to PCA components.
        data_path: Path to training data (file or directory).
        data_format: Format of input data files.
        id_key: Field name for identifying different portfolios.
        portfolio_key: Field name for portfolio data.
        save_folder: Directory for saving trained models.
        save_name: Base name for saved model files.
        save_format: File format for model serialization.
        seed: Random seed for reproducible results.
    """

    model_type: Literal["RS_Binary", "RS_Ranks", "RS_Level0", "RS_LevelMin"] = ConfigField(
        constraint=ChoiceConstraint(["RS_Binary", "RS_Ranks", "RS_Level0", "RS_LevelMin"]),
        doc="模型类型",
        required=True,
    )
    n_components: int = ConfigField(
        default=10, constraint=TypeConstraint(int), doc="PCA降维后的成分数(embedding dim)", required=True
    )
    whiten: bool = ConfigField(default=False, constraint=TypeConstraint(bool), doc="是否对PCA成分进行白化处理")
    data_path: str = ConfigField(
        default=None,
        constraint=TypeConstraint(str),
        preprocessor=path_preprocessor,
        doc="数据路径,支持单个文件或文件夹",
        required=True,
    )
    data_format: str = ConfigField(
        default="csv",
        constraint=ChoiceConstraint(["csv", "json", "binary"]),
        doc="数据格式,支持'csv'、'json'或'binary'",
        required=True,
    )
    id_key: str = ConfigField(default="InvestorID", constraint=TypeConstraint(str), doc="CSV文件中投资者ID列名")
    portfolio_key: str = ConfigField(default="Portfolio", constraint=TypeConstraint(str), doc="CSV文件中投资组合列名")
    proportion1_key: str = ConfigField(
        default="Proportion1", constraint=TypeConstraint(str), doc="CSV文件中持仓比例列名"
    )
    proportion2_key: str = ConfigField(
        default="Proportion2", constraint=TypeConstraint(str), doc="CSV文件中持仓比例2列名"
    )
    save_folder: str = ConfigField(
        default=None,
        constraint=TypeConstraint(str),
        preprocessor=path_preprocessor,
        doc="模型保存文件夹",
        required=True,
    )
    save_name: str = ConfigField(default=None, constraint=TypeConstraint(str), doc="模型保存名称", required=True)
    save_format: str = ConfigField(
        default=".pkl", constraint=ChoiceConstraint([".pkl"]), doc="模型保存格式", required=True
    )
    seed: int = ConfigField(default=42, constraint=TypeConstraint(int), doc="随机种子")


class AssetW2VTrainConfig(Config):
    """Configuration for training Word2Vec models on financial asset sequences.

    This configuration manages the training of Word2Vec embeddings specifically designed
    for financial asset data. It supports both training from scratch and fine-tuning
    pre-trained models with comprehensive hyperparameter control.

    Attributes:
        pretrained_model: Path to existing Word2Vec model for fine-tuning.
        embedding_dim: Dimensionality of word embeddings.
        epochs: Number of training epochs.
        window: Context window size for skip-gram/CBOW.
        min_count: Minimum word frequency threshold.
        sg: Skip-gram vs CBOW model selection.
        sample: Subsampling threshold for frequent words.
        negative_sample: Number of negative samples per positive sample.
        data_path: Path to training data.
        data_format: Format of input data files.
        id_key: Field name for identifying different portfolios.
        portfolio_key: Field name for portfolio data.
        workers: Number of training threads.
        save_folder: Directory for saving trained models.
        save_name: Base name for saved model files.
        save_format: File format for model serialization.
        seed: Random seed for reproducible training.
    """

    pretrained_model: str = ConfigField(
        default=None,
        constraint=OptionalConstraint(TypeConstraint(str)),
        preprocessor=path_preprocessor,
        doc="预训练Word2Vec模型",
    )
    embedding_dim: int = ConfigField(default=100, constraint=TypeConstraint(int), doc="词嵌入维度", required=True)
    epochs: int = ConfigField(default=100, constraint=TypeConstraint(int), doc="训练轮数", required=True)
    window: int = ConfigField(default=5, constraint=TypeConstraint(int), doc="上下文窗口大小", required=True)
    min_count: int = ConfigField(default=5, constraint=TypeConstraint(int), doc="词频最小阈值", required=True)
    sg: int = ConfigField(default=1, constraint=TypeConstraint(int), doc="是否使用Skip-gram模型", required=True)
    sample: float = ConfigField(default=1e-3, constraint=TypeConstraint(float), doc="正采样数量", required=True)
    negative_sample: int = ConfigField(default=5, constraint=TypeConstraint(int), doc="负采样数量", required=True)
    data_path: Union[str, List[str]] = ConfigField(
        default=None,
        constraint=OptionalConstraint(UnionConstraints(TypeConstraint(str), TypeConstraint(list))),
        preprocessor=path_preprocessor,
        doc="数据路径,支持单个文件、文件夹或文件路径列表",
        required=True,
    )
    data_format: str = ConfigField(
        default="csv",
        constraint=ChoiceConstraint(["csv", "json", "binary"]),
        doc="数据格式,支持'csv'、'json'或'binary'",
        required=True,
    )
    id_key: str = ConfigField(
        default="InvestorID", constraint=TypeConstraint(str), doc="ID字段名,用于区分不同Portfolio"
    )
    portfolio_key: str = ConfigField(default="Portfolio", constraint=TypeConstraint(str), doc="Portfolio字段名")
    workers: int = ConfigField(
        default=1,
        constraint=TypeConstraint(int),
        doc="数据加载工作线程数. Note that for a fully deterministically‑reproducible run, you must also limit the model to a single worker thread (workers=1), to eliminate ordering jitter.",
        required=True,
    )
    save_folder: str = ConfigField(
        default=None,
        constraint=TypeConstraint(str),
        preprocessor=path_preprocessor,
        doc="模型保存文件夹",
        required=True,
    )
    save_name: str = ConfigField(default=None, constraint=TypeConstraint(str), doc="模型保存名称", required=True)
    save_format: str = ConfigField(
        default=".model", constraint=ChoiceConstraint([".model", ".txt"]), doc="模型保存格式", required=True
    )
    seed: int = ConfigField(default=42, constraint=TypeConstraint(int), doc="随机种子")


class AssetBERTTrainConfig(Config):
    """Configuration for training Asset BERT models with comprehensive training controls.

    This configuration manages training loop behavior including epoch control,
    gradient clipping, checkpointing, validation/test splitting, and early stopping.
    Hardware acceleration settings (mixed precision, torch.compile, TF32) are managed
    separately in AcceleratorConfig.

    Attributes:
        max_epoches: Maximum number of training epochs.
        clip_grad_norm: Maximum gradient norm for gradient clipping.
        clip_grad_value: Maximum absolute gradient value for clipping.
        detect_anomaly: Whether to enable PyTorch anomaly detection.
        check_per_step: int | Perform validation every specified number of training steps.
        report_per_epoch: int | Output intermediate reports every specified number of epochs.
        save_per_epoch: int | Save the model every specified number of epochs.
        save_folder: str | Directory to save the model.
        save_name: str | Model save filename (without extension).
        save_format: str | Save format, recommended as .safetensors.
        seed: int | Random seed to ensure reproducible experiments.
    """

    max_epoches: int = ConfigField(default=30, constraint=TypeConstraint(int), doc="训练轮数", required=True)
    clip_grad_norm: float = ConfigField(
        default=None, constraint=OptionalConstraint(RangeConstraint(0, None)), doc="梯度裁剪的最大范数"
    )
    clip_grad_value: float = ConfigField(
        default=None, constraint=OptionalConstraint(RangeConstraint(0, None)), doc="梯度裁剪的绝对值"
    )
    detect_anomaly: bool = ConfigField(default=False, constraint=TypeConstraint(bool), doc="是否检测异常")
    check_per_step: int = ConfigField(
        default=300, constraint=TypeConstraint(int) & RangeConstraint(0, None), doc="每n步进行检查"
    )
    report_per_epoch: int = ConfigField(
        default=1, constraint=TypeConstraint(int) & RangeConstraint(0, None), doc="每n个epoch进行报告"
    )
    calculate_contextualized_embeddings: bool = ConfigField(
        default=False, constraint=TypeConstraint(bool), doc="是否计算上下文化embedding"
    )
    save_per_epoch: int = ConfigField(
        default=1, constraint=TypeConstraint(int) & RangeConstraint(0, None), doc="每n个epoch保存"
    )
    keep_best_only: bool = ConfigField(
        default=False,
        constraint=TypeConstraint(bool),
        doc="训练结束后是否只保留best checkpoint和embedding，删除其余epoch的checkpoint和embedding",
    )
    save_folder: str = ConfigField(
        default=None,
        constraint=TypeConstraint(str),
        preprocessor=path_preprocessor,
        doc="保存模型的文件夹",
        required=True,
    )
    save_name: str = ConfigField(default=None, constraint=TypeConstraint(str), doc="保存模型的名称", required=True)
    save_format: Literal[".pt", ".safetensors"] = ConfigField(
        default=".safetensors",
        constraint=ChoiceConstraint([".pt", ".safetensors"]),
        doc="保存模型的格式",
    )
    seed: int = ConfigField(default=42, constraint=TypeConstraint(int), doc="随机种子")
    early_stop_enabled: bool = ConfigField(default=False, constraint=TypeConstraint(bool), doc="是否启用early stopping")
    early_stop_patience: int = ConfigField(
        default=5,
        constraint=TypeConstraint(int) & RangeConstraint(1, None),
        doc="Early stopping容忍轮数（无改进epoch数）",
    )
    early_stop_min_delta: float = ConfigField(
        default=0.0, constraint=TypeConstraint(float) & RangeConstraint(0.0, None), doc="Early stopping最小改进阈值"
    )
    early_stop_mode: Literal["min", "max"] = ConfigField(
        default="min",
        constraint=ChoiceConstraint(["min", "max"]),
        doc="Early stopping优化方向（min=越小越好如loss, max=越大越好如accuracy）",
    )
    early_stop_relative: bool = ConfigField(
        default=True, constraint=TypeConstraint(bool), doc="是否使用相对改进（True=相对改进，False=绝对改进）"
    )
    early_stop_metric: Literal["train_loss", "val_loss", "eval_loss"] = ConfigField(
        default="train_loss",
        constraint=ChoiceConstraint(["train_loss", "val_loss", "eval_loss"]),
        doc="Early stopping监控指标（train_loss=训练loss, val_loss/eval_loss=评估loss）",
    )
    # === Validation/Test Split Configuration ===
    validation_split: Optional[float] = ConfigField(
        default=None,
        constraint=OptionalConstraint(RangeConstraint(0.0, 1.0)),
        doc="验证集比例，None表示不分割。例如0.1表示10%数据作为验证集",
    )
    test_split: Optional[float] = ConfigField(
        default=None,
        constraint=OptionalConstraint(RangeConstraint(0.0, 1.0)),
        doc="测试集比例，None表示不分割。例如0.1表示10%数据作为测试集。用于MP评估等需要三路分割的场景",
    )
    split_method: Literal["sequential", "random"] = ConfigField(
        default="random",
        constraint=ChoiceConstraint(["sequential", "random"]),
        doc="数据集分割方式：sequential=顺序分割，random=随机分割",
    )
    best_metric: Literal["train_loss", "val_loss", "eval_loss"] = ConfigField(
        default="train_loss",
        constraint=ChoiceConstraint(["train_loss", "val_loss", "eval_loss"]),
        doc="用于选择best epoch的指标（train_loss=训练loss, val_loss/eval_loss=评估loss）",
    )

    _cross_constraints = [
        LambdaCrossConstraint(
            lambda cfg: cfg.best_metric == "train_loss" or cfg.validation_split is not None,
            error="best_metric='val_loss'/'eval_loss' requires validation_split to be set",
            name="best_metric_requires_val_split",
        ),
        LambdaCrossConstraint(
            lambda cfg: cfg.early_stop_metric == "train_loss" or cfg.validation_split is not None,
            error="early_stop_metric='val_loss'/'eval_loss' requires validation_split to be set",
            name="early_stop_metric_requires_val_split",
        ),
    ]
