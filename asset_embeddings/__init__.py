"""Asset embeddings for China's A-share market: models, trainers, evaluators and run records.

The package root re-exports the config classes, the dataset and encoder classes, the preparers, the logging
entry points and the `SimplifiedMLM` module so scripts can import them from `asset_embeddings` directly.
"""

from . import configs, datasets, preparers, modules
from .configs import (
    Config,
    ConfigContainer,
    LoggerConfig,
    TokenizerConfig,
    OptimizerConfig,
    DatasetConfig,
    DataLoaderConfig,
    AssetBERTModelConfig,
    SimplifiedMLMModelConfig,
    RunMetadataConfig,
    AssetRSTrainConfig,
    AssetW2VTrainConfig,
    AssetBERTTrainConfig,
    MPTrainConfig,
    RVEvalConfig,
    RCEvalConfig,
)
from .datasets import PortfolioCorpus, MLMDataset, MaskingCollator
from .logger import configure_logging, failure_report
from .preparers import (
    AssetBERT_Preparer,
    Dataset_Preparer,
    DataLoader_Preparer,
    Optimizer_Preparer,
    Tokenizer_Preparer,
    SimplifiedMLM_Preparer,
    Word2Vec_Preparer,
)
from .modules.SimplifiedMLM import SimplifiedMLM


__all__ = [
    "configs",
    "Config",
    "ConfigContainer",
    "LoggerConfig",
    "TokenizerConfig",
    "OptimizerConfig",
    "DatasetConfig",
    "DataLoaderConfig",
    "AssetBERTModelConfig",
    "SimplifiedMLMModelConfig",
    "RunMetadataConfig",
    "AssetRSTrainConfig",
    "AssetW2VTrainConfig",
    "AssetBERTTrainConfig",
    "MPTrainConfig",
    "RVEvalConfig",
    "RCEvalConfig",
    "datasets",
    "PortfolioCorpus",
    "MLMDataset",
    "MaskingCollator",
    "preparers",
    "configure_logging",
    "failure_report",
    "AssetBERT_Preparer",
    "Dataset_Preparer",
    "DataLoader_Preparer",
    "Optimizer_Preparer",
    "Tokenizer_Preparer",
    "SimplifiedMLM_Preparer",
    "Word2Vec_Preparer",
    "modules",
    "SimplifiedMLM",
]
