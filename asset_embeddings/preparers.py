import os
import sys
import inspect
import logging
import logging.handlers
from typing import Optional, Dict, Self

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from safetensors import safe_open
import transformers
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import BertPreTokenizer
from transformers import PreTrainedTokenizerFast, BertConfig, BertForMaskedLM
from gensim.models import Word2Vec, KeyedVectors

from pathlib import Path
from accelerate import Accelerator
from accelerate.utils import TorchDynamoPlugin

from .configs import (
    Config,
    LoggerConfig,
    OptimizerConfig,
    AssetBERTModelConfig,
    DatasetConfig,
    DataLoaderConfig,
    AcceleratorConfig,
    TokenizerConfig,
)
from .logger import (
    ColoredConsoleFormatter,
    ColoredHTMLFormatter,
    TqdmLoggingHandler,
    SysLoggingHandler,
    NotebookLoggingHandler,
)
from .datasets import AssetBERTMLMDataset


def filter_func_kwargs(func, kwargs):
    """
    Filter keyword arguments based on the function signature and print out the dropped arguments.

    :param func: The function whose signature will be checked.
    :param kwargs: Dictionary of keyword arguments.
    :return: Returns a tuple of (filtered keyword arguments dict, dropped keyword arguments dict).
    """
    # get function signature
    if kwargs is None:
        kwargs = {}
    sig = inspect.signature(func)

    filtered_kwargs = {}
    dropped_kwargs = {}

    # Filter out irrelevant kwargs and identify dropped parameters
    for k, v in kwargs.items():
        if k in sig.parameters:
            filtered_kwargs[k] = v
        else:
            dropped_kwargs[k] = v

    return filtered_kwargs, dropped_kwargs


# Lazy-import helpers for niche/GPU optimizer backends. Imported only when the
# corresponding optimizer is selected, so `import asset_embeddings` stays usable
# on CPU-only installs without these packages (see C-H1). sys.modules caches the
# import, so the per-call cost after the first is negligible.
def _bnb():
    import bitsandbytes as bnb

    return bnb


def _lion():
    import lion_pytorch

    return lion_pytorch


def _dadapt():
    import dadaptation

    return dadaptation


class Preparer:
    def __init__(self):
        self.logger: logging.Logger = logging.getLogger("Preparer")
        self.config: Config

    def set_config(self, config: Config) -> Self:
        config.validate()
        self.config = config
        return self

    def get_config(self) -> Config:
        return self.config

    def set_logger(self, logger: logging.Logger) -> Self:
        self.logger = logger
        return self

    def prepare(self): ...


class Logger_Preparer(Preparer):
    def __init__(self):
        super().__init__()
        self.config: LoggerConfig = LoggerConfig()

    def prepare(self) -> logging.Logger:
        cfg = self.config

        logger = logging.getLogger(cfg.log_name)
        logger.setLevel(logging.DEBUG)
        logger.handlers.clear()  # 清除已有的 handlers
        logger.propagate = False  # 避免重复日志

        if cfg.console_stream == "tqdm":  # tqdm.write
            formatter = ColoredConsoleFormatter(
                fmt=cfg.log_format,
                datefmt=cfg.date_format,
                style="%",
                color_target=cfg.color_target,
                enable_colors=cfg.enable_colors,
            )
            stream_handler = TqdmLoggingHandler()
        elif cfg.console_stream == "std":  # sys.stdout
            formatter = ColoredConsoleFormatter(
                fmt=cfg.log_format,
                datefmt=cfg.date_format,
                style="%",
                color_target=cfg.color_target,
                enable_colors=cfg.enable_colors,
            )
            stream_handler = SysLoggingHandler(sys.stdout)
        elif cfg.console_stream == "html":  # IPython.display.HTML
            formatter = ColoredHTMLFormatter(
                fmt=cfg.log_format,
                datefmt=cfg.date_format,
                style="%",
                color_target=cfg.color_target,
                enable_colors=cfg.enable_colors,
            )
            stream_handler = NotebookLoggingHandler()
        else:
            raise ValueError(
                f"Unrecognized console stream type: {cfg.console_stream}. Supported types: `tqdm`, `std`, `html`."
            )
        stream_handler.setLevel(cfg.console_level)
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

        # File logging
        if cfg.log_file:
            if isinstance(cfg.log_file, str):
                log_folder = os.path.dirname(cfg.log_file)
                if log_folder:
                    os.makedirs(log_folder, exist_ok=True)

                file_handler = logging.handlers.RotatingFileHandler(
                    cfg.log_file, mode="a", maxBytes=int(cfg.max_bytes), backupCount=cfg.backup_count, encoding="utf-8"
                )
                file_handler.setLevel(cfg.file_level)
                file_formatter = logging.Formatter(fmt=cfg.log_format, datefmt=cfg.date_format, style="%")
                file_handler.setFormatter(file_formatter)
                logger.addHandler(file_handler)
            elif isinstance(cfg.log_file, list):
                for log_file in cfg.log_file:
                    log_folder = os.path.dirname(log_file)
                    if log_folder:
                        os.makedirs(log_folder, exist_ok=True)

                    file_handler = logging.handlers.RotatingFileHandler(
                        log_file, mode="a", maxBytes=int(cfg.max_bytes), backupCount=cfg.backup_count, encoding="utf-8"
                    )
                    file_handler.setLevel(cfg.file_level)
                    file_formatter = logging.Formatter(fmt=cfg.log_format, datefmt=cfg.date_format, style="%")
                    file_handler.setFormatter(file_formatter)
                    logger.addHandler(file_handler)
        return logger


class AssetBERT_Preparer(Preparer):
    def __init__(self):
        super().__init__()
        self.config: AssetBERTModelConfig = AssetBERTModelConfig()
        self.word2vec_model: Optional[Word2Vec] = None
        self.embedding_df: Optional[pd.DataFrame] = None

    def load_word2vec_model(self, file_path: Optional[str] = None):
        if file_path.endswith(".model"):
            self.word2vec_model = Word2Vec.load(file_path)
        elif file_path.endswith(".txt"):
            self.word2vec_model = KeyedVectors.load_word2vec_format(file_path, binary=False)
        else:
            raise ValueError(f"Unrecognized model file `{file_path}`")

    def load_embedding_csv(self, file_path: Optional[str] = None):
        if file_path.endswith(".csv"):
            self.embedding_df = pd.read_csv(file_path, dtype={"Token": str})
        else:
            raise ValueError(f"Unrecognized embedding file `{file_path}`")

    def prepare(self, tokenizer: PreTrainedTokenizerFast = None) -> BertForMaskedLM:
        bert_config = BertConfig(
            vocab_size=self.config.vocab_size,  # 词表大小
            hidden_size=self.config.hidden_size,  # 隐藏层维度, 必须和嵌入向量长度一致
            num_hidden_layers=self.config.num_hidden_layers,  # BERT 层数
            num_attention_heads=self.config.num_attention_heads,  # 多头注意力
            intermediate_size=self.config.intermediate_size,  # FFN 维度
            max_position_embeddings=self.config.max_position_embeddings,  # 最大位置编码
            type_vocab_size=self.config.type_vocab_size,  # 类型词表大小
        )
        model = BertForMaskedLM(bert_config)
        if self.config.model_checkpoint is not None:
            self.logger.info(f"Loading model from checkpoint: `{os.path.normpath(self.config.model_checkpoint)}`")
            if self.config.model_checkpoint.endswith(".pt"):
                model.load_state_dict(torch.load(self.config.model_checkpoint), strict=False)
            else:
                with safe_open(self.config.model_checkpoint, framework="pt") as f:
                    model_state_dict = model.state_dict()
                    for key in f.keys():
                        if key in model_state_dict:
                            # 如果形状匹配，直接加载
                            if f.get_tensor(key).shape == model_state_dict[key].shape:
                                model_state_dict[key].copy_(f.get_tensor(key))
                            else:
                                self.logger.debug(
                                    f"Skip {key}, shape mismatched: {f.get_tensor(key).shape} != {model_state_dict[key].shape}"
                                )
                        else:
                            self.logger.debug(f"Skip {key}, not found in model state_dict.")
            # model.from_pretrained(self.config.model_checkpoint, ignore_mismatched_sizes=True)
        if self.config.w2v_model is not None:
            self.logger.info(f"Loading Word2Vec as embedding: `{self.config.w2v_model}`")
            self.load_word2vec_model(self.config.w2v_model)

            w2v_vocab_size = len(self.word2vec_model.wv)
            embedding_dim = self.word2vec_model.vector_size

            # self.logger.debug(f"{self.word2vec_model.wv}")

            assert (
                w2v_vocab_size + 3 == model.config.vocab_size
            ), f"Vocab size mismatch: W2V:{len(self.word2vec_model.wv)} BERT:{model.config.vocab_size}"
            assert (
                embedding_dim == model.config.hidden_size
            ), f"Embedding dimension mismatch: Word2Vec:{embedding_dim} BERT:{model.config.hidden_size}"

            # 加载 word2vec 权重
            embedding_matrix = np.zeros((w2v_vocab_size + 3, embedding_dim))
            for i, word in enumerate(self.word2vec_model.wv.index_to_key):
                embedding_matrix[i] = self.word2vec_model.wv[word]

            # 初始化特殊 token
            embedding_matrix[w2v_vocab_size + 0] = np.mean(embedding_matrix, axis=0)  # [MASK] 设为均值
            embedding_matrix[w2v_vocab_size + 1] = np.zeros((embedding_dim,))  # [PAD] 设为零向量
            # [UNK] uses the global numpy RNG, which the trainer seeds
            # (transformers.trainer_utils.set_seed) before model prep — so this draw is
            # reproducible per seed. (release-audit C-L7)
            embedding_matrix[w2v_vocab_size + 2] = np.random.normal(
                embedding_matrix.mean(), embedding_matrix.std(), (embedding_dim,)
            )  # [UNK] 设为随机
            # embedding_matrix[w2v_vocab_size + 3] = embedding_matrix[0]  # [CLS], 第一个token为"0[CLS]"
            # embedding_matrix[w2v_vocab_size + 4] = embedding_matrix[1]  # [SEP], 第二个token为"0[SEP]"

            model.bert.embeddings.word_embeddings.weight.data.copy_(
                torch.tensor(embedding_matrix, dtype=model.bert.embeddings.word_embeddings.weight.data.dtype)
            )

        if self.config.embedding_file is not None:
            self.logger.info(f"Loading embedding from csv: `{os.path.normpath(self.config.embedding_file)}`")
            self.load_embedding_csv(self.config.embedding_file)

            # 提取 Token 和 Embedding 数据
            tokens_from_csv = self.embedding_df["Token"].values
            embeddings_from_csv = self.embedding_df.iloc[:, 1:].values  # 去掉第一列（Token 列）
            embedding_dim_from_csv = embeddings_from_csv.shape[1]

            # 获取模型的词表
            vocab = tokenizer.get_vocab()  # 返回一个字典，键是 token，值是对应的 ID
            vocab_size = len(vocab)  # 词表大小
            embedding_dim = model.bert.embeddings.word_embeddings.weight.shape[1]  # embedding 维度
            assert (
                embedding_dim == embedding_dim_from_csv
            ), f"Embedding dimension mismatch: CSV:{embedding_dim_from_csv} BERT:{embedding_dim}"

            # 转换为字典，方便快速查找
            embedding_dict = {}
            for token, embedding in zip(tokens_from_csv, embeddings_from_csv):
                embedding_dict[str(token)] = embedding

            with torch.no_grad():
                # 初始化一个新的 embedding 权重矩阵
                new_embeddings = model.bert.embeddings.word_embeddings.weight.data.clone()

                # 遍历模型的词表，填充新的 embedding 权重矩阵
                for token, token_id in vocab.items():
                    if token in embedding_dict:
                        # 如果 token 存在于 CSV 文件中，使用其对应的 embedding
                        new_embeddings[token_id] = torch.tensor(embedding_dict[token], dtype=torch.float32)
                    else:
                        # 如果 token 缺失，跳过
                        self.logger.debug(f"Token not found in embedding dict: {token}")
                        pass

                model.bert.embeddings.word_embeddings.weight.data.copy_(new_embeddings)

        if self.config.freeze_embedding:
            self.logger.info("Set Embedding Layer frozen.")
            for param in model.bert.embeddings.parameters():
                param.requires_grad = False
        if self.config.freeze_encoder:
            self.logger.info("Set Encoder Layers frozen.")
            for param in model.bert.encoder.parameters():
                param.requires_grad = False
        self.logger.info("Model prepared.")
        return model


class Tokenizer_Preparer(Preparer):
    def __init__(self):
        super().__init__()
        self.config: TokenizerConfig = TokenizerConfig()
        self.word2vec_model: Optional[Word2Vec] = None
        self.embedding_df: Optional[pd.DataFrame] = None

    def load_word2vec_model(self, file_path: Optional[str] = None):
        if file_path.endswith(".model"):
            self.word2vec_model = Word2Vec.load(file_path)
        elif file_path.endswith(".txt"):
            self.word2vec_model = KeyedVectors.load_word2vec_format(file_path, binary=False)
        else:
            raise ValueError(f"Unrecognized model file `{file_path}`")

    def load_embedding_csv(self, file_path: Optional[str] = None):
        if file_path.endswith(".csv"):
            self.embedding_df = pd.read_csv(file_path)
        else:
            raise ValueError(f"Unrecognized embedding file `{file_path}`")

    def prepare(self) -> PreTrainedTokenizerFast:
        self.logger.info("Preparing Tokenizer...")
        if self.config.pretrained_tokenizer_file is not None:
            self.logger.info(f"Using pretrained tokenizer config file: `{self.config.pretrained_tokenizer_file}`")
            self.vocab_file = None
            tokenizer = PreTrainedTokenizerFast.from_pretrained(self.config.pretrained_tokenizer_file)
        elif self.config.vocab_file is not None:
            self.logger.info(f"Using vocab file: `{self.config.vocab_file}`")
            self.config.pretrained_tokenizer_file = None
            inner_tokenizer = Tokenizer(WordLevel(vocab=self.config.vocab_file, unk_token="[UNK]"))
            inner_tokenizer.pre_tokenizer = BertPreTokenizer()
            bert_tokenizer = PreTrainedTokenizerFast(tokenizer_object=inner_tokenizer)
            bert_tokenizer.add_special_tokens(
                special_tokens_dict={
                    "cls_token": "[CLS]",
                    "sep_token": "[SEP]",
                    "mask_token": "[MASK]",
                    "pad_token": "[PAD]",
                    "unk_token": "[UNK]",
                }
            )
            tokenizer = bert_tokenizer
        elif self.config.w2v_model is not None:
            self.logger.info(f"Using word2vec model: `{self.config.w2v_model}`")
            self.load_word2vec_model(self.config.w2v_model)
            token2index = get_token2index_mapping(self.word2vec_model)
            self.config.pretrained_tokenizer_file = None
            inner_tokenizer = Tokenizer(WordLevel(vocab=token2index, unk_token="[UNK]"))
            inner_tokenizer.pre_tokenizer = BertPreTokenizer()
            bert_tokenizer = PreTrainedTokenizerFast(tokenizer_object=inner_tokenizer)
            bert_tokenizer.add_special_tokens(
                special_tokens_dict={
                    "cls_token": "[CLS]",
                    "sep_token": "[SEP]",
                    "mask_token": "[MASK]",
                    "pad_token": "[PAD]",
                    "unk_token": "[UNK]",
                }
            )
            tokenizer = bert_tokenizer
        elif self.config.embedding_file is not None:
            self.logger.info(f"Using embedding file: `{self.config.embedding_file}`")
            self.load_embedding_csv(self.config.embedding_file)

            # 提取 Token 和 Embedding 数据
            tokens_from_csv = self.embedding_df["Token"].values
            token2index = {}
            for i, token in enumerate(tokens_from_csv):
                if isinstance(token, str):
                    if token.isdigit():
                        token2index[token.zfill(6)] = i
                    else:
                        token2index[token] = i
                elif isinstance(token, int):
                    token2index[str(token).zfill(6)] = i
                elif isinstance(token, np.integer):
                    token2index[str(token).zfill(6)] = i
                else:
                    raise ValueError(f"Unrecognized token type: {type(token)}")
            if "[UNK]" not in token2index:
                # 添加特殊 token
                self.logger.warning("Adding unknown token [UNK] to vocabulary.")
                token2index["[UNK]"] = len(token2index)
            inner_tokenizer = Tokenizer(WordLevel(vocab=token2index, unk_token="[UNK]"))
            inner_tokenizer.pre_tokenizer = BertPreTokenizer()
            bert_tokenizer = PreTrainedTokenizerFast(tokenizer_object=inner_tokenizer)
            bert_tokenizer.add_special_tokens(
                special_tokens_dict={
                    "cls_token": "[CLS]",
                    "sep_token": "[SEP]",
                    "mask_token": "[MASK]",
                    "pad_token": "[PAD]",
                    "unk_token": "[UNK]",
                }
            )
            tokenizer = bert_tokenizer
        else:
            raise ValueError("No valid tokenizer config provided.")
        self.logger.info("Tokenizer prepared.")
        return tokenizer


class Dataset_Preparer(Preparer):
    """Preparer for creating Dataset objects.

    This preparer focuses on creating Dataset objects from configuration,
    separating the concern of "what the data is" from "how to load it".
    """

    def __init__(self):
        super().__init__()
        self.config: DatasetConfig = DatasetConfig()

    def prepare(self, tokenizer: PreTrainedTokenizerFast) -> AssetBERTMLMDataset:
        """Prepare and return a Dataset object.

        Args:
            tokenizer: PreTrainedTokenizerFast for tokenizing the data.

        Returns:
            AssetBERTMLMDataset: The prepared dataset.
        """
        self.logger.info("Preparing Dataset...")
        dataset = AssetBERTMLMDataset(
            path=self.config.data_path,
            tokenizer=tokenizer,
            max_length=self.config.max_length,
            mask_prob=self.config.mask_prob,
            mask_indices=self.config.mask_indices,
            format=self.config.data_format,
            include_proportion=self.config.include_proportion,
            cache_size=self.config.cache_size,
            num_repeats=self.config.num_repeats,
            id_key=self.config.id_key,
            portfolio_key=self.config.portfolio_key,
            proportion1_key=self.config.proportion1_key,
            proportion2_key=self.config.proportion2_key,
        )
        self.logger.info(f"Dataset prepared. Size: {len(dataset)}")
        return dataset


class DataLoader_Preparer(Preparer):
    """Preparer for creating DataLoader objects from Dataset.

    This preparer focuses on wrapping Dataset objects into DataLoader,
    separating the concern of "how to load data in batches" from dataset creation.
    """

    def __init__(self):
        super().__init__()
        self.config: DataLoaderConfig = DataLoaderConfig()

    def prepare(self, dataset: Dataset, shuffle_override: Optional[bool] = None) -> DataLoader:
        """Prepare and return a DataLoader object.

        Args:
            dataset: PyTorch Dataset object to wrap.
            shuffle_override: Override the shuffle setting from config.
                              Useful for forcing shuffle=False on validation loaders.

        Returns:
            DataLoader: The prepared dataloader.
        """
        self.logger.info("Preparing DataLoader...")
        if shuffle_override is not None:
            self.logger.info(f"Overriding shuffle to {shuffle_override}.")
            shuffle = shuffle_override
        else:
            shuffle = self.config.shuffle

        dataloader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=shuffle,
            num_workers=self.config.num_workers,
            persistent_workers=self.config.persistent_workers,
            pin_memory=self.config.pin_memory,
            drop_last=self.config.drop_last,
        )
        self.logger.info(f"Num of batches: {len(dataloader)}.")
        self.logger.info("DataLoader prepared.")
        return dataloader


class Accelerator_Preparer(Preparer):
    """Creates a configured HuggingFace Accelerator instance.

    Handles compile cache environment variables, TF32 flags,
    TorchDynamoPlugin creation, and Accelerator instantiation.

    Does NOT call accelerator.prepare(model, optimizer, ...) — that stays in Trainer.
    """

    def __init__(self):
        super().__init__()
        self.config: AcceleratorConfig = AcceleratorConfig()

    def prepare(self) -> Accelerator:
        """Prepare and return an Accelerator instance.

        Returns:
            Accelerator: The configured Accelerator.
        """
        config = self.config

        # 1. Compile cache
        if config.compile_cache_dir is not None:
            os.environ["TORCHINDUCTOR_CACHE_DIR"] = config.compile_cache_dir
            os.environ["TORCHINDUCTOR_FX_GRAPH_CACHE"] = "1"
            Path(config.compile_cache_dir).mkdir(parents=True, exist_ok=True)
            self.logger.info(f"Compile cache dir: {config.compile_cache_dir}")

        # 2. TF32
        torch.backends.cuda.matmul.allow_tf32 = config.allow_tf32
        torch.backends.cudnn.allow_tf32 = config.allow_tf32
        if config.allow_tf32:
            self.logger.info("TF32 enabled")

        # 3. TorchDynamoPlugin
        dynamo_plugin = None
        if config.use_compile:
            dynamo_plugin = TorchDynamoPlugin(
                backend=config.compile_backend,
                mode=config.compile_mode,
                fullgraph=config.compile_fullgraph,
                dynamic=config.compile_dynamic,
            )
            self.logger.info(
                f"torch.compile enabled: backend={config.compile_backend}, "
                f"mode={config.compile_mode}, dynamic={config.compile_dynamic}"
            )

        # 4. Create Accelerator
        kwargs = dict(
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            dynamo_plugin=dynamo_plugin,
        )
        if config.mixed_precision != "auto":
            kwargs["mixed_precision"] = config.mixed_precision

        accelerator = Accelerator(**kwargs)
        self.logger.info("Accelerator prepared.")
        return accelerator


class Optimizer_Preparer(Preparer):
    def __init__(self) -> None:
        super().__init__()
        self.config: OptimizerConfig = OptimizerConfig()

    def _optimizer_registry(self) -> Dict[str, tuple]:
        """name -> (class_getter, lr_is_none, extra_construct_kwargs, prehook_or_None).

        ``class_getter`` is a zero-arg callable that resolves (and lazily imports)
        the optimizer class; ``prehook`` is an optional bound method run before
        construction for optimizers needing kwarg massaging.
        """
        return {
            "Adam": (lambda: torch.optim.Adam, False, {}, None),
            "AdamW": (lambda: torch.optim.AdamW, False, {}, None),
            "Adam8bit": (lambda: _bnb().optim.Adam8bit, False, {}, None),
            "PagedAdam8bit": (lambda: _bnb().optim.PagedAdam8bit, False, {}, None),
            "AdamW8bit": (lambda: _bnb().optim.AdamW8bit, False, {}, None),
            "PagedAdamW8bit": (lambda: _bnb().optim.PagedAdamW8bit, False, {}, None),
            "Lion": (lambda: _lion().Lion, False, {}, None),
            "Lion8bit": (lambda: _bnb().optim.Lion8bit, False, {}, None),
            "PagedLion8bit": (lambda: _bnb().optim.PagedLion8bit, False, {}, None),
            "DAdaptation": (lambda: _dadapt().DAdaptAdam, False, {}, None),
            "SGDNesterov": (lambda: torch.optim.SGD, False, {"nesterov": True}, self._prehook_sgd_nesterov),
            # SGD8bit also passes nesterov=True and bitsandbytes' SGD8bit requires
            # momentum>0, so it shares the momentum-default guard with SGDNesterov.
            "SGD8bit": (lambda: _bnb().optim.SGD8bit, False, {"nesterov": True}, self._prehook_sgd_nesterov),
            "Adafactor": (lambda: transformers.optimization.Adafactor, True, {}, self._prehook_adafactor),
        }

    def _prehook_sgd_nesterov(self) -> None:
        if "momentum" not in self.config.optimizer_kwargs:
            self.logger.critical("SGD with Nesterov must be with momentum, set momentum to 0.9")
            self.config.optimizer_kwargs["momentum"] = 0.9

    def _prehook_adafactor(self) -> None:
        kwargs = self.config.optimizer_kwargs
        if "relative_step" not in kwargs:
            kwargs["relative_step"] = True  # default
        if not kwargs["relative_step"] and kwargs.get("warmup_init", False):
            self.logger.info("set relative_step to True because warmup_init is True.")
            kwargs["relative_step"] = True
        if kwargs["relative_step"]:
            if self.config.learning_rate != 0.0:
                self.logger.info("learning rate is used as initial_lr.")
            if self.config.lr_scheduler_type != "adafactor":
                self.logger.info("use adafactor_scheduler.")
                self.config.lr_scheduler_type = "adafactor"

    def prepare_optimizer(self, trainable_params) -> torch.optim.Optimizer:
        self.logger.info("Preparing Optimizer...")
        # Normalize optional kwargs so the prehook membership checks don't crash
        # when optimizer_kwargs is left at its None default.
        if self.config.optimizer_kwargs is None:
            self.config.optimizer_kwargs = {}

        spec = self._optimizer_registry().get(self.config.optimizer_type)
        if spec is None:
            raise ValueError(
                f"Unrecognized optimizer type `{self.config.optimizer_type}`. "
                f"Valid types are: {', '.join(self._optimizer_registry())}."
            )
        class_getter, lr_is_none, extra_kwargs, prehook = spec
        if prehook is not None:
            prehook()

        optimizer_class = class_getter()
        filtered_kwargs, invalid_kwargs = filter_func_kwargs(optimizer_class.__init__, self.config.optimizer_kwargs)
        if invalid_kwargs:
            self.logger.warning(f"Invalid kwargs: {invalid_kwargs}")
        lr = None if lr_is_none else self.config.learning_rate
        optimizer = optimizer_class(trainable_params, lr=lr, **extra_kwargs, **filtered_kwargs)

        self.logger.info("Optimizer prepared.")
        return optimizer

    def prepare_lr_scheduler(self, optimizer: torch.optim.Optimizer) -> torch.optim.lr_scheduler.LRScheduler:
        self.logger.info("Preparing LR Scheduler...")
        if self.config.lr_scheduler_type.lower() == "constant":
            lr_scheduler = transformers.optimization.get_constant_schedule_with_warmup(
                optimizer, num_warmup_steps=self.config.lr_scheduler_warmup_steps
            )
        elif self.config.lr_scheduler_type.lower() == "linear":
            lr_scheduler = transformers.optimization.get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=self.config.lr_scheduler_warmup_steps,
                num_training_steps=self.config.lr_scheduler_train_steps,
            )
        elif self.config.lr_scheduler_type.lower() == "cosine":
            lr_scheduler = transformers.optimization.get_cosine_schedule_with_warmup(
                optimizer,
                num_warmup_steps=self.config.lr_scheduler_warmup_steps,
                num_training_steps=self.config.lr_scheduler_train_steps,
                num_cycles=self.config.lr_scheduler_num_cycles,
            )
        elif self.config.lr_scheduler_type.lower() == "cosine_with_restarts":
            lr_scheduler = transformers.optimization.get_cosine_with_hard_restarts_schedule_with_warmup(
                optimizer,
                num_warmup_steps=self.config.lr_scheduler_warmup_steps,
                num_training_steps=self.config.lr_scheduler_train_steps,
                num_cycles=self.config.lr_scheduler_num_cycles,
            )
        elif self.config.lr_scheduler_type.lower() == "polynomial":
            lr_scheduler = transformers.optimization.get_polynomial_decay_schedule_with_warmup(
                optimizer,
                num_warmup_steps=self.config.lr_scheduler_warmup_steps,
                num_training_steps=self.config.lr_scheduler_train_steps,
                power=self.config.lr_scheduler_power,
            )
        elif self.config.lr_scheduler_type.lower() == "adafactor":
            assert (
                type(optimizer) is transformers.optimization.Adafactor
            ), f"Adafactor Scheduler must be used with Adafactor Optimizer. Unexpected optimizer type {type(optimizer)}"
            lr_scheduler = transformers.optimization.AdafactorSchedule(optimizer, initial_lr=self.config.learning_rate)
        else:
            raise ValueError(
                f"Unrecognized lr_scheduler_type `{self.config.lr_scheduler_type}`. Valid types are: constant, linear, cosine, cosine_with_restarts, polynomial and adafactor."
            )

        self.logger.info("LR Scheduler prepared.")
        return lr_scheduler

    def prepare(self, trainable_params):
        optimizer = self.prepare_optimizer(trainable_params=trainable_params)
        lr_scheduler = self.prepare_lr_scheduler(optimizer=optimizer)
        return optimizer, lr_scheduler


def get_token2index_mapping(model: Word2Vec) -> Dict[str, int]:
    """保存 token-index 映射"""
    token2index = {word: i for i, word in enumerate(model.wv.index_to_key)}

    # 追加特殊 token
    special_tokens = ["[MASK]", "[PAD]", "[UNK]"]
    for i, token in enumerate(special_tokens, start=len(token2index)):
        token2index[token] = i

    return token2index


def get_index2token_mapping(model: Word2Vec) -> Dict[int, str]:
    """保存 token-index 映射"""
    index2token = {i: word for i, word in enumerate(model.wv.index_to_key)}

    # 追加特殊 token
    special_tokens = ["[MASK]", "[PAD]", "[UNK]"]
    for i, token in enumerate(special_tokens, start=len(index2token)):
        index2token[i] = token

    return index2token
