"""Preparers, one per training component, each building its object from a `Config`.

A preparer builds the tokenizer, the dataset, the dataloader, the BERT / SimplifiedMLM / Word2Vec model, the
`Accelerator`, or the optimizer with its LR scheduler. The trainers in `asset_embeddings.trainers` and the MP
trainers in `asset_embeddings.evaluators` own one preparer per component and call
`set_config(...).prepare(...)`; a preparer builds and returns the object, it never trains it.
"""

import os
import sys
import inspect
import logging
import logging.handlers
from typing import Any, Callable, Dict, Iterable, List, Optional, Self, Tuple

import numpy as np
import pandas as pd
import torch
from torch.optim.optimizer import ParamsT
from torch.utils.data import Dataset, DataLoader, RandomSampler
from safetensors import safe_open
import transformers
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import BertPreTokenizer
from transformers import PreTrainedTokenizerFast, BertConfig, BertForMaskedLM
from gensim.models import Word2Vec

from pathlib import Path
from accelerate import Accelerator
from accelerate.utils import TorchDynamoPlugin

from .configs import (
    Config,
    LoggerConfig,
    OptimizerConfig,
    AssetBERTModelConfig,
    SimplifiedMLMModelConfig,
    DatasetConfig,
    DataLoaderConfig,
    AcceleratorConfig,
    TokenizerConfig,
    AssetW2VTrainConfig,
)
from .datasets.corpus import PortfolioCorpus
from .datasets.mlm import MLMDataset
from .modules.SimplifiedMLM import SimplifiedMLM


def filter_func_kwargs(func: Callable, kwargs: Optional[Dict[str, Any]]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Split keyword arguments into those named in the signature of `func` and those it does not know.

    Args:
        func: The callable whose signature decides.
        kwargs: Keyword arguments to filter; `None` counts as empty.

    Returns:
        A `(accepted, dropped)` pair of dicts, each in the original order.
    """
    if kwargs is None:
        kwargs = {}
    sig = inspect.signature(func)

    filtered_kwargs = {}
    dropped_kwargs = {}

    for k, v in kwargs.items():
        if k in sig.parameters:
            filtered_kwargs[k] = v
        else:
            dropped_kwargs[k] = v

    return filtered_kwargs, dropped_kwargs


# The optional GPU optimizer backends are imported only when the matching optimizer is selected, so
# `import asset_embeddings` works on CPU-only installs that lack these packages. sys.modules caches the
# module, so calls after the first cost nothing.
def _bnb():
    import bitsandbytes as bnb

    return bnb


def _lion():
    import lion_pytorch

    return lion_pytorch


def _dadapt():
    import dadaptation

    return dadaptation


logger = logging.getLogger(__name__)


class Preparer:
    """Base of every preparer: holds one validated `Config` and builds one component from it in `prepare`.

    Subclasses give `self.config` their own config type in `__init__` and implement `prepare`.
    """

    def __init__(self):
        self.logger: logging.Logger = logger
        self.config: Config

    def set_config(self, config: Config) -> Self:
        """Validate `config`, keep it and return this preparer so that `prepare` can be chained.

        Args:
            config: The config to build from; `config.validate()` runs first, so cross-field errors surface here.

        Returns:
            This preparer.
        """
        config.validate()
        self.config = config
        return self

    def get_config(self) -> Config:
        """Return the config set by `set_config`, or the subclass default when none was set."""
        return self.config

    def prepare(self):
        """Build and return the component; each subclass implements it with its own arguments."""


class SimplifiedMLM_Preparer(Preparer):
    """Builds the `SimplifiedMLM` model of a `SimplifiedMLMModelConfig`.

    Optionally loads a checkpoint, seeds the token embeddings from an embedding CSV, and freezes the
    embedding layer or the head.
    """

    def __init__(self):
        super().__init__()
        self.config: SimplifiedMLMModelConfig = SimplifiedMLMModelConfig()
        self.embedding_df: Optional[pd.DataFrame] = None

    def load_embedding_csv(self, file_path: Optional[str] = None) -> None:
        """Read an embedding CSV (`Token, Emb_0, ...`) into `self.embedding_df`, with `Token` kept as text.

        Args:
            file_path: Path of a `.csv` embedding file.

        Raises:
            ValueError: If the path does not end in `.csv`.
        """
        if file_path.endswith(".csv"):
            self.embedding_df = pd.read_csv(file_path, dtype={"Token": str})
        else:
            raise ValueError(f"Unrecognized embedding file `{file_path}`")

    def prepare(self, tokenizer: PreTrainedTokenizerFast = None) -> SimplifiedMLM:
        """Build the model, then apply the checkpoint, embedding file and freeze flags of the config in that order.

        Args:
            tokenizer: The prepared tokenizer; supplies the `[PAD]` id and the vocabulary the CSV is matched to.

        Returns:
            The model, on CPU, ready for `accelerator.prepare`.

        Raises:
            AssertionError: If the CSV embedding width differs from `hidden_size`.
        """
        model = SimplifiedMLM(
            vocab_size=self.config.vocab_size,
            hidden_size=self.config.hidden_size,
            max_position_embeddings=self.config.max_position_embeddings,
            type_vocab_size=self.config.type_vocab_size,
            pad_token_id=tokenizer.convert_tokens_to_ids("[PAD]"),
        )
        if self.config.model_checkpoint is not None:
            self.logger.info(f"Loading model from checkpoint: `{os.path.normpath(self.config.model_checkpoint)}`")
            if self.config.model_checkpoint.endswith(".pt"):
                model.load_state_dict(torch.load(self.config.model_checkpoint), strict=False)
            else:
                # Key-by-key copy with a shape check is the safetensors counterpart of `strict=False`: a
                # checkpoint built for another vocabulary or hidden size still loads the layers it shares.
                with safe_open(self.config.model_checkpoint, framework="pt") as f:
                    model_state_dict = model.state_dict()
                    for key in f.keys():
                        if key in model_state_dict:
                            if f.get_tensor(key).shape == model_state_dict[key].shape:
                                model_state_dict[key].copy_(f.get_tensor(key))
                            else:
                                self.logger.debug(
                                    f"Skip {key}, shape mismatched: checkpoint shape({f.get_tensor(key).shape}) != model shape({model_state_dict[key].shape})"
                                )
                        else:
                            self.logger.debug(f"Skip {key}, not found in model state_dict.")
        if self.config.embedding_file is not None:
            self.logger.info(f"Loading embedding from csv: `{os.path.normpath(self.config.embedding_file)}`")
            self.load_embedding_csv(self.config.embedding_file)

            tokens_from_csv = self.embedding_df["Token"].values
            embeddings_from_csv = self.embedding_df.iloc[:, 1:].values

            assert (
                embeddings_from_csv.shape[1] == self.config.hidden_size
            ), f"Embedding dimension mismatch: Embedding File:{embeddings_from_csv.shape[1]} BERT:{self.config.hidden_size}"

            vocab = tokenizer.get_vocab()
            embedding_dict = {}
            for token, embedding in zip(tokens_from_csv, embeddings_from_csv):
                embedding_dict[str(token)] = embedding

            with torch.no_grad():
                new_embeddings = model.embeddings.word_embeddings.weight.data.clone()

                for token, token_id in vocab.items():
                    if token in embedding_dict:
                        new_embeddings[token_id] = torch.tensor(embedding_dict[token], dtype=torch.float32)
                    else:
                        # A vocabulary token absent from the CSV keeps the vector it already has.
                        self.logger.debug(f"Token not found in embedding dict: {token}")
                        pass

                model.embeddings.word_embeddings.weight.data.copy_(new_embeddings)
        if self.config.freeze_embedding:
            self.logger.info("Set Embedding Layer frozen.")
            for param in model.embeddings.parameters():
                param.requires_grad = False
        if self.config.freeze_head:
            self.logger.info("Set Head (Linear Layer) frozen.")
            for param in model.head.parameters():
                param.requires_grad = False
        self.logger.info("Model prepared.")
        return model


class AssetBERT_Preparer(Preparer):
    """Builds the `BertForMaskedLM` model of an `AssetBERTModelConfig`.

    Optionally loads a checkpoint, seeds the token embeddings from a Word2Vec model or an embedding CSV, and
    freezes the embedding layer or the encoder.
    """

    def __init__(self):
        super().__init__()
        self.config: AssetBERTModelConfig = AssetBERTModelConfig()
        self.word2vec_model: Optional[Word2Vec] = None
        self.embedding_df: Optional[pd.DataFrame] = None

    def load_word2vec_model(self, file_path: str) -> None:
        """Load a gensim `.model` file saved by the Word2Vec trainer into `self.word2vec_model`.

        Args:
            file_path: Path of the `.model` file.

        Raises:
            ValueError: If the path does not end in `.model`.
        """
        if not file_path.endswith(".model"):
            raise ValueError(f"w2v_model must be a gensim `.model` file saved by the W2V trainer: `{file_path}`")
        self.word2vec_model = Word2Vec.load(file_path)

    def load_embedding_csv(self, file_path: Optional[str] = None) -> None:
        """Read an embedding CSV (`Token, Emb_0, ...`) into `self.embedding_df`, with `Token` kept as text.

        Args:
            file_path: Path of a `.csv` embedding file.

        Raises:
            ValueError: If the path does not end in `.csv`.
        """
        if file_path.endswith(".csv"):
            self.embedding_df = pd.read_csv(file_path, dtype={"Token": str})
        else:
            raise ValueError(f"Unrecognized embedding file `{file_path}`")

    def prepare(self, tokenizer: PreTrainedTokenizerFast = None) -> BertForMaskedLM:
        """Build the model, then apply the checkpoint, Word2Vec vectors, embedding file and freeze flags in that order.

        Args:
            tokenizer: The prepared tokenizer; its vocabulary is what an embedding CSV is matched to.

        Returns:
            The model, on CPU, ready for `accelerator.prepare`.

        Raises:
            AssertionError: If the Word2Vec vocabulary plus the three special tokens is not `vocab_size`, or
                the Word2Vec or CSV vector width differs from `hidden_size`.
        """
        bert_config = BertConfig(
            vocab_size=self.config.vocab_size,
            hidden_size=self.config.hidden_size,  # must equal the width of any Word2Vec / CSV vectors loaded below
            num_hidden_layers=self.config.num_hidden_layers,
            num_attention_heads=self.config.num_attention_heads,
            intermediate_size=self.config.intermediate_size,
            max_position_embeddings=self.config.max_position_embeddings,
            type_vocab_size=self.config.type_vocab_size,
        )
        model = BertForMaskedLM(bert_config)
        if self.config.model_checkpoint is not None:
            self.logger.info(f"Loading model from checkpoint: `{os.path.normpath(self.config.model_checkpoint)}`")
            if self.config.model_checkpoint.endswith(".pt"):
                model.load_state_dict(torch.load(self.config.model_checkpoint), strict=False)
            else:
                # Key-by-key copy with a shape check is the safetensors counterpart of `strict=False`: a
                # checkpoint built for another vocabulary or hidden size still loads the layers it shares.
                with safe_open(self.config.model_checkpoint, framework="pt") as f:
                    model_state_dict = model.state_dict()
                    for key in f.keys():
                        if key in model_state_dict:
                            if f.get_tensor(key).shape == model_state_dict[key].shape:
                                model_state_dict[key].copy_(f.get_tensor(key))
                            else:
                                self.logger.debug(
                                    f"Skip {key}, shape mismatched: {f.get_tensor(key).shape} != {model_state_dict[key].shape}"
                                )
                        else:
                            self.logger.debug(f"Skip {key}, not found in model state_dict.")
        if self.config.w2v_model is not None:
            self.logger.info(f"Loading Word2Vec as embedding: `{self.config.w2v_model}`")
            self.load_word2vec_model(self.config.w2v_model)

            w2v_vocab_size = len(self.word2vec_model.wv)
            embedding_dim = self.word2vec_model.vector_size

            # The BERT vocabulary is the Word2Vec vocabulary followed by [MASK], [PAD], [UNK] in that order
            # (see `get_token2index_mapping`); [CLS] and [SEP] are ordinary corpus tokens with their own vectors.
            assert (
                w2v_vocab_size + 3 == model.config.vocab_size
            ), f"Vocab size mismatch: W2V:{len(self.word2vec_model.wv)} BERT:{model.config.vocab_size}"
            assert (
                embedding_dim == model.config.hidden_size
            ), f"Embedding dimension mismatch: Word2Vec:{embedding_dim} BERT:{model.config.hidden_size}"

            embedding_matrix = np.zeros((w2v_vocab_size + 3, embedding_dim))
            for i, word in enumerate(self.word2vec_model.wv.index_to_key):
                embedding_matrix[i] = self.word2vec_model.wv[word]

            embedding_matrix[w2v_vocab_size + 0] = np.mean(embedding_matrix, axis=0)  # [MASK]: the mean vector
            embedding_matrix[w2v_vocab_size + 1] = np.zeros((embedding_dim,))  # [PAD]: zeros
            # [UNK] is drawn from the global NumPy RNG, which the trainer seeds (`seed_hf_global`) before model
            # preparation, so the draw is reproducible per seed.
            embedding_matrix[w2v_vocab_size + 2] = np.random.normal(
                embedding_matrix.mean(), embedding_matrix.std(), (embedding_dim,)
            )

            model.bert.embeddings.word_embeddings.weight.data.copy_(
                torch.tensor(embedding_matrix, dtype=model.bert.embeddings.word_embeddings.weight.data.dtype)
            )

        if self.config.embedding_file is not None:
            self.logger.info(f"Loading embedding from csv: `{os.path.normpath(self.config.embedding_file)}`")
            self.load_embedding_csv(self.config.embedding_file)

            tokens_from_csv = self.embedding_df["Token"].values
            embeddings_from_csv = self.embedding_df.iloc[:, 1:].values
            embedding_dim_from_csv = embeddings_from_csv.shape[1]

            vocab = tokenizer.get_vocab()
            vocab_size = len(vocab)
            embedding_dim = model.bert.embeddings.word_embeddings.weight.shape[1]
            assert (
                embedding_dim == embedding_dim_from_csv
            ), f"Embedding dimension mismatch: CSV:{embedding_dim_from_csv} BERT:{embedding_dim}"

            embedding_dict = {}
            for token, embedding in zip(tokens_from_csv, embeddings_from_csv):
                embedding_dict[str(token)] = embedding

            with torch.no_grad():
                new_embeddings = model.bert.embeddings.word_embeddings.weight.data.clone()

                for token, token_id in vocab.items():
                    if token in embedding_dict:
                        new_embeddings[token_id] = torch.tensor(embedding_dict[token], dtype=torch.float32)
                    else:
                        # A vocabulary token absent from the CSV keeps the vector it already has.
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


class Word2Vec_Preparer(Preparer):
    """Preparer for the gensim ``Word2Vec`` model of ``asset_embeddings.trainers.w2v``.

    From ``AssetW2VTrainConfig`` it builds the model, a vocabulary whose order and initial vectors are the same
    on every platform, and, when ``pretrained_model`` is set, copies the pretrained vectors in. The trainer only
    trains and saves what comes back.
    """

    def __init__(self):
        super().__init__()
        self.config: AssetW2VTrainConfig = AssetW2VTrainConfig()

    def prepare(self, dataset: Iterable[List[str]]) -> Word2Vec:
        """Build the model and its vocabulary from `dataset`, then copy pretrained vectors in when configured.

        Args:
            dataset: An iterable of token lists; it is iterated once, by gensim's vocabulary count.

        Returns:
            The untrained model with its vocabulary and initial vectors in place.
        """
        self.logger.info("Preparing Word2Vec model...")
        model = Word2Vec(
            vector_size=self.config.embedding_dim,
            window=self.config.window,
            min_count=self.config.min_count,
            sg=self.config.sg,
            sample=self.config.sample,
            negative=self.config.negative_sample,
            seed=self.config.seed,
            workers=self.config.workers,
        )
        self._build_vocab(model, dataset)
        self.logger.info(f"Vocabulary size: {len(model.wv)}, corpus count: {model.corpus_count}")
        if self.config.pretrained_model:
            self._load_pretrained(model, self.config.pretrained_model)
        self.logger.info("Word2Vec model prepared.")
        return model

    def _build_vocab(self, model: Word2Vec, dataset) -> None:
        """Gensim ``build_vocab`` with deterministic ordering and weight init.

        gensim sorts the vocabulary by frequency, but equal-frequency tokens (e.g. [CLS]/[SEP]) end up in a
        platform-dependent order. After gensim has counted, the vocabulary is reordered (freq DESC, token ASC)
        and the weight matrices are re-drawn from ``seed`` so that the same token always gets the same initial
        vector.
        """
        model.build_vocab(dataset)

        wv = model.wv
        counts = wv.expandos["count"]
        order = sorted(range(len(wv.index_to_key)), key=lambda i: (-counts[i], wv.index_to_key[i]))
        if order != list(range(len(order))):
            perm = np.array(order)
            wv.index_to_key = [wv.index_to_key[i] for i in perm]
            wv.key_to_index = {tok: i for i, tok in enumerate(wv.index_to_key)}
            for name in wv.expandos:
                arr = wv.expandos[name]
                if isinstance(arr, np.ndarray) and arr.shape[0] == len(perm):
                    wv.expandos[name] = arr[perm]
            self.logger.info("Vocabulary reordered for cross-platform determinism.")

        # Re-draw the weights from `seed` over the reordered vocabulary, the way gensim's prep_vectors does.
        rng = np.random.default_rng(seed=self.config.seed)
        n, dim = wv.vectors.shape
        wv.vectors = (rng.random((n, dim), dtype=np.float32) * 2.0 - 1.0) / dim
        if hasattr(model, "syn1neg"):
            model.syn1neg = np.zeros((n, dim), dtype=np.float32)
        if hasattr(model, "syn1"):
            model.syn1 = np.zeros((n, dim), dtype=np.float32)

    def _load_pretrained(self, model: Word2Vec, model_path: str) -> None:
        """Copy the pretrained vector in for every token of the built vocabulary the pretrained model knows.

        The other tokens get a uniform draw from a generator seeded with ``seed``, so the result does not depend
        on the caller having seeded NumPy.
        """
        if not model_path.endswith(".model"):
            raise ValueError(
                f"pretrained_model must be a gensim `.model` file saved by the W2V trainer: `{model_path}`"
            )
        self.logger.info(f"Loading pretrained Word2Vec from `{model_path}`")
        pretrained_model = Word2Vec.load(model_path)

        rng = np.random.default_rng(seed=self.config.seed)
        missing = 0
        for word in model.wv.index_to_key:
            if word in pretrained_model.wv:
                model.wv[word] = pretrained_model.wv[word]
            else:
                missing += 1
                self.logger.debug(f"Word `{word}` not found in pretrained model, initializing randomly.")
                model.wv[word] = rng.uniform(-0.25, 0.25, self.config.embedding_dim)
        if missing:
            self.logger.info(f"{missing} of {len(model.wv)} tokens are not in the pretrained model: random init")


class Tokenizer_Preparer(Preparer):
    """Builds the word-level `PreTrainedTokenizerFast` of a `TokenizerConfig`.

    The vocabulary comes from the first source the config sets, in this order: a saved tokenizer directory, a
    vocab file, a Word2Vec model, an embedding CSV.
    """

    def __init__(self):
        super().__init__()
        self.config: TokenizerConfig = TokenizerConfig()
        self.word2vec_model: Optional[Word2Vec] = None
        self.embedding_df: Optional[pd.DataFrame] = None

    def load_word2vec_model(self, file_path: str) -> None:
        """Load a gensim `.model` file saved by the Word2Vec trainer into `self.word2vec_model`.

        Args:
            file_path: Path of the `.model` file.

        Raises:
            ValueError: If the path does not end in `.model`.
        """
        if not file_path.endswith(".model"):
            raise ValueError(f"w2v_model must be a gensim `.model` file saved by the W2V trainer: `{file_path}`")
        self.word2vec_model = Word2Vec.load(file_path)

    def load_embedding_csv(self, file_path: Optional[str] = None) -> None:
        """Read an embedding CSV into `self.embedding_df` with pandas' default dtypes (numeric codes lose their zeros).

        Args:
            file_path: Path of a `.csv` embedding file.

        Raises:
            ValueError: If the path does not end in `.csv`.
        """
        if file_path.endswith(".csv"):
            self.embedding_df = pd.read_csv(file_path)
        else:
            raise ValueError(f"Unrecognized embedding file `{file_path}`")

    def prepare(self) -> PreTrainedTokenizerFast:
        """Build the tokenizer from the first vocabulary source the config sets.

        Every source except a saved tokenizer directory gets the five special tokens `[CLS]`, `[SEP]`, `[MASK]`,
        `[PAD]`, `[UNK]` added; a Word2Vec vocabulary is followed by `[MASK]`, `[PAD]`, `[UNK]` in that order.

        Returns:
            The tokenizer.

        Raises:
            ValueError: If the config sets no vocabulary source, or a CSV token is neither text nor an integer.
        """
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

            tokens_from_csv = self.embedding_df["Token"].values
            token2index = {}
            # Stock codes are six digits; the CSV reader may have turned "000001" into the integer 1, so numeric
            # tokens are zero-padded back to six characters.
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
    """Builds the `MLMDataset` of a `DatasetConfig` over its corpus; masking and batching are left to the loader."""

    def __init__(self):
        super().__init__()
        self.config: DatasetConfig = DatasetConfig()

    def prepare(self, tokenizer: PreTrainedTokenizerFast) -> MLMDataset:
        """Open the corpus of `config.data_path` (CSV files, or a corpus directory memory-mapped) and wrap it.

        Args:
            tokenizer: The prepared tokenizer that encodes each portfolio.

        Returns:
            The dataset of unmasked samples.
        """
        self.logger.info("Preparing Dataset...")
        cfg = self.config
        corpus = PortfolioCorpus.open(
            cfg.data_path,
            include_proportion=cfg.include_proportion,
            id_key=cfg.id_key,
            portfolio_key=cfg.portfolio_key,
            proportion1_key=cfg.proportion1_key,
            proportion2_key=cfg.proportion2_key,
        )
        dataset = MLMDataset(corpus, tokenizer, cfg.max_length, include_proportion=cfg.include_proportion)
        self.logger.info(f"Dataset prepared. Size: {len(dataset)}")
        return dataset


class DataLoader_Preparer(Preparer):
    """Wraps a dataset in a `DataLoader` with the batching and worker settings of a `DataLoaderConfig`."""

    def __init__(self):
        super().__init__()
        self.config: DataLoaderConfig = DataLoaderConfig()

    def prepare(
        self,
        dataset: Dataset,
        shuffle_override: Optional[bool] = None,
        collate_fn: Optional[Callable] = None,
        num_repeats: int = 1,
        name: Optional[str] = None,
    ) -> DataLoader:
        """Build the dataloader over `dataset`.

        Logs one line with the loader's name, batch count and shuffle setting, and a second one only when
        ``shuffle_override`` differs from ``config.shuffle``.

        Args:
            dataset: The dataset to batch.
            shuffle_override: Replaces `config.shuffle` when given; the trainer forces `False` on validation
                loaders and `True` on the training split.
            collate_fn: Turns a list of samples into a batch (the masking collator); None uses PyTorch's default.
            num_repeats: Times the dataset is replayed per pass. Above 1 the loader draws ``len(dataset) *
                num_repeats`` indices: shuffled, as ``RandomSampler`` with that many samples; else the indices in
                order, repeated. At 1 the loader is built without a sampler of its own.
            name: The split the loader serves (``"train"``, ``"validation"``, ...), shown in the log lines.

        Returns:
            The dataloader.
        """
        label = f"DataLoader ({name})" if name else "DataLoader"
        shuffle = self.config.shuffle if shuffle_override is None else shuffle_override
        if shuffle != self.config.shuffle:
            self.logger.info(f"{label}: shuffle {shuffle} instead of the configured {self.config.shuffle}.")

        sampling = {"shuffle": shuffle}
        if num_repeats > 1:
            n = len(dataset)
            sampler = RandomSampler(dataset, num_samples=n * num_repeats) if shuffle else list(range(n)) * num_repeats
            sampling = {"sampler": sampler}
            self.logger.info(f"Replaying the dataset {num_repeats} times per pass ({n * num_repeats} samples).")
        dataloader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            **sampling,
            collate_fn=collate_fn,
            num_workers=self.config.num_workers,
            persistent_workers=self.config.persistent_workers,
            pin_memory=self.config.pin_memory,
            drop_last=self.config.drop_last,
        )
        self.logger.info(f"{label} prepared: {len(dataloader)} batches, shuffle {shuffle}.")
        return dataloader


class Accelerator_Preparer(Preparer):
    """Builds the HuggingFace `Accelerator` of an `AcceleratorConfig`.

    Sets the torch.compile cache location, the TF32 flags and the `TorchDynamoPlugin` before constructing the
    accelerator. Calling `accelerator.prepare(model, optimizer, ...)` stays with the trainer.
    """

    def __init__(self):
        super().__init__()
        self.config: AcceleratorConfig = AcceleratorConfig()

    def prepare(self) -> Accelerator:
        """Build the accelerator and apply the process-wide compile and TF32 settings on the way.

        Returns:
            The accelerator.
        """
        config = self.config

        # Inductor reads these variables when it first compiles, so they are set before the accelerator and
        # its dynamo plugin exist.
        if config.compile_cache_dir is not None:
            os.environ["TORCHINDUCTOR_CACHE_DIR"] = config.compile_cache_dir
            os.environ["TORCHINDUCTOR_FX_GRAPH_CACHE"] = "1"
            Path(config.compile_cache_dir).mkdir(parents=True, exist_ok=True)
            self.logger.info(f"Compile cache dir: {config.compile_cache_dir}")

        torch.backends.cuda.matmul.allow_tf32 = config.allow_tf32
        torch.backends.cudnn.allow_tf32 = config.allow_tf32
        if config.allow_tf32:
            self.logger.info("TF32 enabled")

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

        kwargs = dict(
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            dynamo_plugin=dynamo_plugin,
        )
        # "auto" leaves mixed precision to accelerate's own default (its config file or environment), so the
        # kwarg is passed only when the config sets a value.
        if config.mixed_precision != "auto":
            kwargs["mixed_precision"] = config.mixed_precision

        accelerator = Accelerator(**kwargs)
        self.logger.info("Accelerator prepared.")
        return accelerator


class Optimizer_Preparer(Preparer):
    """Builds the optimizer and LR scheduler of an `OptimizerConfig`.

    Optimizer classes are looked up by name in a registry that imports the optional backends (bitsandbytes,
    lion-pytorch, dadaptation) on demand; `optimizer_kwargs` the class does not accept are dropped with a
    warning.
    """

    def __init__(self) -> None:
        super().__init__()
        self.config: OptimizerConfig = OptimizerConfig()

    def _optimizer_registry(self) -> Dict[str, tuple]:
        """Name -> (class_getter, lr_is_none, extra_construct_kwargs, prehook_or_None).

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
            kwargs["relative_step"] = True  # Adafactor's own default
        if not kwargs["relative_step"] and kwargs.get("warmup_init", False):
            self.logger.info("set relative_step to True because warmup_init is True.")
            kwargs["relative_step"] = True
        if kwargs["relative_step"]:
            if self.config.learning_rate != 0.0:
                self.logger.info("learning rate is used as initial_lr.")
            if self.config.lr_scheduler_type != "adafactor":
                self.logger.info("use adafactor_scheduler.")
                self.config.lr_scheduler_type = "adafactor"

    def prepare_optimizer(self, trainable_params: ParamsT) -> torch.optim.Optimizer:
        """Build the optimizer named by `config.optimizer_type` over `trainable_params`.

        Args:
            trainable_params: The parameters (or parameter groups) to optimize, as the optimizer class accepts them.

        Returns:
            The optimizer; Adafactor is built with `lr=None` and takes its rate from the schedule.

        Raises:
            ValueError: If `config.optimizer_type` is not in the registry.
        """
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
        """Build the transformers schedule named by `config.lr_scheduler_type` for `optimizer`.

        Args:
            optimizer: The optimizer the schedule drives.

        Returns:
            The schedule: constant, linear, cosine, cosine_with_restarts, polynomial or adafactor.

        Raises:
            ValueError: If `config.lr_scheduler_type` is not one of those names.
            AssertionError: If the adafactor schedule is asked for with an optimizer other than Adafactor.
        """
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

    def prepare(self, trainable_params: ParamsT) -> Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LRScheduler]:
        """Build the optimizer and then its LR scheduler.

        Args:
            trainable_params: The parameters (or parameter groups) to optimize.

        Returns:
            The `(optimizer, lr_scheduler)` pair.
        """
        optimizer = self.prepare_optimizer(trainable_params=trainable_params)
        lr_scheduler = self.prepare_lr_scheduler(optimizer=optimizer)
        return optimizer, lr_scheduler


def get_token2index_mapping(model: Word2Vec) -> Dict[str, int]:
    """Token-to-index mapping of a Word2Vec vocabulary, followed by `[MASK]`, `[PAD]`, `[UNK]` in that order.

    Args:
        model: The Word2Vec model whose `wv.index_to_key` fixes the order of the corpus tokens.

    Returns:
        The mapping, whose length is the Word2Vec vocabulary size plus three.
    """
    token2index = {word: i for i, word in enumerate(model.wv.index_to_key)}

    special_tokens = ["[MASK]", "[PAD]", "[UNK]"]
    for i, token in enumerate(special_tokens, start=len(token2index)):
        token2index[token] = i

    return token2index


def get_index2token_mapping(model: Word2Vec) -> Dict[int, str]:
    """Inverse of `get_token2index_mapping`: index to token, the three special tokens last.

    Args:
        model: The Word2Vec model whose `wv.index_to_key` fixes the order of the corpus tokens.

    Returns:
        The mapping, whose length is the Word2Vec vocabulary size plus three.
    """
    index2token = {i: word for i, word in enumerate(model.wv.index_to_key)}

    special_tokens = ["[MASK]", "[PAD]", "[UNK]"]
    for i, token in enumerate(special_tokens, start=len(index2token)):
        index2token[i] = token

    return index2token
