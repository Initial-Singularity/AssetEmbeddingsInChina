"""Asset Word2Vec training (gensim): the trainer behind ``python -m asset_embeddings.scripts.train.w2v``."""

import os
import time
import logging
from datetime import datetime
from typing import Optional

import pandas as pd
from gensim.models import Word2Vec
from tqdm import tqdm

from asset_embeddings.datasets.corpus import PortfolioCorpus, Sentences
from asset_embeddings.configs import AssetW2VTrainConfig
from asset_embeddings.preparers import Word2Vec_Preparer
from asset_embeddings.records import W2VTraining
from asset_embeddings import utils

from .base import BaseTrainer


logger = logging.getLogger(__name__)


class Word2Vec_Trainer(BaseTrainer):
    """Trains a gensim Word2Vec over holding sequences and writes the ``.model`` file, the embedding CSV and the training row.

    The model itself comes from [`Word2Vec_Preparer`][asset_embeddings.preparers.Word2Vec_Preparer]. Outputs:
    ``<save_name>.model``, ``<save_name>_embedding.csv`` (``Token, Embed_1, ..., Embed_N``) and one
    ``train_w2v_summary`` row, refreshed after every epoch.
    """

    def __init__(self):
        super().__init__()
        self.config: AssetW2VTrainConfig = AssetW2VTrainConfig()
        self.dataset: Sentences = None
        self.model: Word2Vec = None
        self.model_preparer: Word2Vec_Preparer = Word2Vec_Preparer()

        self._training: Optional[W2VTraining] = None

    def _configs_changed(self) -> None:
        self.model_preparer.set_config(self.config)

    @property
    def task_name(self) -> str:
        """``finetune_W2V`` when a pretrained model is loaded, else ``pretrain_W2V``."""
        return "finetune_W2V" if self.config.pretrained_model else "pretrain_W2V"

    @property
    def save_folder(self) -> str:
        """``save_folder`` of the config."""
        return self.config.save_folder

    def prepare(self) -> None:
        """Seed, load the holding sequences, build the model and insert the training row."""
        self.logger.info("Preparing Trainer...")
        utils.seed_basic(self.config.seed, include_torch=False)  # gensim/numpy path, no torch RNGs

        corpus = PortfolioCorpus.open(
            self.config.data_path, id_key=self.config.id_key, portfolio_key=self.config.portfolio_key
        )
        self.dataset = corpus.sentences()
        self.model = self.model_preparer.prepare(self.dataset)
        self.logger.info("Trainer Prepared.")

        self.save_config()

        self._training = W2VTraining(epochs_run=0, n_vocab=len(self.model.wv), n_sentences=self.model.corpus_count)
        self.run.save(self._training, replace=True)

    def train(self) -> None:
        """Train one epoch at a time, refreshing the training row after each, then save the model and the embedding."""
        self.logger.info("Training Word2Vec...")
        final_loss = 0.0
        for epoch in tqdm(range(self.config.epochs), desc="Training Progress", unit="epoch"):
            self.model.train(self.dataset, total_examples=self.model.corpus_count, epochs=1)
            final_loss = self.model.get_latest_training_loss()
            self.logger.debug(f"Epoch[{epoch + 1}] Loss: {final_loss}")

            if self._training is not None:
                self._training.epochs_run = epoch + 1
                self.run.save(self._training, replace=True)

        embedding_path = os.path.join(self.config.save_folder, self.config.save_name + "_embedding.csv")
        self.save(self.config.save_folder, self.config.save_name)
        self.save_embedding(embedding_path)

        if self._training is not None:
            self._training.embedding_path = embedding_path
            self.run.save(self._training, replace=True)

    def save(self, save_folder: str, save_name: str) -> None:
        """Write ``<save_name>.model``: the gensim file every other consumer (``pretrained_model``, ``w2v_model``) loads.

        Args:
            save_folder: Target directory.
            save_name: File stem.
        """
        save_path = os.path.join(save_folder, save_name + ".model")
        self.model.save(save_path)
        self.logger.info(f"Word2Vec model saved to `{save_path}`")

    def save_embedding(self, save_path: str) -> None:
        """Write the word vectors as a ``Token, Embed_1, ..., Embed_N`` CSV, rows in vocabulary order.

        Args:
            save_path: Target file; ``.csv`` is appended when missing.
        """
        if save_path[-4:] != ".csv":
            save_path += ".csv"
        embedding_df = pd.DataFrame(
            self.model.wv.vectors, columns=[f"Embed_{i + 1}" for i in range(self.config.embedding_dim)]
        )
        embedding_df.insert(0, "Token", self.model.wv.index_to_key)
        embedding_df.to_csv(save_path, index=False)
        self.logger.info(f"Word2Vec embedding saved to `{save_path}`")
