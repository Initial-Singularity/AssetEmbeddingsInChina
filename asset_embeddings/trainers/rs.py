"""Recommender-system embeddings (PCA on the investor-asset matrix): the trainer behind ``python -m asset_embeddings.scripts.train.rs``."""

import os
import json
import time
import pickle
import logging
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Dict, List, Tuple, Optional


import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.decomposition import PCA

from asset_embeddings import utils
from asset_embeddings.configs import AssetRSTrainConfig
from asset_embeddings.datasets.corpus import PortfolioCorpus
from asset_embeddings.records import RSComponent, RSFit

from .base import BaseTrainer


logger = logging.getLogger(__name__)


class RS_Base(ABC):
    """Holdings of every investor as an investor-by-asset matrix; subclasses decide what a cell holds.

    Args:
        investor_portfolios: ``InvestorID -> (assets, proportions)``; the asset axis is the sorted union of
            all portfolios.
    """

    def __init__(self, investor_portfolios: Dict[str, Tuple[List[str], List[float]]]):
        self.investor_portfolios = investor_portfolios
        self.investors = np.array(list(investor_portfolios.keys()))

        all_assets = set()
        for portfolio, _ in investor_portfolios.values():
            all_assets.update(portfolio)
        self.assets = np.array(sorted(all_assets))

    @abstractmethod
    def transform(self) -> np.ndarray:
        """The ``(n_investors, n_assets)`` matrix this variant feeds to PCA."""
        pass

    def build_matrix(self, fill_value: float = 0) -> np.ndarray:
        """The ``(n_investors, n_assets)`` matrix of proportion1 values, ``fill_value`` where an asset is not held.

        Args:
            fill_value: Cell value of a non-holding.

        Returns:
            A float array in ``self.investors`` by ``self.assets`` order.
        """
        matrix = np.full((len(self.investors), len(self.assets)), fill_value, dtype=float)

        asset_to_idx = {asset: idx for idx, asset in enumerate(self.assets)}

        for investor_idx, investor_id in enumerate(self.investors):
            portfolio, proportions = self.investor_portfolios[investor_id]
            for asset, proportion in zip(portfolio, proportions):
                if asset in asset_to_idx:
                    asset_idx = asset_to_idx[asset]
                    matrix[investor_idx, asset_idx] = proportion

        return matrix


class RS_Binary(RS_Base):
    """Cell is 1 when the investor holds the asset, else 0."""

    def transform(self) -> np.ndarray:
        """Binary holding matrix: any positive proportion becomes 1."""
        matrix = self.build_matrix(fill_value=0)
        matrix = (matrix > 0).astype(int)
        return matrix


class RS_Ranks(RS_Base):
    """Cell is the asset's rank among the investor's holdings by proportion1, scaled into ``[0, 1)``."""

    def transform(self) -> np.ndarray:
        """Rank matrix: within each investor, holdings ranked by proportion (larger stake, larger value)."""
        rank_matrix = np.zeros((len(self.investors), len(self.assets)))
        asset_to_idx = {asset: idx for idx, asset in enumerate(self.assets)}

        for investor_idx, investor_id in enumerate(self.investors):
            portfolio, proportions = self.investor_portfolios[investor_id]
            if len(portfolio) > 0:
                # Ties share the lower rank; (rank - 1) / (n - 0.9) maps ranks into [0, 1), so the smallest
                # holding gets 0, the same cell value as a non-holding.
                ranks = rankdata(proportions, method="min")
                ranks_normalized = (ranks - 1) / (len(ranks) - 0.9)

                for asset, rank in zip(portfolio, ranks_normalized):
                    if asset in asset_to_idx:
                        asset_idx = asset_to_idx[asset]
                        rank_matrix[investor_idx, asset_idx] = rank

        return rank_matrix


class RS_Level0(RS_Base):
    """Cell is the proportion1 value itself, 0 where the asset is not held."""

    def transform(self) -> np.ndarray:
        """Proportion matrix with 0 for a non-holding."""
        return self.build_matrix(fill_value=0).astype(float)


class RS_LevelMin(RS_Base):
    """Cell is the proportion1 value; a non-holding gets the investor's smallest proportion instead of 0."""

    def transform(self) -> np.ndarray:
        """Proportion matrix where each investor's non-holdings are set to that investor's minimum proportion."""
        matrix = self.build_matrix(fill_value=0)

        asset_to_idx = {asset: idx for idx, asset in enumerate(self.assets)}

        for investor_idx, investor_id in enumerate(self.investors):
            portfolio, proportions = self.investor_portfolios[investor_id]
            if len(proportions) > 0:
                min_proportion = min(proportions)

                held_assets = set(portfolio)
                for asset in self.assets:
                    if asset not in held_assets:
                        asset_idx = asset_to_idx[asset]
                        matrix[investor_idx, asset_idx] = min_proportion

        return matrix


class AssetRS_Trainer(BaseTrainer):
    """Fits one PCA on the transposed holding matrix and writes the pickled model, the embedding CSV and the records.

    The ``model_type`` of the config picks the ``RS_*`` matrix variant. Outputs: ``<save_name>_model.pkl``,
    ``<save_name>_embedding.csv`` (``Token, Embed_1, ..., Embed_N``, one row per asset) and one
    ``train_rs_summary`` row with one ``train_rs_components`` row per component.
    """

    def __init__(self):
        super().__init__()
        self.config: AssetRSTrainConfig = AssetRSTrainConfig()

        self.recommender_system: RS_Base
        self.matrix: np.ndarray
        self.pca: PCA

        self.embeddings: np.ndarray

        self._fit: Optional[RSFit] = None

    @property
    def task_name(self) -> str:
        """Every RS variant is one PCA fit: ``fit_RS``."""
        return "fit_RS"

    @property
    def save_folder(self) -> str:
        """``save_folder`` of the config."""
        return self.config.save_folder

    def prepare(self) -> None:
        """Load the portfolios, build the matrix of the configured variant, construct the PCA and insert the ``train_rs_summary`` row.

        Raises:
            ValueError: ``model_type`` names no ``RS_*`` variant.
        """
        self.save_config()

        corpus = PortfolioCorpus.open(
            self.config.data_path,
            include_proportion=True,
            id_key=self.config.id_key,
            portfolio_key=self.config.portfolio_key,
            proportion1_key=self.config.proportion1_key,
            proportion2_key=self.config.proportion2_key,
        )
        investor_portfolios = corpus.holdings(include_proportion=True)

        mem_info = utils.get_memory_usage()
        self.logger.debug(
            f"Current memory usage: {mem_info['used_memory_mb']:.2f} MB ({mem_info['percent_of_total']:.2f}%)."
        )

        if self.config.model_type == "RS_Binary":
            self.recommender_system = RS_Binary(investor_portfolios)
        elif self.config.model_type == "RS_Ranks":
            self.recommender_system = RS_Ranks(investor_portfolios)
        elif self.config.model_type == "RS_Level0":
            self.recommender_system = RS_Level0(investor_portfolios)
        elif self.config.model_type == "RS_LevelMin":
            self.recommender_system = RS_LevelMin(investor_portfolios)
        else:
            raise ValueError(f"Unsupported model type: {self.config.model_type}")

        self.logger.info("Constructing Asset-Investor matrix")
        self.matrix = self.recommender_system.transform()
        self.logger.info(f"Matrix shape: {self.matrix.shape}")
        self.logger.info("Preparing PCA model")
        self.pca = PCA(n_components=self.config.n_components, whiten=self.config.whiten, random_state=self.config.seed)
        self.logger.info("PCA model prepared.")

        self._fit = RSFit(n_investors=int(self.matrix.shape[0]), n_assets=int(self.matrix.shape[1]))
        self.run.save(self._fit, replace=True)

    def train(self) -> None:
        """Fit the PCA on the asset-by-investor matrix, save the model and the embedding, complete the records."""
        self.logger.info("Training PCA model...")
        self.embeddings = self.pca.fit_transform(self.matrix.T)
        self.logger.info("PCA model trained.")
        self.save_model(os.path.join(self.config.save_folder, f"{self.config.save_name}_model.pkl"))
        self.logger.info(
            f"PCA model saved to `{os.path.join(self.config.save_folder, f'{self.config.save_name}_model.pkl')}`"
        )
        embedding_path = os.path.join(self.config.save_folder, f"{self.config.save_name}_embedding.csv")
        self.save_embedding(embedding_path)
        self.logger.info(f"Asset embeddings saved to `{embedding_path}`")

        if self._fit is not None:
            self._fit.embedding_path = embedding_path
            components = [
                RSComponent(component=i, explained_variance_ratio=float(r))
                for i, r in enumerate(self.pca.explained_variance_ratio_, start=1)
            ]
            self.run.save(self._fit, replace=True)
            self.run.save_many(components)

    def save_model(self, save_path: str) -> None:
        """Pickle the fitted PCA.

        Args:
            save_path: Target file; ``.pkl`` is appended when missing.
        """
        if save_path[-4:] != ".pkl":
            save_path += ".pkl"
        with open(save_path, "wb") as f:
            pickle.dump(self.pca, f)

    def save_embedding(self, save_path: str) -> None:
        """Write the asset scores as a ``Token, Embed_1, ..., Embed_N`` CSV, rows in ``assets`` order.

        Args:
            save_path: Target file; ``.csv`` is appended when missing.
        """
        if save_path[-4:] != ".csv":
            save_path += ".csv"
        embedding_df = pd.DataFrame(self.embeddings, columns=[f"Embed_{i+1}" for i in range(self.config.n_components)])
        embedding_df.insert(0, "Token", self.recommender_system.assets)
        embedding_df.to_csv(save_path, index=False)
