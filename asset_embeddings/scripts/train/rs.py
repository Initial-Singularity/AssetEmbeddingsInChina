import os
import sys
import json
import time
import pickle
import logging
import textwrap
import argparse
import traceback
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional


import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.decomposition import PCA

from asset_embeddings import utils
from asset_embeddings.configs import LoggerConfig, AssetRSTrainConfig
from asset_embeddings.preparers import Logger_Preparer
from asset_embeddings.datasets import RSDataset
from asset_embeddings.records import RSTrainRecording, generate_run_id, RecordStore


class RS_Base(ABC):
    def __init__(self, investor_portfolios: Dict[str, Tuple[List[str], List[float]]]):
        """
        Initialize recommender system base class.

        Args:
            investor_portfolios: Dict mapping InvestorID to (list of assets, list of proportions)
        """
        self.investor_portfolios = investor_portfolios
        self.investors = np.array(list(investor_portfolios.keys()))

        # Extract all unique assets from portfolios
        all_assets = set()
        for portfolio, _ in investor_portfolios.values():
            all_assets.update(portfolio)
        self.assets = np.array(sorted(all_assets))

    @abstractmethod
    def transform(self):
        pass

    def build_matrix(self, fill_value=0):
        """构建投资者-资产持有矩阵，使用proportion1值"""
        matrix = np.full((len(self.investors), len(self.assets)), fill_value, dtype=float)

        # Create index mappings for efficient lookup
        asset_to_idx = {asset: idx for idx, asset in enumerate(self.assets)}

        for investor_idx, investor_id in enumerate(self.investors):
            portfolio, proportions = self.investor_portfolios[investor_id]
            for asset, proportion in zip(portfolio, proportions):
                if asset in asset_to_idx:
                    asset_idx = asset_to_idx[asset]
                    matrix[investor_idx, asset_idx] = proportion

        return matrix


class RS_Binary(RS_Base):
    def transform(self):
        """Transform to binary holding matrix (1 if proportion exists, 0 otherwise)"""
        matrix = self.build_matrix(fill_value=0)
        # Binarize: any positive proportion becomes 1
        matrix = (matrix > 0).astype(int)
        return matrix


class RS_Ranks(RS_Base):
    def transform(self):
        """Transform to rank-based matrix (rank by proportion1 values within each investor)"""
        rank_matrix = np.zeros((len(self.investors), len(self.assets)))
        asset_to_idx = {asset: idx for idx, asset in enumerate(self.assets)}

        for investor_idx, investor_id in enumerate(self.investors):
            portfolio, proportions = self.investor_portfolios[investor_id]
            if len(portfolio) > 0:
                # Rank by proportion values (higher proportion = higher rank)
                ranks = rankdata(proportions, method="min")
                # Normalize to approximately [0, 1]
                ranks_normalized = (ranks - 1) / (len(ranks) - 0.9)

                for asset, rank in zip(portfolio, ranks_normalized):
                    if asset in asset_to_idx:
                        asset_idx = asset_to_idx[asset]
                        rank_matrix[investor_idx, asset_idx] = rank

        return rank_matrix


class RS_Level0(RS_Base):
    def transform(self):
        """Use proportion values directly, fill missing with 0"""
        return self.build_matrix(fill_value=0).astype(float)


class RS_LevelMin(RS_Base):
    def transform(self):
        """Use proportion values, fill missing with each investor's minimum proportion"""
        matrix = self.build_matrix(fill_value=0)

        # For each investor, set non-held assets to their minimum proportion value
        asset_to_idx = {asset: idx for idx, asset in enumerate(self.assets)}

        for investor_idx, investor_id in enumerate(self.investors):
            portfolio, proportions = self.investor_portfolios[investor_id]
            if len(proportions) > 0:
                # Find minimum proportion for this investor
                min_proportion = min(proportions)

                # Set non-held assets to minimum proportion
                held_assets = set(portfolio)
                for asset in self.assets:
                    if asset not in held_assets:
                        asset_idx = asset_to_idx[asset]
                        matrix[investor_idx, asset_idx] = min_proportion

        return matrix


class AssetRS_Trainer:
    def __init__(self):
        self.config: AssetRSTrainConfig = AssetRSTrainConfig()
        self.logger: logging.Logger = None

        self.recommender_system: RS_Base
        self.matrix: np.ndarray
        self.pca: PCA

        self.embeddings: np.ndarray

        # Recording system
        self.run_id: str = ""
        self._config_file: Optional[str] = None
        self._record_store: Optional[RecordStore] = None
        self._recording: Optional[RSTrainRecording] = None
        self._start_time: float = 0.0

    @utils.log_exceptions_inclass()
    def set_logger(self, log_name: str = "RS Train", log_file: Optional[str] = None, console_level: str = "INFO"):
        self.logger = (
            Logger_Preparer()
            .set_config(
                LoggerConfig(
                    log_name=log_name,
                    log_file=log_file,
                    console_stream="std",
                    console_level=console_level,
                    file_level="DEBUG",
                    enable_colors=True,
                    color_target="format",
                )
            )
            .prepare()
        )

    @utils.log_exceptions_inclass()
    def load_config(self, config_file: str):
        self.logger.info(f"Loading configuration from `{os.path.normpath(config_file)}`")
        self.config.from_file(config_file)
        self.config.validate()

    @utils.log_exceptions_inclass()
    def load_overrides(self, overrides: Dict[str, Any]):
        self.logger.info(f"Applying overrides: {overrides}")
        self.config.from_dict(overrides)
        self.config.validate()

    @utils.log_exceptions_inclass()
    def prepare(self):
        os.makedirs(self.config.save_folder, exist_ok=True)
        self.logger.info(f"Configs: {self.config.to_dict()}")
        self.config.to_file(os.path.join(self.config.save_folder, "config.json"))
        self.logger.info(f"Config file saved to `{os.path.join(self.config.save_folder, 'config.json')}`")

        self.logger.info(f"Loading data from `{self.config.data_path}`")

        # Use RSDataset to load data with proportions
        dataset = RSDataset(
            path=self.config.data_path,
            format="csv",  # 根据文件路径自动推断
            id_key=self.config.id_key,
            portfolio_key=self.config.portfolio_key,
            proportion1_key=self.config.proportion1_key,
            proportion2_key=self.config.proportion2_key,
        )
        investor_portfolios = dataset.load_data(include_proportion=True)

        self.logger.info("Data loaded successfully.")
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

        # `self.run_id` is generated in main() before prepare() so that
        # error recordings during prepare can reference it.
        self._start_time = time.time()

        # Initialize recording
        if self._record_store:
            self._recording = RSTrainRecording(
                run_id=self.run_id,
                recorded_at=datetime.now().isoformat(),
                config_file=self._config_file,
                config_content=json.dumps(self.config.to_dict(), ensure_ascii=False),
                status="running",
                task="fit_RS",
                model=self.config.model_type,
                seed=self.config.seed,
                n_components=self.config.n_components,
                matrix_shape=str(self.matrix.shape),
            )
            self._record_store.save_or_replace(self._recording, "train_rs_recordings")

    @utils.log_exceptions_inclass()
    def train(self):
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

        # Final recording update
        if self._recording and self._record_store:
            evr = self.pca.explained_variance_ratio_
            self._recording.explained_variance_ratio = json.dumps(evr.tolist())
            self._recording.cumulative_variance = float(evr.sum())
            self._recording.embedding_path = embedding_path
            self._recording.duration_sec = time.time() - self._start_time
            self._recording.status = "completed"
            self._recording.recorded_at = datetime.now().isoformat()
            self._record_store.save_or_replace(self._recording, "train_rs_recordings")

    def save_model(self, save_path: str):
        if save_path[-4:] != ".pkl":
            save_path += ".pkl"
        with open(save_path, "wb") as f:
            pickle.dump(self.pca, f)

    def save_embedding(self, save_path: str):
        if save_path[-4:] != ".csv":
            save_path += ".csv"
        embedding_df = pd.DataFrame(self.embeddings, columns=[f"Embed_{i+1}" for i in range(self.config.n_components)])
        embedding_df.insert(0, "Token", self.recommender_system.assets)
        embedding_df.to_csv(save_path, index=False)


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train an asset recommender system embedding model using dimensionality reduction techniques (PCA).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""
        Configuration File Example:
        ---------------------------
        {
            "model_type": "RS_Binary",
            "n_components": 4,
            "whiten": false,
            "data_path": "data/processed/ShareHoldingIntermediate/2023Q2/2023Q2.csv",
            "save_folder": "model/finetune/AssetRS/RS_Binary/d4/2023Q2",
            "save_name": "AssetRS_Binary_d4_2023Q2",
            "save_format": ".pkl",
            "seed": 42,
            "id_key": "InvestorID",
            "portfolio_key": "Portfolio",
            "proportion1_key": "Proportion1",
            "proportion2_key": "Proportion2"
        }

        Configuration File Descriptions:
        --------------------------------
        model_type     : Recommender system model type. Supported values: "RS_Binary", "RS_Ranks", "RS_Level0", "RS_LevelMin".
        n_components   : Target embedding dimension (number of components after dimensionality reduction).
        whiten         : Whether to whiten the reduced features (normalize variance, e.g., in PCA).
        data_path      : Path to the input portfolio data file or directory (CSV format only).
        save_folder    : Output directory to save the model.
        save_name      : File name (without extension) of the saved model.
        save_format    : File format to save the model (e.g., ".pkl").
        seed           : Random seed for reproducibility.
        id_key         : Column name for investor ID (default: "InvestorID").
        portfolio_key  : Column name for portfolio data (default: "Portfolio").
        proportion1_key  : Column name for proportion1 data (default: "Proportion1").
        proportion2_key  : Column name for proportion2 data (default: "Proportion2").
        """),
    )

    parser.add_argument(
        "--config",
        "-c",
        type=str,
        required=True,
        help="Path to the JSON configuration file. See help below for structure. Required.",
    )
    parser.add_argument(
        "--log",
        "-l",
        type=str,
        nargs="*",
        default=None,
        help="Path(s) to save training logs. If not set, logs only to console. Multiple paths allowed. Optional.",
    )
    parser.add_argument(
        "--result_file",
        "-r",
        type=str,
        default=None,
        help="Path to save structured training records in SQLite format. Optional.",
    )
    parser.add_argument(
        "--override",
        "-o",
        type=str,
        nargs="+",
        default=None,
        help="Override configuration parameters. Format: key=value (e.g., n_components=10 seed=123). Optional.",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        default=False,
        help="Enable verbose (DEBUG) console output. Default: INFO.",
    )

    return parser


def main(args):
    rs_trainer = AssetRS_Trainer()
    console_level = "DEBUG" if args.verbose else "INFO"
    rs_trainer.set_logger(log_file=args.log, console_level=console_level)
    rs_trainer.logger.debug(f"Command: {' '.join(sys.argv)}")
    rs_trainer.load_config(args.config)
    rs_trainer._config_file = args.config
    if args.override:
        rs_trainer.load_overrides(utils.parse_overrides(args.override))

    # Generate run_id and wire record_store BEFORE prepare() so that any
    # failure during prepare() is attributable in the DB even if
    # `_recording` was never constructed.
    rs_trainer.run_id = generate_run_id()
    if args.result_file:
        rs_trainer._record_store = RecordStore(args.result_file, logger=rs_trainer.logger)

    try:
        rs_trainer.prepare()
        rs_trainer.train()
    except Exception:
        if rs_trainer._record_store:
            if rs_trainer._recording is None:
                # prepare() failed before the recording was constructed.
                rs_trainer._recording = RSTrainRecording(
                    run_id=rs_trainer.run_id,
                    recorded_at=datetime.now().isoformat(),
                    config_file=rs_trainer._config_file,
                    config_content=json.dumps(rs_trainer.config.to_dict(), ensure_ascii=False),
                    task="fit_RS",
                    model=rs_trainer.config.model_type,
                )
            rs_trainer._recording.status = "error"
            rs_trainer._recording.error_message = traceback.format_exc()
            rs_trainer._recording.duration_sec = (
                time.time() - rs_trainer._start_time if rs_trainer._start_time else None
            )
            rs_trainer._recording.recorded_at = datetime.now().isoformat()
            rs_trainer._record_store.save_or_replace(rs_trainer._recording, "train_rs_recordings")
        raise


if __name__ == "__main__":
    main(get_parser().parse_args())
