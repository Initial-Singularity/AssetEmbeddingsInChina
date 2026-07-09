import os
import sys
import json
import time
import logging
import argparse
import textwrap
import traceback
from datetime import datetime
from typing import Dict, Literal, Optional, Any

import pandas as pd
import numpy as np
from gensim.models import Word2Vec, KeyedVectors
from tqdm import tqdm

from asset_embeddings.datasets import Word2VecDataset
from asset_embeddings.configs import LoggerConfig, AssetW2VTrainConfig
from asset_embeddings.preparers import Logger_Preparer
from asset_embeddings.records import W2VTrainRecording, generate_run_id, RecordStore
from asset_embeddings import utils


class Word2Vec_Trainer:
    def __init__(self):
        self.dataset: Word2VecDataset = None
        self.model: Word2Vec = None
        self.config: AssetW2VTrainConfig = AssetW2VTrainConfig()
        self.logger: logging.Logger = None

        # Recording system
        self.run_id: str = ""
        self._config_file: Optional[str] = None
        self._record_store: Optional[RecordStore] = None
        self._recording: Optional[W2VTrainRecording] = None
        self._start_time: float = 0.0

    @utils.log_exceptions_inclass()
    def set_logger(self, log_name: str = "W2V Train", log_file: Optional[str] = None, console_level: str = "INFO"):
        self.logger = (
            Logger_Preparer()
            .set_config(
                LoggerConfig(
                    log_name=log_name,
                    log_file=log_file,
                    console_stream="tqdm",
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
    def set_config(self, config_dict: Optional[Dict[str, Any]], **config_kwargs):
        if config_dict is not None:
            self.config.from_kwargs(**config_dict)
        self.config.from_dict(config_kwargs)
        self.config.validate()

    def set_global_seed(self, seed: int):
        # gensim/numpy path — no torch RNGs.
        utils.seed_basic(seed, include_torch=False)

    def _build_vocab_reproducible(self, dataset):
        """Wrap gensim build_vocab with deterministic ordering and weight init.

        gensim's build_vocab sorts by frequency, but equal-frequency tokens
        (e.g. [CLS]/[SEP]) have platform-dependent ordering.  This method:
          1. Lets gensim handle freq counting and internal tables.
          2. Reorders the vocab deterministically: (freq DESC, token ASC).
          3. Reinitializes weight matrices from a fixed seed so that the
             same token always receives the same initial vector.
        """
        self.model.build_vocab(dataset)

        # --- deterministic reorder ---
        wv = self.model.wv
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

        # --- reinitialize weights (mirrors gensim prep_vectors logic) ---
        seed = self.config["seed"]
        rng = np.random.default_rng(seed=seed)
        n, dim = wv.vectors.shape
        wv.vectors = (rng.random((n, dim), dtype=np.float32) * 2.0 - 1.0) / dim
        if hasattr(self.model, "syn1neg"):
            self.model.syn1neg = np.zeros((n, dim), dtype=np.float32)
        if hasattr(self.model, "syn1"):
            self.model.syn1 = np.zeros((n, dim), dtype=np.float32)

    @utils.log_exceptions_inclass()
    def load_pretrained(self, model_path: str, freeze_pretrained: bool = False):
        # Load the pretrained Word2Vec model
        self.logger.info(f"Loading Pretrained Model from `{model_path}`")
        if model_path.endswith(".model"):
            pretrained_model = Word2Vec.load(model_path)
        elif model_path.endswith(".txt"):
            pretrained_model = KeyedVectors.load_word2vec_format(model_path, binary=False)

        current_vocab = sorted(
            list(set(word for sentence in self.dataset for word in sentence))
        )  # construct current vocabulary from the dataset

        # 3. Initialize the word vectors for the new model
        new_vectors = {}
        for word in current_vocab:
            if word in pretrained_model.wv:
                # Use the pretrained word vector
                # self.logger.debug(f"Word `{word}` found in pretrained model, using pretrained vector.")
                new_vectors[word] = pretrained_model.wv[word]
            else:
                # Randomly initialize the word vector
                self.logger.debug(f"Word `{word}` not found in pretrained model, initializing randomly.")
                new_vectors[word] = np.random.uniform(-0.25, 0.25, self.config["embedding_dim"])

        # 5. 将预训练的词向量赋值给新模型
        for word in self.model.wv.index_to_key:
            if word in new_vectors:
                self.model.wv[word] = new_vectors[word]
                if freeze_pretrained and word in pretrained_model.wv:
                    # 冻结预训练词向量的权重
                    self.logger.debug(f"Freezing pretrained word `{word}`")
                    self.model.trainables.syn1neg[self.model.wv.key_to_index[word]] = np.zeros_like(
                        self.model.trainables.syn1neg[self.model.wv.key_to_index[word]]
                    )
            else:
                self.logger.debug(f"Word `{word}` not found in pretrained model, initializing randomly.")
                self.model.wv[word] = np.random.uniform(-0.25, 0.25, self.config["embedding_dim"])

    @utils.log_exceptions_inclass()
    def prepare(self):
        self.logger.info("Preparing Trainer...")
        self.set_global_seed(self.config["seed"])
        self.logger.info("Loading Dataset...")
        self.dataset = Word2VecDataset(
            path=self.config.data_path,
            format=self.config.data_format,
            id_key=self.config.id_key,
            portfolio_key=self.config.portfolio_key,
        )

        self.logger.info("Initializing Model...")
        self.model = Word2Vec(
            vector_size=self.config["embedding_dim"],
            window=self.config["window"],
            min_count=self.config["min_count"],
            sg=self.config["sg"],
            sample=self.config["sample"],
            negative=self.config["negative_sample"],
            seed=self.config["seed"],
            workers=self.config["workers"],
        )

        self.logger.info("Building Vocabulary...")
        self._build_vocab_reproducible(self.dataset)

        self.logger.info(f"Vocabulary Size: {len(self.model.wv)}")
        self.logger.info(f"Corpus Count: {self.model.corpus_count}")

        if self.config["pretrained_model"]:
            self.load_pretrained(self.config["pretrained_model"])

        self.logger.info("Trainer Prepared.")

        # `self.run_id` is generated in main() before prepare() so that
        # error recordings during prepare can reference it.
        self._start_time = time.time()
        self.logger.info(f"Configs: {self.config.to_dict()}")
        os.makedirs(self.config.save_folder, exist_ok=True)
        self.config.to_file(os.path.join(self.config.save_folder, "config.json"))
        self.logger.info(f"Config file saved to `{os.path.join(self.config.save_folder, 'config.json')}`")

        # Initialize recording
        if self._record_store:
            task = "finetune_W2V" if self.config["pretrained_model"] else "pretrain_W2V"
            self._recording = W2VTrainRecording(
                run_id=self.run_id,
                recorded_at=datetime.now().isoformat(),
                config_file=self._config_file,
                config_content=json.dumps(self.config.to_dict(), ensure_ascii=False),
                status="running",
                task=task,
                model="AssetW2V",
                seed=self.config["seed"],
                total_epochs=self.config["epochs"],
                vocab_size=len(self.model.wv),
                corpus_count=self.model.corpus_count,
                embedding_dim=self.config["embedding_dim"],
            )
            self._record_store.save_or_replace(self._recording, "train_w2v_recordings")

    @utils.log_exceptions_inclass()
    def train(self):
        self.logger.info("Training Word2Vec...")
        final_loss = 0.0
        for epoch in tqdm(range(self.config["epochs"]), desc="Training Progress", unit="epoch"):
            self.model.train(self.dataset, total_examples=self.model.corpus_count, epochs=1)
            final_loss = self.model.get_latest_training_loss()
            self.logger.debug(f"Epoch[{epoch+1}] Loss: {final_loss}")

            # Incremental recording update
            if self._recording and self._record_store:
                self._recording.actual_epochs = epoch + 1
                self._recording.final_loss = final_loss
                self._record_store.save_or_replace(self._recording, "train_w2v_recordings")

        embedding_path = os.path.join(self.config["save_folder"], self.config["save_name"] + "_embedding.csv")
        self.save(self.config["save_folder"], self.config["save_name"], self.config["save_format"])
        self.save_embedding(embedding_path)

        # Final recording update
        if self._recording and self._record_store:
            self._recording.embedding_path = embedding_path
            self._recording.duration_sec = time.time() - self._start_time
            self._recording.status = "completed"
            self._recording.recorded_at = datetime.now().isoformat()
            self._record_store.save_or_replace(self._recording, "train_w2v_recordings")

    def save(self, save_folder: str, save_name: str, save_format: Literal[".model", ".txt"] = "model"):
        save_path = os.path.join(save_folder, save_name + save_format)
        if save_format == ".model":
            self.model.save(save_path)
        elif save_format == ".txt":
            self.model.wv.save_word2vec_format(save_path, binary=False)  # 兼容性格式
        self.logger.info(f"Word2Vec model saved to `{save_path}`")

    def save_embedding(self, save_path: str):
        if save_path[-4:] != ".csv":
            save_path += ".csv"
        embedding_df = pd.DataFrame(
            self.model.wv.vectors, columns=[f"Embed_{i+1}" for i in range(self.config["embedding_dim"])]
        )
        embedding_df.insert(0, "Token", self.model.wv.index_to_key)
        embedding_df.to_csv(save_path, index=False)
        self.logger.info(f"Word2Vec embedding saved to `{save_path}`")


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train a Stock Embedding Model using Word2Vec architecture.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""
        Configuration File Example:
        ---------------------------
        {
            "pretrained_model": null,
            "embedding_dim": 4,
            "epochs": 10,
            "window": 5,
            "min_count": 1,
            "sg": 1,
            "sample": 0.001,
            "negative_sample": 10,
            "seed": 42,
            "data_path": "data/processed/ShareHoldingSplit/base",
            "data_format": "csv",
            "workers": 1,
            "save_folder": "model/base/AssetW2V/d4",
            "save_name": "AssetW2V_d4_base",
            "save_format": ".model"
        }

        Configuration File Descriptions:
        -------------------------------
        pretrained_model   : Optional pretrained model path (set to null for training from scratch).
        embedding_dim      : Word embedding dimension (vector size for each stock).
        epochs             : Number of training epochs.
        window             : Context window size for co-occurrence.
        min_count          : Minimum frequency threshold for stocks (1 recommended).
        sg                 : Training mode: 1 for Skip-Gram, 0 for CBOW.
        sample             : Threshold for downsampling frequent stocks (e.g., 0.001).
        negative_sample    : Number of negative samples per positive.
        seed               : Random seed for reproducibility.
        data_path          : Path to input data (CSV, JSON, or binary).
        data_format        : Format of input data: csv, json, binary.
        id_key             : Column name for investor ID (default: "InvestorID").
        portfolio_key      : Column name for portfolio data (default: "Portfolio").
        workers            : Number of parallel workers. (1 recommended for reproducibility)
        save_folder        : Directory to save the trained model.
        save_name          : Filename of the saved model (without extension).
        save_format        : Model save format (e.g., `.model`, `.bin`).
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
        help="Override configuration parameters. Format: key=value (e.g., window=10 seed=123). Optional.",
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
    w2v_trainer = Word2Vec_Trainer()
    console_level = "DEBUG" if args.verbose else "INFO"
    w2v_trainer.set_logger(log_file=args.log, console_level=console_level)
    w2v_trainer.logger.debug(f"Command: {' '.join(sys.argv)}")
    w2v_trainer.load_config(args.config)
    w2v_trainer._config_file = args.config
    if args.override:
        w2v_trainer.load_overrides(utils.parse_overrides(args.override))

    # Generate run_id and wire record_store BEFORE prepare() so that any
    # failure during prepare() is attributable in the DB even if
    # `_recording` was never constructed.
    w2v_trainer.run_id = generate_run_id()
    if args.result_file:
        w2v_trainer._record_store = RecordStore(args.result_file, logger=w2v_trainer.logger)

    try:
        w2v_trainer.prepare()
        w2v_trainer.train()
    except Exception:
        if w2v_trainer._record_store:
            if w2v_trainer._recording is None:
                # prepare() failed before the recording was constructed.
                # `task` derivation must mirror prepare()'s logic.
                task = "finetune_W2V" if w2v_trainer.config["pretrained_model"] else "pretrain_W2V"
                w2v_trainer._recording = W2VTrainRecording(
                    run_id=w2v_trainer.run_id,
                    recorded_at=datetime.now().isoformat(),
                    config_file=w2v_trainer._config_file,
                    config_content=json.dumps(w2v_trainer.config.to_dict(), ensure_ascii=False),
                    task=task,
                    model="AssetW2V",
                )
            w2v_trainer._recording.status = "error"
            w2v_trainer._recording.error_message = traceback.format_exc()
            w2v_trainer._recording.duration_sec = (
                time.time() - w2v_trainer._start_time if w2v_trainer._start_time else None
            )
            w2v_trainer._recording.recorded_at = datetime.now().isoformat()
            w2v_trainer._record_store.save_or_replace(w2v_trainer._recording, "train_w2v_recordings")
        raise


if __name__ == "__main__":
    main(get_parser().parse_args())
