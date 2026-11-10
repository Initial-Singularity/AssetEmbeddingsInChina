"""Managed-portfolio (MP) refines: fit a masked-LM on fixed-index masked holdings and score it on held-out investors.

[`MPTrainer`][asset_embeddings.trainers.mp.MPTrainer] scores the loaded model on the test split (the baseline),
runs the shared epoch loop, restores the best epoch's state and scores the test split again. `MPBertTrainer`
refines an AssetBERT checkpoint; `MPSimplifiedMLMTrainer` fits the attention-free SimplifiedMLM head on a fixed
embedding. Behind ``python -m asset_embeddings.scripts.eval.mp_bert`` / ``mp_smlm``.
"""

import glob
import os
import shutil
from abc import abstractmethod
from datetime import datetime
from typing import Dict, Type

from asset_embeddings.configs import (
    AssetBERTModelConfig,
    Config,
    MPTrainConfig,
    RunMetadataConfig,
    SimplifiedMLMModelConfig,
)
from asset_embeddings.evaluators.mp import score_loader
from asset_embeddings.preparers import AssetBERT_Preparer, Preparer, SimplifiedMLM_Preparer
from asset_embeddings.records import MPResult, TrainingSummary

from .mlm import BaseMLMTrainer, FitSummary


class MPTrainer(BaseMLMTrainer):
    """One MP run: baseline score, refine, restore the best epoch, final score, the ``eval_mp_*`` rows.

    The investor file is split into train, validation and test sets by ``train.validation_split``,
    ``train.test_split`` and ``train.split_method``; ``dataset.mask_indices`` fixes the masked position. When the
    selection loss improves the ``accelerate`` state is saved to ``<save_name>_accelerator_best``; the periodic
    states are deleted after the loop, and the best state is what the final score reloads.
    """

    always_shuffle_train = True

    def __init__(self):
        super().__init__()
        self.baseline: tuple = (0.0, 0.0)
        self.final: tuple = (0.0, 0.0)

    @property
    def task_name(self) -> str:
        """Managed-portfolio refine and evaluation: ``MP``."""
        return "MP"

    @property
    def train_config_class(self) -> Type[Config]:
        """`MPTrainConfig`, shared by both heads."""
        return MPTrainConfig

    def extra_configs(self) -> Dict[str, Config]:
        """The ``run`` section (the period written to the records)."""
        return {"run": RunMetadataConfig()}

    @property
    def best_state_path(self) -> str:
        """Directory of the best epoch's ``accelerate`` state."""
        return self._checkpoint_path("accelerator_best")

    def on_periodic_save(self, epoch: int) -> None:
        """Save the ``accelerate`` state of the epoch."""
        self.save_checkpoint(self._checkpoint_path(f"accelerator_epoch{epoch + 1}"))
        self.logger.info(f"Epoch {epoch + 1} Accelerate Checkpoint saved to {self.config.train.save_folder}")

    def on_best(self, epoch: int) -> None:
        """Save the ``accelerate`` state of the new best epoch over the previous best."""
        self.save_checkpoint(self.best_state_path)
        self.logger.info(f"Epoch {epoch + 1} is the best so far; state saved to `{self.best_state_path}`")

    def on_fit_end(self, summary: FitSummary) -> None:
        """Delete the periodic states; the best state stays."""
        if not self.is_main:
            return
        for folder in glob.glob(self._checkpoint_path("accelerator_epoch*")):
            if os.path.isdir(folder):
                shutil.rmtree(folder)
        self.logger.info(f"Kept only the best state `{self.best_state_path}`")

    def train(self) -> None:
        """Score the test split, refine, reload the best state, score again and write the ``eval_mp_*`` rows.

        Without a test split nothing is scored or recorded (the refine still runs).
        """
        test = self.test_dataloader
        if test is None:
            self.logger.warning("No test dataloader available; the MP score is skipped.")
        else:
            self.logger.info("Scoring the test split before refining (baseline)...")
            self.baseline = score_loader(self.model, test, self.accelerator, desc="Pretrain Test")
            self.logger.info(f"Baseline MP Mean: {self.baseline[0]}, MP Std: {self.baseline[1]}")

        summary = self.fit()
        if test is None:
            return

        self.logger.info(f"Loading the best state from `{self.best_state_path}`")
        self.accelerator.load_state(self.best_state_path)
        self.final = score_loader(self.model, test, self.accelerator, desc="Test")
        if self.is_main:
            self.logger.info(
                f"Epoch({summary.best_epoch}), Test MP Mean: {self.final[0]}, Test MP Std: {self.final[1]}"
            )
            self._save_records(summary)

    def _save_records(self, summary: FitSummary) -> None:
        """The run's ``eval_mp_results`` row (the score before and after refining) and its ``train_summary`` row."""
        cfg = self.config
        result = MPResult(
            period=cfg.run.period,
            metric="mp_score",
            value=float(self.final[0]),
            value_std=float(self.final[1]),
            baseline_value=float(self.baseline[0]),
            baseline_value_std=float(self.baseline[1]),
            n_features=cfg.model.hidden_size,
        )
        training = TrainingSummary(
            epochs_run=summary.epochs_run,
            best_epoch=summary.best_epoch,
            best_loss=summary.best_loss,
            final_train_loss=summary.final_train_loss,
            final_val_loss=summary.final_val_loss,
            final_test_loss=summary.final_test_loss,
            early_stop_epoch=summary.early_stop_epoch,
            n_parameters=self.trainable_params,
            tensorboard_dir=self.tensorboard_dir,
            checkpoint_path=self.best_state_path,
        )
        self.run.save_many([result, training])


class MPBertTrainer(MPTrainer):
    """MP on AssetBERT: refine a checkpoint (``model.model_checkpoint``) on fixed-index masked holdings."""

    @property
    def model_config_class(self) -> Type[Config]:
        """`AssetBERTModelConfig`."""
        return AssetBERTModelConfig

    def make_model_preparer(self) -> Preparer:
        """`AssetBERT_Preparer`."""
        return AssetBERT_Preparer()


class MPSimplifiedMLMTrainer(MPTrainer):
    """MP on the SimplifiedMLM baseline: a linear head over a fixed embedding (``model.embedding_file``)."""

    @property
    def model_config_class(self) -> Type[Config]:
        """`SimplifiedMLMModelConfig`."""
        return SimplifiedMLMModelConfig

    def make_model_preparer(self) -> Preparer:
        """`SimplifiedMLM_Preparer`."""
        return SimplifiedMLM_Preparer()
