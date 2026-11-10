"""AssetBERT masked-LM pretraining and finetuning: the trainer behind ``python -m asset_embeddings.scripts.train.bert``."""

import datetime
import glob
import os
import shutil
import time
from typing import Type

import pandas as pd
import torch
from tqdm import tqdm
from transformers import BertForMaskedLM

from asset_embeddings.configs import (
    AssetBERTModelConfig,
    AssetBERTTrainConfig,
    Config,
    DataLoaderConfig,
    DatasetConfig,
)
from asset_embeddings.datasets.mlm import MaskingCollator
from asset_embeddings.preparers import AssetBERT_Preparer, DataLoader_Preparer, Dataset_Preparer, Preparer
from asset_embeddings.records import TrainingSummary

from .mlm import BaseMLMTrainer, FitSummary


class AssetBERTTrainer(BaseMLMTrainer):
    """Trains AssetBERT as a masked LM over portfolios, under ``accelerate``, and writes checkpoints, embeddings and records.

    Per saved epoch: an ``accelerate`` state directory and ``<save_name>_epoch<n>_embedding.csv``; at each new best
    epoch: ``<save_name>_best/`` (the HF model) and ``<save_name>_best_embedding.csv``; optionally
    ``<save_name>_best_contextual_embedding.csv``. Every embedding CSV is ``Token, Embed_1, ..., Embed_N``. Only a
    validation split is used (the config has no test split). Records: one ``train_summary`` row, written
    after the loop, one ``run_epochs`` row per epoch, and TensorBoard scalars under ``results/tensorboard/``.
    """

    @property
    def model_config_class(self) -> Type[Config]:
        """`AssetBERTModelConfig`."""
        return AssetBERTModelConfig

    @property
    def train_config_class(self) -> Type[Config]:
        """`AssetBERTTrainConfig`."""
        return AssetBERTTrainConfig

    def make_model_preparer(self) -> Preparer:
        """`AssetBERT_Preparer`."""
        return AssetBERT_Preparer()

    @property
    def task_name(self) -> str:
        """``finetune_BERT`` when a checkpoint is loaded, else ``pretrain_BERT`` (runs.task and the recording agree)."""
        if self.config.accelerator.accelerator_checkpoint or self.config.model.model_checkpoint:
            return "finetune_BERT"
        return "pretrain_BERT"

    def save_model(self, save_folder: str) -> None:
        """Save the HF model (weights and config) into ``save_folder``."""
        self.model.save_pretrained(save_folder)

    def _embedding_path(self, suffix: str) -> str:
        return self._checkpoint_path(f"{suffix}_embedding.csv")

    def on_periodic_save(self, epoch: int) -> None:
        """Save the ``accelerate`` state and the static embedding of the epoch."""
        folder = self.config.train.save_folder
        self.save_checkpoint(self._checkpoint_path(f"accelerator_epoch{epoch + 1}"))
        self.logger.info(f"Epoch {epoch + 1} Accelerate Checkpoint saved to {folder}")
        self.save_static_embedding_csv(self.model, self._embedding_path(f"epoch{epoch + 1}"))
        self.logger.info(f"Epoch {epoch + 1} Embedding saved to {folder}")

    def on_best(self, epoch: int) -> None:
        """Save the HF model and the static embedding of the new best epoch."""
        self.save_model(self._checkpoint_path("best"))
        self.save_static_embedding_csv(self.model, self._embedding_path("best"))

    def on_fit_end(self, summary: FitSummary) -> None:
        """Write the training row, apply ``keep_best_only``, and compute the contextual embeddings when configured."""
        if self.is_main:
            self._save_recording(summary)
            if self.config.train.keep_best_only:
                self._remove_periodic_artifacts()
        if self.config.train.calculate_contextualized_embeddings:
            self._contextual_embeddings()

    def train(self) -> None:
        """Run the epoch loop; the artifacts and the training row are written by the hooks."""
        self.fit()

    def _save_recording(self, summary: FitSummary) -> None:
        """The run's ``train_summary`` row, written once after the loop."""
        self.run.save(
            TrainingSummary(
                epochs_run=summary.epochs_run,
                best_epoch=summary.best_epoch,
                best_loss=summary.best_loss,
                final_train_loss=summary.final_train_loss,
                final_val_loss=summary.final_val_loss,
                final_test_loss=summary.final_test_loss,
                early_stop_epoch=summary.early_stop_epoch,
                n_parameters=self.trainable_params,
                tensorboard_dir=self.tensorboard_dir,
                checkpoint_path=self._checkpoint_path("best"),
                embedding_path=self._embedding_path("best"),
            ),
            replace=True,
        )

    def _remove_periodic_artifacts(self) -> None:
        """``keep_best_only``: delete the per-epoch ``accelerate`` states and embeddings."""
        removed_dirs = removed_files = 0
        for d in glob.glob(self._checkpoint_path("accelerator_epoch*")):
            if os.path.isdir(d):
                shutil.rmtree(d)
                removed_dirs += 1
        for f in glob.glob(self._checkpoint_path("epoch*_embedding.csv")):
            if os.path.isfile(f):
                os.remove(f)
                removed_files += 1
        self.logger.info(
            f"keep_best_only: removed {removed_dirs} checkpoint directories and {removed_files} embedding files."
        )

    def _contextual_embeddings(self) -> None:
        """Rebuild the best model from its saved weights and write its contextual embeddings."""
        self.logger.info("Calculating Contextualized Embeddings for the Best Model...")
        cfg = self.config
        best_model: BertForMaskedLM = (
            AssetBERT_Preparer()
            .set_config(
                AssetBERTModelConfig()
                .from_dict(cfg.model.to_dict())
                .from_kwargs(
                    w2v_model=None,
                    model_checkpoint=os.path.join(self._checkpoint_path("best"), "model" + cfg.train.save_format),
                )
            )
            .prepare(self.tokenizer)
        )
        best_model = self.accelerator.prepare(best_model)
        self.save_contextual_embedding_csv(best_model, self._embedding_path("best_contextual"))

    def save_static_embedding_csv(self, model: BertForMaskedLM, save_path: str) -> None:
        """Write the input-embedding matrix as a ``Token, Embed_1, ..., Embed_N`` CSV, rows in tokenizer id order.

        Args:
            model: The model whose input embeddings are written.
            save_path: Target file; ``.csv`` is appended when missing.
        """
        if save_path[-4:] != ".csv":
            save_path += ".csv"
        weights = model.get_input_embeddings().weight.data.cpu().numpy()
        embedding_df = pd.DataFrame(weights, columns=[f"Embed_{i+1}" for i in range(weights.shape[1])])
        token_to_id = self.tokenizer.get_vocab()
        sorted_tokens = [token for token, _ in sorted(token_to_id.items(), key=lambda x: x[1])]
        embedding_df.insert(0, "Token", sorted_tokens)
        embedding_df.to_csv(save_path, index=False)

    def save_contextual_embedding_csv(self, model: BertForMaskedLM, save_path: str, batch_size: int = 64) -> None:
        """Write each token's contextual embedding: its last-layer hidden states over every unmasked portfolio, averaged with the holding proportions as weights.

        The whole pass stays on the GPU: each batch is flattened to its valid tokens and folded into two
        vocabulary-sized accumulators with ``index_add_``, so no per-batch result is stored and the host sees
        one copy at the end. A token that appears in no portfolio keeps its static input embedding.

        Args:
            model: The model whose encoder produces the hidden states.
            save_path: Target file; ``.csv`` is appended when missing.
            batch_size: Portfolios per forward pass; sized to the GPU memory.

        Raises:
            ValueError: The dataset carries no ``proportions2`` (the weights).
        """
        if save_path[-4:] != ".csv":
            save_path += ".csv"

        # Same data, no masking, with proportions: each holding weights its token by the stake.
        no_mask_dataset_config = DatasetConfig()
        no_mask_dataset_config.from_dict(self.config.dataset.to_dict()).from_kwargs(
            mask_prob=None, mask_indices=None, include_proportion=True
        )
        no_mask_dataset = Dataset_Preparer().set_config(no_mask_dataset_config).prepare(self.tokenizer)

        no_mask_dl_config = DataLoaderConfig()
        no_mask_dl_config.from_dict(self.config.dataloader.to_dict()).from_kwargs(batch_size=batch_size)
        no_mask_dataloader = (
            DataLoader_Preparer()
            .set_config(no_mask_dl_config)
            .prepare(no_mask_dataset, collate_fn=MaskingCollator(self.tokenizer), name="contextual")
        )
        no_mask_dataloader = self.accelerator.prepare(no_mask_dataloader)

        self.logger.info(
            f"Processing {len(no_mask_dataloader)} batches (batch_size={batch_size}) for contextual embeddings..."
        )

        vocab_size = len(self.tokenizer)
        hidden_size = self.config.model.hidden_size
        device = self.accelerator.device

        # Running sums per token (weighted embeddings, weights) are all that is kept between batches
        weighted_sum = torch.zeros(vocab_size, hidden_size, dtype=torch.float32, device=device)
        weight_sum = torch.zeros(vocab_size, dtype=torch.float32, device=device)

        model.eval()
        with torch.no_grad():
            for batch_idx, batch in enumerate(
                tqdm(
                    no_mask_dataloader,
                    desc="Extracting contextual embeddings",
                    disable=not self.accelerator.is_local_main_process,
                )
            ):
                input_ids = batch["input_ids"]  # [batch_size, seq_len]
                attention_mask = batch["attention_mask"]  # [batch_size, seq_len]
                proportions2 = batch.get("proportions2", None)

                if proportions2 is None:
                    self.logger.error(
                        "Proportions data (Shareholding proportion relative to tradable A-shares) is required "
                        "for contextual embedding extraction but not found in the dataset. "
                        "Please ensure `include_proportion` is set to True and `proportion2_key` is correctly configured."
                    )
                    raise ValueError("Proportions2 data is required for contextual embedding extraction.")

                outputs = model.bert(input_ids=input_ids, attention_mask=attention_mask)
                hidden_states = outputs.last_hidden_state  # [batch_size, seq_len, hidden_size]

                # Padding positions carry no holding; masking them flattens the batch into one index_add_ call
                valid_mask = input_ids != self.tokenizer.pad_token_id  # [batch_size, seq_len]

                valid_token_ids = input_ids[valid_mask]  # [N_valid]
                valid_embeddings = hidden_states[valid_mask]  # [N_valid, hidden_size]
                valid_weights = proportions2[valid_mask]  # [N_valid]

                weighted_embeddings = valid_embeddings * valid_weights.unsqueeze(1)  # [N_valid, hidden_size]
                weighted_sum.index_add_(0, valid_token_ids, weighted_embeddings)
                weight_sum.index_add_(0, valid_token_ids, valid_weights)

        self.logger.info("Computing weighted average embeddings...")

        # clamp: a token seen nowhere has weight_sum 0 and is replaced by its static embedding below
        final_embeddings = weighted_sum / weight_sum.unsqueeze(1).clamp(min=1e-8)  # [vocab_size, hidden_size]

        tokens_not_found_mask = weight_sum == 0
        num_tokens_not_found = tokens_not_found_mask.sum().item()

        if num_tokens_not_found > 0:
            self.logger.info(f"Found {num_tokens_not_found} tokens not in data, using static embeddings as fallback")
            static_embeddings = model.get_input_embeddings().weight.data  # [vocab_size, hidden_size]
            final_embeddings[tokens_not_found_mask] = static_embeddings[tokens_not_found_mask]

        final_embeddings_cpu = final_embeddings.cpu()  # the one device-to-host copy

        self.logger.info("Creating DataFrame and saving to CSV...")

        token_to_id = self.tokenizer.get_vocab()
        id_to_token = {v: k for k, v in token_to_id.items()}

        tokens = [id_to_token.get(i, f"[UNK_{i}]") for i in range(vocab_size)]

        embedding_df = pd.DataFrame(final_embeddings_cpu.numpy(), columns=[f"Embed_{i+1}" for i in range(hidden_size)])
        embedding_df.insert(0, "Token", tokens)

        embedding_df.to_csv(save_path, index=False)

        num_tokens_processed = (weight_sum > 0).sum().item()
        self.logger.info(f"Contextual embeddings saved to {save_path}")
        self.logger.info(f"Total tokens processed: {num_tokens_processed}/{vocab_size}")
