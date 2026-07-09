import os
import sys
import json
import glob
import time
import shutil
import logging
import datetime
import argparse
import textwrap
import traceback
from typing import Dict, Any, Optional

from tqdm import tqdm

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, Subset, random_split
from torch.utils.tensorboard.writer import SummaryWriter
from transformers import PreTrainedTokenizerFast, BertForMaskedLM
from accelerate import Accelerator

from asset_embeddings import utils
from asset_embeddings.preparers import (
    Logger_Preparer,
    AssetBERT_Preparer,
    Optimizer_Preparer,
    Dataset_Preparer,
    DataLoader_Preparer,
    Tokenizer_Preparer,
    Accelerator_Preparer,
)
from asset_embeddings.configs import (
    LoggerConfig,
    OptimizerConfig,
    AssetBERTModelConfig,
    AssetBERTTrainConfig,
    TokenizerConfig,
    DatasetConfig,
    DataLoaderConfig,
    AcceleratorConfig,
    ConfigContainer,
)
from asset_embeddings.records import BERTTrainRecording, generate_run_id, RecordStore


class AssetBERTMLM_Trainer:
    def __init__(self):
        self.logger: logging.Logger = None

        self.train_dataloader: DataLoader = None
        self.val_dataloader: Optional[DataLoader] = None  # None if no validation split
        self.model: BertForMaskedLM = None
        self.tokenizer: PreTrainedTokenizerFast = None
        self.optimizer: torch.optim.Optimizer = None
        self.lr_scheduler: torch.optim.lr_scheduler.LRScheduler = None

        self.accelerator: Accelerator = None
        self.summary_writer: SummaryWriter = None

        self.config: ConfigContainer = ConfigContainer(
            {
                "model": AssetBERTModelConfig(),
                "tokenizer": TokenizerConfig(),
                "dataset": DatasetConfig(),
                "dataloader": DataLoaderConfig(),
                "accelerator": AcceleratorConfig(),
                "optimizer": OptimizerConfig(),
                "train": AssetBERTTrainConfig(),
            }
        )

        self.tokenizer_preparer: Tokenizer_Preparer = Tokenizer_Preparer()
        self.dataset_preparer: Dataset_Preparer = Dataset_Preparer()
        self.dataloader_preparer: DataLoader_Preparer = DataLoader_Preparer()
        self.model_preparer: AssetBERT_Preparer = AssetBERT_Preparer()
        self.optimizer_preparer: Optimizer_Preparer = Optimizer_Preparer()
        self.accelerator_preparer: Accelerator_Preparer = Accelerator_Preparer()

        # Recording system
        self.run_id: str = ""
        self._config_file: Optional[str] = None
        self._record_store: Optional[RecordStore] = None
        self._recording: Optional[BERTTrainRecording] = None
        self._tensorboard_dir: str = ""
        self._start_time: float = 0.0

    @utils.log_exceptions_inclass()
    def set_logger(self, log_name: str = "BERT Train", log_file: Optional[str] = None, console_level: str = "INFO"):
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
        self.tokenizer_preparer.set_logger(self.logger)
        self.dataset_preparer.set_logger(self.logger)
        self.dataloader_preparer.set_logger(self.logger)
        self.model_preparer.set_logger(self.logger)
        self.optimizer_preparer.set_logger(self.logger)
        self.accelerator_preparer.set_logger(self.logger)

    @utils.log_exceptions_inclass()
    def load_configs(self, config_file: str):
        self.logger.info(f"Loading configuration from `{os.path.normpath(config_file)}`")
        self.config.from_file(os.path.normpath(config_file))
        self.config.validate()

        self.tokenizer_preparer.set_config(self.config.tokenizer)
        self.dataset_preparer.set_config(self.config.dataset)
        self.dataloader_preparer.set_config(self.config.dataloader)
        self.accelerator_preparer.set_config(self.config.accelerator)
        self.model_preparer.set_config(self.config.model)
        self.optimizer_preparer.set_config(self.config.optimizer)

    @utils.log_exceptions_inclass()
    def load_overrides(self, overrides: Dict[str, Any]):
        self.logger.info(f"Applying overrides: {overrides}")
        for key, value in overrides.items():
            self.config.set_nested(key, value)
        self.config.validate()

    @utils.log_exceptions_inclass()
    def prepare(self):
        self.logger.info("Preparing Trainer...")

        self.logger.info(f"Setting Global Seed: {self.config.train.seed} with Device Specific")
        utils.seed_hf_global(self.config.train.seed)

        self.accelerator = self.accelerator_preparer.prepare()

        utils.seed_accelerate(self.config.train.seed)

        # Prepare Tokenizer
        self.logger.info("Preparing Tokenizer...")
        self.tokenizer = self.tokenizer_preparer.prepare()

        # Prepare Dataset and split if validation is configured
        self.logger.info("Preparing Dataset...")
        dataset = self.dataset_preparer.prepare(self.tokenizer)

        if self.config.train.validation_split is not None:
            train_dataset, val_dataset = self._split_dataset(dataset)
            self.logger.info(f"Dataset split: train={len(train_dataset)}, val={len(val_dataset)}")
            self.train_dataloader = self.dataloader_preparer.prepare(train_dataset, shuffle_override=True)
            self.val_dataloader = self.dataloader_preparer.prepare(val_dataset, shuffle_override=False)
        else:
            self.train_dataloader = self.dataloader_preparer.prepare(dataset)
            self.val_dataloader = None

        if self.config.optimizer.lr_scheduler_train_steps is None:
            self.logger.info(
                f"Setting Optimizer Train Steps to DataLoader length * epoches: {len(self.train_dataloader) * self.config.train.max_epoches}"
            )
            self.config.optimizer.lr_scheduler_train_steps = len(self.train_dataloader) * self.config.train.max_epoches

        if self.config.model.vocab_size != len(self.tokenizer):
            self.logger.info(
                f"Updating Model Vocab Size from {self.config.model.vocab_size} to {len(self.tokenizer)} because of Tokenizer"
            )
            self.config.model.vocab_size = len(self.tokenizer)
            self.model_preparer.set_config(self.config.model)

        self.logger.info("Preparing Model...")
        self.model = self.model_preparer.prepare(self.tokenizer)
        trainable_params = list(filter(lambda p: p.requires_grad, self.model.parameters()))
        self.logger.info(f"Trainable Parameters Count: {sum(p.numel() for p in trainable_params)}")
        self.logger.info("Preparing Optimizer and Scheduler...")
        self.optimizer, self.lr_scheduler = self.optimizer_preparer.prepare(trainable_params)

        # Prepare with accelerator
        if self.val_dataloader is not None:
            self.model, self.optimizer, self.train_dataloader, self.val_dataloader, self.lr_scheduler = (
                self.accelerator.prepare(
                    self.model, self.optimizer, self.train_dataloader, self.val_dataloader, self.lr_scheduler
                )
            )
        else:
            self.model, self.optimizer, self.train_dataloader, self.lr_scheduler = self.accelerator.prepare(
                self.model, self.optimizer, self.train_dataloader, self.lr_scheduler
            )

        if self.config.accelerator.accelerator_checkpoint is not None:
            self.logger.info(f"Loading Accelerator Checkpoint from `{self.config.accelerator.accelerator_checkpoint}`")
            self.accelerator.load_state(self.config.accelerator.accelerator_checkpoint)
        # `self.run_id` is generated in main() before prepare() so that
        # error recordings during prepare can reference it.
        self._start_time = time.time()
        if self.accelerator.is_local_main_process:
            os.makedirs(self.config.train.save_folder, exist_ok=True)
            timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            rel = os.path.relpath(self.config.train.save_folder, "checkpoints")
            self._tensorboard_dir = os.path.join(
                "results", "tensorboard", rel, f"{self.config.train.save_name}_{self.run_id}_{timestamp}"
            )
            os.makedirs(self._tensorboard_dir, exist_ok=True)
            self.summary_writer = SummaryWriter(self._tensorboard_dir)
        self.logger.info("Trainer Prepared.")
        self.logger.info(f"Configs: {self.config.to_dict()}")
        self.save_configs(os.path.join(self.config.train.save_folder, "config.json"))
        self.logger.info(f"Config file saved to {os.path.join(self.config.train.save_folder, 'config.json')}")
        # 报告预计显存占用
        if self.accelerator.is_local_main_process:
            self.logger.debug(f"Estimated VRAM Usage (MB): {utils.check_vram(device=None)}")

        # Initialize recording
        if self._record_store and self.accelerator.is_local_main_process:
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            task = (
                "finetune_BERT"
                if self.config.accelerator.accelerator_checkpoint
                or (hasattr(self.config.model, "model_checkpoint") and self.config.model.model_checkpoint)
                else "pretrain_BERT"
            )
            self._recording = BERTTrainRecording(
                run_id=self.run_id,
                recorded_at=datetime.datetime.now().isoformat(),
                config_file=self._config_file,
                config_content=json.dumps(self.config.to_dict(), ensure_ascii=False),
                status="running",
                task=task,
                model="AssetBERT",
                seed=self.config.train.seed,
                total_epochs=self.config.train.max_epoches,
                trainable_params=trainable_params,
                tensorboard_dir=self._tensorboard_dir,
            )
            self._record_store.save_or_replace(self._recording, "train_bert_recordings")

    def _split_dataset(self, dataset: Dataset) -> tuple:
        """Split dataset into train and validation sets based on config.

        Args:
            dataset: The full dataset to split.

        Returns:
            Tuple of (train_dataset, val_dataset).
        """
        val_ratio = self.config.train.validation_split
        total_size = len(dataset)
        val_size = int(total_size * val_ratio)
        train_size = total_size - val_size

        if self.config.train.split_method == "random":
            self.logger.info(f"Splitting dataset randomly: {train_size} train, {val_size} val")
            generator = torch.Generator().manual_seed(self.config.train.seed)
            return random_split(dataset, [train_size, val_size], generator=generator)
        else:  # sequential
            self.logger.info(f"Splitting dataset sequentially: {train_size} train (first), {val_size} val (last)")
            train_indices = list(range(train_size))
            val_indices = list(range(train_size, total_size))
            return Subset(dataset, train_indices), Subset(dataset, val_indices)

    def save_configs(self, save_folder: str):
        self.config.to_file(save_folder)

    def save_checkpoint(self, save_folder: str):
        self.accelerator.save_state(
            save_folder, safe_serialization=True if self.config.train.save_format == ".safetensors" else False
        )

    def save_model(self, save_folder: str):
        self.model.save_pretrained(save_folder)

    @utils.log_exceptions_inclass()
    def train(self):
        self.logger.info("Start Training!")
        self.model.train()

        torch.autograd.set_detect_anomaly(self.config.train.detect_anomaly)
        best_epoch_tracker = utils.MinMaxTracker()

        # Early stopping initialization
        if self.config.train.early_stop_enabled:
            early_stop_tracker = utils.EarlyStopTracker(
                patience=self.config.train.early_stop_patience,
                min_delta=self.config.train.early_stop_min_delta,
                mode=self.config.train.early_stop_mode,
                relative=self.config.train.early_stop_relative,
            )
            mode_str = "relative" if self.config.train.early_stop_relative else "absolute"
            self.logger.debug(
                f"Early stopping enabled: patience={self.config.train.early_stop_patience}, min_delta={self.config.train.early_stop_min_delta} ({mode_str}), mode={self.config.train.early_stop_mode}, metric={self.config.train.early_stop_metric}"
            )

        # Validation check
        if self.config.train.best_metric == "val_loss" and self.val_dataloader is None:
            self.logger.warning(
                "best_metric is set to 'val_loss' but no validation split configured. Falling back to 'train_loss'."
            )
            self.config.train.best_metric = "train_loss"
        if self.config.train.early_stop_metric == "val_loss" and self.val_dataloader is None:
            self.logger.warning(
                "early_stop_metric is set to 'val_loss' but no validation split configured. Falling back to 'train_loss'."
            )
            self.config.train.early_stop_metric = "train_loss"
        self.logger.info(f"Best epoch selection metric: {self.config.train.best_metric}")

        # Training Loop
        for epoch in range(self.config.train.max_epoches):
            # Training Phase
            self.model.train()
            train_loss_tracker = utils.MeanVarianceTracker()
            for step, batch in enumerate(
                tqdm(
                    self.train_dataloader,
                    total=len(self.train_dataloader),
                    desc=f"Epoch [{epoch+1}/{self.config.train.max_epoches}] Train",
                    disable=not self.accelerator.is_local_main_process,
                )
            ):
                with self.accelerator.accumulate(self.model):
                    self.optimizer.zero_grad()
                    input_ids, attention_mask, labels = batch["input_ids"], batch["attention_mask"], batch["labels"]
                    outputs = self.model(input_ids, attention_mask=attention_mask, labels=labels)
                    loss = outputs.loss

                    self.accelerator.backward(loss)

                    # Gradient Clipping
                    if self.accelerator.sync_gradients:
                        if self.config.train.clip_grad_norm is not None:
                            self.accelerator.clip_grad_norm_(self.model.parameters(), self.config.train.clip_grad_norm)
                        if self.config.train.clip_grad_value is not None:
                            self.accelerator.clip_grad_value_(
                                self.model.parameters(), self.config.train.clip_grad_value
                            )

                    self.optimizer.step()
                    self.lr_scheduler.step()

                if step % self.config.train.check_per_step == 0:
                    if self.accelerator.is_local_main_process:
                        self.logger.info(f"Epoch:{epoch+1}, Step:{step+1}, Loss:{loss.item()}")
                        self.logger.debug(f"Estimated VRAM Usage (MB): {utils.check_vram(device=None)}")
                train_loss_tracker.update(loss.item())

            train_loss = train_loss_tracker.mean

            # Validation Phase
            val_loss = None
            if self.val_dataloader is not None:
                val_loss = self.validate(epoch)

            if self.config.train.best_metric == "val_loss" and val_loss is not None:
                metric_for_best = val_loss
            else:
                metric_for_best = train_loss
            if self.accelerator.is_local_main_process:
                best_epoch_tracker.update(metric_for_best, epoch)

            # Reporting
            if epoch % self.config.train.report_per_epoch == 0:
                if self.accelerator.is_local_main_process:
                    report_msg = f"Epoch:{epoch+1}, Train Loss:{train_loss:.6f}"
                    if val_loss is not None:
                        report_msg += f", Val Loss:{val_loss:.6f}"
                    self.logger.info(report_msg)
                    self.summary_writer.add_scalar("Train/Loss", train_loss, epoch + 1)
                    if val_loss is not None:
                        self.summary_writer.add_scalar("Val/Loss", val_loss, epoch + 1)
                    self.summary_writer.flush()

            # Checkpointing
            if epoch % self.config.train.save_per_epoch == 0:
                if self.accelerator.is_local_main_process:
                    self.save_checkpoint(
                        os.path.join(
                            self.config.train.save_folder, f"{self.config.train.save_name}_accelerator_epoch{epoch+1}"
                        )
                    )
                    self.logger.info(f"Epoch {epoch+1} Accelerate Checkpoint saved to {self.config.train.save_folder}")

                    self.save_static_embedding_csv(
                        self.model,
                        os.path.join(
                            self.config.train.save_folder, f"{self.config.train.save_name}_epoch{epoch+1}_embedding.csv"
                        ),
                    )
                    self.logger.info(f"Epoch {epoch+1} Embedding saved to {self.config.train.save_folder}")

            if self.accelerator.is_local_main_process and epoch == best_epoch_tracker.min_idx:
                self.save_model(os.path.join(self.config.train.save_folder, f"{self.config.train.save_name}_best"))
                self.save_static_embedding_csv(
                    self.model,
                    os.path.join(self.config.train.save_folder, f"{self.config.train.save_name}_best_embedding.csv"),
                )

            # Early stopping check
            if self.config.train.early_stop_enabled and self.accelerator.is_local_main_process:
                if self.config.train.early_stop_metric == "val_loss" and val_loss is not None:
                    current_metric = val_loss
                else:
                    current_metric = train_loss
                early_stop_tracker.update(current_metric, epoch)

                if early_stop_tracker.improved:
                    self.logger.debug(
                        f"Epoch {epoch+1}: {self.config.train.early_stop_metric} "
                        f"improvement: {early_stop_tracker.improvement:.6f}. "
                    )
                else:
                    self.logger.debug(
                        f"Epoch {epoch+1}: No improvement in {self.config.train.early_stop_metric}. "
                        f"Patience: {early_stop_tracker.patience_counter}/{self.config.train.early_stop_patience}"
                    )

                if early_stop_tracker.should_stop:
                    self.logger.debug(
                        f"Early stopping triggered at epoch {epoch+1}. "
                        f"Best {self.config.train.early_stop_metric}: {early_stop_tracker.best_metric:.6f} at epoch {early_stop_tracker.best_epoch + 1}"
                    )
                    # Save final checkpoint if not already saved this epoch
                    if epoch % self.config.train.save_per_epoch != 0:
                        self.save_checkpoint(
                            os.path.join(
                                self.config.train.save_folder,
                                f"{self.config.train.save_name}_accelerator_epoch{epoch+1}",
                            )
                        )
                        self.save_static_embedding_csv(
                            self.model,
                            os.path.join(
                                self.config.train.save_folder,
                                f"{self.config.train.save_name}_epoch{epoch+1}_embedding.csv",
                            ),
                        )
                        self.logger.info(f"Early stop checkpoint saved to {self.config.train.save_folder}")
                    self._early_stopped_flag = True
                    break

        if self.accelerator.is_local_main_process:
            self.summary_writer.close()
            self.logger.info(
                f"Epoch {best_epoch_tracker.min_idx + 1} is the best epoch (based on {self.config.train.best_metric})."
            )
            self.logger.info("Training Finished!")

            # Update recording with final training info
            if self._recording and self._record_store:
                self._recording.actual_epochs = epoch + 1
                self._recording.best_epoch = best_epoch_tracker.min_idx + 1
                self._recording.best_train_loss = best_epoch_tracker.min
                self._recording.final_train_loss = train_loss
                if val_loss is not None:
                    self._recording.final_val_loss = val_loss
                if (
                    self.config.train.early_stop_enabled
                    and hasattr(self, "_early_stopped_flag")
                    and self._early_stopped_flag
                ):
                    self._recording.early_stopped = True
                    self._recording.early_stop_epoch = epoch + 1
                self._recording.checkpoint_path = os.path.join(
                    self.config.train.save_folder, f"{self.config.train.save_name}_best"
                )
                self._recording.embedding_path = os.path.join(
                    self.config.train.save_folder, f"{self.config.train.save_name}_best_embedding.csv"
                )
                self._recording.duration_sec = time.time() - self._start_time
                self._recording.status = "completed"
                self._recording.recorded_at = datetime.datetime.now().isoformat()
                self._record_store.save_or_replace(self._recording, "train_bert_recordings")

            # 清理非best的checkpoint和embedding
            if self.config.train.keep_best_only:
                save_folder = self.config.train.save_folder
                save_name = self.config.train.save_name
                removed_dirs = 0
                removed_files = 0
                for d in glob.glob(os.path.join(save_folder, f"{save_name}_accelerator_epoch*")):
                    if os.path.isdir(d):
                        shutil.rmtree(d)
                        removed_dirs += 1
                for f in glob.glob(os.path.join(save_folder, f"{save_name}_epoch*_embedding.csv")):
                    if os.path.isfile(f):
                        os.remove(f)
                        removed_files += 1
                self.logger.info(
                    f"keep_best_only: removed {removed_dirs} checkpoint directories and {removed_files} embedding files."
                )

        # 计算上下文化embedding
        if self.config.train.calculate_contextualized_embeddings:
            self.logger.info("Calculating Contextualized Embeddings for the Best Model...")
            best_model: BertForMaskedLM = (
                AssetBERT_Preparer()
                .set_config(
                    AssetBERTModelConfig()
                    .from_dict(self.config.model.to_dict())
                    .from_kwargs(
                        w2v_model=None,
                        model_checkpoint=os.path.join(
                            self.config.train.save_folder,
                            f"{self.config.train.save_name}_best",
                            "model" + self.config.train.save_format,
                        ),
                    )
                )
                .prepare(self.tokenizer)
            )
            best_model = self.accelerator.prepare(best_model)
            self.save_contextual_embedding_csv(
                best_model,
                os.path.join(
                    self.config.train.save_folder, f"{self.config.train.save_name}_best_contextual_embedding.csv"
                ),
            )

    @utils.log_exceptions_inclass()
    def validate(self, epoch: int) -> float:
        """Run validation and return average validation loss.

        Args:
            epoch: Current epoch number (for progress bar display).

        Returns:
            Average validation loss.
        """
        self.model.eval()
        val_loss_tracker = utils.MeanVarianceTracker()

        with torch.no_grad():
            for batch in tqdm(
                self.val_dataloader,
                total=len(self.val_dataloader),
                desc=f"Epoch [{epoch+1}/{self.config.train.max_epoches}] Val",
                disable=not self.accelerator.is_local_main_process,
            ):
                input_ids, attention_mask, labels = batch["input_ids"], batch["attention_mask"], batch["labels"]
                outputs = self.model(input_ids, attention_mask=attention_mask, labels=labels)
                val_loss_tracker.update(outputs.loss.item())

        return val_loss_tracker.mean

    def save_static_embedding_csv(self, model: BertForMaskedLM, save_path: str):
        # 根据tokenizer和bert的embedding层保存embedding
        if save_path[-4:] != ".csv":
            save_path += ".csv"
        embedding_df = pd.DataFrame(
            model.get_input_embeddings().weight.data.cpu().numpy(),
            columns=[f"Embed_{i+1}" for i in range(self.config.model.hidden_size)],
        )
        token_to_id = self.tokenizer.get_vocab()
        sorted_tokens = [token for token, _ in sorted(token_to_id.items(), key=lambda x: x[1])]
        embedding_df.insert(0, "Token", sorted_tokens)
        embedding_df.to_csv(save_path, index=False)

    @utils.log_exceptions_inclass()
    def save_contextual_embedding_csv(self, model: BertForMaskedLM, save_path: str, batch_size: int = 64):
        """
        优化版：保存基于上下文信息的embedding，通过持股比例加权平均得到

        性能优化要点：
        1. 增大batch_size，充分利用GPU并行计算（10x+ speedup）
        2. 消除Python循环，使用张量向量化操作（5x+ speedup）
        3. 使用torch.index_add_在线聚合，避免存储所有中间结果（2x+ speedup）
        4. 减少CPU-GPU数据传输次数

        预估加速比：20-50x（取决于数据规模和GPU型号）

        Args:
            save_path: 保存路径
            batch_size: 批处理大小（建议32-64，根据GPU显存调整）
        """
        if save_path[-4:] != ".csv":
            save_path += ".csv"

        # 创建无mask的数据集配置
        no_mask_dataset_config = DatasetConfig()
        no_mask_dataset_config.from_dict(self.config.dataset.to_dict()).from_kwargs(
            mask_prob=None, mask_indices=None, include_proportion=True
        )
        no_mask_dataset = (
            Dataset_Preparer().set_logger(self.logger).set_config(no_mask_dataset_config).prepare(self.tokenizer)
        )

        # 创建无mask的数据加载器配置（使用传入的batch_size）
        no_mask_dl_config = DataLoaderConfig()
        no_mask_dl_config.from_dict(self.config.dataloader.to_dict()).from_kwargs(batch_size=batch_size)
        no_mask_dataloader = (
            DataLoader_Preparer().set_logger(self.logger).set_config(no_mask_dl_config).prepare(no_mask_dataset)
        )
        no_mask_dataloader = self.accelerator.prepare(no_mask_dataloader)

        self.logger.info(
            f"Processing {len(no_mask_dataloader)} batches (batch_size={batch_size}) for contextual embeddings..."
        )

        vocab_size = len(self.tokenizer)
        hidden_size = self.config.model.hidden_size
        device = self.accelerator.device

        # 累加器：存储加权embedding之和与权重之和
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

                # 向量化处理整个batch，消除Python循环
                # 创建mask：过滤掉padding token
                valid_mask = input_ids != self.tokenizer.pad_token_id  # [batch_size, seq_len]

                # 展平所有有效token（一次性处理整个batch）
                valid_token_ids = input_ids[valid_mask]  # [N_valid]
                valid_embeddings = hidden_states[valid_mask]  # [N_valid, hidden_size]
                valid_weights = proportions2[valid_mask]  # [N_valid]

                # 计算加权embedding
                weighted_embeddings = valid_embeddings * valid_weights.unsqueeze(1)  # [N_valid, hidden_size]
                weighted_sum.index_add_(0, valid_token_ids, weighted_embeddings)
                weight_sum.index_add_(0, valid_token_ids, valid_weights)

        self.logger.info("Computing weighted average embeddings...")

        # 计算加权平均：weighted_sum / weight_sum
        # 使用clamp避免除零
        final_embeddings = weighted_sum / weight_sum.unsqueeze(1).clamp(min=1e-8)  # [vocab_size, hidden_size]

        # 处理未出现的token：使用静态embedding作为fallback
        tokens_not_found_mask = weight_sum == 0
        num_tokens_not_found = tokens_not_found_mask.sum().item()

        if num_tokens_not_found > 0:
            self.logger.info(f"Found {num_tokens_not_found} tokens not in data, using static embeddings as fallback")
            static_embeddings = model.get_input_embeddings().weight.data  # [vocab_size, hidden_size]
            final_embeddings[tokens_not_found_mask] = static_embeddings[tokens_not_found_mask]

        final_embeddings_cpu = final_embeddings.cpu()  # 只在最后进行一次CPU传输

        # 创建DataFrame
        self.logger.info("Creating DataFrame and saving to CSV...")

        token_to_id = self.tokenizer.get_vocab()
        id_to_token = {v: k for k, v in token_to_id.items()}

        tokens = [id_to_token.get(i, f"[UNK_{i}]") for i in range(vocab_size)]

        embedding_df = pd.DataFrame(final_embeddings_cpu.numpy(), columns=[f"Embed_{i+1}" for i in range(hidden_size)])
        embedding_df.insert(0, "Token", tokens)

        # 保存到CSV
        embedding_df.to_csv(save_path, index=False)

        num_tokens_processed = (weight_sum > 0).sum().item()
        self.logger.info(f"Contextual embeddings saved to {save_path}")
        self.logger.info(f"Total tokens processed: {num_tokens_processed}/{vocab_size}")


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train BERT model with Masked Language Modeling (MLM) on stock data.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""
        Configuration File Example:
        ---------------------------
        {
            "model": {
                "model_type": "BERT",
                "model_checkpoint": null,
                "w2v_model": "model/finetune/AssetW2V/d4/2023Q2/AssetW2V_d4_2023Q2.model",
                "embedding_file": null,
                "vocab_size": 5236,
                "hidden_size": 4,
                "num_hidden_layers": 4,
                "num_attention_heads": 2,
                "intermediate_size": 512,
                "max_position_embeddings": 64,
                "type_vocab_size": 1,
                "freeze_embedding": false,
                "freeze_encoder": false
            },
            "tokenizer": {
                "w2v_model": "model/finetune/AssetW2V/d4/2023Q2/AssetW2V_d4_2023Q2.model",
                "vocab_file": null,
                "finetune_tokenizer_file": null,
                "alias_file": null
            },
            "dataset": {
                "data_path": "data/processed/MutualFundShareHoldings/finetune/2023Q2.csv",
                "data_format": "csv",
                "id_key": "InvestorID",
                "portfolio_key": "Portfolio",
                "proportion1_key": "Proportion1",
                "proportion2_key": "Proportion2",
                "include_proportion": true,
                "max_length": 64,
                "num_repeats": 1,
                "mask_prob": 0.15,
                "mask_indices": null,
                "cache_size": 400
            },
            "dataloader": {
                "batch_size": 128,
                "shuffle": true,
                "num_workers": 4,
                "pin_memory": false
            },
            "optimizer": {
                "optimizer_type": "PagedAdam8bit",
                "optimizer_kwargs": null,
                "learning_rate": 0.001,
                "lr_scheduler_type": "cosine",
                "lr_scheduler_warmup_steps": 0,
                "lr_scheduler_train_steps": null,
                "lr_scheduler_num_cycles": 0.5,
                "lr_scheduler_power": 0
            },
            "accelerator": {
                "mixed_precision": "auto",
                "gradient_accumulation_steps": 1,
                "accelerator_checkpoint": null,
                "use_compile": false,
                "compile_backend": "inductor",
                "compile_mode": "reduce-overhead",
                "compile_fullgraph": false,
                "compile_dynamic": false,
                "compile_cache_dir": null,
                "allow_tf32": false
            },
            "train": {
                "max_epoches": 5,
                "clip_grad_norm": null,
                "clip_grad_value": null,
                "detect_anomaly": false,
                "check_per_step": 200,
                "report_per_epoch": 1,
                "save_per_epoch": 1,
                "keep_best_only": false,
                "calculate_contextualized_embeddings": true,
                "save_folder": "model/finetune/AssetBERT/d4/2023Q2",
                "save_name": "AssetBERT_d4_2023Q2",
                "save_format": ".safetensors",
                "seed": 42,
                "early_stop_enabled": false,
                "early_stop_patience": 5,
                "early_stop_min_delta": 0.0,
                "early_stop_mode": "min",
                "early_stop_relative": true,
                "early_stop_metric": "train_loss",
                "validation_split": null,
                "split_method": "random",
                "best_metric": "train_loss"
            }
        }

        Configuration File Descriptions:
        --------------------------------
        model.model_type               : Model type, fixed as "BERT".
        model.model_checkpoint         : Pretrained model path; null means training from scratch.
        model.w2v_model                : Path to Word2Vec model for initializing embeddings.
        model.embedding_file           : Optional CSV with embedding weights; null if unused.
        model.vocab_size               : Number of asset tokens in the vocabulary.
        model.hidden_size              : BERT hidden dimension (embedding size).
        model.num_hidden_layers        : Number of transformer encoder layers.
        model.num_attention_heads      : Number of attention heads (must divide hidden size).
        model.intermediate_size        : Size of feedforward layers.
        model.max_position_embeddings  : Max number of assets (i.e., sequence length).
        model.type_vocab_size          : Usually 1 for asset sequences.
        model.freeze_embedding         : If true, freeze embedding layer weights.
        model.freeze_encoder           : If true, freeze encoder parameters.

        tokenizer.w2v_model            : Path to Word2Vec model (for tokenizer).
        tokenizer.vocab_file           : Optional vocab file from transformers.
        tokenizer.pretrained_tokenizer_file : Optional saved tokenizer path.
        tokenizer.alias_file           : Optional JSON/CSV for stock alias mapping.

        dataset.data_path              : Input data directory or file.
        dataset.data_format            : Format: "csv", "json", or "binary".
        dataset.id_key                 : Column name for investor ID (default: "InvestorID").
        dataset.portfolio_key          : Column name for portfolio data (default: "Portfolio").
        dataset.proportion1_key        : Column name for shareholding proportion relative to total shares.
        dataset.proportion2_key        : Column name for shareholding proportion relative to tradable A-shares.
        dataset.include_proportion     : Whether to include proportion information.
        dataset.max_length             : Sequence length (should match model.max_position_embeddings).
        dataset.num_repeats            : Number of times to repeat the dataset per epoch.
        dataset.mask_prob              : Masked LM probability.
        dataset.mask_indices           : Fixed mask positions; mutually exclusive with mask_prob.
        dataset.cache_size             : Cache size for stream loading.

        dataloader.batch_size          : Number of samples per batch.
        dataloader.shuffle             : Whether to shuffle training data.
        dataloader.num_workers         : Number of dataloader workers.
        dataloader.persistent_workers  : Whether to use persistent workers.
        dataloader.pin_memory          : Pin memory (if using CUDA).

        optimizer.optimizer_type       : Optimizer name. Supports Adam, AdamW, Lion, etc.
        optimizer.optimizer_kwargs     : Optional dictionary for optimizer arguments.
        optimizer.learning_rate        : Base learning rate.
        optimizer.lr_scheduler_type    : Scheduler: constant, linear, cosine, polynomial, etc.
        optimizer.lr_scheduler_warmup_steps : Number of warmup steps.
        optimizer.lr_scheduler_train_steps  : Total steps (null for auto).
        optimizer.lr_scheduler_num_cycles   : For cosine/cosine_with_restarts.
        optimizer.lr_scheduler_power   : Power for polynomial scheduler.

        accelerator.mixed_precision    : "no", "fp16", "bf16", or "auto".
        accelerator.gradient_accumulation_steps : Grad accumulation steps.
        accelerator.accelerator_checkpoint : Checkpoint path for accelerator framework.
        accelerator.use_compile        : Enable torch.compile. Default: false.
        accelerator.compile_backend    : Compile backend: "inductor", "eager", "aot_eager".
        accelerator.compile_mode       : Compile mode: "default", "reduce-overhead", "max-autotune".
        accelerator.compile_fullgraph  : Require full graph compilation (no graph breaks).
        accelerator.compile_dynamic    : Enable dynamic shapes (for variable batch size).
        accelerator.compile_cache_dir  : Persistent compile cache directory. null = PyTorch default.
        accelerator.allow_tf32         : Enable TF32 on Ampere/Ada GPUs. Default: false.

        train.max_epoches              : Number of training epochs.
        train.clip_grad_norm           : Max gradient norm (L2).
        train.clip_grad_value          : Max gradient value (absolute).
        train.detect_anomaly           : Enable autograd anomaly detection.
        train.check_per_step           : Steps between evaluations.
        train.report_per_epoch         : Epochs between logging.
        train.save_per_epoch           : Epochs between saving.
        train.calculate_contextualized_embeddings : Whether to compute contextualized embeddings after training.
        train.save_folder              : Folder to save model.
        train.save_name                : Name of saved model.
        train.save_format              : Format of model file (e.g., ".safetensors").
        train.seed                     : Random seed.
        train.early_stop_enabled       : Whether to enable early stopping.
        train.early_stop_patience      : Number of epochs without improvement before stopping.
        train.early_stop_min_delta     : Minimum improvement threshold.
        train.early_stop_mode          : Optimization direction ("min" or "max").
        train.early_stop_relative      : Whether to use relative improvement.
        train.early_stop_metric        : Metric to monitor ("train_loss" or "val_loss").
        train.validation_split         : Validation split ratio (null for no split, e.g., 0.1 for 10%).
        train.split_method             : Split method ("random" or "sequential").
        train.best_metric              : Metric for best epoch selection ("train_loss" or "val_loss").
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
        help="Override configuration parameters. Format: key=value (e.g., train.max_epoches=10 optimizer.learning_rate=0.001). Use dot notation for nested keys. Optional.",
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
    os.environ["TF_ENABLE_ONEDNN_OPTS"] = "1"
    trainer = AssetBERTMLM_Trainer()
    console_level = "DEBUG" if args.verbose else "INFO"
    trainer.set_logger(log_file=args.log, console_level=console_level)
    trainer.logger.debug(f"Command: {' '.join(sys.argv)}")
    trainer.load_configs(config_file=args.config)
    trainer._config_file = args.config

    # Apply overrides if provided
    if args.override:
        trainer.load_overrides(utils.parse_overrides(args.override))

    # Generate run_id and wire record_store BEFORE prepare() so that any
    # failure during prepare() is attributable in the DB even if
    # `_recording` was never constructed (it's currently built late in
    # prepare(), gated on accelerator + record_store).
    trainer.run_id = generate_run_id()
    if args.result_file:
        trainer._record_store = RecordStore(args.result_file, logger=trainer.logger)

    try:
        trainer.prepare()
        trainer.train()
    except Exception:
        is_main = trainer.accelerator is None or trainer.accelerator.is_local_main_process
        if is_main and trainer._record_store:
            if trainer._recording is None:
                # prepare() failed before the recording was constructed.
                # Build a minimal one with required fields so the failure
                # is still queryable. `task` derivation must mirror the
                # logic in prepare().
                task = (
                    "finetune_BERT"
                    if (
                        trainer.config.accelerator.accelerator_checkpoint
                        or (hasattr(trainer.config.model, "model_checkpoint") and trainer.config.model.model_checkpoint)
                    )
                    else "pretrain_BERT"
                )
                trainer._recording = BERTTrainRecording(
                    run_id=trainer.run_id,
                    recorded_at=datetime.datetime.now().isoformat(),
                    config_file=trainer._config_file,
                    config_content=json.dumps(trainer.config.to_dict(), ensure_ascii=False),
                    task=task,
                    model="AssetBERT",
                )
            trainer._recording.status = "error"
            trainer._recording.error_message = traceback.format_exc()
            trainer._recording.duration_sec = time.time() - trainer._start_time if trainer._start_time else None
            trainer._recording.recorded_at = datetime.datetime.now().isoformat()
            trainer._record_store.save_or_replace(trainer._recording, "train_bert_recordings")
        raise


if __name__ == "__main__":
    main(get_parser().parse_args())
