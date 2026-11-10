"""The masked-LM training loop shared by AssetBERT pre/finetraining and the MP refines.

[`BaseMLMTrainer`][asset_embeddings.trainers.mlm.BaseMLMTrainer] builds the tokenizer, the dataset and its split,
the loaders, the model and the optimizer under ``accelerate``, and runs the epoch loop: a training pass, a scoring
pass on the held-out splits the selection needs, the report, the ``run_epochs`` row, periodic saves, best-epoch
tracking and early stopping. Subclasses choose the model and its config (which declares the splits), and what
happens at a periodic save, at a new best epoch and after the loop.
"""

import datetime
import os
import time
from abc import abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple, Type

import torch
from accelerate import Accelerator
from torch.utils.data import DataLoader, Dataset, Subset, random_split
from torch.utils.tensorboard.writer import SummaryWriter
from tqdm import tqdm
from transformers import PreTrainedTokenizerFast

from asset_embeddings import utils
from asset_embeddings.datasets.mlm import split_masking
from asset_embeddings.configs import (
    AcceleratorConfig,
    Config,
    ConfigContainer,
    DataLoaderConfig,
    DatasetConfig,
    OptimizerConfig,
    TokenizerConfig,
)
from asset_embeddings.preparers import (
    Accelerator_Preparer,
    DataLoader_Preparer,
    Dataset_Preparer,
    Optimizer_Preparer,
    Preparer,
    Tokenizer_Preparer,
)

from .base import BaseTrainer

#: The held-out split each selection metric reads, and the name of that split in messages.
_METRIC_SPLIT = {"val_loss": ("val", "validation"), "test_loss": ("test", "test")}


def split_dataset(
    dataset: Dataset,
    val: Optional[float],
    test: Optional[float],
    method: str,
    seed: int,
) -> Tuple[Dataset, Optional[Dataset], Optional[Dataset]]:
    """Split ``dataset`` into train, validation and test parts.

    Each part's size is ``int(len(dataset) * share)``; a share that is None or 0, or a part that rounds to no row,
    means the part does not exist. ``"random"``: one ``random_split`` over the existing parts in the order train,
    val, test, drawn from ``torch.Generator().manual_seed(seed)``. ``"sequential"``: consecutive slices in the same
    order. Without any val or test part the dataset itself is the training part.

    Args:
        dataset: The full dataset.
        val: Validation share.
        test: Test share.
        method: ``"random"`` or ``"sequential"``.
        seed: Seed of the random split.

    Returns:
        ``(train, val, test)``; val and test are None when absent.

    Raises:
        ValueError: When the val and test shares leave no training data.
    """
    val_share, test_share = val or 0.0, test or 0.0
    if val_share + test_share >= 1.0:
        raise ValueError(f"validation_split ({val_share}) + test_split ({test_share}) must be < 1.0")
    total = len(dataset)
    val_size, test_size = int(total * val_share), int(total * test_share)
    train_size = total - val_size - test_size
    if val_size == 0 and test_size == 0:
        return dataset, None, None

    sizes = [train_size] + [n for n in (val_size, test_size) if n > 0]
    if method == "random":
        parts = list(random_split(dataset, sizes, generator=torch.Generator().manual_seed(seed)))
    else:
        bounds = [0]
        for n in sizes:
            bounds.append(bounds[-1] + n)
        parts = [Subset(dataset, list(range(lo, hi))) for lo, hi in zip(bounds[:-1], bounds[1:])]
    train_part = parts.pop(0)
    val_part = parts.pop(0) if val_size > 0 else None
    test_part = parts.pop(0) if test_size > 0 else None
    return train_part, val_part, test_part


@dataclass(frozen=True)
class LossStats:
    """Mean and standard deviation of the per-batch losses of one pass."""

    mean: float
    std: float


@dataclass(frozen=True)
class FitSummary:
    """What the epoch loop produced.

    Attributes:
        epochs_run: Epochs completed (fewer than configured after an early stop).
        best_epoch: The selected epoch, 1-based (0 on processes other than the local main one).
        best_metric: The selection metric.
        best_loss: The selection loss at the best epoch.
        final_train_loss: Mean training loss of the last epoch.
        final_val_loss: Mean validation loss of the last epoch; None when the validation split was not scored.
        final_test_loss: Mean test loss of the last epoch; None when the test split was not scored.
        early_stop_epoch: The epoch at which early stopping ended the loop, 1-based; None otherwise.
    """

    epochs_run: int
    best_epoch: int
    best_metric: str
    best_loss: float
    final_train_loss: float
    final_val_loss: Optional[float]
    final_test_loss: Optional[float]
    early_stop_epoch: Optional[int]


class BaseMLMTrainer(BaseTrainer):
    """A masked-LM trainer under ``accelerate``: preparation, the epoch loop, checkpoint and best-epoch hooks.

    Each epoch scores the validation split when there is one, and the test split when ``best_metric`` or the
    enabled ``early_stop_metric`` is ``test_loss``. The config has already required the split a metric reads; a
    share that rounds to no row is refused when the loop starts.

    Hooks: ``model_config_class``, ``train_config_class``, ``make_model_preparer``, ``extra_configs``,
    ``on_periodic_save``, ``on_best``, ``on_fit_end``.
    """

    #: Whether the training loader always shuffles; otherwise it does whenever a split exists and follows
    #: ``dataloader.shuffle`` when the whole dataset is the training part.
    always_shuffle_train: bool = False

    def __init__(self):
        super().__init__()
        self.train_dataloader: Optional[DataLoader] = None
        self.val_dataloader: Optional[DataLoader] = None
        self.test_dataloader: Optional[DataLoader] = None
        self.model: Any = None
        self.tokenizer: Optional[PreTrainedTokenizerFast] = None
        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.lr_scheduler: Any = None
        self.accelerator: Optional[Accelerator] = None
        self.summary_writer: Optional[SummaryWriter] = None

        self.config: ConfigContainer = ConfigContainer(
            {
                **self.extra_configs(),
                "model": self.model_config_class(),
                "tokenizer": TokenizerConfig(),
                "dataset": DatasetConfig(),
                "dataloader": DataLoaderConfig(),
                "accelerator": AcceleratorConfig(),
                "optimizer": OptimizerConfig(),
                "train": self.train_config_class(),
            }
        )
        self.tokenizer_preparer = Tokenizer_Preparer()
        self.dataset_preparer = Dataset_Preparer()
        self.dataloader_preparer = DataLoader_Preparer()
        self.model_preparer: Preparer = self.make_model_preparer()
        self.optimizer_preparer = Optimizer_Preparer()
        self.accelerator_preparer = Accelerator_Preparer()

        self.trainable_params: int = 0
        self.tensorboard_dir: str = ""
        self.start_time: float = 0.0

    @property
    @abstractmethod
    def model_config_class(self) -> Type[Config]:
        """The config class of the ``model`` section."""

    @property
    @abstractmethod
    def train_config_class(self) -> Type[Config]:
        """The config class of the ``train`` section."""

    @abstractmethod
    def make_model_preparer(self) -> Preparer:
        """The preparer that builds the model from the ``model`` section."""

    def extra_configs(self) -> Dict[str, Config]:
        """Config sections placed before the shared ones; none by default."""
        return {}

    def on_periodic_save(self, epoch: int) -> None:
        """Save at a periodic epoch (and at the early-stop epoch when it was not one); main process only."""

    def on_best(self, epoch: int) -> None:
        """React to a new best epoch; main process only."""

    def on_fit_end(self, summary: FitSummary) -> None:
        """Run after the epoch loop, on every process."""

    @property
    def save_folder(self) -> str:
        """``train.save_folder`` of the config."""
        return self.config.train.save_folder

    def _configs_changed(self) -> None:
        self.tokenizer_preparer.set_config(self.config.tokenizer)
        self.dataset_preparer.set_config(self.config.dataset)
        self.dataloader_preparer.set_config(self.config.dataloader)
        self.accelerator_preparer.set_config(self.config.accelerator)
        self.model_preparer.set_config(self.config.model)
        self.optimizer_preparer.set_config(self.config.optimizer)

    @property
    def is_main(self) -> bool:
        """Whether this is the local main process (the one that logs, saves and records)."""
        return self.accelerator.is_local_main_process

    def _checkpoint_path(self, suffix: str) -> str:
        return os.path.join(self.config.train.save_folder, f"{self.config.train.save_name}_{suffix}")

    def save_checkpoint(self, save_folder: str) -> None:
        """Save the ``accelerate`` state (model, optimizer, scheduler, RNG) into ``save_folder``."""
        self.accelerator.save_state(save_folder, safe_serialization=self.config.train.save_format == ".safetensors")

    def prepare(self) -> None:
        """Seed and build tokenizer, dataset split, loaders, model and optimizer, wrap them in ``accelerate``.

        The order is fixed because the random state depends on it. Derives what the config leaves open: the
        scheduler's step count from the loader length and epochs, the model's vocabulary size from the tokenizer.
        The TensorBoard writer exists only on the local main process.
        """
        cfg = self.config
        self.logger.info("Preparing Trainer...")
        self.logger.info(f"Setting Global Seed: {cfg.train.seed} with Device Specific")
        utils.seed_hf_global(cfg.train.seed)
        self.accelerator = self.accelerator_preparer.prepare()
        utils.seed_accelerate(cfg.train.seed)
        self.tokenizer = self.tokenizer_preparer.prepare()
        dataset = self.dataset_preparer.prepare(self.tokenizer)

        train, val, test = split_dataset(
            dataset,
            cfg.train.validation_split,
            getattr(cfg.train, "test_split", None),
            cfg.train.split_method,
            cfg.train.seed,
        )
        # The held-out losses are compared across epochs, so their masks stay fixed; training masks are redrawn.
        train_masking, held_out_masking = split_masking(
            self.tokenizer, cfg.dataset.mask_prob, cfg.dataset.mask_indices, cfg.train.seed
        )
        split = val is not None or test is not None
        self.logger.info(
            f"Dataset split ({cfg.train.split_method}): train={len(train)}, val={len(val) if val else 0}, "
            f"test={len(test) if test else 0}"
        )
        shuffle = True if split or self.always_shuffle_train else None
        self.train_dataloader = self.dataloader_preparer.prepare(
            train,
            shuffle_override=shuffle,
            collate_fn=train_masking,
            num_repeats=cfg.dataset.num_repeats,
            name="train",
        )
        self.val_dataloader = (
            self.dataloader_preparer.prepare(
                val, shuffle_override=False, collate_fn=held_out_masking, name="validation"
            )
            if val
            else None
        )
        self.test_dataloader = (
            self.dataloader_preparer.prepare(test, shuffle_override=False, collate_fn=held_out_masking, name="test")
            if test
            else None
        )

        if cfg.optimizer.lr_scheduler_train_steps is None:
            steps = len(self.train_dataloader) * cfg.train.max_epochs
            self.logger.info(f"Setting Optimizer Train Steps to DataLoader length * epochs: {steps}")
            cfg.optimizer.lr_scheduler_train_steps = steps
        if cfg.model.vocab_size != len(self.tokenizer):
            self.logger.info(
                f"Updating Model Vocab Size from {cfg.model.vocab_size} to {len(self.tokenizer)} because of Tokenizer"
            )
            cfg.model.vocab_size = len(self.tokenizer)
            self.model_preparer.set_config(cfg.model)

        self.model = self.model_preparer.prepare(self.tokenizer)
        trainable = [p for p in self.model.parameters() if p.requires_grad]
        self.logger.info(f"Trainable Parameters Count: {sum(p.numel() for p in trainable)}")
        self.optimizer, self.lr_scheduler = self.optimizer_preparer.prepare(trainable)
        self._accelerate()

        if cfg.accelerator.accelerator_checkpoint is not None:
            self.logger.info(f"Loading Accelerator Checkpoint from `{cfg.accelerator.accelerator_checkpoint}`")
            self.accelerator.load_state(cfg.accelerator.accelerator_checkpoint)
        self.trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.start_time = time.time()
        if self.is_main:
            os.makedirs(cfg.train.save_folder, exist_ok=True)
            timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            rel = os.path.relpath(cfg.train.save_folder, "checkpoints")
            self.tensorboard_dir = os.path.join(
                "results", "tensorboard", rel, f"{cfg.train.save_name}_{self.run.run_id}_{timestamp}"
            )
            os.makedirs(self.tensorboard_dir, exist_ok=True)
            self.summary_writer = SummaryWriter(self.tensorboard_dir)
        self.logger.info("Trainer Prepared.")
        self.save_config()
        if self.is_main:
            self._probe_resources()

    def _accelerate(self) -> None:
        """``accelerator.prepare`` over the model, optimizer, present loaders and scheduler, in that order."""
        loaders = [(name, loader) for name, loader in self._loaders() if loader is not None]
        prepared = self.accelerator.prepare(
            self.model, self.optimizer, *(loader for _, loader in loaders), self.lr_scheduler
        )
        self.model, self.optimizer, self.lr_scheduler = prepared[0], prepared[1], prepared[-1]
        for (name, _), loader in zip(loaders, prepared[2:-1]):
            setattr(self, f"{name}_dataloader", loader)

    def _loaders(self) -> List[Tuple[str, Optional[DataLoader]]]:
        return [("train", self.train_dataloader), ("val", self.val_dataloader), ("test", self.test_dataloader)]

    def _metric_roles(self) -> List[str]:
        """The config fields naming a tracked metric: ``best_metric``, and ``early_stop_metric`` when it is on."""
        return ["best_metric"] + (["early_stop_metric"] if self.config.train.early_stop_enabled else [])

    @property
    def scored_splits(self) -> List[str]:
        """The held-out splits scored every epoch: ``"val"`` when it exists, ``"test"`` when a metric reads it."""
        splits = ["val"] if self.val_dataloader is not None else []
        metrics = [getattr(self.config.train, role) for role in self._metric_roles()]
        if self.test_dataloader is not None and "test_loss" in metrics:
            splits.append("test")
        return splits

    def _check_metric_splits(self) -> None:
        """Refuse a metric whose split holds no rows (a positive share that rounds to zero rows).

        Raises:
            ValueError: Naming the metric and the empty split.
        """
        loaders = {"val": self.val_dataloader, "test": self.test_dataloader}
        for role in self._metric_roles():
            metric = getattr(self.config.train, role)
            if metric in _METRIC_SPLIT and loaders[_METRIC_SPLIT[metric][0]] is None:
                raise ValueError(f"{role} is '{metric}' but the {_METRIC_SPLIT[metric][1]} split holds no rows")

    def _train_epoch(self, epoch: int) -> LossStats:
        """One pass over the training loader: forward, backward, clipping on synchronizing steps, optimizer step."""
        cfg = self.config.train
        self.model.train()
        tracker = utils.MeanVarianceTracker()
        for step, batch in enumerate(
            tqdm(
                self.train_dataloader,
                total=len(self.train_dataloader),
                desc=f"Epoch [{epoch + 1}/{cfg.max_epochs}] Train",
                disable=not self.is_main,
            )
        ):
            with self.accelerator.accumulate(self.model):
                self.optimizer.zero_grad()
                outputs = self.model(batch["input_ids"], attention_mask=batch["attention_mask"], labels=batch["labels"])
                loss = outputs.loss
                self.accelerator.backward(loss)
                # Clip only on a synchronizing step: under gradient accumulation the earlier steps hold partial sums.
                if self.accelerator.sync_gradients:
                    if cfg.clip_grad_norm is not None:
                        self.accelerator.clip_grad_norm_(self.model.parameters(), cfg.clip_grad_norm)
                    if cfg.clip_grad_value is not None:
                        self.accelerator.clip_grad_value_(self.model.parameters(), cfg.clip_grad_value)
                self.optimizer.step()
                self.lr_scheduler.step()
            if step % cfg.log_every_steps == 0 and self.is_main:
                self.logger.info(f"Epoch:{epoch + 1}, Step:{step + 1}, Loss:{loss.item()}")
                self._probe_resources()
            tracker.update(loss.item())
        return LossStats(tracker.mean, tracker.std())

    def _eval_epoch(self, loader: DataLoader, epoch: int, split: str) -> LossStats:
        """Mean and std of the per-batch losses over ``loader``, without gradients."""
        self.model.eval()
        tracker = utils.MeanVarianceTracker()
        with torch.no_grad():
            for batch in tqdm(
                loader,
                total=len(loader),
                desc=f"Epoch [{epoch + 1}/{self.config.train.max_epochs}] {split.capitalize()}",
                disable=not self.is_main,
            ):
                outputs = self.model(batch["input_ids"], attention_mask=batch["attention_mask"], labels=batch["labels"])
                tracker.update(outputs.loss.item())
        return LossStats(tracker.mean, tracker.std())

    def _report(self, epoch: int, train: LossStats, held_out: Dict[str, LossStats]) -> None:
        """One log line and the TensorBoard scalars (``Loss/train``, ``Loss/val``, ``Loss/test``) of an epoch."""
        line = f"Epoch {epoch + 1}: train loss {train.mean:.6f} (std {train.std:.6f})"
        self.summary_writer.add_scalar("Loss/train", train.mean, epoch + 1)
        for split, stats in held_out.items():
            line += f", {split} loss {stats.mean:.6f} (std {stats.std:.6f})"
            self.summary_writer.add_scalar(f"Loss/{split}", stats.mean, epoch + 1)
        self.logger.info(line)
        self.summary_writer.flush()

    def fit(self) -> FitSummary:
        """Run the epoch loop and return its summary.

        Per epoch: the training pass, the scoring pass on each of `scored_splits`, the report every
        ``report_every_epochs`` epochs, a ``run_epochs`` row, ``on_periodic_save`` every ``save_every_epochs``
        epochs, ``on_best`` when the selection loss reaches a new minimum, and early stopping when enabled (the
        stopping epoch is saved through ``on_periodic_save`` when it was not a periodic epoch). Finally
        ``on_fit_end``.

        Raises:
            ValueError: When a metric's split holds no rows.
        """
        cfg = self.config.train
        self._check_metric_splits()
        self.logger.info("Start Training!")
        self.model.train()
        torch.autograd.set_detect_anomaly(cfg.detect_anomaly)

        splits = self.scored_splits
        loaders = {"val": self.val_dataloader, "test": self.test_dataloader}
        self.logger.info(f"Best epoch selection metric: {cfg.best_metric}")
        best_tracker = utils.MinMaxTracker()
        early_stop = None
        if cfg.early_stop_enabled:
            early_stop = utils.EarlyStopTracker(
                patience=cfg.early_stop_patience,
                min_delta=cfg.early_stop_min_delta,
                mode=cfg.early_stop_mode,
                relative=cfg.early_stop_relative,
            )
            self.logger.debug(
                f"Early stopping enabled: patience={cfg.early_stop_patience}, min_delta={cfg.early_stop_min_delta} "
                f"({'relative' if cfg.early_stop_relative else 'absolute'}), mode={cfg.early_stop_mode}, "
                f"metric={cfg.early_stop_metric}"
            )

        early_stop_epoch = None
        for epoch in range(cfg.max_epochs):
            train = self._train_epoch(epoch)
            held_out = {split: self._eval_epoch(loaders[split], epoch, split) for split in splits}
            losses = {"train_loss": train.mean, **{f"{split}_loss": held_out[split].mean for split in splits}}

            if self.is_main:
                best_tracker.update(losses[cfg.best_metric], epoch)
                if epoch % cfg.report_every_epochs == 0:
                    self._report(epoch, train, held_out)
                self.run.log_epoch(
                    epoch + 1, train.mean, val_loss=losses.get("val_loss"), test_loss=losses.get("test_loss")
                )
                if epoch % cfg.save_every_epochs == 0:
                    self.on_periodic_save(epoch)
                if epoch == best_tracker.min_idx:
                    self.on_best(epoch)

            if early_stop is not None and self.is_main:
                early_metric = cfg.early_stop_metric
                early_stop.update(losses[early_metric], epoch)
                if early_stop.improved:
                    self.logger.debug(f"Epoch {epoch + 1}: {early_metric} improvement: {early_stop.improvement:.6f}")
                else:
                    self.logger.debug(
                        f"Epoch {epoch + 1}: No improvement in {early_metric}. "
                        f"Patience: {early_stop.patience_counter}/{cfg.early_stop_patience}"
                    )
                if early_stop.should_stop:
                    self.logger.debug(
                        f"Early stopping triggered at epoch {epoch + 1}. Best {early_metric}: "
                        f"{early_stop.best_metric:.6f} at epoch {early_stop.best_epoch + 1}"
                    )
                    if epoch % cfg.save_every_epochs != 0:
                        self.on_periodic_save(epoch)
                    early_stop_epoch = epoch + 1
                    break

        # Only the main process tracks the best epoch; the others report epoch 0.
        best_idx = best_tracker.min_idx
        summary = FitSummary(
            epochs_run=epoch + 1,
            best_epoch=0 if best_idx is None else best_idx + 1,
            best_metric=cfg.best_metric,
            best_loss=best_tracker.min,
            final_train_loss=train.mean,
            final_val_loss=losses.get("val_loss"),
            final_test_loss=losses.get("test_loss"),
            early_stop_epoch=early_stop_epoch,
        )
        if self.is_main:
            self.summary_writer.close()
            self.logger.info(f"Epoch {summary.best_epoch} is the best epoch (based on {cfg.best_metric}).")
            self.logger.info("Training Finished!")
        self.on_fit_end(summary)
        return summary
