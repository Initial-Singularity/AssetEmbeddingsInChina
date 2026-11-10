"""The shape every trainer shares: its config, the run it records into, and the config artifact it writes."""

import logging
import os
from abc import ABC, abstractmethod
from typing import Any, Dict

from ..configs import ConfigContainer
from ..records import Run


class BaseTrainer(ABC):
    """Common ground of the training classes and the two MP trainers.

    A subclass sets ``self.config`` (a ``Config`` or a ``ConfigContainer``) in its ``__init__`` and implements
    ``task_name``, ``save_folder``, ``prepare`` and ``train``. The base owns what was copied between them
    before: the module logger, the placeholder [`Run`][asset_embeddings.records.Run], ``attach``, config
    loading with overrides, and ``save_config`` (the ``Configs: {...}`` line and ``config.json`` next to the
    artifacts).

    Logging convention: the trainer logs the phases it owns (``Preparing Trainer...`` / ``Trainer Prepared.``),
    its own decisions and where it saved; datasets and preparers report what they load and build themselves.
    """

    config: Any

    def __init__(self) -> None:
        self.logger: logging.Logger = logging.getLogger(type(self).__module__)  # the subclass's module logger
        # Records: `main()` attaches the real Run; the placeholder writes nowhere.
        self.run: Run = Run(None, script=type(self).__module__)

    @property
    @abstractmethod
    def task_name(self) -> str:
        """``runs.task`` of this trainer (``pretrain_W2V``, ``fit_RS``, ``MP``, ...)."""

    @property
    @abstractmethod
    def save_folder(self) -> str:
        """Where the artifacts and ``config.json`` of this run go."""

    def attach(self, run: Run) -> None:
        """Bind the run that owns this trainer's records (``runs`` row, recording, epochs)."""
        self.run = run

    def _probe_resources(self) -> None:
        """One DEBUG line when the run's VRAM high-water mark moved (``ResourceProbe.changed``), else nothing."""
        line = self.run.resources.changed()
        if line:
            self.logger.debug(line)

    def load_configs(self, config_file: str) -> None:
        """Load the config file into ``self.config``, validate it and hand it to the preparers.

        Args:
            config_file: Path of the JSON / YAML config.
        """
        self.logger.info(f"Loading configuration from `{os.path.normpath(config_file)}`")
        self.config.from_file(os.path.normpath(config_file))
        self.config.validate()
        self._configs_changed()

    def load_overrides(self, overrides: Dict[str, Any]) -> None:
        """Apply CLI overrides on top of the loaded config, validate and hand it to the preparers.

        Args:
            overrides: Field values to set; a ``ConfigContainer`` takes dotted keys (``train.seed``).
        """
        self.logger.info(f"Applying overrides: {overrides}")
        if isinstance(self.config, ConfigContainer):
            for key, value in overrides.items():  # dotted keys: "train.seed"
                self.config.set_nested(key, value)
        else:
            self.config.from_dict(overrides)
        self.config.validate()
        self._configs_changed()

    def _configs_changed(self) -> None:
        """Hook after a load or override: hand the (sub-)configs to the preparers."""

    def save_config(self) -> None:
        """Log the effective config and write it as ``config.json`` into ``save_folder``."""
        os.makedirs(self.save_folder, exist_ok=True)
        path = os.path.join(self.save_folder, "config.json")
        self.config.to_file(path)
        self.logger.info(f"Configs: {self.config.to_dict()}")
        self.logger.info(f"Config file saved to `{path}`")

    @abstractmethod
    def prepare(self) -> None:
        """Build everything ``train`` needs from the config: data, model, optimizer, the per-run training row."""
        ...

    @abstractmethod
    def train(self) -> None:
        """Run the training and write the artifacts and the final recording."""
        ...
