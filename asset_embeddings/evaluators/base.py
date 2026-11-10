"""The evaluator lifecycle shared by every benchmark: config, run, per-period loop, records, summary.

[`BaseEvaluator`][asset_embeddings.evaluators.base.BaseEvaluator] knows nothing about ridges or features. It
assembles each period's firm universe through the data layer (``evaluators.data``), joins the extra sources, hands
the frame to ``evaluate_period`` and saves the records the period returns as soon as it completes. A CLI drives an
instance as ``attach(run)``, ``load_configs``, ``load_overrides``, ``prepare``, ``eval``. The evaluations over
feature columns sit on ``FeatureEvaluator`` (``evaluators.feature``).
"""

import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional, Type

from tqdm import tqdm

from asset_embeddings.configs import Config
from asset_embeddings.records import Record, Run, SampleStep

from .data import PeriodAssembler, PeriodFrame, PeriodSource


@dataclass
class PeriodOutcome:
    """What one period (or the summary) produced.

    Attributes:
        value: The primary score, logged and averaged by the default summary (may be NaN).
        records: The rows to save, of any record classes, in order.
    """

    value: float
    records: List[Record] = field(default_factory=list)


class BaseEvaluator(ABC):
    """Template of an evaluation over the periods of a config.

    A subclass declares its config class and task name, the benchmark columns its target needs, optionally extra
    per-period sources (``sources``) and the assembler (``make_assembler``), and computes each period in
    ``evaluate_period``. The summary hook (``summarize``) logs the mean value by default; ``after_periods`` runs
    once the summary is saved.
    """

    def __init__(self):
        self.logger: logging.Logger = logging.getLogger(type(self).__module__)  # the subclass's module logger
        self.config: Config = self.config_class()
        self.extra_sources: List[PeriodSource] = []
        self.assembler: Optional[PeriodAssembler] = None
        # Records: the CLI attaches the real Run; the placeholder writes nowhere.
        self.run: Run = Run(None, script=type(self).__module__)

    @property
    @abstractmethod
    def config_class(self) -> Type[Config]:
        """The `Config` subclass this evaluator is configured by."""
        ...

    @property
    @abstractmethod
    def task_name(self) -> str:
        """Task label written to the ``runs`` row and every result row."""
        ...

    @property
    def benchmark_columns(self) -> List[str]:
        """Benchmark columns the target needs beyond the stock id."""
        return []

    def sources(self) -> List[PeriodSource]:
        """Per-period sources joined onto the primary frame, in order; none by default."""
        return []

    @abstractmethod
    def evaluate_period(self, pf: PeriodFrame) -> PeriodOutcome:
        """Compute one period and return its value and records."""
        ...

    def attach(self, run: Run) -> None:
        """Bind the run that owns this evaluation's records."""
        self.run = run

    def load_configs(self, config_file: str) -> None:
        """Load the JSON config file into ``self.config`` and validate it."""
        self.logger.info(f"Loading configuration from `{os.path.normpath(config_file)}`")
        self.config.from_file(config_file)
        self.config.validate()

    def load_overrides(self, overrides: Dict[str, Any]) -> None:
        """Apply ``--override`` key/value pairs on top of the loaded config and validate again."""
        self.logger.info(f"Applying overrides: {overrides}")
        self.config.from_dict(overrides)
        self.config.validate()

    def make_assembler(self) -> PeriodAssembler:
        """The assembler of each period's primary frame: the firm universe and the target columns."""
        return PeriodAssembler(self.config, self.benchmark_columns)

    def prepare(self) -> None:
        """Build the assembler and the extra sources from the config.

        The config has already checked that every path names one file per period, so nothing here can mismatch.
        No global RNG is seeded: every random draw of an evaluation (placebo, CV folds, bootstrap, permutations)
        takes its own generator from ``config.seed``.
        """
        self.logger.info("Preparing Evaluator...")
        self.logger.debug(f"Configuration: {self.config.to_dict()}")
        self.assembler = self.make_assembler()
        self.extra_sources = self.sources()
        self.logger.info(f"{len(self.config.periods)} period(s); seed {self.config.seed}.")

    def iter_periods(self) -> Iterator[PeriodFrame]:
        """Assemble each period's primary frame, then join the extra sources onto it in order."""
        for i in range(len(self.config.periods)):
            pf = self.assembler.assemble(i)
            for source in self.extra_sources:
                pf = source.join(pf, i)
            yield pf

    @staticmethod
    def build_sample(pf: PeriodFrame) -> List[SampleStep]:
        """The period's funnel as ``eval_sample`` rows, numbered in the order the steps ran."""
        return [
            SampleStep(period=pf.period, step_no=i, step=step, n_rows=n_rows)
            for i, (step, n_rows) in enumerate(pf.sample.steps, start=1)
        ]

    def save(self, outcome: PeriodOutcome) -> None:
        """Write the outcome's records to the run in one transaction."""
        self.run.save_many(outcome.records)

    def summarize(self, outcomes: List[PeriodOutcome]) -> PeriodOutcome:
        """Log the mean of the period values and return it (no records by default)."""
        mean = sum(o.value for o in outcomes) / len(outcomes)
        self.logger.info(f"Mean {self.task_name}: {mean}")
        self.logger.info(f"{self.task_name} Benchmark Evaluation Completed.")
        return PeriodOutcome(value=mean)

    def after_periods(self) -> None:
        """Checks across the periods, run after the summary is saved; nothing by default."""

    def eval(self) -> float:
        """Run the period loop, saving each period's records as it completes; return the summary value."""
        outcomes: List[PeriodOutcome] = []
        for pf in tqdm(
            self.iter_periods(), total=len(self.config.periods), desc=f"Calculating {self.task_name} Scores"
        ):
            outcome = self.evaluate_period(pf)
            self.logger.info(f"{self.task_name} Score for period {pf.period}: {outcome.value}")
            self.save(outcome)
            outcomes.append(outcome)
        summary = self.summarize(outcomes)
        self.save(summary)
        self.after_periods()
        return summary.value
