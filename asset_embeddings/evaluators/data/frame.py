"""The value one period's data assembly produces: the frame and the statistics of how it was built."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import pandas as pd


@dataclass
class SampleFunnel:
    """The sample funnel of one period: every file read, join and drop, in order, with its row count.

    File steps (``embedding_file``, ``benchmark_file``, ``char_file``, ``group_file``) hold the
    rows of the file; the other steps hold the rows left after the step. The vocabulary is
    ``SAMPLE_STEPS`` of ``asset_embeddings.records``.

    Attributes:
        steps: ``(step, n_rows)`` in the order the steps ran.
    """

    steps: List[Tuple[str, int]] = field(default_factory=list)

    def add(self, step: str, n_rows: int) -> None:
        """Append one step."""
        self.steps.append((step, int(n_rows)))

    def rows(self, step: str) -> Optional[int]:
        """Row count of ``step``; None when it did not run."""
        for name, n in self.steps:
            if name == step:
                return n
        return None


@dataclass(frozen=True)
class GroupStats:
    """One group of a period's group file.

    Attributes:
        n_firms: Distinct firms of the group file in the group.
        value_mean: Mean grouping value of those rows; None under the categorical grouping, where every row of a
            group holds the same value.
        value_median: Median grouping value (None under the categorical grouping).
        value_min: Smallest grouping value (None under the categorical grouping).
        value_max: Largest grouping value (None under the categorical grouping).
    """

    n_firms: int
    value_mean: Optional[float] = None
    value_median: Optional[float] = None
    value_min: Optional[float] = None
    value_max: Optional[float] = None


@dataclass
class PeriodGroups:
    """The group assignment of one period, made on the group file's own universe.

    Attributes:
        labels: Every group label in order, including groups without firms.
        members: The firms of each label (group-file tokens).
        stats: The statistics of each label that has firms.
        n_total: Rows of the period's group file (the share denominator).
    """

    labels: List[str]
    members: Dict[str, Set[str]]
    stats: Dict[str, GroupStats]
    n_total: int


@dataclass
class PeriodFrame:
    """One period's assembled data.

    Attributes:
        period: The period label of the config.
        frame: ``Token``, the target columns and the feature columns.
        sample: How the frame was built: its sample funnel.
        groups: The group assignment when a group file joined onto the frame; None otherwise.
    """

    period: str
    frame: pd.DataFrame
    sample: SampleFunnel
    groups: Optional[PeriodGroups] = None
