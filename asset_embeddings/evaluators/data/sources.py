"""Per-period files joined onto the finished primary frame (after the placebo draw, the dummies and the NA drop)."""

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Optional, Sequence, Union

import pandas as pd

from .files import resolve_period_files
from .frame import GroupStats, PeriodFrame, PeriodGroups
from .grouping import assign_groups

logger = logging.getLogger(__name__)

PathSpec = Union[str, Path, List[str], None]


class PeriodSource(ABC):
    """A file per period, joined onto each period's frame.

    Args:
        path: The config value of the source's path field, one file per period.
        name: The config field name, for messages.
    """

    def __init__(self, path: PathSpec, name: str):
        self.path = path
        self.name = name
        self.files: List[Path] = resolve_period_files(path)

    @abstractmethod
    def join(self, pf: PeriodFrame, i: int) -> PeriodFrame:
        """Join period ``i``'s file onto ``pf`` and return the updated frame."""
        ...


def read_firm_file(path: Path, stock_col: str, columns: Sequence[str], name: str) -> pd.DataFrame:
    """One per-period firm file: ``stock_col`` as a six-digit ``Token`` column, then ``columns``.

    Args:
        path: The file.
        stock_col: Its stock id column.
        columns: The other columns to keep.
        name: The config field the file comes from, for messages.

    Returns:
        ``Token`` and ``columns``, one row per stock.

    Raises:
        ValueError: When the file lacks a column, or lists a stock twice (a duplicate would copy that firm in the
            join).
    """
    df = pd.read_csv(path, dtype={stock_col: str})
    missing = [c for c in [stock_col, *columns] if c not in df.columns]
    if missing:
        raise ValueError(f"{name} file {path} lacks column(s) {missing}; found {list(df.columns)}")
    df = df[[stock_col, *columns]].rename(columns={stock_col: "Token"})
    df["Token"] = df["Token"].str.zfill(6)
    dup = df["Token"][df["Token"].duplicated()]
    if len(dup):
        raise ValueError(f"{name} file {path} lists {len(dup)} stock(s) twice, e.g. {sorted(dup.unique())[:5]}")
    return df.reset_index(drop=True)


class LabelFileSource(PeriodSource):
    """A per-period firm label, inner-joined onto the frame: firms without a label leave the sample.

    The join comes after the primary assembly, so the placebo draw and the NA drop are those of the frame. The
    labeled rows of the file go to the ``label_file`` step of the funnel and the rows left by the join to
    ``label_joined``.

    Args:
        path: ``label_path``, one file per period.
        stock_col: Stock id column of the label file.
        label_col: The label column, added to the frame under its own name.
    """

    def __init__(self, path: PathSpec, stock_col: str, label_col: str):
        super().__init__(path, "label_path")
        self.stock_col = stock_col
        self.label_col = label_col

    def join(self, pf: PeriodFrame, i: int) -> PeriodFrame:
        """Join period ``i``'s labeled firms onto ``pf``."""
        labels = read_firm_file(self.files[i], self.stock_col, [self.label_col], self.name)
        labels = labels.dropna(subset=[self.label_col])
        pf.sample.add("label_file", len(labels))
        pf.frame = pf.frame.merge(labels, on="Token", how="inner")
        pf.sample.add("label_joined", len(pf.frame))
        logger.debug(f"{pf.period}: {len(labels)} labeled firms in the file, {len(pf.frame)} after the join")
        return pf


GROUP_COLUMN = "group"


class GroupFileSource(PeriodSource):
    """The grouped evaluation's group file: firms assigned to groups by a value, joined as a ``group`` column.

    Groups are assigned on the file's own universe (every firm it lists), not on the firms of the frame. The join
    is a lookup that keeps every row of the frame (firms absent from the file get no group), because the penalty
    chosen on the whole cross section (``alpha_scope="cross_section"``) needs every row. The file's row count goes to
    the ``group_file`` step of the funnel and the assignment with its per-group statistics to ``pf.groups``.

    Args:
        path: ``group_path``, one file per period.
        mode: ``"quantile"``, ``"fixed"`` or ``"categorical"``.
        n_quantiles: Number of quantile groups.
        bins: Inner cut points of the fixed bins.
        labels: Explicit fixed-bin labels.
        stock_col: Stock id column of the group file.
        value_col: Grouping value column of the group file.
    """

    def __init__(
        self,
        path: PathSpec,
        mode: str,
        n_quantiles: Optional[int] = None,
        bins: Optional[Sequence[float]] = None,
        labels: Optional[Sequence[str]] = None,
        *,
        stock_col: str,
        value_col: str,
    ):
        super().__init__(path, "group_path")
        self.mode = mode
        self.n_quantiles = n_quantiles
        self.bins = bins
        self.labels = labels
        self.stock_col = stock_col
        self.value_col = value_col

    def read(self, i: int) -> pd.DataFrame:
        """Period ``i``'s group file: ``Token`` (six-digit stock ids) and the grouping value."""
        return read_firm_file(self.files[i], self.stock_col, [self.value_col], self.name)

    def assign(self, df: pd.DataFrame) -> PeriodGroups:
        """Group the file's firms and summarize each group's grouping values (not under the categorical grouping)."""
        labels, groups = assign_groups(df[self.value_col], self.mode, self.n_quantiles, self.bins, self.labels)
        members, stats = {}, {}
        for label in labels:
            rows = df[groups == label]
            members[label] = set(rows["Token"])
            if len(rows) and self.mode == "categorical":
                stats[label] = GroupStats(n_firms=len(members[label]))
            elif len(rows):
                values = rows[self.value_col]
                stats[label] = GroupStats(
                    n_firms=len(members[label]),
                    value_mean=float(values.mean()),
                    value_median=float(values.median()),
                    value_min=float(values.min()),
                    value_max=float(values.max()),
                )
            logger.debug(f"  {label}: {len(rows)} stocks")
        return PeriodGroups(labels=labels, members=members, stats=stats, n_total=len(df))

    def join(self, pf: PeriodFrame, i: int) -> PeriodFrame:
        """Add the ``group`` column (NaN for firms outside the file) and attach the assignment."""
        pf.groups = self.assign(self.read(i))
        lookup = {token: label for label in pf.groups.labels for token in pf.groups.members[label]}
        pf.frame = pf.frame.assign(**{GROUP_COLUMN: pf.frame["Token"].map(lookup)})
        pf.sample.add("group_file", pf.groups.n_total)
        return pf
