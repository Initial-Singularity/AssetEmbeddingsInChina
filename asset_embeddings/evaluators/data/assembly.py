"""Build one period's primary frame: the firm universe, then (for a feature evaluation) the feature columns.

[`PeriodAssembler`][asset_embeddings.evaluators.data.assembly.PeriodAssembler] reads the benchmark's target
columns, inner-joins the embedding when the period has an embedding file (otherwise the benchmark's firms are the
universe) and drops rows with a missing value. It reads only the data fields of the config.

[`FeatureAssembler`][asset_embeddings.evaluators.data.assembly.FeatureAssembler] adds the feature columns through
the assembler's extension points: the characteristics when ``"chars"`` is a feature or ``align_chars`` is set,
``IndustryCode`` when ``"industry"`` is a feature (both from the benchmark file, or from ``char_benchmark_path``),
the placebo columns and the industry dummies. Every read column enters the NA drop, which is how ``align_chars``
restricts the firms.

The placebo columns are drawn before the NA drop, one row per firm of the joined frame in file order, from a
generator of the period's own: ``np.random.default_rng([seed, crc32(period)])``. Seeding by the period label
rather than its position in ``periods`` makes a period's draw the same whether it is evaluated alone or after
other periods.
"""

import logging
import os
import zlib
from pathlib import Path
from typing import Dict, FrozenSet, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from asset_embeddings.configs import Config

from .files import resolve_period_files
from .frame import PeriodFrame, SampleFunnel

logger = logging.getLogger(__name__)

N_PLACEBO = 4


def _unique(columns: Sequence[str]) -> List[str]:
    return list(dict.fromkeys(columns))


class PeriodAssembler:
    """Builds each period's firm universe from the config's embedding and benchmark files.

    The funnel steps are ``embedding_file``, ``benchmark_file`` and ``joined`` (with an embedding file) or
    ``benchmark_file`` alone, then ``na_dropped``. A subclass adds columns through three extension points, called
    in this order: ``extra_benchmark_columns``, ``join_extra`` and ``transform``.

    Args:
        config: The evaluation config (its data fields: paths, periods, column names).
        benchmark_columns: The benchmark columns the target needs beyond the stock id.
    """

    def __init__(self, config: Config, benchmark_columns: Sequence[str]):
        self.config = config
        self.target_cols = _unique([config.benchmark_stock_col, *benchmark_columns])
        self.embedding_files: List[Path] = resolve_period_files(config.embedding_path)
        self.benchmark_files: List[Path] = resolve_period_files(config.benchmark_path)

    def assemble(self, i: int) -> PeriodFrame:
        """Read, join and clean period ``i`` of ``config.periods``.

        Returns:
            The period's frame with a ``Token`` column, the target columns and the subclass's columns.
        """
        period = self.config.periods[i]
        frame, sample = self._read(i)
        frame = self.transform(frame, period)
        frame = self._drop_na(frame, sample)
        return PeriodFrame(period=period, frame=frame, sample=sample)

    def extra_benchmark_columns(self, i: int) -> List[str]:
        """Columns read from period ``i``'s benchmark file beside the target columns; none here."""
        return []

    def join_extra(self, frame: pd.DataFrame, sample: SampleFunnel, key: str, i: int) -> pd.DataFrame:
        """Join further files onto the frame after the embedding join, on ``key``; nothing here."""
        return frame

    def transform(self, frame: pd.DataFrame, period: str) -> pd.DataFrame:
        """Add or replace columns before the NA drop; nothing here."""
        return frame

    def _read(self, i: int) -> Tuple[pd.DataFrame, SampleFunnel]:
        """Read the benchmark, inner-join the embedding (when the period has one) and the subclass's files."""
        stock = self.config.benchmark_stock_col
        embedding_file: Optional[Path] = self.embedding_files[i] if self.embedding_files else None
        benchmark_file = self.benchmark_files[i]
        sample = SampleFunnel()

        columns = _unique(self.target_cols + self.extra_benchmark_columns(i))
        logger.debug(f"Loading benchmark data from `{os.path.normpath(benchmark_file)}` with columns: {columns}")
        benchmark = pd.read_csv(benchmark_file, dtype={stock: str}, usecols=columns)

        if embedding_file is not None:
            logger.debug(f"Loading embedding data from `{os.path.normpath(embedding_file)}`")
            embedding = pd.read_csv(embedding_file, dtype={self.config.embedding_token_col: str})
            frame = pd.merge(embedding, benchmark, left_on="Token", right_on=stock, how="inner")
            frame = frame.drop(columns=[stock])
            sample.add("embedding_file", len(embedding))
            sample.add("benchmark_file", len(benchmark))
            sample.add("joined", len(frame))
            logger.debug(
                f"After merge: {len(frame)} rows retained (embedding {len(embedding)}, benchmark {len(benchmark)})"
            )
            key = "Token"
        else:
            frame = benchmark
            key = stock
            sample.add("benchmark_file", len(benchmark))

        frame = self.join_extra(frame, sample, key, i)

        if embedding_file is None:
            frame = frame.rename(columns={stock: "Token"})
        return frame, sample

    def _drop_na(self, frame: pd.DataFrame, sample: SampleFunnel) -> pd.DataFrame:
        """Drop rows with any missing value; the rows left are the ``na_dropped`` step."""
        n_before = len(frame)
        na_ratios = frame.isna().mean()
        frame = frame.dropna()
        n_dropped = n_before - len(frame)
        if n_dropped > 0:
            logger.debug(f"Dropped {n_dropped} rows due to NA values ({n_dropped / n_before * 100:.2f}%)")
            logger.debug(f"NA ratios before dropna:\n{na_ratios[na_ratios > 0]}")
        sample.add("na_dropped", len(frame))
        return frame


class FeatureAssembler(PeriodAssembler):
    """The firm universe plus the feature columns of ``config.features``.

    Reads the feature fields of the config itself (``features``, ``align_chars``, ``char_benchmark_path``). The
    separate characteristics file adds the funnel steps ``char_file`` and ``char_joined``.

    Args:
        config: The evaluation config (data and feature fields).
        benchmark_columns: The benchmark columns the target needs beyond the stock id.
        char_cols: The benchmark's characteristic columns.

    Attributes:
        industry_cols: The ``Ind_*`` dummy columns of each assembled period, by period label (empty sets without
            the ``"industry"`` feature).
    """

    def __init__(self, config: Config, benchmark_columns: Sequence[str], char_cols: Sequence[str]):
        super().__init__(config, benchmark_columns)
        self.char_cols = list(char_cols)
        self.char_files: List[Path] = resolve_period_files(config.char_benchmark_path)
        self.industry_cols: Dict[str, FrozenSet[str]] = {}

    @property
    def _chars(self) -> bool:
        return "chars" in self.config.features or self.config.align_chars

    @property
    def _industry(self) -> bool:
        return "industry" in self.config.features

    def _char_file(self, i: int) -> Optional[Path]:
        return self.char_files[i] if self.char_files else None

    def _feature_source_columns(self) -> List[str]:
        """The characteristics and ``IndustryCode`` columns this config reads."""
        return (self.char_cols if self._chars else []) + (["IndustryCode"] if self._industry else [])

    def extra_benchmark_columns(self, i: int) -> List[str]:
        """The characteristics and ``IndustryCode``, unless the period has a separate characteristics file."""
        return [] if self._char_file(i) is not None else self._feature_source_columns()

    def join_extra(self, frame: pd.DataFrame, sample: SampleFunnel, key: str, i: int) -> pd.DataFrame:
        """Inner-join the separate characteristics file when the period has one and the config reads from it."""
        char_file = self._char_file(i)
        if char_file is None or not self._feature_source_columns():
            return frame
        stock = self.config.benchmark_stock_col
        chars, n_chars = self._read_chars(char_file)
        if key == "Token":
            chars = chars.rename(columns={stock: "Token"})
        n_before = len(frame)
        frame = pd.merge(frame, chars, on=key, how="inner")
        sample.add("char_file", n_chars)
        sample.add("char_joined", len(frame))
        logger.debug(f"After char_benchmark join: {len(frame)} rows retained (dropped {n_before - len(frame)})")
        return frame

    def transform(self, frame: pd.DataFrame, period: str) -> pd.DataFrame:
        """The placebo draw, then the industry dummies."""
        frame = self._add_placebo(frame, period)
        frame, self.industry_cols[period] = self._add_industry_dummies(frame)
        return frame

    def _read_chars(self, char_file) -> Tuple[pd.DataFrame, int]:
        """Load the characteristics and ``IndustryCode`` this config reads from the separate file.

        This file is the sole source of feature columns when ``char_benchmark_path`` is set, so a missing column
        is refused here, where the message can name it.

        Raises:
            ValueError: When the file lacks the stock column, a declared characteristic, or ``IndustryCode``.
        """
        stock = self.config.benchmark_stock_col
        present = pd.read_csv(char_file, nrows=0).columns
        need = _unique([stock] + self._feature_source_columns())
        missing = [c for c in need if c not in present]
        if missing:
            raise ValueError(
                f"char_benchmark_path file `{os.path.normpath(str(char_file))}` is missing required columns: {missing}"
            )
        logger.debug(f"Loading char benchmark from `{os.path.normpath(str(char_file))}` with columns: {need}")
        chars = pd.read_csv(char_file, dtype={stock: str}, usecols=need)
        return chars, len(chars)

    def _add_placebo(self, frame: pd.DataFrame, period: str) -> pd.DataFrame:
        """With the ``"placebo"`` feature, add ``Placebo_1..4`` drawn uniformly on [-1, 1] from the period's generator."""
        if "placebo" in self.config.features:
            # crc32 because Python's str hash is salted per process.
            rng = np.random.default_rng([self.config.seed, zlib.crc32(period.encode())])
            for i in range(1, N_PLACEBO + 1):
                frame[f"Placebo_{i}"] = rng.uniform(-1, 1, size=len(frame))
        return frame

    def _add_industry_dummies(self, frame: pd.DataFrame) -> Tuple[pd.DataFrame, FrozenSet[str]]:
        """With the ``"industry"`` feature, replace ``IndustryCode`` by ``Ind_*`` dummies (first level dropped)."""
        if not self._industry:
            return frame, frozenset()
        dummies = pd.get_dummies(frame["IndustryCode"], prefix="Ind", drop_first=True)
        frame = pd.concat([frame, dummies], axis=1).drop(columns=["IndustryCode"])
        logger.debug(f"Created {len(dummies.columns)} industry dummy columns")
        return frame, frozenset(dummies.columns)
