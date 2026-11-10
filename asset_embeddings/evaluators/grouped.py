"""Grouped RV and RC evaluations: the benchmark ridge fitted within groups of firms, with bootstrap intervals.

Firms are grouped per period by a value of a group file (institutional coverage: quantile groups or fixed bins).
Each (period, group) cell is scored like its parent benchmark (on the cross section, or under holdout scoring by the
held-out R² aggregated over firm splits), with the target optionally winsorized within the cell, and a percentile
bootstrap interval around the score. After the last period, permutation tests compare the
groups' per-period scores. Run from ``python -m asset_embeddings.scripts.eval.rv_grouped`` / ``rc_grouped``.
"""

from abc import abstractmethod
from typing import Any, Dict, Iterable, List, Optional, Tuple, Type

import numpy as np
import pandas as pd

from asset_embeddings.configs import Config, GroupedRCEvalConfig, GroupedRVEvalConfig
from asset_embeddings.records import (
    GroupedRCResult,
    GroupedRCTest,
    GroupedResult,
    GroupedRidgeSplit,
    GroupedRVResult,
    GroupedRVTest,
    GroupedTest,
    Record,
    Result,
)

from .base import PeriodOutcome
from .benchmark import RidgeBenchmarkEvaluator
from .data import PeriodFrame, PeriodSource
from .data.grouping import auto_bin_labels, category_sort_key
from .data.sources import GroupFileSource
from .estimation import (
    CV_ALPHA_GRID,
    Penalty,
    RidgeEstimator,
    UnscorableSplit,
    bootstrap_interval,
    make_estimator,
    permutation_test,
)
from .rc import RCBenchmarkEvaluator
from .rv import RVBenchmarkEvaluator

#: A cell below this many firms (in the group file, or after the merge) is skipped: the 10-fold CV needs 10 rows.
MIN_CELL = 10


def permutation_pairs(labels: List[str]) -> List[Tuple[str, str]]:
    """Adjacent groups in label order, then the first against the last when there are more than two groups."""
    pairs = [(labels[i], labels[i + 1]) for i in range(len(labels) - 1)]
    if len(labels) > 2:
        pairs.append((labels[0], labels[-1]))
    return pairs


class GroupedRidgeEvaluator(RidgeBenchmarkEvaluator):
    """The grouped variant of a ridge benchmark; combined with RV or RC, whose target and design it reuses.

    Per period the penalty is selected once on the full cross section (``alpha_scope="cross_section"``) or per cell
    (``"cell"``), on a grid floored at ``cv_alpha_min``; each cell's score comes with a bootstrap interval that
    rescores resamples of the cell at the selected penalty. Under holdout scoring the penalty is one per firm split
    (selected on the split's training firms of the period or of the cell), every split of a scored cell is
    recorded, a resample keeps the firms' tokens, so the copies of a firm share its fold, and a resample that
    cannot be scored is left out of the interval.
    """

    @property
    @abstractmethod
    def test_class(self) -> Type[GroupedTest]:
        """The record class of the permutation tests."""
        ...

    @property
    def group_labels(self) -> List[str]:
        """Every group label the config defines, in order.

        Empty under the categorical grouping, whose groups are the values found in the group files.
        """
        cfg = self.config
        if cfg.grouping == "fixed":
            return list(cfg.bin_labels) if cfg.bin_labels is not None else auto_bin_labels(cfg.bins)
        if cfg.grouping == "categorical":
            return []
        return [f"G{i + 1}" for i in range(cfg.n_quantiles)]

    def ordered_labels(self, scored: Iterable[str]) -> List[str]:
        """The labels the permutation tests run over: the config's, or the scored categories in value order."""
        if self.config.grouping == "categorical":
            return sorted(set(scored), key=category_sort_key)
        return self.group_labels

    def sources(self) -> List[PeriodSource]:
        """The parent's sources, then the group file."""
        cfg = self.config
        return super().sources() + [
            GroupFileSource(
                cfg.group_path,
                cfg.grouping,
                cfg.n_quantiles,
                cfg.bins,
                cfg.bin_labels,
                stock_col=cfg.group_stock_col,
                value_col=cfg.group_col,
            )
        ]

    def penalty_grid(self) -> np.ndarray:
        """The CV grid restricted to ``alpha >= cv_alpha_min``; the full grid when unset, 0, or above every point.

        On the well-covered groups the cross-validated penalty comes out low, low enough that the in-sample score
        runs well ahead of the held-out one; the floor forces a heavier penalty there. The thinly covered groups
        already select penalties above it.
        """
        floor = self.config.cv_alpha_min
        if not floor:
            return CV_ALPHA_GRID
        floored = CV_ALPHA_GRID[CV_ALPHA_GRID >= floor]
        return floored if len(floored) else CV_ALPHA_GRID

    def make_estimator(self) -> RidgeEstimator:
        """The estimator ``scoring`` asks for, on the floored grid."""
        return make_estimator(self.config, self.penalty_grid())

    def winsorize(self, y):
        """Clip the target to its ``[p, 100 - p]`` percentiles when ``winsor_pct`` is p; unchanged when unset or 0."""
        p = self.config.winsor_pct
        if not p:
            return y
        lo, hi = np.percentile(y, [p, 100 - p])
        return np.clip(y, lo, hi)

    def cell_problem(self, frame: pd.DataFrame) -> Tuple[np.ndarray, Any, np.ndarray, Dict[str, Any]]:
        """Design matrix, winsorized target, stock ids and target diagnostics of one (possibly resampled) frame."""
        frame, y, diagnostics = self.target(frame)
        return self.design(frame), self.winsorize(y), frame["Token"].to_numpy(), diagnostics

    def select_penalty(self, frame: pd.DataFrame) -> Penalty:
        """The configured penalty, or the cross-validated one on ``frame``."""
        X, y, ids, _ = self.cell_problem(frame)
        return self.estimator.select_penalty(X, y, ids)

    def cell_value(self, frame: pd.DataFrame, penalty: Penalty) -> float:
        """The cell's score at ``penalty``; the value the bootstrap recomputes on each resample."""
        X, y, ids, _ = self.cell_problem(frame)
        return self.estimator.score(X, y, ids, penalty).value

    def evaluate_period(self, pf: PeriodFrame) -> PeriodOutcome:
        """Score every group of the period with its bootstrap interval; the value is the mean cell score."""
        cfg = self.config
        period, frame, groups = pf.period, pf.frame, pf.groups
        period_penalty: Optional[Penalty] = None
        if cfg.alpha_scope == "cross_section":
            period_penalty = self.select_penalty(frame)
            self.logger.info(f"  {period}: alpha={float(np.median(period_penalty.alphas)):.6g} (n={len(frame)})")

        records: List[Record] = []
        scores = []
        for label in groups.labels:
            members = groups.members[label]
            if len(members) < MIN_CELL:
                self.logger.warning(f"Group {label} {period}: only {len(members)} tokens, skipping")
                continue
            cell = frame[frame["Token"].isin(members)]
            if len(cell) < MIN_CELL:
                self.logger.warning(f"Group {label} {period}: only {len(cell)} merged rows, skipping")
                continue

            try:
                penalty = period_penalty if period_penalty is not None else self.select_penalty(cell)
                X, y, ids, diagnostics = self.cell_problem(cell)
                fit = self.estimator.score(X, y, ids, penalty)
            except UnscorableSplit as err:
                self.logger.warning(f"Group {label} {period}: {err}, skipping")
                continue
            self.log_diagnostics(diagnostics)
            score = fit.value

            n_unscored = 0

            def resample_value(resample: pd.DataFrame) -> Optional[float]:
                nonlocal n_unscored
                try:
                    return self.cell_value(resample, penalty)
                except UnscorableSplit:
                    n_unscored += 1
                    return None

            interval = bootstrap_interval(resample_value, cell, cfg.n_bootstrap, cfg.seed)
            if n_unscored:
                self.logger.warning(
                    f"Group {label} {period}: {n_unscored} of {cfg.n_bootstrap} resamples could not be scored, "
                    f"left out of the interval"
                )
            ci = f"[{interval.lower:.4f}, {interval.upper:.4f}]" if interval else "[CI skipped]"
            self.logger.info(f"{label} {period} : R^2={score:.4f} {ci} n={len(cell)}")
            scores.append(score)

            stats = groups.stats[label]
            records.append(
                self.result_class(
                    period=period,
                    group_label=label,
                    metric=fit.metric,
                    value=float(score),
                    ci_lower=interval.lower if interval else None,
                    ci_upper=interval.upper if interval else None,
                    # float() because a fixed ridge_alpha read from JSON may be an integer (1000).
                    ridge_alpha=float(fit.alpha),
                    n_scored=fit.n_scored,
                    n_features=len(self.features.columns(cell)),
                    n_group_firms=stats.n_firms,
                    group_value_mean=stats.value_mean,
                    group_value_median=stats.value_median,
                    group_value_min=stats.value_min,
                    group_value_max=stats.value_max,
                )
            )
            records.extend(
                GroupedRidgeSplit(
                    period=period,
                    group_label=label,
                    split=s.split,
                    value=float(s.value),
                    value_sklearn=float(s.value_sklearn),
                    alpha=float(s.alpha),
                    n_train=int(s.n_train),
                    n_test=int(s.n_test),
                )
                for s in fit.splits
            )

        value = float(np.mean(scores)) if scores else float("nan")
        return PeriodOutcome(value=value, records=records + self.build_sample(pf))

    def summarize(self, outcomes: List[PeriodOutcome]) -> PeriodOutcome:
        """Permutation tests between groups on their per-period scores, recorded; per-group means logged."""
        cfg = self.config
        by_group: Dict[str, List[float]] = {label: [] for label in self.group_labels}
        for outcome in outcomes:
            for record in outcome.records:
                if isinstance(record, GroupedResult):
                    by_group.setdefault(record.group_label, []).append(record.value)

        records = []
        labels = self.ordered_labels(label for label, scores in by_group.items() if scores)
        for a, b in permutation_pairs(labels):
            if by_group[a] and by_group[b]:
                statistic, p = permutation_test(by_group[a], by_group[b], cfg.n_permutation, cfg.seed)
                self.logger.info(f"Permutation test {a} vs {b}: p={p:.4f}")
                test = self.test_class(
                    group_a=a,
                    group_b=b,
                    statistic=statistic,
                    p_value=p,
                    n_a=len(by_group[a]),
                    n_b=len(by_group[b]),
                )
                records.append(test)

        for label, scores in by_group.items():
            if scores:
                self.logger.info(f"  {label}: mean R^2={np.mean(scores):.4f} (n_periods={len(scores)})")
        all_scores = [s for scores in by_group.values() for s in scores]
        self.logger.info(f"{self.task_name} Evaluation Completed.")
        return PeriodOutcome(value=float(np.mean(all_scores)) if all_scores else float("nan"), records=records)


class GroupedRVEvaluator(GroupedRidgeEvaluator, RVBenchmarkEvaluator):
    """Grouped relative valuation: the RV valuation residual fitted within coverage groups."""

    @property
    def config_class(self) -> Type[Config]:
        """`GroupedRVEvalConfig`: the RV fields plus the grouping fields."""
        return GroupedRVEvalConfig

    @property
    def task_name(self) -> str:
        """``"GROUPED_RV"``."""
        return "GROUPED_RV"

    @property
    def result_class(self) -> Type[Result]:
        """`GroupedRVResult`."""
        return GroupedRVResult

    @property
    def test_class(self) -> Type[GroupedTest]:
        """`GroupedRVTest`."""
        return GroupedRVTest


class GroupedRCEvaluator(GroupedRidgeEvaluator, RCBenchmarkEvaluator):
    """Grouped return comovement: the RC next-quarter return fitted within coverage groups."""

    @property
    def config_class(self) -> Type[Config]:
        """`GroupedRCEvalConfig`: the RC fields plus the grouping fields."""
        return GroupedRCEvalConfig

    @property
    def task_name(self) -> str:
        """``"GROUPED_RC"``."""
        return "GROUPED_RC"

    @property
    def result_class(self) -> Type[Result]:
        """`GroupedRCResult`."""
        return GroupedRCResult

    @property
    def test_class(self) -> Type[GroupedTest]:
        """`GroupedRCTest`."""
        return GroupedRCTest
