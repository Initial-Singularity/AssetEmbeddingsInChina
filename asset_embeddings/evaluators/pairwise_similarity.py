"""Pairwise similarity: how much closer two firms' embeddings are when the firms share a label or an industry.

Per period, the cosine similarity of the raw embedding vectors of every unordered firm pair is regressed by OLS on a
constant and one indicator per ``match_on`` column (1 when the two firms share the value), optionally with their
interactions, solved from per-pattern cell statistics (``estimation.pairwise``). After the last period each term's
coefficients are aggregated Fama-MacBeth style (``estimation.panel``). Run from
``python -m asset_embeddings.scripts.eval.pairwise_similarity``.
"""

import fnmatch
import math
from typing import Dict, List, Optional, Type

import numpy as np

from asset_embeddings.configs import Config, PairwiseSimilarityEvalConfig
from asset_embeddings.records import PairwiseCoef, PairwiseFit, PairwiseResult, Record

from .base import BaseEvaluator, PeriodOutcome
from .data import LabelFileSource, PeriodFrame, PeriodSource
from .estimation import cell_statistics, fama_macbeth, ols_from_cells, term_names


def _or_none(value: float) -> Optional[float]:
    return None if math.isnan(value) else value


class PairwiseSimilarityEvaluator(BaseEvaluator):
    """Fama-MacBeth regression of pairwise embedding similarity on same-value indicators.

    The evaluation sits on the lifecycle layer, not the feature layer: its frame is the firm universe (the raw
    embedding joined with the benchmark's ``match_on`` columns, no characteristics, no scaling) plus the label;
    firms missing any of them leave the sample.
    """

    @property
    def config_class(self) -> Type[Config]:
        """`PairwiseSimilarityEvalConfig`: the data and label fields plus ``match_on`` and ``interaction``."""
        return PairwiseSimilarityEvalConfig

    @property
    def task_name(self) -> str:
        """``"PAIRWISE"``."""
        return "PAIRWISE"

    @property
    def benchmark_columns(self) -> List[str]:
        """The ``match_on`` columns other than the label, read from the benchmark file."""
        return [c for c in self.config.match_on if c != self.config.label_col]

    def sources(self) -> List[PeriodSource]:
        """The label file."""
        cfg = self.config
        return [LabelFileSource(cfg.label_path, cfg.label_stock_col, cfg.label_col)]

    @property
    def terms(self) -> List[str]:
        """The regression terms, in coefficient order."""
        return term_names(list(self.config.match_on), self.config.interaction)

    def evaluate_period(self, pf: PeriodFrame) -> PeriodOutcome:
        """One period's regression: the fit row, one coefficient row per term and the funnel.

        Raises:
            ValueError: When the design is singular (a ``match_on`` column holds a single value).
        """
        cfg = self.config
        match_on = list(cfg.match_on)
        columns = [c for c in pf.frame.columns if fnmatch.fnmatch(c, cfg.embedding_value_pattern)]
        E = pf.frame[columns].to_numpy(dtype=float)
        cells = cell_statistics(E, pf.frame[match_on].to_numpy())
        beta, r2, n_pairs, mean_sim = ols_from_cells(cells, self.terms, match_on, period=pf.period)
        self.logger.info(
            f"{pf.period}: {len(E)} firms, {n_pairs} pairs, mean similarity {mean_sim:.4f}, R2 {r2:.4f}, "
            + ", ".join(f"{t} {b:+.4f}" for t, b in zip(self.terms, beta))
        )
        records: List[Record] = [
            PairwiseFit(period=pf.period, r2=r2, n_pairs=n_pairs, mean_sim=mean_sim, n_firms=len(E))
        ]
        records.extend(PairwiseCoef(period=pf.period, term=t, coef=float(b)) for t, b in zip(self.terms, beta))
        return PeriodOutcome(value=float(beta[1]), records=records + self.build_sample(pf))

    def summarize(self, outcomes: List[PeriodOutcome]) -> PeriodOutcome:
        """The Fama-MacBeth row of every term; the value is the first ``match_on`` term's mean coefficient."""
        coefs: Dict[str, List[float]] = {t: [] for t in self.terms}
        for outcome in outcomes:
            for record in outcome.records:
                if isinstance(record, PairwiseCoef):
                    coefs[record.term].append(record.coef)
        records = []
        for term in self.terms:
            fm = fama_macbeth(coefs[term])
            self.logger.info(f"FM {term}: {fm['mean']:+.4f} (se {fm['se']:.4f}, t {fm['t']:.2f}, p {fm['p']:.3g})")
            records.append(
                PairwiseResult(
                    term=term,
                    metric="fm_coef",
                    value=fm["mean"],
                    se=_or_none(fm["se"]),
                    t=_or_none(fm["t"]),
                    p=_or_none(fm["p"]),
                    n_periods=fm["n"],
                )
            )
        first = records[1].value if len(records) > 1 else float("nan")
        return PeriodOutcome(value=first, records=records)
