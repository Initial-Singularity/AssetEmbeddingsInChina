"""Portfolio data for the trainers: the CSV format, the corpus and the masked-LM samples.

``portfolio_csv`` reads and writes the portfolio files; [`PortfolioCorpus`][asset_embeddings.datasets.corpus.PortfolioCorpus]
holds them as CSR arrays (in memory or memory-mapped) with the W2V and RS views; [`MLMDataset`][asset_embeddings.datasets.mlm.MLMDataset]
and [`MaskingCollator`][asset_embeddings.datasets.mlm.MaskingCollator] build the AssetBERT and MP batches from it.
"""

from .corpus import PortfolioCorpus
from .mlm import MaskingCollator, MLMDataset
from .portfolio_csv import read_portfolio_csv, resolve_data_files, write_portfolio_csv

__all__ = [
    "MaskingCollator",
    "MLMDataset",
    "PortfolioCorpus",
    "read_portfolio_csv",
    "resolve_data_files",
    "write_portfolio_csv",
]
