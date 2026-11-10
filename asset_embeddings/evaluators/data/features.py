"""Which columns of a period's frame are the features, in the order of ``config.features``, and their scaling."""

import fnmatch
import logging
from typing import List, Sequence

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from asset_embeddings.configs import Config

logger = logging.getLogger(__name__)


class FeatureSpec:
    """The feature columns and their scaling for one evaluation config.

    Each entry of ``config.features`` contributes its block, in list order: ``"embedding"`` the columns matching
    ``embedding_value_pattern``, ``"placebo"`` the ``Placebo_*`` columns, ``"chars"`` the benchmark's
    characteristics, ``"industry"`` the ``Ind_*`` dummies.

    Args:
        config: The evaluation config.
        char_cols: The benchmark's characteristic columns.
    """

    def __init__(self, config: Config, char_cols: Sequence[str]):
        self.config = config
        self.char_cols = list(char_cols)

    def columns(self, frame: pd.DataFrame) -> List[str]:
        """The feature columns of ``frame``, in selection order."""
        cfg = self.config
        # The dummies are read off the frame: each period carries its own set.
        blocks = {
            "embedding": lambda: [c for c in frame.columns if fnmatch.fnmatch(c, cfg.embedding_value_pattern)],
            "placebo": lambda: [c for c in frame.columns if fnmatch.fnmatch(c, "Placebo_*")],
            "chars": lambda: list(self.char_cols),
            "industry": lambda: [c for c in frame.columns if c.startswith("Ind_")],
        }
        cols = [c for feature in cfg.features for c in blocks[feature]()]
        logger.debug(f"Feature columns ({len(cols)}): {cols[:5]}{'...' if len(cols) > 5 else ''}")
        return cols

    def matrix(self, frame: pd.DataFrame) -> np.ndarray:
        """The feature columns of ``frame`` scaled per ``config.normalization``.

        Raises:
            ValueError: On an unknown normalization name.
        """
        return self.normalize(frame[self.columns(frame)])

    def normalize(self, features: pd.DataFrame) -> np.ndarray:
        """Scale the columns: ``standard`` (StandardScaler), ``minmax`` (MinMaxScaler) or ``none``.

        Raises:
            ValueError: On an unknown normalization name.
        """
        mode = self.config.normalization
        if mode == "standard":
            scaler = StandardScaler()
        elif mode == "minmax":
            scaler = MinMaxScaler()
        elif mode == "none":
            return features.values
        else:
            raise ValueError(f"Unknown normalization strategy: {mode}")
        return scaler.fit_transform(features)
