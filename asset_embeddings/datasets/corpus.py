"""Every portfolio of a set of portfolio CSV files, read once into compressed-row (CSR) arrays.

Portfolio ``i`` is ``codes[offsets[i]:offsets[i + 1]]``, indices into the sorted stock ``vocab``; the proportions,
when read, are aligned with ``codes``. Records keep the order of the files (``resolve_data_files``) and of the rows
in each file. A corpus can be saved as ``.npy`` arrays with a ``corpus.json`` manifest and opened memory-mapped, so
data larger than memory is paged by the operating system and DataLoader workers share the pages. The W2V
sentences and the RS holdings are views of it; the BERT samples are built by ``datasets.mlm``.
"""

import json
import logging
from collections import Counter
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Tuple, Union

import numpy as np

from .portfolio_csv import PathLike, read_portfolio_csv, resolve_data_files

logger = logging.getLogger(__name__)

FORMAT_VERSION = 1
MANIFEST = "corpus.json"
_ARRAYS = ("ids", "offsets", "codes", "vocab")
_PROPORTIONS = ("proportions1", "proportions2")


class Sentences:
    """The W2V view: one ``[CLS] code ... [SEP]`` token list per portfolio, iterable any number of times."""

    def __init__(self, corpus: "PortfolioCorpus"):
        self.corpus = corpus

    def __iter__(self) -> Iterator[List[str]]:
        """Yield the sentences in record order."""
        for i in range(len(self.corpus)):
            yield ["[CLS]", *self.corpus.portfolio(i), "[SEP]"]


class PortfolioCorpus:
    """Portfolios as CSR arrays.

    Args:
        ids: Investor id per portfolio (strings).
        offsets: ``n + 1`` boundaries into ``codes`` (int64).
        codes: Index into ``vocab`` per holding, all portfolios concatenated (int32).
        vocab: The stock codes, sorted.
        proportions1: Proportion of total shares per holding (float64), or None.
        proportions2: Proportion of tradable A-shares per holding (float64), or None.
        sources: The files the corpus was read from.
    """

    def __init__(
        self,
        ids: np.ndarray,
        offsets: np.ndarray,
        codes: np.ndarray,
        vocab: np.ndarray,
        proportions1: Optional[np.ndarray] = None,
        proportions2: Optional[np.ndarray] = None,
        sources: Sequence[str] = (),
    ):
        self.ids = ids
        self.offsets = offsets
        self.codes = codes
        self.vocab = vocab
        self.proportions1 = proportions1
        self.proportions2 = proportions2
        self.sources = list(sources)

    @classmethod
    def from_csv(
        cls, path: Union[PathLike, List[PathLike]], include_proportion: bool = False, **column_keys: str
    ) -> "PortfolioCorpus":
        """Read portfolio CSV files into memory.

        Args:
            path: A file, a directory or a list of them (``resolve_data_files``).
            include_proportion: Whether to read the two proportion columns.
            **column_keys: ``id_key``, ``portfolio_key``, ``proportion1_key``, ``proportion2_key`` of the files.

        Returns:
            The corpus.
        """
        keys = {"id_key": "InvestorID", "portfolio_key": "Portfolio"}
        keys.update({"proportion1_key": "Proportion1", "proportion2_key": "Proportion2"})
        keys.update(column_keys)
        files = resolve_data_files(path)
        ids: List[str] = []
        tokens: List[str] = []
        lengths: List[int] = []
        p1: List[float] = []
        p2: List[float] = []
        for f in files:
            df = read_portfolio_csv(f, include_proportion, **keys)
            ids.extend(df[keys["id_key"]])
            for portfolio in df[keys["portfolio_key"]]:
                tokens.extend(portfolio)
                lengths.append(len(portfolio))
            if include_proportion:
                for values in df[keys["proportion1_key"]]:
                    p1.extend(values)
                for values in df[keys["proportion2_key"]]:
                    p2.extend(values)
        vocab, codes = np.unique(np.array(tokens, dtype=str), return_inverse=True)
        offsets = np.zeros(len(lengths) + 1, dtype=np.int64)
        np.cumsum(lengths, out=offsets[1:])
        corpus = cls(
            ids=np.array(ids, dtype=str),
            offsets=offsets,
            codes=codes.astype(np.int32),
            vocab=vocab,
            proportions1=np.array(p1, dtype=np.float64) if include_proportion else None,
            proportions2=np.array(p2, dtype=np.float64) if include_proportion else None,
            sources=[str(f) for f in files],
        )
        logger.info(
            f"Corpus: {len(corpus):,} portfolios, {corpus.n_holdings:,} holdings, {len(vocab):,} stocks "
            f"from {len(files)} file(s)"
        )
        return corpus

    @classmethod
    def open(
        cls, path: Union[PathLike, List[PathLike]], include_proportion: bool = False, **column_keys: str
    ) -> "PortfolioCorpus":
        """Open a saved corpus directory memory-mapped, or read CSV files.

        Args:
            path: A directory holding ``corpus.json``, or anything ``from_csv`` reads.
            include_proportion: Whether the proportions are needed.
            **column_keys: Column names of the CSV files (unused for a saved corpus).

        Returns:
            The corpus.

        Raises:
            ValueError: Proportions are asked of a saved corpus built without them.
        """
        if not isinstance(path, list) and (Path(path) / MANIFEST).is_file():
            corpus = cls.load(path)
            if include_proportion and not corpus.has_proportions:
                raise ValueError(
                    f"Corpus {path} was built without proportions; rebuild it with --include_proportion or set "
                    f"include_proportion false"
                )
            return corpus
        return cls.from_csv(path, include_proportion, **column_keys)

    def __len__(self) -> int:
        return len(self.offsets) - 1

    @property
    def n_holdings(self) -> int:
        """Holdings over all portfolios."""
        return int(self.offsets[-1])

    @property
    def has_proportions(self) -> bool:
        """Whether the proportion arrays were read."""
        return self.proportions1 is not None

    def _span(self, i: int) -> slice:
        return slice(int(self.offsets[i]), int(self.offsets[i + 1]))

    def portfolio(self, i: int) -> List[str]:
        """The stock codes of portfolio ``i``, in file order."""
        return self.vocab[self.codes[self._span(i)]].tolist()

    def proportions(self, i: int) -> Tuple[np.ndarray, np.ndarray]:
        """The two proportion arrays of portfolio ``i``.

        Raises:
            ValueError: The corpus holds no proportions.
        """
        if not self.has_proportions:
            raise ValueError("This corpus was read without proportions")
        span = self._span(i)
        return self.proportions1[span], self.proportions2[span]

    def sentences(self) -> Sentences:
        """The W2V view: ``[CLS] + codes + [SEP]`` per portfolio, restartable."""
        return Sentences(self)

    def holdings(self, include_proportion: bool = False) -> Dict[str, Union[List[str], Tuple[List[str], List[float]]]]:
        """The RS view: the portfolios keyed by investor id, in record order.

        Args:
            include_proportion: Whether each value is ``(codes, proportions1)`` instead of the codes.

        Returns:
            ``{investor_id: codes}`` or ``{investor_id: (codes, proportions1)}``.

        Raises:
            ValueError: An investor id occurs more than once (two portfolios cannot share a row of the matrix).
        """
        ids = [str(i).strip() for i in self.ids]
        repeated = {k: n for k, n in Counter(ids).items() if n > 1}
        if repeated:
            sample = dict(list(repeated.items())[:5])
            raise ValueError(f"Investor ids occur more than once (id: count): {sample}")
        if include_proportion:
            return {k: (self.portfolio(i), self.proportions(i)[0].tolist()) for i, k in enumerate(ids)}
        return {k: self.portfolio(i) for i, k in enumerate(ids)}

    def save(self, directory: PathLike) -> Path:
        """Write the arrays as ``.npy`` files and the ``corpus.json`` manifest.

        Args:
            directory: Created if missing.

        Returns:
            The directory.
        """
        out = Path(directory)
        out.mkdir(parents=True, exist_ok=True)
        names = _ARRAYS + (_PROPORTIONS if self.has_proportions else ())
        for name in names:
            np.save(out / f"{name}.npy", np.asarray(getattr(self, name)))
        manifest = {
            "format_version": FORMAT_VERSION,
            "sources": self.sources,
            "n_portfolios": len(self),
            "n_holdings": self.n_holdings,
            "n_stocks": len(self.vocab),
            "include_proportion": self.has_proportions,
        }
        (out / MANIFEST).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        return out

    @classmethod
    def load(cls, directory: PathLike, mmap: bool = True) -> "PortfolioCorpus":
        """Open a saved corpus.

        Args:
            directory: Holds ``corpus.json`` and the arrays.
            mmap: Map ``offsets``, ``codes`` and the proportions from disk instead of reading them.

        Returns:
            The corpus.

        Raises:
            FileNotFoundError: The manifest or an array is missing.
            ValueError: The manifest's format version or counts disagree with the arrays.
        """
        d = Path(directory)
        manifest_path = d / MANIFEST
        if not manifest_path.is_file():
            raise FileNotFoundError(f"No {MANIFEST} in {d}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("format_version") != FORMAT_VERSION:
            raise ValueError(
                f"Corpus {d} has format version {manifest.get('format_version')}, expected {FORMAT_VERSION}"
            )
        names = _ARRAYS + (_PROPORTIONS if manifest["include_proportion"] else ())
        arrays = {}
        for name in names:
            path = d / f"{name}.npy"
            if not path.is_file():
                raise FileNotFoundError(f"Corpus {d} lacks {path.name}")
            mode = "r" if mmap and name not in ("ids", "vocab") else None
            arrays[name] = np.load(path, mmap_mode=mode)
        corpus = cls(sources=manifest["sources"], **arrays)
        checks = {
            "n_portfolios": (len(corpus), len(arrays["ids"])),
            "n_holdings": (corpus.n_holdings, len(arrays["codes"])),
            "n_stocks": (len(arrays["vocab"]),),
        }
        for key, values in checks.items():
            if any(v != manifest[key] for v in values):
                raise ValueError(f"Corpus {d}: {key} is {manifest[key]} in {MANIFEST} but {values} in the arrays")
        for name in _PROPORTIONS:
            if name in arrays and len(arrays[name]) != corpus.n_holdings:
                raise ValueError(f"Corpus {d}: {name} has {len(arrays[name])} values for {corpus.n_holdings} holdings")
        return corpus
