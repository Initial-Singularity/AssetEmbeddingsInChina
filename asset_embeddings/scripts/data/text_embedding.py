import os
import sys
import json
import time
import hashlib
import logging
import argparse
import warnings
import re
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_random_exponential,
)
from tqdm import tqdm

from asset_embeddings.configs import LoggerConfig
from asset_embeddings.preparers import Logger_Preparer
from asset_embeddings.utils.log_utils import log_exceptions_inclass
from asset_embeddings import utils

warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", None)


# Provider registry: those whose `supported_native_dims()` set excludes any
# value in target_dims=[4..128]. Listed here so `validate_args` can fail-fast
# even before the corresponding client class is implemented (session-03).
NATIVE_LOWDIM_UNSUPPORTED_PROVIDERS: Set[str] = {"cohere", "voyage", "gemini"}


# -------------------- Name Loader --------------------
class NameLoader:
    """Load CSMAR `TRD_Co.csv` and emit canonical name schema.

    Output DataFrame columns (in order):
        Stkcd, name_zh, name_zh_short, name_en, name_source

    - `Stkcd` is normalized to a 6-digit zero-padded string.
    - `name_zh`        ← Conme       (full company Chinese name, e.g. "贵州茅台酒股份有限公司")
    - `name_zh_short`  ← Stknme      (short security name,        e.g. "贵州茅台")
    - `name_en`        ← Conme_en    (English name)
    - `name_source`    = "csmar"     (constant for this loader)

    The CSMAR `Listdt` column is *not* parsed — it contains a sentinel
    `'0000-00-00'` for some rows that would break date parsing, and the
    name pipeline does not need it.
    """

    def __init__(
        self,
        logger: logging.Logger,
        csmar_path: str,
        source_stkcd_column: str = "Stkcd",
        source_stknme_column: str = "Stknme",
        source_conme_column: str = "Conme",
        source_conme_en_column: str = "Conme_en",
    ):
        self.csmar_path: Path = Path(csmar_path)
        self.source_stkcd_column: str = source_stkcd_column
        self.source_stknme_column: str = source_stknme_column
        self.source_conme_column: str = source_conme_column
        self.source_conme_en_column: str = source_conme_en_column
        self.logger: logging.Logger = logger

    @log_exceptions_inclass("logger")
    def load(self) -> pd.DataFrame:
        if not self.csmar_path.is_file():
            raise FileNotFoundError(f"CSMAR file not found: {self.csmar_path}")

        self.logger.info(f"Loading CSMAR name table from {self.csmar_path}")

        df = pd.read_csv(
            self.csmar_path,
            dtype={self.source_stkcd_column: str},
            encoding="utf-8-sig",
        )

        required = [
            self.source_stkcd_column,
            self.source_stknme_column,
            self.source_conme_column,
            self.source_conme_en_column,
        ]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(
                f"CSMAR file {self.csmar_path} missing required columns: {missing}. "
                f"Found columns: {list(df.columns)}"
            )

        # Normalize Stkcd → 6-digit zero-padded string
        df[self.source_stkcd_column] = df[self.source_stkcd_column].astype(str).str.strip().str.zfill(6)

        out = pd.DataFrame(
            {
                "Stkcd": df[self.source_stkcd_column],
                "name_zh": df[self.source_conme_column],
                "name_zh_short": df[self.source_stknme_column],
                "name_en": df[self.source_conme_en_column],
            }
        )
        out["name_source"] = "csmar"

        # Stats
        n_rows = len(out)
        n_dup_stkcd = int(out["Stkcd"].duplicated().sum())
        n_zh_missing = int(out["name_zh"].isna().sum())
        n_zh_short_missing = int(out["name_zh_short"].isna().sum())
        n_en_missing = int(out["name_en"].isna().sum() + (out["name_en"] == "").sum())

        self.logger.info(
            f"Loaded {n_rows:,} stocks; "
            f"missing name_zh={n_zh_missing}, name_zh_short={n_zh_short_missing}, "
            f"name_en (NaN or empty)={n_en_missing}, duplicate Stkcd={n_dup_stkcd}"
        )
        if n_dup_stkcd > 0:
            self.logger.warning(
                f"Found {n_dup_stkcd} duplicate Stkcd in CSMAR; downstream cache " f"will deduplicate by Stkcd."
            )

        return out[["Stkcd", "name_zh", "name_zh_short", "name_en", "name_source"]]


# -------------------- Valid Stocks Loader --------------------
class ValidStocksLoader:
    """Load `valid_stocks.json` files written by `data_sharehold.py` per period.

    Layout expected:
        {valid_stocks_dir}/{period}/valid_stocks.json   # JSON list[str]

    Stand-alone JSON files directly under `valid_stocks_dir` (e.g. `summary.json`
    written by ShareholdingSaver) are ignored; only direct subdirectories are
    scanned.
    """

    def __init__(self, logger: logging.Logger, valid_stocks_dir: str, periods: Optional[List[str]] = None):
        self.valid_stocks_dir: Path = Path(valid_stocks_dir)
        self.periods: Optional[List[str]] = periods
        self.logger: logging.Logger = logger

    @log_exceptions_inclass("logger")
    def load(self) -> Dict[str, List[str]]:
        if not self.valid_stocks_dir.is_dir():
            raise FileNotFoundError(f"valid_stocks_dir does not exist or is not a directory: {self.valid_stocks_dir}")

        period_filter: Optional[Set[str]] = set(self.periods) if self.periods is not None else None

        result: Dict[str, List[str]] = {}
        for child in sorted(self.valid_stocks_dir.iterdir()):
            if not child.is_dir():
                continue
            period = child.name
            if period_filter is not None and period not in period_filter:
                continue
            stocks_file = child / "valid_stocks.json"
            if not stocks_file.is_file():
                self.logger.debug(f"Skipping {period}: no valid_stocks.json under {child}")
                continue

            with open(stocks_file, "r", encoding="utf-8") as f:
                codes = json.load(f)
            normalized = sorted({str(c).strip().zfill(6) for c in codes})
            result[period] = normalized
            self.logger.debug(f"{period}: {len(normalized):,} valid stocks")

        # Optional warning when --periods explicitly requested missing periods
        if period_filter is not None:
            missing_periods = period_filter - set(result.keys())
            if missing_periods:
                self.logger.warning(
                    f"Requested periods not found under {self.valid_stocks_dir}: " f"{sorted(missing_periods)}"
                )

        union_size = len({c for codes in result.values() for c in codes})
        self.logger.info(f"Loaded valid_stocks for {len(result)} period(s) " f"({union_size:,} unique stocks in union)")
        return result


# -------------------- Missing Stocks Reporter --------------------
def write_missing_stocks_report(
    logger: logging.Logger,
    valid_stocks_per_period: Dict[str, List[str]],
    csmar_codes: Set[str],
    output_path,
) -> int:
    """Write CSV of stocks present in `valid_stocks_per_period` but absent
    from `csmar_codes`.

    Output schema: ``Stkcd, periods`` where ``periods`` is a comma-separated
    list of period strings sorted ascending. Rows are sorted by ``Stkcd``.

    Returns the number of missing stocks (rows written).
    """
    missing: Dict[str, List[str]] = {}
    for period in sorted(valid_stocks_per_period.keys()):
        for code in valid_stocks_per_period[period]:
            if code in csmar_codes:
                continue
            missing.setdefault(code, []).append(period)

    rows = [{"Stkcd": code, "periods": ",".join(missing[code])} for code in sorted(missing.keys())]
    df = pd.DataFrame(rows, columns=["Stkcd", "periods"])

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, encoding="utf-8")

    logger.info(f"Missing stocks report: {len(df):,} Stkcd in valid_stocks but not in CSMAR " f"-> {output_path}")
    return len(df)


# -------------------- Embedding Cache --------------------
class EmbeddingCache:
    """Append-only JSONL cache mapping `key -> embedding`.

    Each line: ``{"key": str, "embedding": [float, ...]}``.

    Key derivation (see :meth:`make_key`)::

        sha1(f"{provider}|{model}|{lang}|{native_dim}|{text}").hexdigest()[:16]

    `native_dim` is the dimension actually requested in the API call (PCA
    path → `native_max_dim`; native path → `target_dim`). Same provider /
    model / lang at the same `native_dim` share a cache, regardless of which
    target_dim downstream consumes the result.
    """

    def __init__(self, logger: logging.Logger, cache_path: str):
        self.logger = logger
        self.cache_path = Path(cache_path)
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self._memory: Dict[str, np.ndarray] = {}
        if self.cache_path.is_file():
            with open(self.cache_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    obj = json.loads(line)
                    self._memory[obj["key"]] = np.asarray(obj["embedding"], dtype=np.float32)
            self.logger.info(f"EmbeddingCache: loaded {len(self._memory):,} entries from {self.cache_path}")
        else:
            self.logger.info(f"EmbeddingCache: starting fresh at {self.cache_path}")

    @staticmethod
    def make_key(provider: str, model: str, lang: str, native_dim: int, text: str) -> str:
        raw = f"{provider}|{model}|{lang}|{native_dim}|{text}"
        return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]

    def has(self, key: str) -> bool:
        return key in self._memory

    def get(self, key: str) -> Optional[np.ndarray]:
        return self._memory.get(key)

    def add(self, key: str, embedding) -> None:
        if key in self._memory:
            return
        vec = np.asarray(embedding, dtype=np.float32)
        self._memory[key] = vec
        with open(self.cache_path, "a", encoding="utf-8") as f:
            f.write(json.dumps({"key": key, "embedding": vec.tolist()}, ensure_ascii=False) + "\n")
            f.flush()
            os.fsync(f.fileno())

    def __len__(self) -> int:
        return len(self._memory)


# -------------------- Provider Adapter ABC --------------------
class BaseEmbeddingClient(ABC):
    """Abstract base for embedding-provider adapters.

    Subclasses must set:
        - ``name``: short provider id ("openai", "cohere", ...)
        - ``model``: full model id
        - ``native_max_dim``: native maximum embedding dimension
    and implement :meth:`supported_native_dims` and :meth:`embed`.

    Subclasses may set:
        - ``max_batch_size``: hard ceiling on texts per ``embed()`` call (the
          provider's per-request quota — e.g. Cohere v4 = 96, Voyage = 128,
          Gemini = 100). ``None`` (default) means no provider-side cap.
          ``EmbeddingProcessor`` clamps ``--batch-size`` against this.
        - ``min_request_interval``: minimum seconds between consecutive
          ``embed()`` calls to stay under provider RPM quotas (Gemini-style:
          100 requests/min regardless of batch size). ``None`` = no throttle.
        - ``max_inputs_per_minute``: cap on cumulative texts/min sent to the
          provider (Cohere-style: 2,000 inputs/min regardless of how many
          calls you split them across). ``EmbeddingProcessor`` derives an
          equivalent inter-call interval from this and ``batch_size``.
        ``EmbeddingProcessor`` honors the stricter of the two interval
        sources between every pair of consecutive ``embed()`` calls.
    """

    name: str
    model: str
    native_max_dim: int
    max_batch_size: Optional[int] = None
    min_request_interval: Optional[float] = None
    max_inputs_per_minute: Optional[int] = None

    @abstractmethod
    def supported_native_dims(self) -> Optional[Iterable[int]]:
        """The set of `dim` values the API natively accepts.

        - ``None``           → continuous: any d in [1, native_max_dim].
        - non-empty set      → discrete: only those dims.
        - empty ``set()``    → no native dim parameter (must use full size).
        """

    @abstractmethod
    def embed(self, texts: List[str], dim: Optional[int] = None) -> np.ndarray:
        """Embed `texts` and return shape ``(len(texts), out_dim)``.

        ``dim=None`` → return ``native_max_dim``.
        ``dim=N``    → return ``(len, N)``; if N is not supported natively,
                       implementations should raise ``ValueError``.
        """

    def supports_dim(self, dim: int) -> bool:
        sup = self.supported_native_dims()
        if sup is None:
            return 1 <= dim <= self.native_max_dim
        return dim in set(sup)


# -------------------- OpenAI Adapter --------------------
class OpenAIEmbeddingClient(BaseEmbeddingClient):
    """OpenAI embeddings: text-embedding-3-{small,large}.

    Both support the `dimensions` API parameter for any d ≤ native_max_dim,
    so :meth:`supported_native_dims` returns ``None``.
    """

    name = "openai"
    _NATIVE_MAX = {
        "text-embedding-3-small": 1536,
        "text-embedding-3-large": 3072,
    }

    def __init__(self, logger: logging.Logger, model: str, api_key: Optional[str] = None, max_retries: int = 6):
        # Lazy import so `--mode names` works without `openai` installed.
        import openai

        self._openai = openai
        self.logger = logger
        self.model = model
        if model in self._NATIVE_MAX:
            self.native_max_dim = self._NATIVE_MAX[model]
        else:
            self.native_max_dim = 1536
            logger.warning(
                f"OpenAIEmbeddingClient: unknown model '{model}', " f"assuming native_max_dim={self.native_max_dim}"
            )
        api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not provided (pass --api-key or set the env var)")
        self.client = openai.OpenAI(api_key=api_key)
        self.max_retries = max_retries
        self._retryable_excs = (
            openai.RateLimitError,
            openai.APITimeoutError,
            openai.APIConnectionError,
            openai.InternalServerError,
        )

        # Fail-fast on RateLimitError(code=insufficient_quota): account is out
        # of credit, retrying never recovers — would just burn 6 retries × 1-20s
        # of wait_random_exponential before giving up. Same logic as Gemini's
        # PerDay quotaId check (see _gemini_stop below).
        def _is_retryable(exc):
            if not isinstance(exc, self._retryable_excs):
                return False
            if isinstance(exc, openai.RateLimitError) and self._is_insufficient_quota(exc):
                return False
            return True

        self._retry_decorator = retry(
            wait=wait_random_exponential(min=1, max=20),
            stop=stop_after_attempt(self.max_retries),
            retry=retry_if_exception(_is_retryable),
            reraise=True,
        )

    @staticmethod
    def _is_insufficient_quota(exc) -> bool:
        """True iff a 429 RateLimitError carries error.code == 'insufficient_quota'.

        OpenAI uses the same HTTP 429 for two very different conditions:
        (a) RPM/TPM throttle — recoverable by waiting;
        (b) ``insufficient_quota`` — account out of credit, never recoverable
        until the user tops up at platform.openai.com. Distinguishing them
        avoids burning the retry budget on (b).
        """
        # 1. Structured: openai SDK exposes .body or .response.json()['error']
        body = getattr(exc, "body", None)
        if isinstance(body, dict):
            err = body.get("error") if isinstance(body.get("error"), dict) else body
            if isinstance(err, dict) and err.get("code") == "insufficient_quota":
                return True
        # 2. Fallback: scan stringified exception (covers older SDK shapes)
        return "insufficient_quota" in str(exc)

    def supported_native_dims(self) -> Optional[Iterable[int]]:
        return None

    def embed(self, texts: List[str], dim: Optional[int] = None) -> np.ndarray:
        openai = self._openai

        @self._retry_decorator
        def _call():
            kwargs: Dict[str, Any] = dict(input=texts, model=self.model)
            if dim is not None:
                kwargs["dimensions"] = dim
            resp = self.client.embeddings.create(**kwargs)
            return np.array([r.embedding for r in resp.data], dtype=np.float32)

        try:
            return _call()
        except openai.AuthenticationError as e:
            raise RuntimeError(f"OpenAI authentication failed (check API key). Detail: {e}") from e
        except openai.PermissionDeniedError as e:
            raise RuntimeError(f"OpenAI permission denied. Detail: {e}") from e
        except openai.RateLimitError as e:
            # _is_retryable returned False → either insufficient_quota or
            # max_retries exhausted on a real 429 burst. Distinguish for the
            # error message so the user knows whether to top up or wait.
            if self._is_insufficient_quota(e):
                raise RuntimeError(
                    f"OpenAI insufficient_quota — account is out of credit. "
                    f"Top up at https://platform.openai.com/ → Settings → "
                    f"Billing → Add to credit balance. Detail: {e}"
                ) from e
            raise


# -------------------- Cohere Adapter --------------------
class CohereEmbeddingClient(BaseEmbeddingClient):
    """Cohere embeddings (e.g. embed-v4.0) via `cohere>=5.15` ClientV2.

    Discrete `output_dimension` set: {256, 512, 1024, 1536}. None of the
    project's target_dims (4..128) are in this set, so the native transform
    path is always skipped — see `validate_args` fail-fast.
    """

    name = "cohere"
    # Cohere v2 embed endpoint hard-caps texts at 96/request (verified against
    # API error: "total number of texts must be at most 96").
    max_batch_size = 96
    # Cohere docs (https://docs.cohere.com/docs/rate-limits): both trial and
    # production keys are capped at 2,000 inputs/min for text embeddings.
    # 1,840 = 2,000 with 8% headroom (avoids edge cases when our send timing
    # drifts late in the minute window). With batch=96 → ~3.13s/call.
    max_inputs_per_minute = 1840
    # Only embed-v4.0 is in scope for this project. v3 models have a
    # different output_dimension policy (max 1024, no discrete subset);
    # adding them here would mis-advertise supported_native_dims().
    _NATIVE_MAX = {
        "embed-v4.0": 1536,
    }

    def __init__(
        self, logger: logging.Logger, model: str = "embed-v4.0", api_key: Optional[str] = None, max_retries: int = 6
    ):
        import cohere

        self._cohere = cohere
        self.logger = logger
        self.model = model
        self.native_max_dim = self._NATIVE_MAX.get(model, 1536)
        if model not in self._NATIVE_MAX:
            logger.warning(
                f"CohereEmbeddingClient: unknown model '{model}', "
                f"assuming native_max_dim={self.native_max_dim} and v4-style "
                f"discrete output_dimension set. Verify before relying on "
                f"supported_native_dims()."
            )
        api_key = api_key or os.getenv("COHERE_API_KEY")
        if not api_key:
            raise ValueError("COHERE_API_KEY not provided (pass --api-key or set the env var)")
        self.client = cohere.ClientV2(api_key)
        self.max_retries = max_retries
        self._retry_decorator = retry(
            wait=wait_random_exponential(min=1, max=20),
            stop=stop_after_attempt(self.max_retries),
            reraise=True,
        )

    def supported_native_dims(self) -> Optional[Iterable[int]]:
        return {256, 512, 1024, 1536}

    def embed(self, texts: List[str], dim: Optional[int] = None) -> np.ndarray:
        actual_dim = dim if dim is not None else self.native_max_dim
        if actual_dim not in self.supported_native_dims():
            raise ValueError(
                f"Cohere {self.model} only supports dims "
                f"{sorted(self.supported_native_dims())}; got dim={actual_dim}. "
                f"Use the PCA path for non-native dims."
            )

        @self._retry_decorator
        def _call():
            resp = self.client.embed(
                model=self.model,
                # Stock names are documents being indexed for downstream
                # similarity comparison, not user queries — match Voyage's
                # input_type="document".
                input_type="search_document",
                embedding_types=["float"],
                texts=texts,
                output_dimension=actual_dim,
            )
            # Pydantic alias: API field "float" → python attribute `float_`.
            return np.array(resp.embeddings.float_, dtype=np.float32)

        return _call()


# -------------------- Voyage Adapter --------------------
class VoyageEmbeddingClient(BaseEmbeddingClient):
    """Voyage AI embeddings via `voyageai>=0.3`.

    Implementation note: voyageai 0.3.x does not expose the supported-dims
    set programmatically. Hard-coded from current public docs (verified
    against SDK signature: `embed(..., output_dimension: Optional[int])`).
    Update if Voyage publishes new dim values.
    """

    name = "voyage"
    # Voyage embeddings API caps at 128 texts/request per public docs.
    max_batch_size = 128

    SUPPORTED_DIMS_BY_MODEL = {
        "voyage-3-large": {256, 512, 1024, 2048},
        "voyage-3": {256, 512, 1024, 2048},
        "voyage-3.5": {256, 512, 1024, 2048},
    }
    _NATIVE_MAX = {
        "voyage-3-large": 2048,
        "voyage-3": 1024,
        "voyage-3.5": 1024,
    }

    # Voyage unpaid (no payment method on file) gives 3 RPM + 10K TPM. After the
    # SDK's first RateLimitError mentioning "have not yet added your payment
    # method", we set min_request_interval to this value so subsequent batches
    # in the run pace themselves at <3 RPM.
    _UNPAID_RPM_INTERVAL = 21.0

    def __init__(
        self, logger: logging.Logger, model: str = "voyage-3-large", api_key: Optional[str] = None, max_retries: int = 6
    ):
        import voyageai

        self._voyageai = voyageai
        self.logger = logger
        self.model = model
        self.native_max_dim = self._NATIVE_MAX.get(model, 2048 if "large" in model else 1024)
        if model not in self._NATIVE_MAX:
            logger.warning(
                f"VoyageEmbeddingClient: unknown model '{model}', " f"assuming native_max_dim={self.native_max_dim}"
            )
        api_key = api_key or os.getenv("VOYAGE_API_KEY")
        if not api_key:
            raise ValueError("VOYAGE_API_KEY not provided (pass --api-key or set the env var)")
        self.client = voyageai.Client(api_key=api_key)
        self.max_retries = max_retries
        self._unpaid_warned = False  # one-time warning guard

        # Custom wait: on a Voyage RateLimitError that mentions the unpaid-tier
        # marker, mutate self.min_request_interval (read fresh by the
        # EmbeddingProcessor before each batch) and sleep enough for the
        # current retry to clear the 3-RPM window. Other errors fall back to
        # exponential backoff.
        _exp = wait_random_exponential(min=1, max=20)

        def _voyage_wait(rcs):
            outcome = rcs.outcome
            exc = outcome.exception() if outcome and outcome.failed else None
            if isinstance(exc, voyageai.error.RateLimitError) and "have not yet added your payment method" in str(exc):
                if not self._unpaid_warned:
                    self.logger.warning(
                        f"Voyage unpaid tier detected (3 RPM, 10K TPM). "
                        f"Engaging {self._UNPAID_RPM_INTERVAL:.0f}s/call throttle "
                        f"for the rest of this run. Add a payment method at "
                        f"https://dashboard.voyageai.com/ to lift to 2,000 RPM "
                        f"(no charge until you exceed the free 200M-token quota)."
                    )
                    self._unpaid_warned = True
                self.min_request_interval = self._UNPAID_RPM_INTERVAL
                return self._UNPAID_RPM_INTERVAL + 1.0
            return _exp(rcs)

        self._retry_decorator = retry(
            wait=_voyage_wait,
            stop=stop_after_attempt(self.max_retries),
            reraise=True,
        )

    def supported_native_dims(self) -> Optional[Iterable[int]]:
        return self.SUPPORTED_DIMS_BY_MODEL.get(self.model, {256, 512, 1024, 2048})

    def embed(self, texts: List[str], dim: Optional[int] = None) -> np.ndarray:
        actual_dim = dim if dim is not None else self.native_max_dim
        if actual_dim not in self.supported_native_dims():
            raise ValueError(
                f"Voyage {self.model} supports dims " f"{sorted(self.supported_native_dims())}; got dim={actual_dim}."
            )

        @self._retry_decorator
        def _call():
            resp = self.client.embed(
                texts=texts,
                model=self.model,
                input_type="document",
                output_dimension=actual_dim,
            )
            return np.array(resp.embeddings, dtype=np.float32)

        return _call()


# -------------------- Gemini Adapter --------------------
class GeminiEmbeddingClient(BaseEmbeddingClient):
    """Google Gemini embeddings via `google-genai>=1.0`.

    `embed_content(model=..., contents=..., config=EmbedContentConfig(
        output_dimensionality=...))` returns a response whose `.embeddings`
    is a list of `ContentEmbedding(values=[float, ...])`.

    SDK signature verified against google-genai 1.75.0 (2026-05); update if
    upstream renames `output_dimensionality` or the response shape.
    """

    name = "gemini"
    # google-genai batch_embed caps at 100 contents per request.
    max_batch_size = 100
    # Free tier: 100 RPM for gemini-embedding-001 (verified via 429
    # RESOURCE_EXHAUSTED with metric EmbedContentRequestsPerMinutePerProjectPerModel).
    # 0.65s/req ≈ 92/min, leaving headroom under the 100 RPM cap. Paid-tier
    # users on Tier 1+ (1k+ RPM) can override this attribute to None.
    min_request_interval = 0.65

    # gemini-embedding-001: docs list discrete supported dims.
    # text-embedding-004: only native 768 (no output_dimensionality control).
    SUPPORTED_DIMS_BY_MODEL = {
        "gemini-embedding-001": {768, 1536, 3072},
        "text-embedding-004": {768},
    }
    _NATIVE_MAX = {
        "gemini-embedding-001": 3072,
        "text-embedding-004": 768,
    }

    def __init__(
        self,
        logger: logging.Logger,
        model: str = "gemini-embedding-001",
        api_key: Optional[str] = None,
        max_retries: int = 6,
    ):
        from google import genai
        from google.genai import errors as gerrors
        from google.genai import types as gtypes

        self._gtypes = gtypes
        self._gerrors = gerrors
        self.logger = logger
        self.model = model
        self.native_max_dim = self._NATIVE_MAX.get(model, 3072)
        if model not in self._NATIVE_MAX:
            logger.warning(
                f"GeminiEmbeddingClient: unknown model '{model}', " f"assuming native_max_dim={self.native_max_dim}"
            )
        api_key = api_key or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not provided (pass --api-key or set the env var)")
        self.client = genai.Client(api_key=api_key)
        self.max_retries = max_retries

        # Custom wait: on 429 RESOURCE_EXHAUSTED, parse the API-specified
        # `retryDelay` and sleep that long (+1s buffer). Other failures fall
        # back to exponential backoff. Critical because Gemini free-tier 429s
        # carry retryDelays of 30-60s, which exceed wait_random_exponential's
        # default max=20s — without honoring retryDelay every retry would hit
        # the quota wall again immediately.
        _exp = wait_random_exponential(min=1, max=20)

        def _gemini_wait(rcs):
            outcome = rcs.outcome
            exc = outcome.exception() if outcome and outcome.failed else None
            if isinstance(exc, gerrors.ClientError) and getattr(exc, "code", None) == 429:
                delay = self._parse_retry_delay(exc)
                if delay is not None:
                    self.logger.warning(
                        f"Gemini 429 RESOURCE_EXHAUSTED: API requested retry "
                        f"after {delay:.1f}s; sleeping (+1s buffer)."
                    )
                    return delay + 1.0
                self.logger.warning(
                    "Gemini 429 RESOURCE_EXHAUSTED: no retryDelay in response; " "falling back to 60s wait."
                )
                return 60.0
            return _exp(rcs)

        # Custom stop: certain 429s are non-recoverable by waiting and must
        # break the retry loop immediately. Two known cases:
        #   1. PerDay quotaId — free-tier daily cap, resets at midnight Pacific.
        #      Cache is append-only, so re-running tomorrow continues where
        #      we stopped.
        #   2. ``"prepayment credits are depleted"`` — paid tier with prepay
        #      mode and zero balance. Same pattern as OpenAI's
        #      ``insufficient_quota``. Top up or switch to pay-as-you-go.
        # Both would otherwise burn ~6 × (retryDelay or 60s) of useless wait.
        def _gemini_stop(rcs):
            if rcs.attempt_number >= self.max_retries:
                return True
            outcome = rcs.outcome
            exc = outcome.exception() if outcome and outcome.failed else None
            if isinstance(exc, gerrors.ClientError) and getattr(exc, "code", None) == 429:
                quota_id = self._parse_quota_id(exc) or ""
                if "PerDay" in quota_id:
                    self.logger.error(
                        f"Gemini DAILY quota exhausted ({quota_id}). Free tier "
                        f"is 1,000 requests/day for gemini-embedding-001 and "
                        f"resets at midnight Pacific. Retrying now is futile; "
                        f"aborting this batch. Cache is preserved (append-only "
                        f"JSONL), so re-running after the reset will continue "
                        f"where it stopped. To remove the daily cap, upgrade "
                        f"the API project to Tier 1+ in Google AI Studio."
                    )
                    return True
                if self._is_prepayment_depleted(exc):
                    self.logger.error(
                        "Gemini prepayment credits depleted. Top up at "
                        "https://ai.studio/projects → Billing, or switch to "
                        "pay-as-you-go (no minimum prepay required). "
                        "Retrying is futile; aborting this batch. Cache is "
                        "preserved (append-only JSONL)."
                    )
                    return True
            return False

        self._retry_decorator = retry(
            wait=_gemini_wait,
            stop=_gemini_stop,
            reraise=True,
        )

    @staticmethod
    def _parse_retry_delay(exc) -> Optional[float]:
        """Best-effort extraction of RetryInfo.retryDelay (seconds) from a Gemini 429.

        ``ClientError(status_code, response_json, response)`` stores the
        decoded body in ``args[1]``. When that fails (older SDK or different
        shape), fall back to regex-scanning the stringified exception for
        ``'retryDelay': '42s'``.
        """
        # 1. Structured access via args[1] = response_json
        args = getattr(exc, "args", ())
        if len(args) >= 2 and isinstance(args[1], dict):
            error = args[1].get("error", {}) or {}
            for d in error.get("details", []) or []:
                if isinstance(d, dict) and d.get("@type", "").endswith("RetryInfo"):
                    raw = d.get("retryDelay", "")
                    if isinstance(raw, str) and raw.endswith("s"):
                        try:
                            return float(raw[:-1])
                        except ValueError:
                            pass
        # 2. Fallback: regex on the stringified exception
        m = re.search(r"['\"]retryDelay['\"]\s*:\s*['\"]([\d.]+)s['\"]", str(exc))
        if m:
            try:
                return float(m.group(1))
            except ValueError:
                pass
        return None

    @staticmethod
    def _parse_quota_id(exc) -> Optional[str]:
        """Extract QuotaFailure.violations[0].quotaId from a Gemini 429.

        Per-minute and per-day quotas are surfaced via different quotaIds
        (e.g. ``EmbedContentRequestsPerMinutePerProjectPerModel-FreeTier``
        vs ``EmbedContentRequestsPerDayPerProjectPerModel-FreeTier``) which
        determines whether retrying is even possible.
        """
        # 1. Structured: args[1].error.details[i] with @type ending in QuotaFailure
        args = getattr(exc, "args", ())
        if len(args) >= 2 and isinstance(args[1], dict):
            error = args[1].get("error", {}) or {}
            for d in error.get("details", []) or []:
                if isinstance(d, dict) and d.get("@type", "").endswith("QuotaFailure"):
                    for v in d.get("violations", []) or []:
                        qid = v.get("quotaId") if isinstance(v, dict) else None
                        if isinstance(qid, str):
                            return qid
        # 2. Fallback: regex on the stringified exception
        m = re.search(r"['\"]quotaId['\"]\s*:\s*['\"]([^'\"]+)['\"]", str(exc))
        if m:
            return m.group(1)
        return None

    @staticmethod
    def _is_prepayment_depleted(exc) -> bool:
        """True iff a 429 carries error.message saying prepayment is exhausted.

        Surfaces as ``RESOURCE_EXHAUSTED`` with a message like
        ``"Your prepayment credits are depleted. Please go to AI Studio..."``.
        Distinct from PerDay quotaId — this happens on PAID projects in
        prepay mode, not free-tier daily caps. Switching to pay-as-you-go or
        topping up the prepayment unblocks; retrying alone never does.
        """
        msg = ""
        # 1. Structured: args[1].error.message
        args = getattr(exc, "args", ())
        if len(args) >= 2 and isinstance(args[1], dict):
            error = args[1].get("error", {}) or {}
            msg = error.get("message", "") or ""
        # 2. Fallback: stringified exception
        if not msg:
            msg = str(exc)
        msg = msg.lower()
        return "prepayment" in msg and ("deplete" in msg or "exhaust" in msg)

    def supported_native_dims(self) -> Optional[Iterable[int]]:
        return self.SUPPORTED_DIMS_BY_MODEL.get(self.model, {self.native_max_dim})

    def embed(self, texts: List[str], dim: Optional[int] = None) -> np.ndarray:
        actual_dim = dim if dim is not None else self.native_max_dim
        if actual_dim not in self.supported_native_dims():
            raise ValueError(
                f"Gemini {self.model} supports dims " f"{sorted(self.supported_native_dims())}; got dim={actual_dim}."
            )

        @self._retry_decorator
        def _call():
            cfg = self._gtypes.EmbedContentConfig(output_dimensionality=actual_dim)
            resp = self.client.models.embed_content(
                model=self.model,
                contents=texts,
                config=cfg,
            )
            arr = np.array([e.values for e in resp.embeddings], dtype=np.float32)
            # Gemini does NOT auto-renormalize when output_dimensionality <
            # native_max_dim (https://ai.google.dev/gemini-api/docs/embeddings).
            # Renormalize to keep the same row-norm contract the other 4
            # hosted/local adapters satisfy. No-op when already unit-norm.
            norms = np.linalg.norm(arr, axis=1, keepdims=True)
            return arr / np.clip(norms, a_min=1e-12, a_max=None)

        return _call()


# -------------------- Local HF Adapter --------------------
class LocalHFEmbeddingClient(BaseEmbeddingClient):
    """Local HuggingFace embedding models (Qwen3-Embedding, BGE-m3).

    Both families are Matryoshka models, so `supported_native_dims()` returns
    `None` (any d ≤ native_max_dim is valid via truncation + L2 renorm).

    Pooling depends on the model family:
        - Qwen3-Embedding: last non-padding token's hidden state.
        - BGE-m3:          [CLS] (first token) hidden state.
    """

    name = "local"

    MODEL_PROFILES: Dict[str, Dict[str, Any]] = {
        "Qwen/Qwen3-Embedding-0.6B": dict(native_max_dim=1024, pooling="last-token", approx_params=0.6e9),
        "Qwen/Qwen3-Embedding-4B": dict(native_max_dim=2560, pooling="last-token", approx_params=4.0e9),
        "Qwen/Qwen3-Embedding-8B": dict(native_max_dim=4096, pooling="last-token", approx_params=8.0e9),
        "BAAI/bge-m3": dict(native_max_dim=1024, pooling="cls", approx_params=0.57e9),
    }

    DTYPE_BYTES = {"float32": 4, "fp32": 4, "float16": 2, "fp16": 2, "bfloat16": 2, "bf16": 2}

    def __init__(
        self,
        logger: logging.Logger,
        model: str,
        device: str = "cuda:0",
        hf_dtype: str = "bfloat16",
        batch_size_internal: int = 32,
    ):
        if model not in self.MODEL_PROFILES:
            raise ValueError(f"Unsupported local model '{model}'. " f"Supported: {list(self.MODEL_PROFILES.keys())}")
        prof = self.MODEL_PROFILES[model]

        self.logger = logger
        self.model = model
        self.native_max_dim = prof["native_max_dim"]
        self.pooling = prof["pooling"]
        self.device = device
        self.batch_size_internal = batch_size_internal

        import torch
        from transformers import AutoModel, AutoTokenizer

        self._torch = torch

        # VRAM precheck (design.md §4.3): warn but don't abort.
        if device.startswith("cuda"):
            dtype_bytes = self.DTYPE_BYTES.get(hf_dtype)
            if dtype_bytes is None:
                logger.info(f"VRAM check skipped: hf_dtype={hf_dtype} not in size table")
            else:
                required_gb = prof["approx_params"] * dtype_bytes / 1e9 * 1.3
                try:
                    idx = int(device.split(":")[1]) if ":" in device else 0
                    available_gb = torch.cuda.get_device_properties(idx).total_memory / 1e9
                    if required_gb > available_gb * 0.9:
                        logger.warning(
                            f"LocalHFEmbeddingClient: estimated VRAM need "
                            f"{required_gb:.1f} GB exceeds 90% of {device} "
                            f"({available_gb:.1f} GB). Likely OOM. Consider "
                            f"--hf-dtype int8/int4 or a smaller model."
                        )
                    else:
                        logger.info(
                            f"VRAM check ok: need ~{required_gb:.1f} GB, " f"have {available_gb:.1f} GB on {device}"
                        )
                except (RuntimeError, IndexError, AssertionError) as e:
                    logger.warning(f"VRAM check skipped (CUDA query failed): {e}")

        torch_dtype_map = {
            "float32": torch.float32,
            "fp32": torch.float32,
            "float16": torch.float16,
            "fp16": torch.float16,
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
        }
        if hf_dtype not in torch_dtype_map:
            # int8/int4 are listed in the CLI choices as escape hatches in the
            # VRAM-warning message, but real quantization would need
            # bitsandbytes (or similar) wired into from_pretrained. Fail loudly
            # rather than silently downgrade to bf16.
            raise NotImplementedError(
                f"hf_dtype={hf_dtype!r} is not yet implemented in "
                f"LocalHFEmbeddingClient. Use one of "
                f"{sorted(torch_dtype_map.keys())} for now."
            )
        torch_dtype = torch_dtype_map[hf_dtype]

        logger.info(f"Loading {model} on {device} (dtype={hf_dtype})")
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.hf_model = AutoModel.from_pretrained(model, torch_dtype=torch_dtype).to(device).eval()

    def supported_native_dims(self) -> Optional[Iterable[int]]:
        # Matryoshka: any d in [1, native_max_dim].
        return None

    def embed(self, texts: List[str], dim: Optional[int] = None) -> np.ndarray:
        torch = self._torch
        actual_dim = dim if dim is not None else self.native_max_dim
        if actual_dim > self.native_max_dim:
            raise ValueError(f"dim={actual_dim} > native_max_dim={self.native_max_dim} for {self.model}")
        if actual_dim < 1:
            raise ValueError(f"dim={actual_dim} must be >= 1")

        outs: List[np.ndarray] = []
        with torch.no_grad():
            for i in range(0, len(texts), self.batch_size_internal):
                batch = texts[i : i + self.batch_size_internal]
                enc = self.tokenizer(batch, padding=True, truncation=True, return_tensors="pt").to(self.device)
                hidden = self.hf_model(**enc).last_hidden_state  # (B, L, H)

                if self.pooling == "cls":
                    vec = hidden[:, 0, :]
                elif self.pooling == "last-token":
                    lengths = enc["attention_mask"].sum(dim=1) - 1  # (B,)
                    vec = hidden[torch.arange(hidden.size(0), device=hidden.device), lengths, :]
                elif self.pooling == "mean":
                    mask = enc["attention_mask"].unsqueeze(-1).to(hidden.dtype)
                    vec = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
                else:
                    raise ValueError(f"Unknown pooling: {self.pooling}")

                # Matryoshka truncation + L2 renormalization. Cast to fp32
                # before normalize so the result is unit-norm in fp32 (bf16
                # normalize drifts ~1e-3 per row).
                vec = vec[:, :actual_dim].float()
                vec = torch.nn.functional.normalize(vec, p=2, dim=1)
                outs.append(vec.cpu().numpy())

        return np.concatenate(outs, axis=0).astype(np.float32)


# -------------------- Embedding Processor --------------------
class EmbeddingProcessor:
    """Batched embedding driver: cache → API → cache.

    A single `process()` call corresponds to one `native_dim` (PCA path uses
    `native_dim=None` to get `native_max_dim`; native path passes the
    `target_dim`).
    """

    def __init__(
        self,
        logger: logging.Logger,
        client: BaseEmbeddingClient,
        cache: EmbeddingCache,
        lang: str,
        native_dim: Optional[int],
        batch_size: int = 100,
    ):
        self.logger = logger
        self.client = client
        self.cache = cache
        self.lang = lang
        self.native_dim = native_dim
        cap = getattr(client, "max_batch_size", None)
        if cap is not None and batch_size > cap:
            logger.warning(f"{client.name} caps texts/request at {cap}; clamping " f"batch_size {batch_size} -> {cap}.")
            batch_size = cap
        self.batch_size = batch_size
        self.effective_dim = native_dim if native_dim is not None else client.native_max_dim
        self.hits = 0
        self.misses = 0

    @log_exceptions_inclass("logger")
    def process(self, df: pd.DataFrame, text_col: str) -> np.ndarray:
        out_dim = self.effective_dim
        result = np.zeros((len(df), out_dim), dtype=np.float32)

        keys: List[str] = []
        miss_indices: List[int] = []
        miss_texts: List[str] = []
        for i, raw_text in enumerate(df[text_col].tolist()):
            text = "" if pd.isna(raw_text) else str(raw_text)
            key = EmbeddingCache.make_key(self.client.name, self.client.model, self.lang, out_dim, text)
            keys.append(key)
            cached = self.cache.get(key)
            if cached is not None:
                if cached.shape[0] != out_dim:
                    raise ValueError(
                        f"Cached embedding for key {key} has dim {cached.shape[0]} "
                        f"but expected {out_dim}; cache file may be stale."
                    )
                result[i] = cached
                self.hits += 1
            else:
                miss_indices.append(i)
                miss_texts.append(text)

        desc = f"Embedding {self.client.name}/{self.client.model}/{self.lang}/d{out_dim}"
        last_call_at: Optional[float] = None
        last_logged_interval: Optional[float] = None
        for batch_start in tqdm(
            range(0, len(miss_texts), self.batch_size),
            desc=desc,
            unit="batch",
            disable=len(miss_texts) == 0,
        ):
            batch_idx = miss_indices[batch_start : batch_start + self.batch_size]
            batch_text = miss_texts[batch_start : batch_start + self.batch_size]
            # Re-read throttle attributes each iteration so adapters can
            # surface tighter limits learned mid-run from API errors (e.g.
            # Voyage's unpaid-tier RateLimitError → bumps min_request_interval).
            rpm_interval = getattr(self.client, "min_request_interval", None) or 0.0
            inputs_cap = getattr(self.client, "max_inputs_per_minute", None)
            inputs_interval = self.batch_size * 60.0 / inputs_cap if inputs_cap else 0.0
            interval = max(rpm_interval, inputs_interval) or None
            # Log only when the effective interval changes (avoids spam).
            if interval and interval >= 0.5 and interval != last_logged_interval:
                self.logger.info(
                    f"Throttle: sleeping >={interval:.2f}s between calls "
                    f"(min_request_interval={rpm_interval or 0:.2f}s, "
                    f"inputs/min cap-derived={inputs_interval:.2f}s "
                    f"@ batch={self.batch_size})."
                )
                last_logged_interval = interval
            if interval and last_call_at is not None:
                wait = interval - (time.monotonic() - last_call_at)
                if wait > 0:
                    time.sleep(wait)
            batch_emb = self.client.embed(batch_text, dim=self.native_dim)
            last_call_at = time.monotonic()
            if batch_emb.shape != (len(batch_text), out_dim):
                raise ValueError(
                    f"client.embed returned shape {batch_emb.shape}, " f"expected {(len(batch_text), out_dim)}"
                )
            for j, idx in enumerate(batch_idx):
                vec = batch_emb[j].astype(np.float32)
                result[idx] = vec
                self.cache.add(keys[idx], vec)
                self.misses += 1

        self.logger.info(
            f"EmbeddingProcessor done ({self.client.name}/{self.client.model}/"
            f"{self.lang}/d{out_dim}): total={len(df)}, "
            f"hits={self.hits}, misses={self.misses}"
        )
        return result


# -------------------- PCA Reducer --------------------
class PcaReducer:
    """Per-target-dim independent PCA fits.

    For each ``d`` in ``target_dims`` runs ``PCA(n_components=d).fit_transform(X)``
    as its own call. Same ``random_state`` is used for every call so a given
    ``(X, d)`` is reproducible — but **different ``d`` values do NOT nest**
    (under sklearn's default randomized SVD, the d=4 estimate is not equal to
    d=8's first 4 columns, even with the same seed).

    Drops nesting deliberately: the user wanted each target_dim to be its own
    independent estimate of top-d principal components, so caller code must not
    rely on ``out[8][:, :4] == out[4]``. If that coincidence is needed, switch
    PCA's solver to 'full' (deterministic, exact nesting).

    Skips any ``d > min(n_samples, n_features)`` with a warning rather than
    raising — useful for the per-period flavor where small periods may not
    support the largest target_dim.
    """

    def __init__(self, logger: logging.Logger, target_dims: List[int], random_state: int = 42):
        self.logger = logger
        self.target_dims = sorted(set(target_dims))
        self.random_state = random_state

    @log_exceptions_inclass("logger")
    def fit_transform_all(self, X: np.ndarray) -> Dict[int, np.ndarray]:
        n_samples, n_features = X.shape
        result: Dict[int, np.ndarray] = {}
        for d in self.target_dims:
            max_supported = min(n_samples, n_features)
            if d > max_supported:
                self.logger.warning(
                    f"PCA: skipping d={d} (n_components > min(n_samples={n_samples}, "
                    f"n_features={n_features})={max_supported})"
                )
                continue
            pca = PCA(n_components=d, random_state=self.random_state)
            transformed = pca.fit_transform(X)
            cum = pca.explained_variance_ratio_.sum()
            self.logger.debug(f"PCA fit (d={d}) on {X.shape}: cum-explained-var={cum:.4f}")
            result[d] = transformed.astype(np.float32, copy=False)
        if not result:
            raise ValueError(f"PcaReducer: every target_dim was skipped for input shape {X.shape}")
        return result


# -------------------- Per-Period Slicer --------------------
class PerPeriodSlicer:
    """Slice a full (n_stocks, d) matrix into per-period DataFrames.

    Precomputes the Token→row-position mapping once at construction so that
    multiple `slice()` calls (one per (transform, target_dim)) reuse the same
    indices instead of rebuilding them from `stock_names`.

    Stocks in a period universe but missing from `stock_names` (already
    reported by the names stage) are silently dropped.
    """

    def __init__(self, logger: logging.Logger, stock_names: pd.DataFrame, universe_per_period: Dict[str, List[str]]):
        self.logger = logger
        # First-occurrence position per Token (defensive against duplicate Stkcd).
        token_to_pos: Dict[str, int] = {}
        for i, tok in enumerate(stock_names["Stkcd"].tolist()):
            if tok not in token_to_pos:
                token_to_pos[tok] = i
        self.period_positions: Dict[str, np.ndarray] = {}
        self.period_tokens: Dict[str, List[str]] = {}
        for period, codes in universe_per_period.items():
            positions = [token_to_pos[c] for c in codes if c in token_to_pos]
            if not positions:
                self.logger.warning(f"Period {period}: 0 stocks intersect with stock_names; skipping")
                continue
            self.period_positions[period] = np.asarray(positions, dtype=np.int64)
            self.period_tokens[period] = [c for c in codes if c in token_to_pos]

    @log_exceptions_inclass("logger")
    def slice(self, embeddings: np.ndarray) -> Dict[str, pd.DataFrame]:
        d = embeddings.shape[1]
        emb_cols = [f"Emb_{i}" for i in range(d)]
        result: Dict[str, pd.DataFrame] = {}
        for period, positions in self.period_positions.items():
            sub_mat = embeddings[positions]
            df = pd.DataFrame(sub_mat, columns=emb_cols)
            df.insert(0, "Token", self.period_tokens[period])
            result[period] = df
            self.logger.debug(f"Period {period}: sliced {len(df):,} stocks")
        return result


# -------------------- Embedding Saver --------------------
class EmbeddingSaver:
    """Write per-period embedding CSVs and a single config.json snapshot."""

    def __init__(self, logger: logging.Logger, subdir: Path, run_config: dict):
        self.logger = logger
        self.subdir = Path(subdir)
        self.subdir.mkdir(parents=True, exist_ok=True)
        self.run_config = run_config

    @log_exceptions_inclass("logger")
    def save(self, sliced: Dict[str, pd.DataFrame]) -> None:
        with open(self.subdir / "config.json", "w", encoding="utf-8") as f:
            json.dump(self.run_config, f, indent=2, ensure_ascii=False)

        for period, df in tqdm(
            sliced.items(),
            desc=f"Saving {self.subdir.name}",
            unit="period",
            disable=len(sliced) == 0,
        ):
            period_dir = self.subdir / period
            period_dir.mkdir(parents=True, exist_ok=True)
            df.to_csv(period_dir / "embedding.csv", index=False, encoding="utf-8")

        self.logger.info(f"EmbeddingSaver: wrote {len(sliced)} period(s) to {self.subdir}")


# -------------------- CLI guard --------------------
def validate_args(args, logger: logging.Logger) -> None:
    """Start-up validation. Catches combinations that would silently produce no
    output (e.g. Cohere with `--transforms native` only, since none of
    target_dims=[4..128] are in Cohere's discrete native dim set)."""
    transforms = list(getattr(args, "transforms", []))
    has_pca = any(t.startswith("pca") for t in transforms)
    if getattr(args, "provider", None) in NATIVE_LOWDIM_UNSUPPORTED_PROVIDERS and not has_pca:
        raise ValueError(
            f"Provider '{args.provider}' does not support any of "
            f"target_dims={list(args.target_dims)} natively (its supported "
            f"native dims start at 256+). Running --transforms {transforms} "
            f"alone would produce zero outputs. Add --transforms pca-universal "
            f"or --transforms pca-perperiod (or both)."
        )
    logger.debug(
        f"validate_args OK: provider={getattr(args, 'provider', None)}, "
        f"transforms={list(getattr(args, 'transforms', []))}, "
        f"target_dims={list(getattr(args, 'target_dims', []))}"
    )


# -------------------- Helpers --------------------
def _file_md5(path: str, chunk_size: int = 1 << 20) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def _redact_api_key(value: Optional[str]) -> Optional[str]:
    if not value:
        return value
    # Echo only first 8 chars + ellipsis (safety §7).
    return value[:8] + "..."


def _safe_args(args) -> Dict[str, Any]:
    """Render argparse Namespace as JSON-serializable dict with API key redacted."""
    out: Dict[str, Any] = {}
    for k, v in vars(args).items():
        if k == "api_key":
            out[k] = _redact_api_key(v)
        elif isinstance(v, Path):
            out[k] = str(v)
        else:
            out[k] = v
    return out


_LANG_TEXT_COL = {"zh": "name_zh", "zh-short": "name_zh_short", "en": "name_en"}


# -------------------- Pipeline orchestration --------------------
def run_pipeline(
    args,
    logger: logging.Logger,
    client: BaseEmbeddingClient,
    name_df: pd.DataFrame,
    valid_stocks: Dict[str, List[str]],
    started_at: str,
) -> None:
    """Run the embedding pipeline for one (provider, model, language) invocation.

    Factored out of `main()` so tests can inject a mock `BaseEmbeddingClient`.
    """
    validate_args(args, logger)

    text_col = _LANG_TEXT_COL[args.language]
    cache_dir = Path(args.cache_folder)
    output_root = Path(args.output_folder)
    model_slug = args.model.replace("/", "-")
    target_dims = list(args.target_dims)
    csmar_md5 = _file_md5(args.csmar_info)
    cli_args_safe = _safe_args(args)

    seed = int(getattr(args, "seed", 42))

    def _embed(native_dim: Optional[int], cache_filename: str) -> np.ndarray:
        cache = EmbeddingCache(logger, str(cache_dir / cache_filename))
        proc = EmbeddingProcessor(
            logger=logger,
            client=client,
            cache=cache,
            lang=args.language,
            native_dim=native_dim,
            batch_size=args.batch_size,
        )
        return proc.process(name_df, text_col)

    transforms = list(args.transforms)
    needs_full_native = ("pca-universal" in transforms) or ("pca-perperiod" in transforms)
    full_native_max: Optional[np.ndarray] = None
    if needs_full_native:
        full_native_max = _embed(
            native_dim=None,
            cache_filename=(f"{client.name}_{model_slug}_{args.language}" f"_max_d{client.native_max_dim}.jsonl"),
        )

    slicer = PerPeriodSlicer(logger=logger, stock_names=name_df, universe_per_period=valid_stocks)
    finished_at_placeholder = "<filled-at-save>"

    def _make_run_config(transform_name: str, d: int, n_periods: int) -> dict:
        return {
            "provider": client.name,
            "model": client.model,
            "language": args.language,
            "transform": transform_name,
            "dim": d,
            "target_dims": target_dims,
            "batch_size": args.batch_size,
            "seed": seed,
            "n_stocks": int(len(name_df)),
            "n_periods": int(n_periods),
            "csmar_md5": csmar_md5,
            "cli_args": cli_args_safe,
            "started_at": started_at,
            "finished_at": datetime.now(timezone.utc).isoformat(),
        }

    # ---- pca-universal: fit once on full CSMAR, slice per period ----
    if "pca-universal" in transforms:
        pca_universal = PcaReducer(
            logger=logger,
            target_dims=target_dims,
            random_state=seed,
        ).fit_transform_all(full_native_max)
        for d, mat in pca_universal.items():
            sliced = slicer.slice(mat)
            cfg = _make_run_config("pca-universal", d, len(sliced))
            subdir = output_root / f"{client.name}_{model_slug}_{args.language}_pca-universal_d{d}"
            EmbeddingSaver(logger=logger, subdir=subdir, run_config=cfg).save(sliced)

    # ---- pca-perperiod: fit on each period's valid_stocks subset ----
    if "pca-perperiod" in transforms:
        # Collect per-period DataFrames keyed by target_dim.
        perperiod_per_d: Dict[int, Dict[str, pd.DataFrame]] = {d: {} for d in target_dims}
        for period, positions in slicer.period_positions.items():
            period_subset = full_native_max[positions]
            tokens = slicer.period_tokens[period]
            try:
                period_results = PcaReducer(
                    logger=logger,
                    target_dims=target_dims,
                    random_state=seed,
                ).fit_transform_all(period_subset)
            except ValueError:
                logger.warning(
                    f"pca-perperiod: period {period} skipped entirely (n_stocks="
                    f"{len(period_subset)} too small for any target_dim)"
                )
                continue
            for d, mat in period_results.items():
                emb_cols = [f"Emb_{i}" for i in range(d)]
                df = pd.DataFrame(mat, columns=emb_cols)
                df.insert(0, "Token", tokens)
                perperiod_per_d[d][period] = df
        for d, period_dfs in perperiod_per_d.items():
            if not period_dfs:
                logger.warning(f"pca-perperiod d={d}: no period had enough stocks; no output written")
                continue
            cfg = _make_run_config("pca-perperiod", d, len(period_dfs))
            subdir = output_root / f"{client.name}_{model_slug}_{args.language}_pca-perperiod_d{d}"
            EmbeddingSaver(logger=logger, subdir=subdir, run_config=cfg).save(period_dfs)

    # ---- native: per target_dim API call, slice per period ----
    if "native" in transforms:
        for target_dim in args.target_dims:
            if not client.supports_dim(target_dim):
                logger.warning(
                    f"Skipping native transform for d={target_dim}: not supported "
                    f"by {client.name}/{client.model} (native_max_dim={client.native_max_dim})"
                )
                continue
            mat = _embed(
                native_dim=target_dim,
                cache_filename=f"{client.name}_{model_slug}_{args.language}_native_d{target_dim}.jsonl",
            )
            sliced = slicer.slice(mat)
            cfg = _make_run_config("native", target_dim, len(sliced))
            subdir = output_root / f"{client.name}_{model_slug}_{args.language}_native_d{target_dim}"
            EmbeddingSaver(logger=logger, subdir=subdir, run_config=cfg).save(sliced)

    logger.info("PIPELINE COMPLETED SUCCESSFULLY")


def _resolve_api_key(args) -> Optional[str]:
    """Resolve the API key with precedence: --api-key > --api-key-env > provider default env var.

    Returns None to let the client constructor fall back to its provider-default
    env var (e.g. ``OPENAI_API_KEY``). Raises if ``--api-key-env`` is given but
    that env var is unset, so a typo is loud rather than silent.
    """
    if getattr(args, "api_key", None):
        return args.api_key
    env_name = getattr(args, "api_key_env", None)
    if env_name:
        val = os.getenv(env_name)
        if not val:
            raise ValueError(f"--api-key-env={env_name} is not set in the environment.")
        return val
    return None


def build_client(args, logger: logging.Logger) -> BaseEmbeddingClient:
    """Provider → client factory.

    Hosted providers (`openai|cohere|voyage|gemini`) get their API key resolved
    by `_resolve_api_key(args)` (precedence: ``--api-key`` > ``--api-key-env``
    > provider-default env var). `local` loads HF weights with ``--device`` /
    ``--hf-dtype``.
    """
    api_key = _resolve_api_key(args)
    if args.provider == "openai":
        return OpenAIEmbeddingClient(
            logger=logger,
            model=args.model,
            api_key=api_key,
            max_retries=args.max_retries,
        )
    if args.provider == "cohere":
        return CohereEmbeddingClient(
            logger=logger,
            model=args.model,
            api_key=api_key,
            max_retries=args.max_retries,
        )
    if args.provider == "voyage":
        return VoyageEmbeddingClient(
            logger=logger,
            model=args.model,
            api_key=api_key,
            max_retries=args.max_retries,
        )
    if args.provider == "gemini":
        return GeminiEmbeddingClient(
            logger=logger,
            model=args.model,
            api_key=api_key,
            max_retries=args.max_retries,
        )
    if args.provider == "local":
        # `--batch-size` controls both the outer EmbeddingProcessor batch and
        # the inner GPU mini-batch for local models, per design.md §5.
        return LocalHFEmbeddingClient(
            logger=logger,
            model=args.model,
            device=args.device,
            hf_dtype=args.hf_dtype,
            batch_size_internal=args.batch_size,
        )
    raise ValueError(f"Unknown provider: {args.provider}")


# -------------------- CLI --------------------
def parse_args():
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", "-c", type=str, default=None)
    pre_args, _ = pre_parser.parse_known_args()

    parser = argparse.ArgumentParser(description="Text embedding pipeline: name resolution + provider embeddings.")

    parser.add_argument(
        "--config",
        "-c",
        type=str,
        default=None,
        help="JSON config file path. Values become argparse defaults; explicit "
        "CLI flags still override. Hyphenated keys (e.g. `csmar-info`) and "
        "underscored keys (`csmar_info`) are both accepted. Optional.",
    )

    # --- Mode ---
    parser.add_argument(
        "--mode",
        type=str,
        choices=["names", "full"],
        default="full",
        help="`names` runs only the name-resolution stage and exits. "
        "`full` runs the full embedding pipeline. Default: full",
    )

    # --- Data sources ---
    parser.add_argument(
        "--csmar-info", type=str, default=None, help="Path to CSMAR `TRD_Co.csv`. Required (via CLI or --config)."
    )
    parser.add_argument(
        "--valid-stocks-dir",
        type=str,
        default=None,
        help="Directory containing `{period}/valid_stocks.json` (output of "
        "`data_sharehold.py`). Required (via CLI or --config).",
    )
    parser.add_argument(
        "--periods",
        type=str,
        nargs="+",
        default=None,
        help="Subset of periods to process. Default: all periods present in " "valid-stocks-dir.",
    )

    # --- Output paths ---
    parser.add_argument(
        "--names-output",
        type=str,
        default="data/text_embeddings/names",
        help="Directory for names-stage outputs (stock_names.csv, "
        "missing_stocks.csv). Default: data/text_embeddings/names",
    )
    parser.add_argument(
        "--cache-folder",
        type=str,
        default="data/text_embeddings/cache",
        help="Directory for embedding API cache (jsonl). " "Default: data/text_embeddings/cache",
    )
    parser.add_argument(
        "--output-folder",
        type=str,
        default="data/text_embeddings/embeddings",
        help="Root output folder for embedding artifacts. " "Default: data/text_embeddings/embeddings",
    )

    # --- Provider / model selection (used by --mode full; session-02+) ---
    parser.add_argument(
        "--provider",
        type=str,
        choices=["openai", "cohere", "voyage", "gemini", "local"],
        default=None,
        help="Embedding provider. Required for --mode full.",
    )
    parser.add_argument("--model", type=str, default=None, help="Provider-specific model id. Required for --mode full.")
    parser.add_argument(
        "--language",
        type=str,
        choices=["zh", "zh-short", "en"],
        default="zh",
        help="`zh` (Conme), `zh-short` (Stknme), or `en` (Conme_en). Default: zh",
    )
    parser.add_argument(
        "--transforms",
        type=str,
        nargs="+",
        choices=["native", "pca-universal", "pca-perperiod"],
        default=["native", "pca-universal"],
        help="Dimensionality strategies to produce. "
        "`pca-universal` fits PCA once on the full CSMAR name set and slices "
        "the result for each period; `pca-perperiod` fits PCA "
        "independently on each period's valid_stocks subset (analogous "
        "to asset-embedding's per-period training). "
        "Default: native pca-universal",
    )
    parser.add_argument(
        "--target-dims",
        type=int,
        nargs="+",
        default=[4, 8, 10, 16, 32, 64, 128],
        help="Target embedding dimensions (aligned with main_rerun). " "Default: 4 8 10 16 32 64 128",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed threaded into PCA's random_state. Default: 42"
    )

    # --- API / batching knobs (session-02+) ---
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Batch size: API batch for hosted providers, also the GPU " "mini-batch for local HF models. Default: 100",
    )
    parser.add_argument("--max-retries", type=int, default=6, help="Max retries per batch. Default: 6")
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="API key (literal string). Highest precedence; if set, "
        "--api-key-env is ignored. If unset, falls back to --api-key-env, "
        "then to the provider-default env var (OPENAI_API_KEY, "
        "COHERE_API_KEY, VOYAGE_API_KEY, GOOGLE_API_KEY).",
    )
    parser.add_argument(
        "--api-key-env",
        type=str,
        default=None,
        help="Name of an environment variable to read the API key from. "
        "Used only when --api-key is unset. Errors loudly if the named "
        "var is itself unset. Default: None (use provider-specific var).",
    )

    # --- Local model settings (session-03 LocalHFEmbeddingClient) ---
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device for local HF models (ignored for hosted providers). Default: cuda:0",
    )
    parser.add_argument(
        "--hf-dtype",
        type=str,
        choices=["bfloat16", "float16", "float32", "int8", "int4"],
        default="bfloat16",
        help="Local HF model dtype. Default: bfloat16",
    )

    # --- Behavior on names missing from CSMAR ---
    parser.add_argument(
        "--missing-name-fallback",
        type=str,
        choices=["skip", "akshare", "error"],
        default="skip",
        help="What to do (in --mode full) for stocks in valid_stocks but missing "
        "from CSMAR. `skip` (default) drops them with a warning; `error` aborts; "
        "`akshare` is reserved for a future name backfill (currently raises "
        "NotImplementedError).",
    )

    # --- Logging ---
    parser.add_argument(
        "--verbose", "-v", type=utils.str2bool, default=False, help="Enable DEBUG-level console logging. Default: False"
    )
    parser.add_argument("--log", "-l", type=str, default=None, help="Log file path. Default: None (no log file)")

    if pre_args.config:
        with open(pre_args.config, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        if not isinstance(cfg, dict):
            raise ValueError(f"--config {pre_args.config} must be a JSON object, got {type(cfg).__name__}")
        normalized = {k.replace("-", "_"): v for k, v in cfg.items()}
        unknown = set(normalized) - {a.dest for a in parser._actions}
        if unknown:
            raise ValueError(
                f"--config {pre_args.config} has unknown keys: {sorted(unknown)}. "
                f"Allowed: {sorted({a.dest for a in parser._actions})}"
            )
        parser.set_defaults(**normalized)

    args = parser.parse_args()

    if args.csmar_info is None or args.valid_stocks_dir is None:
        parser.error("--csmar-info and --valid-stocks-dir are required " "(via CLI or via --config).")

    return args


def main():
    args = parse_args()

    # Load .env so OPENAI_API_KEY etc. become available before client construction.
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass

    log_name = "TextEmbedding-Names" if args.mode == "names" else "TextEmbedding"
    logger = (
        Logger_Preparer()
        .set_config(
            LoggerConfig(
                log_name=log_name,
                log_file=args.log,
                console_stream="tqdm",
                console_level="DEBUG" if args.verbose else "INFO",
                file_level="DEBUG",
                enable_colors=True,
                color_target="format",
            )
        )
        .prepare()
    )

    logger.debug(f"Command: {' '.join(sys.argv)}")
    started_at = datetime.now(timezone.utc).isoformat()

    name_loader = NameLoader(logger=logger, csmar_path=args.csmar_info)
    name_df = name_loader.load()

    valid_stocks_loader = ValidStocksLoader(
        logger=logger,
        valid_stocks_dir=args.valid_stocks_dir,
        periods=args.periods,
    )
    valid_stocks = valid_stocks_loader.load()

    names_dir = Path(args.names_output)
    names_dir.mkdir(parents=True, exist_ok=True)

    stock_names_path = names_dir / "stock_names.csv"
    name_df.to_csv(stock_names_path, index=False, encoding="utf-8")
    logger.info(f"Wrote {len(name_df):,} rows to {stock_names_path}")

    missing_stocks_path = names_dir / "missing_stocks.csv"
    missing_count = write_missing_stocks_report(
        logger,
        valid_stocks,
        set(name_df["Stkcd"]),
        missing_stocks_path,
    )
    logger.info(
        f"Names stage complete: {len(name_df):,} stocks in CSMAR, "
        f"{missing_count:,} missing from CSMAR but present in valid_stocks."
    )

    if args.mode == "names":
        logger.info("Mode = 'names', exit. Use --mode full for the embedding pipeline " "(session-02 onward).")
        return

    # Resolve stocks present in valid_stocks but missing from CSMAR (full mode only;
    # names mode above always emits missing_stocks.csv for inspection).
    if missing_count > 0:
        if args.missing_name_fallback == "error":
            raise ValueError(
                f"{missing_count} stock(s) are in valid_stocks but missing from CSMAR "
                f"(see {missing_stocks_path}); aborting per --missing-name-fallback=error. "
                f"Use 'skip' to drop them with a warning instead."
            )
        if args.missing_name_fallback == "akshare":
            raise NotImplementedError(
                "--missing-name-fallback=akshare (akshare name backfill) is reserved but not "
                "yet implemented. Use 'skip' (drop missing stocks) or 'error' (abort)."
            )
        logger.warning(
            f"{missing_count} stock(s) missing from CSMAR will be dropped " f"(--missing-name-fallback=skip)."
        )

    if args.provider is None or args.model is None:
        raise ValueError(
            "--mode full requires --provider and --model. " f"Got provider={args.provider}, model={args.model}."
        )

    client = build_client(args, logger)
    run_pipeline(args, logger, client, name_df, valid_stocks, started_at)


if __name__ == "__main__":
    main()
