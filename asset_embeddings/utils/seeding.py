"""Centralized random-seeding helpers — the single source of seeding policy.

Three deliberately distinct scopes:

* :func:`seed_basic` — python ``random`` + ``numpy`` (+ optional ``torch``/cuDNN).
  Used by non-HuggingFace code paths: the data distributor and the
  Word2Vec trainer.
* :func:`seed_hf_global` — the HuggingFace/``transformers`` deterministic block,
  run **before** the ``Accelerator`` is constructed.
* :func:`seed_accelerate` — ``accelerate`` per-device seeding, run **after** the
  ``Accelerator`` is constructed (``device_specific`` needs the process index).

``transformers``/``accelerate`` are imported lazily inside the functions so that
importing this module stays cheap on CPU-only / no-HF environments (see C-H1).
"""

import os
import random

import numpy as np
import torch


def seed_basic(seed: int, *, include_torch: bool = True, deterministic: bool = True) -> None:
    """Seed python ``random``, ``numpy`` and (optionally) ``torch`` + cuDNN.

    Args:
        seed: The seed value.
        include_torch: When True, also seed ``torch`` (CPU + all CUDA devices).
            Set False for numpy/sklearn-only paths that never touch torch RNGs.
        deterministic: When True (and ``include_torch``), force cuDNN into
            deterministic, non-benchmarking mode.
    """
    random.seed(seed)
    np.random.seed(seed)
    if include_torch:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False


def seed_hf_global(seed: int) -> None:
    """Deterministic global seeding for HuggingFace training, pre-``Accelerator``.

    Seeds ``transformers`` (which covers python/numpy/torch), pins the hash and
    cuBLAS workspace, and forces cuDNN determinism. Call this *before*
    constructing the ``Accelerator``; pair with :func:`seed_accelerate` after.
    """
    import transformers

    transformers.trainer_utils.set_seed(seed, deterministic=True)
    os.environ["PYTHONHASHSEED"] = str(seed)  # 禁止hash随机化
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"  # 禁止cublas工作空间随机化
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_accelerate(seed: int) -> None:
    """Per-device ``accelerate`` seeding, run **after** the ``Accelerator`` exists.

    ``device_specific=True`` offsets the seed by the process index, so the
    distributed state must already be initialized when this is called.
    """
    import accelerate

    accelerate.utils.set_seed(seed, device_specific=True, deterministic=True)
