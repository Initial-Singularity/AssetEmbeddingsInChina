"""Seeding helpers: the one place that decides which random generators a run seeds, and when.

`seed_basic` covers Python `random` and NumPy (plus torch and cuDNN on request) for the paths without
HuggingFace: the data distributor, the RV/RC evaluators and the Word2Vec trainer. `seed_hf_global` is the
transformers deterministic block that runs before the `Accelerator` is built, and `seed_accelerate` the
per-device accelerate seeding that runs after it. transformers and accelerate are imported inside the
functions so that importing this module stays cheap on CPU-only installs without them.
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
    constructing the ``Accelerator``; pair with `seed_accelerate` after.
    """
    import transformers

    transformers.trainer_utils.set_seed(seed, deterministic=True)
    # The running interpreter fixed its hash seed at start-up; this reaches the child processes spawned
    # later (DataLoader workers).
    os.environ["PYTHONHASHSEED"] = str(seed)
    # CUDA needs a fixed cuBLAS workspace for deterministic results; torch's deterministic mode checks it.
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_accelerate(seed: int) -> None:
    """Per-device ``accelerate`` seeding, run **after** the ``Accelerator`` exists.

    ``device_specific=True`` offsets the seed by the process index, so the
    distributed state must already be initialized when this is called.
    """
    import accelerate

    accelerate.utils.set_seed(seed, device_specific=True, deterministic=True)
