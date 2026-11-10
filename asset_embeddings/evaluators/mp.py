"""The managed-portfolio (MP) score: how well a masked-LM identifies the held-out holding of a portfolio."""

import math
from typing import Tuple

import numpy as np
import torch
from accelerate import Accelerator
from torch.utils.data import DataLoader
from tqdm import tqdm

from asset_embeddings import utils


def mp_token_scores(logits: torch.Tensor, labels: torch.Tensor) -> np.ndarray:
    """Per-token MP score at the masked positions of one batch.

    Args:
        logits: Model output, ``[batch_size, seq_len, vocab_size]``.
        labels: Target ids, ``[batch_size, seq_len]``; ``-100`` marks an unmasked position.

    Returns:
        1-D array of ``mp_i = 1 + log p_i / log(V)`` over the masked tokens, ``V`` the vocabulary size; empty when
        the batch has no masked position. A uniform guess scores ``log p = -log(V)``, so ``mp_i`` is 0 at chance and 1
        at certainty.
    """
    mask_positions = labels != -100
    masked_logits = logits[mask_positions]  # [num_masked, vocab_size]
    masked_labels = labels[mask_positions]  # [num_masked]
    if masked_labels.numel() == 0:
        return np.empty(0, dtype=float)
    log_probs = torch.log_softmax(masked_logits, dim=-1)
    true_log_probs = log_probs[torch.arange(len(masked_labels), device=masked_labels.device), masked_labels]
    mp = 1.0 + true_log_probs / math.log(logits.size(-1))
    return mp.detach().cpu().numpy()


def score_loader(
    model: torch.nn.Module, loader: DataLoader, accelerator: Accelerator, desc: str = "Eval"
) -> Tuple[float, float]:
    """Mean and standard deviation of the per-token MP score over every masked token of ``loader``.

    The statistics are over tokens, not batches, so batches with different numbers of masked tokens carry their
    proper weight.

    Args:
        model: The masked-LM, called as ``model(input_ids, attention_mask=..., labels=...)``.
        loader: The loader to score.
        accelerator: Decides which process shows the progress bar.
        desc: Progress-bar label.

    Returns:
        ``(mean, std)``.
    """
    tracker = utils.MeanVarianceTracker()
    model.eval()
    with torch.no_grad():
        for batch in tqdm(loader, total=len(loader), desc=desc, disable=not accelerator.is_local_main_process):
            labels = batch["labels"]
            outputs = model(batch["input_ids"], attention_mask=batch["attention_mask"], labels=labels)
            for mp_i in mp_token_scores(outputs.logits, labels):
                tracker.update(mp_i)
    return tracker.mean, tracker.std()
