"""Masked-LM samples for AssetBERT and the MP refines: unmasked samples from a corpus, masked by a collator.

[`MLMDataset`][asset_embeddings.datasets.mlm.MLMDataset] turns portfolio ``i`` of a
[`PortfolioCorpus`][asset_embeddings.datasets.corpus.PortfolioCorpus] into ``[CLS] ids [SEP]`` cut or padded to
``max_length``; [`MaskingCollator`][asset_embeddings.datasets.mlm.MaskingCollator] masks a list of samples and
stacks them into a batch. Masking happens in the collator, in the process that fetched the samples (a DataLoader
worker or the main process), sample by sample in batch order, so the random draws come in the same order as when
each sample was masked as it was read.
"""

import logging
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerFast

from .corpus import PortfolioCorpus

logger = logging.getLogger(__name__)

BATCH_KEYS = ("input_ids", "attention_mask", "labels", "proportions1", "proportions2")
# Holdings counted per pass over a memory-mapped corpus, so the count never materializes the whole codes array.
COUNT_CHUNK = 1 << 22


class MLMDataset(Dataset):
    """Unmasked masked-LM samples, one per portfolio of a corpus.

    At construction every stock code of the corpus is mapped once to its tokenizer id (codes the tokenizer does not
    know map to ``[UNK]``), and the unknown codes and the portfolios longer than ``max_length`` are logged.

    Args:
        corpus: The portfolios.
        tokenizer: The tokenizer whose ids the samples carry.
        max_length: Sequence length; a longer portfolio is cut (dropping ``[SEP]``), a shorter one padded.
        include_proportion: Whether the samples carry the corpus proportions (else zeros).

    Raises:
        ValueError: ``include_proportion`` on a corpus without proportions.
    """

    def __init__(
        self,
        corpus: PortfolioCorpus,
        tokenizer: PreTrainedTokenizerFast,
        max_length: int,
        include_proportion: bool = False,
    ):
        if include_proportion and not corpus.has_proportions:
            raise ValueError("include_proportion is set but the corpus was read without proportions")
        self.corpus = corpus
        self.max_length = int(max_length)
        self.include_proportion = include_proportion
        self.cls_id = tokenizer.cls_token_id
        self.sep_id = tokenizer.sep_token_id
        self.pad_id = tokenizer.pad_token_id or 0
        self.special_ids = torch.tensor(sorted(tokenizer.all_special_ids), dtype=torch.long)
        self.token_of_code = np.asarray(tokenizer.convert_tokens_to_ids(corpus.vocab.tolist()), dtype=np.int64)
        self._log_coverage(tokenizer)

    def _log_coverage(self, tokenizer: PreTrainedTokenizerFast) -> None:
        unknown = self.token_of_code == tokenizer.unk_token_id
        n_unknown_codes = int(unknown.sum())
        if n_unknown_codes:
            codes = self.corpus.codes
            n_unknown_holdings = sum(
                int(unknown[np.asarray(codes[lo : lo + COUNT_CHUNK])].sum()) for lo in range(0, len(codes), COUNT_CHUNK)
            )
            logger.info(
                f"MLMDataset: {n_unknown_holdings} holding(s) of {n_unknown_codes} code(s) are unknown to the "
                f"tokenizer and map to [UNK]"
            )
        lengths = np.diff(np.asarray(self.corpus.offsets))
        n_cut = int((lengths + 2 > self.max_length).sum())
        if n_cut:
            logger.info(
                f"MLMDataset: {n_cut} of {len(lengths)} portfolio(s) are longer than max_length {self.max_length} "
                f"with [CLS] and [SEP] and are cut ({n_cut / len(lengths):.1%})"
            )

    def __len__(self) -> int:
        return len(self.corpus)

    def __getitem__(self, i: int) -> Dict[str, torch.Tensor]:
        """Sample ``i`` unmasked: ``index``, ``input_ids``, ``attention_mask``, ``special_tokens_mask``, proportions.

        The proportions are zero at ``[CLS]``, ``[SEP]`` and padding, and zeros throughout without proportions.
        """
        if i < 0 or i >= len(self):
            raise IndexError(f"Index {i} is out of range [0, {len(self) - 1}]")
        lo, hi = int(self.corpus.offsets[i]), int(self.corpus.offsets[i + 1])
        ids = self.token_of_code[np.asarray(self.corpus.codes[lo:hi])].tolist()
        input_ids = [self.cls_id] + ids + [self.sep_id]
        n = len(input_ids)
        proportions = []
        if self.include_proportion:
            for column in (self.corpus.proportions1, self.corpus.proportions2):
                values = torch.zeros(n, dtype=torch.float32)
                values[1:-1] = torch.tensor(np.asarray(column[lo:hi]), dtype=torch.float32)
                proportions.append(values)
        if n > self.max_length:
            input_ids = input_ids[: self.max_length]
            attention_mask = [1] * self.max_length
            proportions = [p[: self.max_length] for p in proportions]
        else:
            pad = self.max_length - n
            input_ids = input_ids + [self.pad_id] * pad
            attention_mask = [1] * n + [0] * pad
            proportions = [torch.cat([p, torch.zeros(pad, dtype=torch.float32)]) for p in proportions]
        if not proportions:
            proportions = [torch.zeros(self.max_length, dtype=torch.float32) for _ in range(2)]
        ids_tensor = torch.tensor(input_ids, dtype=torch.long)
        return {
            "index": i,
            "input_ids": ids_tensor,
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "special_tokens_mask": torch.isin(ids_tensor, self.special_ids),
            "proportions1": proportions[0],
            "proportions2": proportions[1],
        }


class MaskingCollator:
    """Masks a list of `MLMDataset` samples and stacks them into a batch.

    Policies, by the arguments: ``mask_indices`` set: those positions become ``[MASK]`` (MP); ``mask_prob`` with
    ``seed``: BERT-style masks drawn from a generator seeded by ``(seed, sample index)``, the same on every pass
    (validation and test); ``mask_prob`` alone: BERT-style masks from torch's global generator (training); neither:
    no masking. Special-token positions are never masked; the proportions are zeroed at special and masked
    positions, so the input does not reveal the size of a holding the model must predict.

    Args:
        tokenizer: Defines ``[MASK]`` and the vocabulary size for the random replacements.
        mask_prob: Probability that a non-special position is selected.
        mask_indices: Fixed positions to mask.
        seed: The training seed; with ``mask_prob``, masks fixed per sample.
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerFast,
        mask_prob: Optional[float] = None,
        mask_indices: Optional[Sequence[int]] = None,
        seed: Optional[int] = None,
    ):
        self.mask_token_id = tokenizer.convert_tokens_to_ids("[MASK]")
        self.vocab_size = len(tokenizer)
        self.mask_prob = mask_prob
        self.mask_indices = list(mask_indices) if mask_indices is not None else None
        self.seed = seed

    @property
    def policy(self) -> str:
        """``"positions"``, ``"fixed_per_sample"``, ``"random"`` or ``"none"``."""
        if self.mask_indices is not None:
            return "positions"
        if self.mask_prob is not None:
            return "fixed_per_sample" if self.seed is not None else "random"
        return "none"

    def __call__(self, samples: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """Mask each sample in list order and stack ``input_ids``, ``attention_mask``, ``labels`` and the proportions."""
        policy = self.policy
        masked = []
        for sample in samples:
            input_ids = sample["input_ids"].clone()
            p1 = sample["proportions1"].clone()
            p2 = sample["proportions2"].clone()
            special = sample["special_tokens_mask"]
            if policy == "random":
                labels = self._mask_random(input_ids, special, p1, p2, generator=None)
            elif policy == "fixed_per_sample":
                state = np.random.SeedSequence([self.seed, int(sample["index"])]).generate_state(1, dtype=np.uint64)[0]
                labels = self._mask_random(input_ids, special, p1, p2, torch.Generator().manual_seed(int(state)))
            else:
                labels = self._mask_positions(input_ids, special, p1, p2, self.mask_indices or [])
            masked.append((input_ids, sample["attention_mask"], labels, p1, p2))
        return {key: torch.stack([m[k] for m in masked]) for k, key in enumerate(BATCH_KEYS)}

    def _mask_random(
        self,
        input_ids: torch.Tensor,
        special: torch.Tensor,
        p1: torch.Tensor,
        p2: torch.Tensor,
        generator: Optional[torch.Generator],
    ) -> torch.Tensor:
        """BERT-style masking in place (80% ``[MASK]``, 10% random token, 10% unchanged); returns the labels."""
        labels = input_ids.clone()
        probability_matrix = torch.full(labels.shape, self.mask_prob)
        probability_matrix.masked_fill_(special, value=0.0)
        mask_positions = torch.bernoulli(probability_matrix, generator=generator).bool()
        labels[~mask_positions] = -100

        # torch.where returns the selected positions in ascending order, so slicing it directly would give the
        # earliest holdings [MASK] and the last ones the random/unchanged treatment; shuffle first so the 80/10/10
        # split is independent of position.
        mask_indices = torch.where(mask_positions)[0]
        mask_indices = mask_indices[torch.randperm(len(mask_indices), generator=generator)]

        mask_count = int(0.8 * len(mask_indices))
        input_ids[mask_indices[:mask_count]] = self.mask_token_id

        random_count = int(0.1 * len(mask_indices))
        if random_count > 0:
            random_indices = mask_indices[mask_count : mask_count + random_count]
            random_tokens = torch.randint(
                self.vocab_size, (len(random_indices),), dtype=torch.long, generator=generator
            )
            input_ids[random_indices] = random_tokens

        self._zero_weights(p1, p2, special, mask_positions)
        return labels

    def _mask_positions(
        self,
        input_ids: torch.Tensor,
        special: torch.Tensor,
        p1: torch.Tensor,
        p2: torch.Tensor,
        positions: Sequence[int],
    ) -> torch.Tensor:
        """Mask the given positions with ``[MASK]`` in place (past the end and special positions skipped)."""
        labels = input_ids.clone()
        selected = torch.zeros_like(input_ids, dtype=torch.float)
        for idx in positions:
            if idx < len(selected):
                selected[idx] = 1.0
        selected.masked_fill_(special, value=0.0)
        mask_positions = selected.bool()
        labels[~mask_positions] = -100
        input_ids[mask_positions] = self.mask_token_id
        self._zero_weights(p1, p2, special, mask_positions)
        return labels

    @staticmethod
    def _zero_weights(p1: torch.Tensor, p2: torch.Tensor, special: torch.Tensor, masked: torch.Tensor) -> None:
        for p in (p1, p2):
            p[special] = 0.0
            p[masked] = 0.0


def split_masking(
    tokenizer: PreTrainedTokenizerFast, mask_prob: Optional[float], mask_indices: Optional[Sequence[int]], seed: int
) -> Tuple[MaskingCollator, MaskingCollator]:
    """The collators of the training split and of the held-out splits.

    With ``mask_indices`` every split masks those positions; otherwise the training split draws fresh masks and
    the validation and test splits keep masks fixed per sample, so their losses compare across epochs.

    Args:
        tokenizer: The tokenizer.
        mask_prob: The dataset's masking probability.
        mask_indices: The dataset's fixed positions.
        seed: The training seed.

    Returns:
        ``(training collator, held-out collator)``.
    """
    train = MaskingCollator(tokenizer, mask_prob=mask_prob, mask_indices=mask_indices)
    held_out = MaskingCollator(tokenizer, mask_prob=mask_prob, mask_indices=mask_indices, seed=seed)
    return train, held_out
