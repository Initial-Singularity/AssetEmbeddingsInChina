# Quickstart

Run the full pretrain–finetune pipeline end-to-end on synthetic data in about a minute (no CSMAR data, no GPU)
and watch the fine-tuned BERT embedding pull ahead of the pretrained baseline. This is the paper's core
mechanism in miniature.

## Setup

```bash
git clone https://github.com/Initial-Singularity/AssetEmbeddingsInChina
cd AssetEmbeddingsInChina
uv sync --extra notebook
```

## Run it

```bash
uv run jupyter lab examples/quickstart.ipynb
# or, headless:
uv run python examples/quickstart.py
```

The notebook uses the synthetic dataset shipped in `examples/data/` — every number is generated, so no
external data is needed.

## What it does

The notebook runs the four-stage chain of W2V, BERT-PT, BERT-FT and RV — the same four stages the paper uses, on
toy data:

1. **W2V** — train a Skip-gram Word2Vec on the synthetic portfolio "sentences".
2. **BERT-PT** (pretrain) — train BERT on the pooled history, its embedding layer initialized from the
   W2V matrix (the **warm start**).
3. **BERT-FT** (finetune) — refine BERT on the finetune-period data (one quarter, 2024Q2), initialized from the pretrained model
   (the **pretrain–finetune** step).
4. **RV** — score all three embeddings (W2V, BERT-PT, BERT-FT) on the relative-valuation benchmark.

## What you should see

BERT-FT lands a clear **+0.07 $R^2$** over the PT-trained baselines on the same benchmark: the quarterly
fine-tune captures time-varying structure the static baselines miss. (Exact numbers vary with the
synthetic seed, but the ordering is stable.)

## Next steps

- [Method & experiments](../explanation/method.md) — how each stage maps to the paper, and the full experiment catalog.
- [Train](../how-to/train.md) / [Evaluate](../how-to/evaluate.md) — the per-stage configs and CLIs.
