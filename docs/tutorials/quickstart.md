# Quickstart

Run the full pretrain–finetune training pipeline end-to-end on synthetic data in about a minute — no CSMAR data, no GPU — and come away with four asset-embedding CSVs (W2V-PT, W2V-FT, BERT-PT, BERT-FT). This is the paper's estimation in miniature.

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

The notebook runs the **W2V-PT → W2V-FT → BERT-PT → BERT-FT** chain — the same training stages the paper uses, on toy data:

1. **W2V-PT** — train a Skip-gram Word2Vec on the synthetic portfolio "sentences".
2. **W2V-FT** — continue training W2V on the fine-tune period, starting from the pretrained vectors
   (**PT→FT**).
3. **BERT-PT** (pretrain) — train BERT on the pooled history, its embedding layer warm-started from the
   W2V-PT matrix (**W2V→BERT**).
4. **BERT-FT** (finetune) — refine BERT per "quarter", the encoder initialized from the pretrained model
   (**PT→FT**) and the embedding layer warm-started from W2V-FT (**W2V→BERT**).

Each stage writes its embeddings as a `(Token, Embed_1..Embed_d)` CSV — the deliverable of the pipeline.

## What you should see

The run ends with a summary table of the four embedding CSVs (model, training data, shape, path) and a quick peek at what they encode: the top-5 cosine neighbours of one probe stock in each space. The synthetic universe has latent cluster structure, so a stock's neighbours should stay within its cluster; the fine-tuned models pick up the FT-period drift together, while the initializations (W2V→BERT, PT→FT) keep all four embeddings comparable with one another. (Exact neighbours vary with the synthetic seed, but the cluster structure is stable.)

## Next steps

- [Method & experiments](../explanation/method.md) — how each stage maps to the paper.
- [Train](../how-to/train.md) — the per-stage configs and CLIs.
