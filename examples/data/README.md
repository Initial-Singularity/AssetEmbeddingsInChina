# Example data — synthetic

This directory holds the synthetic dataset used by `examples/quickstart.{ipynb,py}`.
**Every number here is generated — none of it
comes from CSMAR or any other commercial data vendor.** It exists purely so the
example can ship inside a public repository.

## Files

| File | Rows | Description |
|---|---|---|
| `portfolio_pretrained.csv` | 1500 | PT-period investor portfolios. Schema: `InvestorID`, `Portfolio` (comma-separated stock codes), `Proportion1`, `Proportion2` (comma-separated weights). |
| `portfolio_finetune.csv`   | 1500 | FT-period portfolios — a drifted snapshot of the same world. Same schema. |
| `benchmark.csv`            | 2499 | Per-stock valuation/return features at 2024Q2, generated from FT-period factors. Schema: `Stkcd`, `Trdmnt`, `be`, `me`, `be_me`, `beta_60m`, `at_gr1`, `gp_at`, `ret_12_1`, `qret`, `div_at`, `qret_next`. |

## How it is shaped

The data is sampled from a latent factor model designed to mirror the
*statistical structure* of A-share shareholding + valuation data **and** to
make the PT-FT distinction observable in downstream RV:

- A universe of ~2500 stock codes drawn from realistic A-share board ranges
  (Shanghai main `600xxx/601xxx/603xxx`, Shenzhen main `000xxx`, SME `002xxx`,
  ChiNext `300xxx/301xxx`, STAR `688xxx`).
- 18 latent "industry-like" clusters with centroids in a 10-D factor space.
- **Two stock-factor matrices**:
  - `stock_factors_pt` = cluster_centroid + per-stock jitter (PT period)
  - `stock_factors_ft` = stock_factors_pt + cluster-level drift + per-stock noise (FT period)
- Investors prefer 1–3 clusters; PT portfolios are sampled via softmax similarity
  against `stock_factors_pt`, FT portfolios against `stock_factors_ft`.
- Benchmark features (`be`, `me`, `be_me`, `beta_60m`, `at_gr1`, `gp_at`,
  `ret_12_1`, `qret`, `div_at`, `qret_next`) are continuous mappings of
  `stock_factors_ft` onto random direction vectors in latent space, plus
  realistic noise. NaN patterns roughly match real shareholding-quarter data
  (`beta_60m` ~32% NaN, `div_at` ~63% NaN for non-payers, etc.).

This design has two consequences relevant to downstream evaluation:

1. **W2V and BERT-PT (both PT-trained) plateau together at RV ≈ 0.52.**
   They recover the PT-period cluster structure, which captures most of the
   value/momentum/growth signal in the benchmark — but they cannot see the
   cluster-level drift, so a residual gap remains.

2. **BERT-FT lifts RV to ≈ 0.60.** Fine-tuning on FT-period portfolios lets
   it pick up the drift direction and align its representation with the
   FT-aligned benchmark.

## Reproducing this dataset

The exact generation recipe is **not shipped in the repository**. The CSVs here are the canonical artifact. Treat
them as a deterministic, redistributable demo fixture; do **not** rely on them
for anything other than verifying the pipeline runs.

The data-preparation step for real CSMAR data is not part of this repository; real inputs must follow the same schemas as the files above.
