# Method & experiments

This page maps the manuscript to the code — which file estimates which equation, which config field wires which stage — and then walks through the paper's experiments in order, each with the families (config template and run script) that produce it. The symbols follow the manuscript's notation.

## Method (paper and code)

### Portfolios as sentences

Each investor's quarter-$t$ portfolio is a rank-ordered sequence of holdings — a *sentence* — and each stock is a *token*. Rank order (largest holding first) carries economic meaning, just as word order does in language. This is what lets the asset-pricing problem be attacked with NLP architectures.

| Paper | Code |
|---|---|
| Holdings to rank-ordered portfolio sequences | Prepared outside this repository: holdings grouped by investor/period, sorted by holding proportion, written as the `Portfolio` sequence of the portfolio CSV format |
| Serialize portfolios for training | `asset_embeddings.datasets.portfolio_csv` ([API](../reference/api/datasets.md)) |
| Stock/token-id vocabulary | `Tokenizer_Preparer` builds a `PreTrainedTokenizerFast` ([Preparers](preparers.md)) |

### Three architectures

The repo estimates the embedding sequence $\{\boldsymbol{x}_{a,t}\}$ with three architectures.

#### Recommender system (RS)

A truncated SVD of the demeaned holdings matrix $\tilde{\boldsymbol{H}}_t$:

$$\tilde{\boldsymbol{H}}_t = \boldsymbol{U}_t\,\boldsymbol{\Sigma}_t\,\boldsymbol{V}_t^{\top},
\qquad \boldsymbol{X}_t^{\text{RS}} = \boldsymbol{V}_{t,d}\,\boldsymbol{\Sigma}_{t,d}.$$

The four $(\phi,\psi)$ encodings (RS-Binary / RS-Ranks / RS-Level0 / RS-LevelMin) are the `model_type` choices.

Code: `asset_embeddings/trainers/rs.py` (CLI `asset_embeddings/scripts/train/rs.py`), `AssetRSTrainConfig` (`model_type ∈ {RS_Binary, RS_Ranks, RS_Level0, RS_LevelMin}`). RS is fit independently per quarter: it has no iterative parameters to pretrain or warm-start.

#### Word2Vec (W2V)

Skip-gram over the rank-ordered portfolio sequences:

$$\mathcal{L}_{\text{W2V}} = -\sum_{(i,t)\in D}\ \sum_{c=1}^{L_{i,t}}\ \sum_{0<|j-c|\le w}
\log \mathbb{P}\!\big(a_{i,t}(j)\mid a_{i,t}(c)\big).$$

The trained input matrix $\boldsymbol{E}_{\text{in}}^{*}$ gives the asset embeddings.

Code: `asset_embeddings/trainers/w2v.py` (CLI `asset_embeddings/scripts/train/w2v.py`; gensim Skip-gram/CBOW + negative sampling), `AssetW2VTrainConfig`.

#### BERT

A Transformer encoder trained with a masked-language-modeling objective at mask rate $p_{\text{mask}}=0.15$:

$$\mathcal{L}_{\text{MLM}} = -\sum_{(i,t)\in D}\ \mathbb{E}_{M_{i,t}}\sum_{j\in M_{i,t}}
\log \mathbb{P}_\theta\!\big(a_{i,t}(j)\mid \tilde{\pi}_{i,t}\big).$$

The asset embedding at quarter $t$ is the row $[\boldsymbol{E}_{\text{emb}}^{*}]_a$; the encoder and head are discarded after training.

Code: `asset_embeddings/trainers/bert.py` (CLI `asset_embeddings/scripts/train/bert.py`); `asset_embeddings.modules.BertEmbeddings`; the random masked-position split is `asset_embeddings.datasets.mlm.MaskingCollator`; `AssetBERTTrainConfig`.

### Pretrain–finetune paradigm

The aim is a *time-varying* embedding: one vector per stock per quarter, such that the change from one quarter to the next is itself informative. This runs into an identifiability problem. An embedding learned by a neural model is only pinned down up to an arbitrary rotation, reflection, and relabeling of its axes — train on the same data twice from different random starting points and you get two embeddings that are equally valid yet look entirely different. So if each quarter were trained independently, the quarter-to-quarter difference $\boldsymbol{x}_{a,t+1}-\boldsymbol{x}_{a,t}$ (stock $a$'s vector at $t{+}1$ minus its vector at $t$) would mostly measure that arbitrary coordinate drift, not any real change in the stock.

The remedy is to give every quarter the **same starting point**. The model is *pretrained* once on the pooled earlier-quarter corpus, producing a single base solution; each evaluation quarter is then *fine-tuned* from that shared base rather than from scratch. Because all quarters set out from the same anchor and move only a little, their embeddings stay in a common coordinate frame, and $\boldsymbol{x}_{a,t+1}-\boldsymbol{x}_{a,t}$ comes to reflect genuine movement of the stock.

A BERT here has two trainable parts to initialize — an **embedding layer** (one vector per stock) and a Transformer **encoder** — and "starting from the shared base" is exactly what two fields in the BERT config set (`templates/train_main.json`):

| Stage | `model.model_checkpoint` (seeds the encoder) | `model.w2v_model` = `tokenizer.w2v_model` (seeds the embedding layer) |
|---|---|---|
| **BERT pretrain** | `null` — encoder trained from scratch | `checkpoints/pretrained/AssetW2V/d{dim}/AssetW2V_d{dim}_pretrained.model` |
| **BERT finetune, quarter $t$** | `checkpoints/pretrained/AssetBERT/d{dim}/AssetBERT_d{dim}_pretrained_best/model.safetensors` | `checkpoints/finetune/AssetW2V/d{dim}/{t}/AssetW2V_d{dim}_{t}.model` |

- **`model.model_checkpoint`** seeds the encoder. At pretrain it is `null`, so the encoder is trained from scratch on the pooled corpus. At fine-tune it is the pretrained BERT, so the encoder is carried over and only nudged ; this is what keeps every quarter's solution in the same shared frame. (The masked-language-model prediction head is re-initialized each time; it is discarded after training anyway.)
- **`model.w2v_model`** seeds the embedding layer: `AssetBERT_Preparer` copies the vectors of a trained Word2Vec model into BERT's embedding matrix (see [Preparers](preparers.md)) ; this is the **warm start**. At pretrain it is the W2V trained on the pooled corpus. At fine-tune the seeding runs *after* the checkpoint load, so quarter $t$'s own W2V overwrites the embedding layer while the encoder stays as loaded from the pretrained BERT.
- **`model.embedding_file`** does the same embedding seeding from an embedding CSV instead of a Word2Vec model, and is `null` in the main pipeline.

So each fine-tuned quarter is *the shared pretrained encoder plus that quarter's own embedding layer*, adjusted by a short fine-tune on the quarter's holdings. The shared encoder is what holds the per-quarter embeddings in one comparable coordinate frame across time. See [Train](../how-to/train.md) for the full per-stage config.

### Training setup

One BERT configuration serves every dimension; the settings below are what `templates/train_main.json` and `templates/train_timesplit_robust.json` render, and every other training template follows them.

| | Pretraining | Fine-tuning |
|---|---|---|
| Architecture | 4 layers, 2 heads, FFN 512, context 256 (the same for every $d$) | same, encoder loaded from the pretrained checkpoint |
| Data | pooled earlier quarters | all portfolios of one quarter |
| Epochs | up to 40, cosine schedule | 30, cosine schedule |
| Selection | best epoch by validation loss on a 10 % random split (seed 42); early stop after 5 epochs without a 0.1 % relative improvement | best epoch by training loss, no validation split, no early stop |
| Optimizer | AdamW, lr $10^{-3}$, 500 warmup steps, gradient clipping 1.0 | PagedAdam8bit, lr $10^{-3}$, no warmup, gradient clipping 1.0 |
| Batch / masking | batch 128, mask probability 0.15 | batch 32, mask probability 0.15 |

Pretraining is selected on held-out data because a training-loss rule cannot tell a converged run from one that diverged after a few epochs. Fine-tuning keeps the original rule so that the quarter's embedding sees every portfolio of the quarter and the MP score reads as the highest attainable prediction level.

### The three benchmarks

Embeddings are scored on three complementary axes. The estimators and their math are in [Evaluate](../how-to/evaluate.md); the mapping:

| Benchmark | Code |
|---|---|
| **RV** — relative valuation (cross-sectional explanatory $R^2$ of valuation residuals on embeddings, a ridge fit on every firm of the quarter) | `asset_embeddings.evaluators.rv` · `asset_embeddings/scripts/eval/rv.py` · `RVEvalConfig` |
| **RC** — return comovement (cross-sectional predictive $R^2$ of next-quarter returns on embeddings) | `asset_embeddings.evaluators.rc` · `asset_embeddings/scripts/eval/rc.py` · `RCEvalConfig` |
| **MP** — managed portfolios (masked-asset normalized log-likelihood) | `asset_embeddings/trainers/mp.py` (`MPBertTrainer` / `MPSimplifiedMLMTrainer`; CLIs `asset_embeddings/scripts/eval/mp_bert.py` / `mp_smlm.py`); `asset_embeddings.evaluators.mp.mp_token_scores` returns the per-token normalized log-likelihood; `MPResult` |

RV and RC are fitted in sample by default; `scoring: "holdout"` on either config is the separate firm-split option, scored on held-out firms over seeded splits (the [`holdout` family](#extensions), an extension the paper does not use). Temporal RC (`asset_embeddings.evaluators.rc_temporal` · `asset_embeddings/scripts/eval/rc_temporal.py` · `TemporalRCEvalConfig`) is a further option: the RC slope fitted on the previous quarter, or the mean of the slopes of every earlier quarter, is applied unchanged to the next, in raw coordinates or after a Procrustes alignment of the embeddings (the [`rc_temporal` family](#extensions)).

### Text-embedding comparison

The commercial-LLM baseline — embedding firm *names* instead of holdings — comes from OpenAI / Cohere / Voyage / Gemini / Qwen / BGE embeddings prepared outside this repository, then scored through the same RV/RC/MP evaluators.

## Experiments

The experiments are listed in the order of the paper, section by section and then appendix by appendix. Each entry states the question the paper asks there, what the analysis varies and what it holds fixed, and the experiment families that produce the numbers.

A **family** is a [`fast_generator`](../how-to/generate-configs.md) template that expands the config grid, plus a `scripts/run/*.sh` runner that sets the family's `runs.experiment` label (`EXP=…`) and runs the grid; how to launch, filter and re-run a family is in [Run experiments](../how-to/run-experiments.md). The label equals the family name except where an entry says otherwise. Eval-only families train no new models: they read the checkpoints of `main` (or `timesplit_robust`) from a different angle.

Dimensions are $d \in \{4,8,10,16,32,64,128\}$ throughout. The main split pretrains on 2005Q1–2023Q1 and evaluates on the six quarters 2023Q2–2024Q3; the 50/50 time split pretrains on 2005Q1–2020Q3 and evaluates on the sixteen quarters 2020Q4–2024Q3.

### §3 Data

**Question.** Mutual funds disclose full holdings in Q2/Q4 and only their top ten in Q1/Q3, so the observed panel zigzags between the two regimes. The sample has to stay comparable across quarters for one model to be pretrained on history and finetuned on each evaluation quarter.

**What is done.** A $(10,10)$ co-holding screen keeps a holding when the stock has at least ten holders and the investor holds at least ten stocks, and the panel is split chronologically: the first 80 % of holding observations for pretraining, the rest for finetuning and evaluation (cutoff 2023Q1). The 50/50 split of Appendix J comes from the same pass.

| Family | Role | Template | Runner |
|---|---|---|---|
| data preprocessing | Shareholding sequences, benchmark data, industry labels, the state-ownership labels of §7.1, the 80/20 and 50/50 splits; prepared outside this repository. | — | — |

### §4 Benchmark performance (RV, RC, MP)

**Question.** Do asset embeddings learned from China's partial institutional holdings explain valuations, predict return comovement and capture the similarity investors act on, and how do the three architectures rank? In the US evidence of Gabaix et al., the recommender system leads and BERT stands out only on masked-asset prediction.

**What is done.** Six embedding models (BERT, W2V, RS-Binary, RS-Ranks, RS-Level0, RS-LevelMin) at seven dimensions are scored on the three benchmarks over the six evaluation quarters, against firm-characteristics baselines. For RV and RC, BERT is also combined with the Shenwan Level-1 industry dummies and the characteristics, seven regressor sets in all, to test whether the embedding carries information beyond both. MP scores every model under the same BERT-shape refine procedure, so that only the embedding layer differs.

| Family | Role | Template | Runner |
|---|---|---|---|
| **main** | Trains W2V, then BERT pretrain, then BERT finetune per quarter, and the four RS variants; scores RV, RC (with and without characteristics) and MP. BERT follows the [training setup](#training-setup). | `train_main.json` · `eval_main.json` | `train_main.sh` · `eval_main.sh` |
| **industry** | Industry dummies alone, with characteristics, with BERT, and with both. Reuses `main` embeddings. | `eval_industry.json` | `eval_industry.sh` |

### §5.1 Effect of pretrain–finetune

**Question.** Is the pretrain–finetune paradigm needed? In the US it adds nothing, but in China the observed holdings change sharply from one quarter to the next under the hybrid disclosure rules.

**What is done.** BERT is compared in three forms at every dimension: the full paradigm, the same model trained each quarter from random initialization (without pretraining or the Word2Vec warm start), and the pretrained model with no quarterly finetune (without finetuning).

| Family | Role | Template | Runner |
|---|---|---|---|
| **no_pt_init** (label `pt_strategy_ablation/no_pt_init`) | Without pretraining: each quarter's BERT starts from random initialization, embedding layer included; the quarter's Word2Vec model supplies only the vocabulary. | `train_no_pt_init.json` · `eval_no_pt_init.json` | `train_no_pt_init.sh` · `eval_no_pt_init.sh` |
| **static** | Without finetuning: the pretrained BERT (and W2V) embedding scored in every evaluation quarter. Reuses `main` checkpoints. | `eval_static.json` | `eval_static.sh` |

### §5.2 Effect of warm-start

**Question.** Given that pretrain–finetune matters, does it also matter to seed BERT's embedding layer with the Word2Vec embedding, which has already grouped assets by co-holding?

**What is done.** BERT is trained with and without the W2V warm-start at both stages (pretraining from the pooled W2V, each quarter's finetune from that quarter's W2V), everything else fixed. The family runs the full 2×2 over the two stages; the paper reports the with-both against without-either contrast and notes that the two stages are complementary.

| Family | Role | Template | Runner |
|---|---|---|---|
| **w2v_init_ablation** | 2×2 over the warm-start at pretraining and at finetuning. | `train_w2v_init_ablation.json` · `eval_w2v_init_ablation.json` | `train_w2v_init_ablation.sh` · `eval_w2v_init_ablation.sh` |

### §6 Asset versus text embeddings

**Question.** Commercial text embeddings of firm names encode industry and other public descriptions. Do they match holdings-based embeddings, and where do the two differ?

**What is done.** Nine text-embedding models embed each firm's Chinese name; the vectors are reduced to $d$ by per-quarter PCA and scored on the same three benchmarks as the asset embeddings.

| Family | Role | Template | Runner |
|---|---|---|---|
| **text_embedding** | Scores the name embeddings (OpenAI, Cohere, Voyage, Gemini, Qwen, BGE; fetched and reduced outside this repository) through RV, RC and MP. | `eval_text_embedding.json` | `eval_text_embedding.sh` |

### §7.1 State ownership and the geometry of asset embeddings

**Question.** Does the embedding, learned from holdings alone, separate state-controlled firms from private ones, and how much of the cross section does that distinction organize?

**What is done.** Each firm-year is labeled state-owned or private from the nature of its ultimate controller in the CSMAR controlling-shareholder file, a label prepared outside this repository. A cross-validated logistic regression recovers the label from the embedding, scored by its AUC per model and dimension, against the four RV characteristics. RV and RC are re-estimated within state-owned and within private firms. The cosine similarity of every firm pair is regressed on same-ownership and same-industry indicators, and the per-quarter coefficients are averaged Fama-MacBeth style over the six quarters.

| Family | Role | Template | Runner |
|---|---|---|---|
| **state_ownership** | The label probe, the grouped RV and RC with `grouping: "categorical"` on `SOE`, and the pairwise-similarity regression for the six models at seven dimensions, plus the characteristics baselines (171 runs). Reuses `main` embeddings. See [Label probe](../how-to/evaluate.md#label-probe), [Grouped evaluation](../how-to/evaluate.md#grouped-evaluation) and [Pairwise similarity](../how-to/evaluate.md#pairwise-similarity). | `eval_state_ownership.json` | `eval_state_ownership.sh` |

### §7.4 Institutional coverage and the informativeness of embeddings

**Question.** Many A-share firms are disclosed by only a handful of institutions. Are the embeddings informative for this thinly covered tail?

**What is done.** Each quarter the firms are sorted into quartiles by the number of disclosing institutions, and RV and RC are re-estimated within each group, next to a characteristics baseline recomputed in the same group.

| Family | Role | Template | Runner |
|---|---|---|---|
| **coverage** | RV and RC grouped by coverage for the six models at seven dimensions, in quantile and fixed-bin modes, plus the characteristics baseline. Reuses `main` embeddings. See [Evaluate](../how-to/evaluate.md#grouped-evaluation). | `eval_coverage.json` | `eval_coverage.sh` |

### §7.5 How much disclosure do asset embeddings need?

**Question.** China reveals full holdings twice a year and a thinner slice in between. If a disclosure rule governed the whole history, how much could it withhold before the finetuned embedding no longer improves on the embedding that was never finetuned under full disclosure?

**What is done.** A disclosure regime is applied to every quarter of the unscreened holdings, the pretraining history and the evaluation quarters alike. On the depth axis a fund reveals its top-$N$ positions by value, while each stock's top-10 holders with at least 0.3 % of its capital are revealed on the firm side; on the breadth axis only a fixed share of institutions discloses. A random control keeps the same share of records by an i.i.d. draw. No co-holding screen is applied: the rule is the screen. Every level of the three ladders trains its own W2V and BERT chain on the 50/50 split ($d=16$ by default) and is scored on RV and RC over the sixteen evaluation quarters.

| Family | Role | Template | Runner |
|---|---|---|---|
| **disclosure_regime** (label `disclosure_regime/{level}/s{seed}`) | Uses the regime panels prepared outside this repository (depth `top{N}`, breadth `keep{r}_s{draw}`, random `rand{r}_s{draw}`), trains one chain per level and training seed, and scores RV and RC for the finetuned and never-finetuned embeddings; `DIM=128` runs the $d=128$ grid. | `train_disclosure_regime.json` · `eval_disclosure_regime.json` | `train_disclosure_regime.sh` · `eval_disclosure_regime.sh` |

### Appendix A: Estimation and computational details

The hyperparameter tables and training curves describe the `main` family, whose BERT settings are the [training setup](#training-setup) above.

### Appendix B: Skip-gram vs CBOW

**Question.** Gabaix et al. use CBOW and the paper uses skip-gram. Does the choice matter, for the stand-alone W2V embedding and for BERT's warm-start?

**What is done.** The W2V objective is switched between skip-gram and CBOW with everything else fixed, and both the W2V embedding and the BERT built on it are scored.

| Family | Role | Template | Runner |
|---|---|---|---|
| **w2v_model_ablation** | CBOW against skip-gram, as the stand-alone embedding and as BERT's warm start. | `train_w2v_model_ablation.json` · `eval_w2v_model_ablation.json` | `train_w2v_model_ablation.sh` · `eval_w2v_model_ablation.sh` |

### Appendix E: Ridge penalty sensitivity

**Question.** Do the RV and RC scores, and the model ranking, depend on the fixed ridge penalty $\lambda = 0.5$?

**What is done.** The main grid is rescored at nine fixed penalties over $[0, 10]$ and under 10-fold within-quarter cross-validation.

| Family | Role | Template | Runner |
|---|---|---|---|
| **ridge_robust** | RV and RC over the penalty sweep and the CV penalty. Reuses `main` embeddings. | `eval_ridge_robust.json` | `eval_ridge_robust.sh` |

### Appendix F: Downstream architecture and encoder initialization in MP

**Question.** Gabaix et al. evaluate BERT with its own encoder but W2V and RS with a linear head (SMLM). Does the downstream architecture drive the MP ranking, and does BERT's warm-started encoder?

**What is done.** The five non-BERT embeddings are scored under both SMLM and the BERT-shape refine procedure (F.1–F.3). F.4 pairs the BERT embedding with a randomly initialized encoder, the start every other model gets.

| Family | Role | Template | Runner |
|---|---|---|---|
| **main** (SMLM rows) | MP under the SMLM head, run alongside the refine-procedure rows of §4. | `eval_main.json` | `eval_main.sh` |
| **mp_bert_random_encoder** | The BERT embedding under a random encoder in the refine procedure (F.4). | `eval_mp_bert_random_encoder.json` | `eval_mp_bert_random_encoder.sh` |

### Appendix G: Mask position sensitivity

**Question.** MP masks the rank-2 holding. Do the model ranking and the shape of the score depend on which rank is masked?

**What is done.** $\mathrm{MP}_k$ is re-estimated at $k \in \{1,2,3,5,10\}$ for every model and dimension.

| Family | Role | Template | Runner |
|---|---|---|---|
| **mask_position** | MP at five masked ranks, under both the refine procedure and SMLM. | `eval_mask_position.json` | `eval_mask_position.sh` |

### Appendix H: Predictive evaluation

**Question.** RV and MP read the embedding and the target in the same quarter. Does the quarter-$t$ embedding remain informative for quarter $t+1$?

**What is done.** Pred-RV regresses the quarter-$(t+1)$ valuation residual on the quarter-$t$ embedding; Pred-MP refines on quarter-$(t+1)$ holdings with the quarter-$t$ embedding frozen. Both use the five adjacent quarter pairs of the main window.

| Family | Role | Template | Runner |
|---|---|---|---|
| **pred** | Pred-RV and Pred-MP. Reuses `main` checkpoints. | `eval_pred.json` | `eval_pred.sh` |

### Appendix I: Text-embedding robustness

**Question.** Does §6 depend on reading firm names in Chinese, or on the per-quarter PCA?

**What is done.** The text comparison is re-run with Chinese and English names and three reductions (PCA fitted once on the pooled cross section, PCA refitted each quarter, native truncation) at all seven dimensions.

| Family | Role | Template | Runner |
|---|---|---|---|
| **text_embedding** | The family of §6, over both languages and all reductions. | `eval_text_embedding.json` | `eval_text_embedding.sh` |

### Appendix J: Extended-window robustness

**Question.** Does the §4 ranking hold over a longer evaluation window, at the cost of a shorter pretraining history?

**What is done.** The whole pipeline is re-run on the 50/50 split and evaluated over sixteen quarters.

| Family | Role | Template | Runner |
|---|---|---|---|
| **timesplit_robust** | Train and evaluate on the 50/50 split. BERT follows the [training setup](#training-setup). | `train_timesplit_robust.json` · `eval_timesplit_robust.json` | `train_timesplit_robust.sh` · `eval_timesplit_robust.sh` |

## Extensions

Families the paper does not report. They reuse the same data, checkpoints and evaluators and are kept as options.

| Family | What it does | Template(s) | Script(s) |
|---|---|---|---|
| **holdout** | RV and RC of the `main` grid on firm splits (`scoring: "holdout"`): 20 seeded 80/20 splits per quarter, scored on the held-out firms, in two penalty tiers run back to back, `fixed` (`ridge_alpha` 0.5) and `cv` (`ridge_alpha` null, RidgeCV inside each training split). The period's value is the median over the splits; per-split scores go to `eval_ridge_splits`. See [Evaluate](../how-to/evaluate.md#holdout-evaluation). | `eval_holdout_main.json` | `eval_holdout_main.sh` |
| **rc_temporal** | Temporal RC of the `main` (2023Q2–2024Q3) and `timesplit_robust` (2020Q4–2024Q3) grids: the RC slope fitted on quarter $T-1$ (`previous`) or the mean of the slopes of every earlier quarter (`expanding`) applied unchanged to quarter $T$ (out-of-period), with the embeddings in raw coordinates or Procrustes-aligned onto the fitted quarter, in the `fixed` and `cv` penalty tiers; six models × seven dimensions × two alignments, the characteristics baseline and the pretrained BERT and Word2Vec embeddings without finetuning, 792 runs. Reuses the pretrained and finetuned embeddings of `main` and `timesplit_robust`. See [Evaluate](../how-to/evaluate.md#temporal-rc). | `eval_rc_temporal.json` | `eval_rc_temporal.sh` |
| **contextual** | BERT contextual (size-weighted) embeddings against the static embedding rows. | `eval_contextual.json` | `eval_contextual.sh` |
| **timesplit_pred** / **timesplit_static** | Predictive and without-finetuning evaluations on the 50/50 checkpoints. | `eval_timesplit_pred.json` · `eval_timesplit_static.json` | `eval_timesplit_pred.sh` · `eval_timesplit_static.sh` |
| **w2v_model_ablation_timesplit** / **mp_bert_random_encoder_timesplit** | Appendix B and Appendix F.4 on the 50/50 split. | `train_w2v_model_ablation_timesplit.json` · `eval_w2v_model_ablation_timesplit.json` · `eval_mp_bert_random_encoder_timesplit.json` | `train_w2v_model_ablation_timesplit.sh` · `eval_w2v_model_ablation_timesplit.sh` · `eval_mp_bert_random_encoder_timesplit.sh` |
| **semiannual_robust** | Train and evaluate on Q2/Q4 quarters only. | `train_semiannual_robust.json` · `eval_semiannual_robust.json` | `train_semiannual_robust.sh` · `eval_semiannual_robust.sh` |
| **semiannual_timesplit** | The Q2/Q4 subset of the 50/50 split. | `train_semiannual_timesplit.json` · `eval_semiannual_timesplit.json` | `train_semiannual_timesplit.sh` · `eval_semiannual_timesplit.sh` |
| **expansion** | Cumulative pretraining: for each quarter $t$, pretrain from scratch on all data before $t$, then finetune on $t$. | `train_expansion.json` · `eval_expansion.json` | `train_expansion.sh` · `eval_expansion.sh` |
| **sliding_window** N{8,20,40} (label `sliding_window/N{8,20,40}`) | Pretrain on only the most recent N quarters (2, 5 or 10 years). | `train_sliding_window_N{8,20,40}.json` · `eval_sliding_window_N{8,20,40}.json` | `train_sliding_window.sh` · `eval_sliding_window.sh` |
