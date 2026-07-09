# Method & experiments

This page maps the manuscript to the code — which file estimates which equation, which config field wires which stage — and then catalogs every experiment family (its purpose, its config template, its run script). The symbols follow the manuscript's notation.

## Method (paper and code)

### The demand system and asset embeddings

Let $h_{i,a,t}$ be the dollar holding of investor $i$ in asset $a$ in quarter $t$. Following Gabaix et al. (2025), a demand system relates holdings to a latent asset vector:

$$h_{i,a,t} = \boldsymbol{\lambda}_i^{\top}\boldsymbol{x}_{a,t} + \delta_i + \delta_a + \epsilon_{i,a,t},$$

where $\boldsymbol{x}_{a,t}$ is a $d$-dimensional latent vector for asset $a$ in quarter $t$ — the *asset embedding*, the sole object the pipeline estimates — $\boldsymbol{\lambda}_i$ captures investor $i$'s preference, and $\delta_i$, $\delta_a$ are investor and asset fixed effects. The economic intuition is that investors assign similar portfolio weights to assets with similar embeddings.

Word2Vec and BERT do not use the holdings matrix directly; they exploit the ordinal information in holdings. For each investor-quarter, holdings are sorted by portfolio weight in descending order into the sequence

$$\pi_{i,t} = \big(a_{i,t}(1),\, a_{i,t}(2),\, \ldots,\, a_{i,t}(L_{i,t})\big),$$

where $a_{i,t}(k)$ is the asset in the $k$-th largest position. With this ordering a portfolio reads as a sentence — a sequence of asset tokens — and larger reported holdings reveal how an investor prioritizes assets within her portfolio.

| Paper | Code |
|---|---|
| Holdings to rank-ordered portfolio sequences $\pi_{i,t}$ | `scripts/data/sharehold.py` — groups holdings by investor/period, sorts by holding proportion, emits the `Portfolio` sequence ([CLI](../reference/cli.md#data-sharehold)) |
| Serialize portfolios for training | `asset_embeddings.datasets.PortfolioDataEncoder` ([API](../reference/api/datasets.md)) |
| Stock/token-id vocabulary | `Tokenizer_Preparer` builds a `PreTrainedTokenizerFast` ([Preparers](preparers.md)) |

### Three models

The pipeline estimates $\boldsymbol{x}_{a,t}$ with three models. They share a common objective — represent each asset as a $d$-dimensional vector recovered from co-holding patterns — but differ in the structure they exploit: a linear factorization of the unordered holdings matrix (recommender system), local window-based co-occurrence in the weight-ranked portfolio (Word2Vec), and full-portfolio, nonlinear context via a transformer (BERT).

#### Recommender system (RS)

Each quarter, principal component analysis is applied to the observed holdings panel $\boldsymbol{D}_t$, and the first $d$ principal components give the embeddings. Because China's disclosure rules are hybrid — firms report only their top-ten shareholders, and mutual funds disclose only their top-ten holdings in Q1/Q3 — an asset that does not appear in an investor's observed portfolio is not necessarily unheld; the investor may simply fall outside the top ten. The entries of $\boldsymbol{D}_t$ therefore distinguish reported from unreported holdings:

$$d_{i,a,t} = \begin{cases}\phi(h_{i,a,t}) & \text{if } a \text{ is reported in } i\text{'s quarter-}t \text{ portfolio},\\ \psi & \text{otherwise},\end{cases}$$

with four $(\phi,\psi)$ variants — RS-Binary, RS-Ranks, RS-Level0, RS-LevelMin — that treat the unreported entries differently. For the level variants, investor and asset fixed effects are removed from the entries before the PCA, so the embeddings reflect active deviations rather than investor wealth or asset size.

Code: `scripts/train/rs.py`, `AssetRSTrainConfig` (`model_type ∈ {RS_Binary, RS_Ranks, RS_Level0, RS_LevelMin}`). RS is fit independently per quarter: its embeddings come from an eigendecomposition rather than an iterative optimization, so there is no initialization through which the pretrain–finetune paradigm below could enter.

#### Word2Vec (W2V)

Words appearing in similar contexts take on similar meanings — "bread" and "butter" often appear together and receive similar vectors. The skip-gram model applies the same idea to portfolios, learning embeddings such that assets held at nearby weights in the same portfolios are close in the embedding space. With context-window radius $w$:

$$\max_{\boldsymbol{E}_\text{in},\,\boldsymbol{E}_\text{out}}\ \sum_{(i,t)\in C}\ \sum_{c=1}^{L_{i,t}}\ \sum_{0<|j-c|\le w} \log P\!\big(a_{i,t}(j)\mid a_{i,t}(c)\big).$$

The trained input matrix $\boldsymbol{E}_{\text{in}}^{*}$ gives the asset embeddings. The paper adopts skip-gram; Gabaix et al. use the continuous-bag-of-words variant, and the `w2v_model_ablation` family below compares the two.

Code: `scripts/train/w2v.py` (gensim Skip-gram/CBOW + negative sampling), `AssetW2VTrainConfig`.

#### BERT

A transformer encoder trained with a masked-language-modeling objective: predict masked assets from the rest of the portfolio. Where Word2Vec sees only a fixed window around each position, BERT conditions on the entire remaining portfolio. With $M_{i,t}$ the masked positions and $\tilde{\pi}_{i,t}$ the masked sequence:

$$\max_{\boldsymbol{E}_\text{emb},\,\theta_\text{enc},\,\boldsymbol{W}_\text{pred}}\ \sum_{(i,t)\in C}\ \mathbb{E}_{M_{i,t}}\sum_{j\in M_{i,t}} \log P_\theta\!\big(a_{i,t}(j)\mid \tilde{\pi}_{i,t}\big).$$

The asset embedding at quarter $t$ is the row $[\boldsymbol{E}_{\text{emb}}^{*}]_a$; the encoder and head are discarded after training.

Code: `scripts/train/bert.py`; `asset_embeddings.modules.BertEmbeddings`; the random masked-position split is `asset_embeddings.datasets.TokenMasker`; `AssetBERTTrainConfig` (mask rate `dataset.mask_prob = 0.15`).

### Pretrain–finetune paradigm

In estimating the Word2Vec and BERT embeddings, the paper departs from Gabaix et al. (2025) in two respects. Both departures answer the same underlying question: how can one estimate latent demand factors when holdings are only partially observed? China's hybrid disclosure rules act as a censoring filter on the true holdings, and the two departures are designed for exactly the places where that censoring binds.

**First departure: pretrain–finetune instead of fitting each quarter independently.** Gabaix et al. train Word2Vec and BERT each quarter on its own. A potential issue is that the resulting embeddings from different quarters are not directly comparable: in the demand system, $\boldsymbol{\lambda}_i^{\top}\boldsymbol{x}_{a,t}$ is invariant to a rotation $\Lambda$ — rotating $\boldsymbol{x}_{a,t}$ and counter-rotating $\boldsymbol{\lambda}_i$ yields the same fit — so each quarter's solution is identified only up to a rotation. When the rotations differ between quarters, the difference $\boldsymbol{x}_{a,t+1} - \boldsymbol{x}_{a,t}$ need not represent a true change in the asset. The pretrain–finetune paradigm makes the rotations more likely to be the same across quarters: a single pretraining on the pooled historical corpus produces a base embedding, and initializing every quarter's estimation from this base makes the quarterly solutions stay, loosely speaking, in one common coordinate frame. In sum, pretraining learns the common, stable holding structure across quarters, while finetuning captures the quarter-specific variations on top of it. (Gabaix et al. find that pretraining does not improve performance in the uniformly disclosed US market; the paper shows this is not the case in China.)

**Second departure: warm-start BERT's embedding layer from Word2Vec.** Rather than initializing at random (cold-start), BERT's embedding layer is set to the trained Word2Vec embeddings at both the pretraining and finetuning stages. The Word2Vec embeddings already place assets with similar co-holding profiles close together, so BERT begins from a meaningful geometry instead of noise. This accelerates convergence and — the point that matters most under censored disclosure — gives thinly observed assets (those appearing in only a handful of reported portfolios in Q1/Q3) a sensible representation from the outset. It follows the standard transfer-learning practice of initializing a richer model from pretrained representations.

Without either departure, the performance of the BERT embeddings generally deteriorates — the `w2v_init_ablation` and `no_pt_init` families below isolate each one.

A BERT here has two trainable parts to initialize — an **embedding layer** (one vector per stock) and a transformer **encoder** — and the two departures are exactly what two fields in the BERT config set (`templates/train_main.json`):

| Stage | `model.model_checkpoint` (seeds the encoder) | `model.w2v_model` = `tokenizer.w2v_model` (seeds the embedding layer) |
|---|---|---|
| **BERT pretrain** | `null` — encoder trained from scratch | `checkpoints/pretrained/AssetW2V/d{dim}/AssetW2V_d{dim}_pretrained.model` |
| **BERT finetune, quarter $t$** | `checkpoints/pretrained/AssetBERT/d{dim}/AssetBERT_d{dim}_pretrained_best/model.safetensors` | `checkpoints/finetune/AssetW2V/d{dim}/{t}/AssetW2V_d{dim}_{t}.model` |

- **`model.model_checkpoint`** seeds the encoder. At pretrain it is `null`, so the encoder is trained from scratch on the pooled corpus. At fine-tune it is the pretrained BERT, so the encoder is carried over and only nudged — the **PT→FT** initialization link of the first departure. (The masked-language-model prediction head is re-initialized each time; it is discarded after training anyway.)
- **`model.w2v_model`** seeds the embedding layer: `AssetBERT_Preparer` copies the vectors of a trained Word2Vec model into BERT's embedding matrix (see [Preparers](preparers.md)) — the **W2V→BERT** warm-start of the second departure. At pretrain it is the W2V trained on the pooled corpus. At fine-tune the seeding runs *after* the checkpoint load, so quarter $t$'s own W2V overwrites the embedding layer while the encoder stays as loaded from the pretrained BERT.
- **`model.embedding_file`** does the same embedding seeding from an embedding CSV instead of a Word2Vec model, and is `null` in the main pipeline.

So each fine-tuned quarter is *the shared pretrained encoder plus that quarter's own embedding layer*, adjusted by a short fine-tune on the quarter's holdings. See [Train](../how-to/train.md) for the full per-stage config. The companion paper studies how well the resulting quarterly embeddings explain firm valuations and return correlations.

### Text-embedding comparison

The commercial-LLM baseline — embedding firm *names* instead of holdings — is the `asset_embeddings.scripts.data.text_embedding` pipeline (OpenAI / Cohere / Voyage / Gemini / Qwen / BGE). See [Text embeddings](../how-to/text-embeddings.md).

## Experiments

Every experiment is a **family**: a [`fast_generator`](../how-to/generate-configs.md) template that expands the config grid, plus a `scripts/run/*.sh` that runs it through the `try_run` + marker mechanism (`-k`/`-e` select subsets). Each family produces its own set of quarterly embedding sequences.

Dimensions are $d \in \{4,8,10,16,32,64,128\}$ throughout; quarters are 2023Q2–2024Q3 for `main` and 2020Q4–2024Q3 for the time-split families.

### Main pipeline

| Family | What it does | Template | Script |
|---|---|---|---|
| **main** | Train W2V, then BERT-PT, then BERT-FT (and RS) across architectures × dims × quarters; the headline embedding sequences. | `train_main.json` | `train_main.sh` |

### Robustness (alternative training designs)

| Family | What it does | Template | Script |
|---|---|---|---|
| **timesplit_robust** | 50/50 pretrain/finetune split (a more balanced alternative to the headline pipeline), extended to 16 quarters. | `train_timesplit_robust.json` | `train_timesplit_robust.sh` |
| **semiannual_robust** | Train on Q2/Q4 only. | `train_semiannual_robust.json` | `train_semiannual_robust.sh` |
| **semiannual_timesplit** | Q2/Q4 subset of the 50/50 time-split corpus. | `train_semiannual_timesplit.json` | `train_semiannual_timesplit.sh` |
| **expansion** | Cumulative pretrain: per quarter $t$, retrain from scratch on all data before $t$, then finetune on $t$. | `train_expansion.json` | `train_expansion.sh` |
| **sliding_window** N{8,20,40} | Pretrain on only the most recent N quarters (2 / 5 / 10 yr). | `train_sliding_window_N{8,20,40}.json` | `train_sliding_window.sh` |

### Ablations (what drives the result)

| Family | What it does | Template | Script |
|---|---|---|---|
| **w2v_init_ablation** | 2×2 factorial over W2V initialization (PT w2v-init × FT w2v-reinit); isolates the **W2V→BERT** warm-start. | `train_w2v_init_ablation.json` | `train_w2v_init_ablation.sh` |
| **no_pt_init** | BERT finetuned from random init each quarter (no pretrained BERT checkpoint); isolates the **PT→FT** initialization. | `train_no_pt_init.json` | `train_no_pt_init.sh` |
| **w2v_model_ablation** (+ `_timesplit`) | CBOW vs. Skip-gram W2V, all else frozen — does the W2V algorithm matter through the BERT chain? | `train_w2v_model_ablation.json` (+ `_timesplit`) | `train_w2v_model_ablation.sh` (+ `_timesplit`) |

### Text-embedding baseline

| Family | What it does | Template | Script |
|---|---|---|---|
| **text_embedding** | LLM embeddings of firm names (OpenAI / Cohere / Voyage / Gemini / Qwen / BGE). | `data_text_embedding.json` | `data_text_embedding.sh` |
