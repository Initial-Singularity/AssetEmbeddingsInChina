# Method & experiments

This page maps the manuscript to the code — which file estimates which equation, which config field wires
which stage — and then catalogs every experiment family (its purpose, its config template, its run
script). The symbols follow the manuscript's notation.

## Method (paper and code)

### Portfolios as sentences

Each investor's quarter-$t$ portfolio is a rank-ordered sequence of holdings — a *sentence* — and each
stock is a *token*. Rank order (largest holding first) carries economic meaning, just as word order does
in language. This is what lets the asset-pricing problem be attacked with NLP architectures.

| Paper | Code |
|---|---|
| Holdings to rank-ordered portfolio sequences | `scripts/data/sharehold.py` — groups holdings by investor/period, sorts by holding proportion, emits the `Portfolio` sequence ([CLI](../reference/cli.md#data-sharehold)) |
| Serialize portfolios for training | `asset_embeddings.datasets.PortfolioDataEncoder` ([API](../reference/api/datasets.md)) |
| Stock/token-id vocabulary | `Tokenizer_Preparer` builds a `PreTrainedTokenizerFast` ([Preparers](preparers.md)) |

### Three architectures

The repo estimates the embedding sequence $\{\boldsymbol{x}_{a,t}\}$ with three architectures.

#### Recommender system (RS)

A truncated SVD of the demeaned holdings matrix $\tilde{\boldsymbol{H}}_t$:

$$\tilde{\boldsymbol{H}}_t = \boldsymbol{U}_t\,\boldsymbol{\Sigma}_t\,\boldsymbol{V}_t^{\top},
\qquad \boldsymbol{X}_t^{\text{RS}} = \boldsymbol{V}_{t,d}\,\boldsymbol{\Sigma}_{t,d}.$$

The four $(\phi,\psi)$ encodings (RS-Binary / RS-Ranks / RS-Level0 / RS-LevelMin) are the
`model_type` choices.

Code: `scripts/train/rs.py`, `AssetRSTrainConfig` (`model_type ∈ {RS_Binary, RS_Ranks, RS_Level0, RS_LevelMin}`).
RS is fit independently per quarter and stands outside the coupling below.

#### Word2Vec (W2V)

Skip-gram over the rank-ordered portfolio sequences:

$$\mathcal{L}_{\text{W2V}} = -\sum_{(i,t)\in D}\ \sum_{c=1}^{L_{i,t}}\ \sum_{0<|j-c|\le w}
\log \mathbb{P}\!\big(a_{i,t}(j)\mid a_{i,t}(c)\big).$$

The trained input matrix $\boldsymbol{E}_{\text{in}}^{*}$ gives the asset embeddings.

Code: `scripts/train/w2v.py` (gensim Skip-gram/CBOW + negative sampling), `AssetW2VTrainConfig`.

#### BERT

A Transformer encoder trained with a masked-language-modeling objective at mask rate $p_{\text{mask}}=0.15$:

$$\mathcal{L}_{\text{MLM}} = -\sum_{(i,t)\in D}\ \mathbb{E}_{M_{i,t}}\sum_{j\in M_{i,t}}
\log \mathbb{P}_\theta\!\big(a_{i,t}(j)\mid \tilde{\pi}_{i,t}\big).$$

The asset embedding at quarter $t$ is the row $[\boldsymbol{E}_{\text{emb}}^{*}]_a$; the encoder and head
are discarded after training.

Code: `scripts/train/bert.py`; `asset_embeddings.modules.BertEmbeddings`; the random masked-position split is
`asset_embeddings.datasets.TokenMasker`; `AssetBERTTrainConfig`.

### Pretrain–finetune coupling

The aim is a *time-varying* embedding: one vector per stock per quarter, such that the change from one
quarter to the next is itself informative. This runs into an identifiability problem. An embedding learned
by a neural model is only pinned down up to an arbitrary rotation, reflection, and relabelling of its
axes — train on the same data twice from different random starting points and you get two embeddings that
are equally valid yet look entirely different. So if each quarter were trained independently, the
quarter-to-quarter difference $\boldsymbol{x}_{a,t+1}-\boldsymbol{x}_{a,t}$ (stock $a$'s vector at $t{+}1$
minus its vector at $t$) would mostly measure that arbitrary coordinate drift, not any real change in the
stock.

The remedy is to give every quarter the **same starting point**. The model is *pretrained* once on the pooled earlier-quarter corpus, producing a single base solution; each subsequent quarter is then *fine-tuned* from that shared base rather than from scratch. Because all quarters set out from the same anchor and move only a little, their embeddings stay in a common coordinate frame, and $\boldsymbol{x}_{a,t+1}-\boldsymbol{x}_{a,t}$ comes to reflect genuine movement of the stock.

A BERT here has two trainable parts to initialize — an **embedding layer** (one vector per stock) and a
Transformer **encoder** — and "starting from the shared base" is exactly what two fields in the BERT
config set (`templates/train_main.json`):

| Stage | `model.model_checkpoint` (seeds the encoder) | `model.w2v_model` = `tokenizer.w2v_model` (seeds the embedding layer) |
|---|---|---|
| **BERT pretrain** | `null` — encoder trained from scratch | `checkpoints/pretrained/AssetW2V/d{dim}/AssetW2V_d{dim}_pretrained.model` |
| **BERT finetune, quarter $t$** | `checkpoints/pretrained/AssetBERT/d{dim}/AssetBERT_d{dim}_pretrained_best/model.safetensors` | `checkpoints/finetune/AssetW2V/d{dim}/{t}/AssetW2V_d{dim}_{t}.model` |

- **`model.model_checkpoint`** seeds the encoder. At pretrain it is `null`, so the encoder is trained from
  scratch on the pooled corpus. At fine-tune it is the pretrained BERT, so the encoder is carried over and
  only nudged — this is the **PT-to-FT** link, the part that keeps every quarter's solution tied to the
  same shared frame. (The masked-language-model prediction head is re-initialized each time; it is
  discarded after training anyway.)
- **`model.w2v_model`** seeds the embedding layer: `AssetBERT_Preparer` copies the vectors of a trained
  Word2Vec model into BERT's embedding matrix (see [Preparers](preparers.md)) — the **W2V-to-BERT** link.
  At pretrain it is the W2V trained on the pooled corpus. At fine-tune the seeding runs *after* the
  checkpoint load, so quarter $t$'s own W2V overwrites the embedding layer while the encoder stays as
  loaded from the pretrained BERT.
- **`model.embedding_file`** does the same embedding seeding from an embedding CSV instead of a Word2Vec
  model, and is `null` in the main pipeline.

So each fine-tuned quarter is *the shared pretrained encoder plus that quarter's own embedding layer*, adjusted by a short fine-tune on the quarter's holdings. The shared encoder is what holds the per-quarter embeddings in one comparable coordinate frame across time. See [Train](../how-to/train.md) for the full per-stage config. The companion paper studies how well the resulting quarterly embeddings explain firm valuations and return correlations.

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
| **w2v_init_ablation** | 2×2 factorial over W2V initialization (PT w2v-init × FT w2v-reinit); isolates the **W2V-to-BERT** coupling. | `train_w2v_init_ablation.json` | `train_w2v_init_ablation.sh` |
| **no_pt_init** | BERT finetuned from random init each quarter (no pretrained BERT checkpoint); isolates the **PT-to-FT** coupling. | `train_no_pt_init.json` | `train_no_pt_init.sh` |
| **w2v_model_ablation** (+ `_timesplit`) | CBOW vs. Skip-gram W2V, all else frozen — does the W2V algorithm matter through the BERT chain? | `train_w2v_model_ablation.json` (+ `_timesplit`) | `train_w2v_model_ablation.sh` (+ `_timesplit`) |

### Text-embedding baseline

| Family | What it does | Template | Script |
|---|---|---|---|
| **text_embedding** | LLM embeddings of firm names (OpenAI / Cohere / Voyage / Gemini / Qwen / BGE). | `data_text_embedding.json` | `data_text_embedding.sh` |
