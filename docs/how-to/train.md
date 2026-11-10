# Training

> Part of [AssetEmbeddingsInChina](../index.md). See also [Installation](install.md), [Evaluation](evaluate.md).

Three model families are supported:

- **Recommender Systems (RS)** — traditional methods like PCA, suitable for low-dimensional representations.
- **Word2Vec (W2V)** — shallow neural network that treats each portfolio (holdings sorted by weight from largest to smallest) as a sentence and predicts the assets held alongside each asset within a context window.
- **BERT** — Transformer model that generates context-aware embeddings, suitable for modeling ambiguous scenarios like "Is Apple a tech company or a fruit?"

All training entry points share the same six-flag CLI (also in the [CLI reference](../reference/cli.md#train-eval-shared)):

```bash
uv run python -m asset_embeddings.scripts.train.<model> \
    --config CONFIG [--log [LOG ...]] [--result_file RESULT_FILE] \
    [--experiment EXPERIMENT] [--override KEY=VALUE ...] [--verbose]
```

| Flag | Type | Description |
|---|---|---|
| `--config`, `-c` | `str` | Path to the JSON config file (examples below). **Required.** |
| `--log`, `-l` | `str` (0+) | Log-file path(s). Optional. |
| `--result_file`, `-r` | `str` | SQLite path for structured results. Optional. |
| `--experiment`, `-e` | `str` | Experiment family stored in the `runs` table (default: derived from the config path, e.g. `configs/main/...` gives `main`). The runners pass `-e "$EXP"`. Optional. |
| `--override`, `-o` | `str` (1+) | Override config fields, `key=value` with dot notation (e.g. `train.max_epochs=10 optimizer.learning_rate=0.001`). |
| `--verbose`, `-v` | flag | DEBUG console output (default `INFO`). |

**What a run writes.** Every run writes its effective config as `config.json` into `save_folder`, next to the model files listed under each model below; BERT runs also write TensorBoard scalars under `results/tensorboard/`. With `--result_file`, every run inserts one `runs` row (the config stored once, provenance such as script, git commit and hostname, status, wall time, `peak_ram_mb` / `peak_vram_mb`) and one per-run training row: `train_summary` for BERT (epochs run, best epoch and loss, final losses, the best checkpoint and embedding paths), `train_w2v_summary` or `train_rs_summary` (plus one `train_rs_components` row per component); BERT training also writes one `run_epochs` row per epoch (`train_loss`, `val_loss`, `test_loss`). The checkpoints and embedding CSVs themselves go under `save_folder`, by convention `checkpoints/...`. Table schemas and the read path (`load_runs`, `load_epochs`, `load_details`) are in the [records system](../explanation/records-system.md).

## AssetRS

`asset_embeddings.scripts.train.rs` trains recommender-system-type embedding
models, corresponding to the first method in *Asset Embeddings*. It learns
low-dimensional asset embeddings from an investor-asset matrix using PCA.

```bash
uv run python -m asset_embeddings.scripts.train.rs -c config.json
```

### Config example

```json
{
    "model_type": "RS_Binary",
    "n_components": 4,
    "whiten": false,
    "data_path": "data/processed/MutualFundShareHoldings/finetune/2023Q2.csv",
    "data_format": "csv",
    "id_key": "InvestorID",
    "portfolio_key": "Portfolio",
    "proportion1_key": "Proportion1",
    "proportion2_key": "Proportion2",
    "save_folder": "checkpoints/finetune/AssetRS/RS_Binary/d4/2023Q2",
    "save_name": "AssetRS_Binary_d4_2023Q2",
    "save_format": ".pkl",
    "seed": 42
}
```

### Config fields

| Field             | Type   | Description                                                  |
| ----------------- | ------ | ------------------------------------------------------------ |
| `model_type`      | `str`  | RS embedding model type. Supported: `RS_Binary`, `RS_Ranks`, `RS_Level0`, `RS_LevelMin`. |
| `n_components`    | `int`  | Embedding dimension (target reduced dimension). |
| `whiten`          | `bool` | Whether to whiten the dimensionality-reduction result. |
| `data_path`       | `str`  | Single file or directory (a list is rejected). |
| `data_format`     | `str`  | `csv`, the only accepted value. |
| `id_key`          | `str`  | Investor ID key. |
| `portfolio_key`   | `str`  | Portfolio key. |
| `proportion1_key` | `str`  | Proportion (relative to total shares) key. |
| `proportion2_key` | `str`  | Proportion (relative to tradable A-shares) key. |
| `save_folder`     | `str`  | Save directory. |
| `save_name`       | `str`  | File stem: the run writes `<save_name>_model.pkl` (the fitted PCA) and `<save_name>_embedding.csv`. |
| `save_format`     | `str`  | Must be `.pkl` (the only allowed value). |
| `seed`            | `int`  | Random seed. |

> Note: RS uses proportion data when constructing the investor-asset matrix,
> so the source portfolio CSV must carry the proportion columns.

## AssetW2V

`asset_embeddings.scripts.train.w2v` trains asset-level Word2Vec models.
Holding sequences are treated as sentences and stock codes as words; low-dimensional
representations are learned through contextual co-occurrence. Implemented via
Gensim with Skip-Gram / CBOW, negative sampling, and high-frequency downsampling.

This corresponds to *Asset Word2Vec* in the paper, suitable for modeling asset
co-occurrence structure and initializing more complex models (e.g. AssetBERT).

```bash
uv run python -m asset_embeddings.scripts.train.w2v -c config.json
```

### Config example

```json
{
    "pretrained_model": "checkpoints/pretrained/AssetW2V/d4/AssetW2V_d4_pretrained.model",
    "embedding_dim": 4,
    "epochs": 10,
    "window": 5,
    "min_count": 1,
    "sg": 1,
    "sample": 0.001,
    "negative_sample": 10,
    "seed": 42,
    "data_path": "data/processed/MutualFundShareHoldings/finetune/2023Q2.csv",
    "data_format": "csv",
    "id_key": "InvestorID",
    "portfolio_key": "Portfolio",
    "workers": 1,
    "save_folder": "checkpoints/finetune/AssetW2V/d4/2023Q2",
    "save_name": "AssetW2V_d4_2023Q2"
}
```

### Config fields

| Field              | Type    | Description                                                  |
| ------------------ | ------- | ------------------------------------------------------------ |
| `pretrained_model` | `str`   | Optional `.model` file saved by an earlier W2V run (incremental training); `null` for from-scratch. |
| `embedding_dim`    | `int`   | Word embedding dimension. |
| `epochs`           | `int`   | Number of training epochs. |
| `window`           | `int`   | Context window size. |
| `min_count`        | `int`   | Ignore stocks appearing fewer times than this (recommended: 1). |
| `sg`               | `int`   | Skip-Gram (1) or CBOW (0). |
| `sample`           | `float` | High-frequency downsampling threshold. |
| `negative_sample`  | `int`   | Number of negatives per positive sample. |
| `seed`             | `int`   | Random seed. |
| `data_path`        | `str` or `list[str]` | Input data path(s). |
| `data_format`      | `str`   | `csv`, the only accepted value. |
| `id_key`           | `str`   | Investor ID key. |
| `portfolio_key`    | `str`   | Portfolio key. |
| `workers`          | `int`   | Number of parallel threads. |
| `save_folder`      | `str`   | Save directory. |
| `save_name`        | `str`   | File stem: the run writes the model as `<save_name>.model` and the embeddings as `<save_name>_embedding.csv`. |

> W2V does not consume proportion data, so the `proportion*_key` fields are
> not present.

## AssetBERT

AssetBERT is a Transformer-based embedding model that learns context-aware
representations (contextualized asset embeddings) of assets from holding
sequences. It is the AssetBERT model of the paper (shortened to BERT in
figures and tables).

We use AssetW2V embeddings to initialize BERT's embedding layer; the
tokenizer vocabulary (excluding special tokens) is shared with AssetW2V.

```bash
uv run python -m asset_embeddings.scripts.train.bert -c config.json
```

### Config example

The finetune stage of `templates/train_main.json` for d4 and 2023Q2: it starts from the pretrained BERT checkpoint and that quarter's W2V model, and it extracts contextualized embeddings.

```json
{
    "model": {
        "model_type": "BERT",
        "model_checkpoint": "checkpoints/pretrained/AssetBERT/d4/AssetBERT_d4_pretrained_best/model.safetensors",
        "w2v_model": "checkpoints/finetune/AssetW2V/d4/2023Q2/AssetW2V_d4_2023Q2.model",
        "embedding_file": null,
        "vocab_size": 5236,
        "hidden_size": 4,
        "num_hidden_layers": 4,
        "num_attention_heads": 2,
        "intermediate_size": 512,
        "max_position_embeddings": 256,
        "type_vocab_size": 1,
        "freeze_embedding": false,
        "freeze_encoder": false
    },
    "tokenizer": {
        "w2v_model": "checkpoints/finetune/AssetW2V/d4/2023Q2/AssetW2V_d4_2023Q2.model",
        "vocab_file": null,
        "pretrained_tokenizer_file": null,
        "alias_file": null
    },
    "dataset": {
        "data_path": "data/processed/MutualFundShareHoldings/finetune/2023Q2.csv",
        "data_format": "csv",
        "id_key": "InvestorID",
        "portfolio_key": "Portfolio",
        "proportion1_key": "Proportion1",
        "proportion2_key": "Proportion2",
        "include_proportion": true,
        "max_length": 256,
        "num_repeats": 1,
        "mask_prob": 0.15,
        "mask_indices": null
    },
    "dataloader": {
        "batch_size": 32,
        "shuffle": true,
        "num_workers": 4,
        "persistent_workers": true,
        "pin_memory": true,
        "drop_last": false
    },
    "optimizer": {
        "optimizer_type": "PagedAdam8bit",
        "optimizer_kwargs": null,
        "learning_rate": 0.001,
        "lr_scheduler_type": "cosine",
        "lr_scheduler_warmup_steps": 0,
        "lr_scheduler_train_steps": null,
        "lr_scheduler_num_cycles": 0.5,
        "lr_scheduler_power": 0
    },
    "accelerator": {
        "mixed_precision": "auto",
        "gradient_accumulation_steps": 1,
        "accelerator_checkpoint": null,
        "use_compile": true,
        "compile_backend": "inductor",
        "compile_mode": "default",
        "compile_fullgraph": false,
        "compile_dynamic": false,
        "compile_cache_dir": null,
        "allow_tf32": true
    },
    "train": {
        "max_epochs": 30,
        "validation_split": null,
        "split_method": "random",
        "best_metric": "train_loss",
        "clip_grad_norm": 1.0,
        "clip_grad_value": null,
        "detect_anomaly": false,
        "log_every_steps": 200,
        "report_every_epochs": 1,
        "calculate_contextualized_embeddings": true,
        "save_every_epochs": 1,
        "save_folder": "checkpoints/finetune/AssetBERT/d4/2023Q2",
        "save_name": "AssetBERT_d4_2023Q2",
        "save_format": ".safetensors",
        "seed": 42,
        "keep_best_only": true
    }
}
```

### Config fields

The config is in JSON format and groups fields into modules:

- `model`: model structure and initialization.
- `tokenizer`: asset sequence encoding.
- `dataset`: dataset definition and preprocessing.
- `dataloader`: PyTorch DataLoader behavior.
- `optimizer`: optimizer and LR scheduler.
- `accelerator`: HuggingFace Accelerator, mixed precision, `torch.compile`.
- `train`: training control, validation/test split, early stopping.

| Module         | Field                         | Type        | Description                                                                 |
| -------------- | ----------------------------- | ----------- | --------------------------------------------------------------------------- |
| `model`        | `model_type`                  | `str`       | Fixed as `"BERT"`. |
|                | `model_checkpoint`            | `str`       | Pretrained path; `null` from scratch. |
|                | `w2v_model`                   | `str`       | Word2Vec model for vocab + embedding init; `null` for random. |
|                | `embedding_file`              | `str`       | Embedding init CSV; `null` if not used. |
|                | `vocab_size`                  | `int`       | Asset vocabulary size. |
|                | `hidden_size`                 | `int`       | BERT embedding dim. |
|                | `num_hidden_layers`           | `int`       | Number of Transformer encoder layers. |
|                | `num_attention_heads`         | `int`       | Number of attention heads; must divide `hidden_size`. |
|                | `intermediate_size`           | `int`       | FFN hidden size; the project uses 512 for every embedding dimension. |
|                | `max_position_embeddings`     | `int`       | Max sequence length, `[CLS]` and `[SEP]` included. |
|                | `type_vocab_size`             | `int`       | Number of token-type (segment) ids. Set to 1. |
|                | `freeze_embedding`            | `bool`      | Freeze embedding-layer weights. |
|                | `freeze_encoder`              | `bool`      | Freeze encoder; fine-tune embeddings + LM head only. |
| `tokenizer`    | `w2v_model`                   | `str`       | Word2Vec path for vocab/index mapping. |
|                | `vocab_file`                  | `str`       | HuggingFace-format vocab file. |
|                | `pretrained_tokenizer_file`   | `str`       | Saved tokenizer directory, loaded with `from_pretrained`. Checked first, then `vocab_file`, `w2v_model`, `embedding_file`. |
|                | `embedding_file`              | `str`       | Embedding-init CSV consistent with `model.embedding_file`. |
|                | `alias_file`                  | `str`       | Reserved; nothing reads it, so it has no effect. |
| `dataset`      | `data_path`                   | `str` or `list[str]` | Portfolio CSV file(s) or directories, or a corpus directory written by `PortfolioCorpus.save` (opened memory-mapped). |
|                | `data_format`                 | `str`       | `csv`, the only accepted value. |
|                | `id_key`                      | `str`       | Investor ID key. |
|                | `portfolio_key`               | `str`       | Portfolio key. |
|                | `proportion1_key`             | `str`       | Proportion (total shares) key. |
|                | `proportion2_key`             | `str`       | Proportion (tradable A-shares) key. |
|                | `include_proportion`          | `bool`      | Fill each sample's `proportions1` / `proportions2` tensors from the `proportion*_key` columns (zeros when false). |
|                | `max_length`                  | `int`       | Sequence length, `[CLS]` and `[SEP]` included, so at most `max_length - 2` holdings; longer portfolios are cut (their smallest holdings, since portfolios are ordered by proportion; the cut drops `[SEP]`). The dataset logs how many portfolios are cut and how many holdings carry a code the tokenizer does not know (they become `[UNK]`). Should match `model.max_position_embeddings`. |
|                | `mask_prob`                   | `float`     | MLM mask ratio. Mutually exclusive with `mask_indices`. Training samples draw fresh masks every epoch; validation and test samples are masked from a generator seeded by `train.seed` and the sample's index, so every evaluation of a held-out loss sees the same masked inputs and epochs compare on equal terms. |
|                | `mask_indices`                | `list[int]` | Fixed mask positions. Mutually exclusive with `mask_prob`. |
|                | `num_repeats`                 | `int`       | Times the training split is replayed per epoch, through the sampler; the validation and test splits are not repeated. Default: `1`. |
| `dataloader`   | `batch_size`                  | `int`       | Batch size. |
|                | `shuffle`                     | `bool`      | Whether to shuffle. |
|                | `num_workers`                 | `int`       | Loader processes. |
|                | `persistent_workers`          | `bool`      | Reduce per-epoch worker restart. Default: `true`. |
|                | `pin_memory`                  | `bool`      | Pin to RAM for GPU transfer. Default: `true`. |
|                | `drop_last`                   | `bool`      | Drop last incomplete batch. Default: `false`. |
| `optimizer`    | `optimizer_type`              | `str`       | `Adam`, `AdamW`, `Adam8bit`, `PagedAdam8bit`, `AdamW8bit`, `PagedAdamW8bit`, `Lion`, `Lion8bit`, `PagedLion8bit`, `DAdaptation`, `SGDNesterov`, `SGD8bit`, `Adafactor`. |
|                | `optimizer_kwargs`            | `dict`      | Optional extras. |
|                | `learning_rate`               | `float`     | Initial LR. |
|                | `lr_scheduler_type`           | `str`       | `constant`, `linear`, `cosine`, `cosine_with_restarts`, `polynomial`, `adafactor`. |
|                | `lr_scheduler_warmup_steps`   | `int`       | Warmup steps. |
|                | `lr_scheduler_train_steps`    | `int`       | Total steps; `null` auto-computes. |
|                | `lr_scheduler_num_cycles`     | `float`     | For cosine schedulers. |
|                | `lr_scheduler_power`          | `float`     | For polynomial scheduler. |
| `accelerator`  | `mixed_precision`             | `str`       | `'no'`, `'fp16'`, `'bf16'`, `'fp8'`, `'auto'` (`'auto'` leaves it to the accelerate default). |
|                | `gradient_accumulation_steps` | `int`       | Grad accumulation. |
|                | `accelerator_checkpoint`      | `str`       | Accelerator checkpoint path. |
|                | `use_compile`                 | `bool`      | Enable `torch.compile`. |
|                | `compile_backend`             | `str`       | `"inductor"`, `"eager"`, `"aot_eager"`. |
|                | `compile_mode`                | `str`       | `"default"`, `"reduce-overhead"`, `"max-autotune"`. |
|                | `compile_fullgraph`           | `bool`      | Require full-graph compile. |
|                | `compile_dynamic`             | `bool`      | Enable dynamic shapes. |
|                | `compile_cache_dir`           | `str`       | Persistent compile cache. |
|                | `allow_tf32`                  | `bool`      | TF32 on Ampere/Ada. |
| `train`        | `max_epochs`                  | `int`       | Max training epochs (at least 1). |
|                | `validation_split`            | `float`     | Validation share, e.g. `0.1`; `null` or `0` means no validation split. AssetBERT's `train` block (`AssetBERTTrainConfig`, pretraining and finetuning alike) has no test split; the MP `train` block (`MPTrainConfig`) adds `test_split`. |
|                | `split_method`                | `str`       | `"random"` or `"sequential"`. |
|                | `best_metric`                 | `str`       | `"train_loss"` or `"val_loss"`; `"val_loss"` needs `validation_split > 0`, checked when the config loads. |
|                | `clip_grad_norm`              | `float`     | L2 grad-clip. |
|                | `clip_grad_value`             | `float`     | Abs grad-clip. |
|                | `detect_anomaly`              | `bool`      | Anomaly detection (debugging). |
|                | `log_every_steps`             | `int`       | Log the training loss every n steps. |
|                | `report_every_epochs`         | `int`       | Log the epoch losses and write them to TensorBoard every n epochs. |
|                | `calculate_contextualized_embeddings` | `bool` | Extract size-weighted contextualized embeddings post-training. The data must carry both proportion columns; the extraction pass turns proportions on by itself. |
|                | `save_every_epochs`           | `int`       | Save a checkpoint (and the embeddings) every n epochs. |
|                | `keep_best_only`              | `bool`      | Drop non-best checkpoints/embeddings at end. |
|                | `save_folder`                 | `str`       | Save directory. |
|                | `save_name`                   | `str`       | Filename (no extension). |
|                | `save_format`                 | `str`       | `.safetensors` or `.pt`, for the per-epoch accelerate state. The best model is always saved as `model.safetensors`, so `.pt` is incompatible with `calculate_contextualized_embeddings` (the extraction looks for `model.pt`). |
|                | `seed`                        | `int`       | Random seed. |
|                | `early_stop_enabled`          | `bool`      | Enable early stopping. |
|                | `early_stop_patience`         | `int`       | Patience epochs. |
|                | `early_stop_min_delta`        | `float`     | Improvement threshold (rel if `early_stop_relative=true`). |
|                | `early_stop_mode`             | `str`       | `"min"` / `"max"`. |
|                | `early_stop_relative`         | `bool`      | Relative vs absolute improvement. |
|                | `early_stop_metric`           | `str`       | `"train_loss"` or `"val_loss"`, with the same split rule as `best_metric`. |

### Static vs. contextualized embeddings

AssetBERT provides two complementary asset representations.

**Static embeddings** are read directly from the trained embedding layer.
Each token maps to a fixed vector, analogous to standard BERT word embeddings,
summarizing asset-level characteristics across all training data. Vocabulary
matches AssetW2V (modulo BERT special tokens).

**Contextualized embeddings** are derived from BERT hidden states processing
real investor holding sequences. Following Gabaix et al. (2024), we compute the
size-weighted average

$$\frac{\sum_i S_{i,a} \cdot x_{i,a}}{\sum_i S_{i,a}}$$

over investors $i$ for each asset $a$, where:

- $x_{i,a}$ is the BERT hidden state for asset $a$ in investor $i$'s sequence,
- $S_{i,a}$ is the investor's shareholding proportion relative to tradable
  A-shares (the `proportion2_key` field).

An asset that appears in no portfolio keeps its static embedding.

To enable contextualized extraction:

1. Set `train.calculate_contextualized_embeddings=true`.
2. Make sure the data carries both proportion columns named by `dataset.proportion1_key` and `dataset.proportion2_key`. The extraction pass reads them itself, whatever `dataset.include_proportion` says.
3. Keep `train.save_format` at `.safetensors`.

Both kinds of embeddings are saved as CSV with `Token` plus `Embed_1...Embed_N`
columns. Naming distinguishes them:

- Static: `{save_name}_best_embedding.csv`
- Contextualized: `{save_name}_best_contextual_embedding.csv`

### Outputs

Everything goes under `train.save_folder`:

- `{save_name}_best/`: the best epoch's HuggingFace model (`config.json`, `model.safetensors`); a finetune run loads `{save_name}_best/model.safetensors` as `model.model_checkpoint`.
- `{save_name}_best_embedding.csv` and, when enabled, `{save_name}_best_contextual_embedding.csv`.
- `{save_name}_accelerator_epoch{n}/` (accelerate state) and `{save_name}_epoch{n}_embedding.csv` for each saved epoch; `keep_best_only` deletes them at the end.
- `config.json`: the effective config.
