# 训练

> 隶属于 [AssetEmbeddings](../index.md)。另见 [安装](install.md)、[数据获取](prepare-data.md)。

支持三个模型族：

- **推荐系统（Recommender Systems, RS）**——PCA 等传统方法，适合低维表示。
- **Word2Vec（W2V）**——浅层神经网络，利用投资者持仓的顺序（按权重从大到小排序）来预测“被遮蔽”的资产。
- **BERT**——Transformer 模型，生成上下文感知的嵌入，适合刻画“Apple 是科技公司还是水果？”这类含歧义的情形。

所有训练入口共享同一套五参数 CLI（亦见 [CLI 参考](../reference/cli.md#train-eval-shared)）：

```bash
uv run python -m asset_embeddings.scripts.train.<model> \
    --config CONFIG [--log [LOG ...]] [--result_file RESULT_FILE] \
    [--override KEY=VALUE ...] [--verbose]
```

| Flag | Type | 说明 |
|---|---|---|
| `--config`, `-c` | `str` | JSON 配置文件路径（示例见下）。**必填。** |
| `--log`, `-l` | `str`（0+） | 日志文件路径。可选。 |
| `--result_file`, `-r` | `str` | 存放结构化结果的 SQLite 路径。可选。 |
| `--override`, `-o` | `str`（1+） | 覆盖配置字段，`key=value` 且支持点号嵌套（如 `train.max_epoches=10 optimizer.learning_rate=0.001`）。 |
| `--verbose`, `-v` | flag | DEBUG 级控制台输出（默认 `INFO`）。 |

指定 `--result_file` 时，每次运行还会经由 [记录系统](../explanation/records-system.md) 写入一份逐运行的训练摘要（`train_{bert,w2v,rs}_recordings`）。

## AssetRS

`asset_embeddings.scripts.train.rs` 训练推荐系统类的嵌入模型，对应 *Asset Embeddings* 中的第一种方法。它用 PCA 从投资者-资产矩阵中学习低维资产嵌入。

```bash
uv run python -m asset_embeddings.scripts.train.rs -c config.json
```

### 配置示例

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

### 配置字段

| Field             | Type   | 说明                                                  |
| ----------------- | ------ | ------------------------------------------------------------ |
| `model_type`      | `str`  | RS 嵌入模型类型。支持：`RS_Binary`、`RS_Ranks`、`RS_Level0`、`RS_LevelMin`。 |
| `n_components`    | `int`  | 嵌入维度（目标降维维数）。 |
| `whiten`          | `bool` | 是否对降维结果做白化。 |
| `data_path`       | `str` 或 `list[str]` | 单个文件、目录，或文件路径列表。 |
| `data_format`     | `str`  | 输入数据格式：`csv`、`json`、`binary`。 |
| `id_key`          | `str`  | 投资者 ID 键。 |
| `portfolio_key`   | `str`  | 组合键。 |
| `proportion1_key` | `str`  | 比例（相对总股本）键。 |
| `proportion2_key` | `str`  | 比例（相对流通 A 股）键。 |
| `save_folder`     | `str`  | 保存目录。 |
| `save_name`       | `str`  | 模型保存文件名（无扩展名）。 |
| `save_format`     | `str`  | 推荐 `.pkl`。 |
| `seed`            | `int`  | 随机种子。 |

> 注：RS 在构造投资者-资产矩阵时会用到比例数据，因此源数据必须是在 `asset_embeddings.scripts.data.sharehold` 中带 `--include_proportion True` 标志生成的。

## AssetW2V

`asset_embeddings.scripts.train.w2v` 训练资产级 Word2Vec 模型。持仓序列被当作句子、股票代码当作词，通过上下文共现学习低维表示。基于 Gensim 实现，支持 Skip-Gram / CBOW、负采样与高频下采样。

它对应论文中的 *Asset Word2Vec*，适合刻画资产共现结构、并为更复杂的模型（如 AssetBERT）做初始化。

```bash
uv run python -m asset_embeddings.scripts.train.w2v -c config.json
```

### 配置示例

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
    "save_folder": "checkpoints/finetune/AssetW2V/d4",
    "save_name": "AssetW2V_d4_2023Q2",
    "save_format": ".model"
}
```

### 配置字段

| Field              | Type    | 说明                                                  |
| ------------------ | ------- | ------------------------------------------------------------ |
| `pretrained_model` | `str`   | 可选的预训练模型路径（增量训练）；`null` 表示从零训练。 |
| `embedding_dim`    | `int`   | 词嵌入维度。 |
| `epochs`           | `int`   | 训练轮数。 |
| `window`           | `int`   | 上下文窗口大小。 |
| `min_count`        | `int`   | 忽略出现次数少于此值的股票（推荐：1）。 |
| `sg`               | `int`   | Skip-Gram（1）或 CBOW（0）。 |
| `sample`           | `float` | 高频下采样阈值。 |
| `negative_sample`  | `int`   | 每个正样本对应的负样本数。 |
| `seed`             | `int`   | 随机种子。 |
| `data_path`        | `str` 或 `list[str]` | 输入数据路径。 |
| `data_format`      | `str`   | `csv`、`json`、`binary`。 |
| `id_key`           | `str`   | 投资者 ID 键。 |
| `portfolio_key`    | `str`   | 组合键。 |
| `workers`          | `int`   | 并行线程数。 |
| `save_folder`      | `str`   | 保存目录。 |
| `save_name`        | `str`   | 保存文件名（无扩展名）。 |
| `save_format`      | `str`   | `.model` 或 `.txt`。 |

> W2V 不消费比例数据，因此不存在 `proportion*_key` 字段。

## AssetBERT

AssetBERT 是基于 Transformer 的嵌入模型，从持仓序列中学习资产的上下文感知表示（contextualized asset embeddings）。即论文中的 AssetBERT（图表中简写为 BERT）。

我们用 AssetW2V 嵌入初始化 BERT 的嵌入层；分词器词表（不含特殊词元）与 AssetW2V 共享。

```bash
uv run python -m asset_embeddings.scripts.train.bert -c config.json
```

### 配置示例

```json
{
    "model": {
        "model_type": "BERT",
        "model_checkpoint": null,
        "w2v_model": "checkpoints/finetune/AssetW2V/d4/2023Q2/AssetW2V_d4_2023Q2.model",
        "embedding_file": null,
        "vocab_size": 5236,
        "hidden_size": 4,
        "num_hidden_layers": 4,
        "num_attention_heads": 2,
        "intermediate_size": 512,
        "max_position_embeddings": 64,
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
        "include_proportion": false,
        "max_length": 64,
        "mask_prob": 0.15,
        "mask_indices": null,
        "num_repeats": 1,
        "cache_size": 400
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
        "use_compile": false,
        "compile_backend": "inductor",
        "compile_mode": "reduce-overhead",
        "compile_fullgraph": false,
        "compile_dynamic": false,
        "compile_cache_dir": null,
        "allow_tf32": true
    },
    "train": {
        "max_epoches": 30,
        "validation_split": 0.1,
        "test_split": 0.1,
        "split_method": "random",
        "best_metric": "val_loss",
        "clip_grad_norm": null,
        "clip_grad_value": null,
        "detect_anomaly": false,
        "check_per_step": 200,
        "report_per_epoch": 1,
        "calculate_contextualized_embeddings": true,
        "save_per_epoch": 1,
        "save_folder": "checkpoints/finetune/AssetBERT/d4/2023Q2",
        "save_name": "AssetBERT_d4_2023Q2",
        "save_format": ".safetensors",
        "seed": 42,
        "early_stop_enabled": false,
        "early_stop_patience": 5,
        "early_stop_min_delta": 0.0,
        "early_stop_mode": "min",
        "early_stop_relative": true,
        "early_stop_metric": "val_loss"
    }
}
```

### 配置字段

配置为 JSON 格式，按模块分组字段：

- `model`：模型结构与初始化。
- `tokenizer`：资产序列编码。
- `dataset`：数据集定义与预处理。
- `dataloader`：PyTorch DataLoader 行为。
- `optimizer`：优化器与学习率调度。
- `accelerator`：HuggingFace Accelerator、混合精度、`torch.compile`。
- `train`：训练控制、验证/测试切分、早停。

| 模块         | Field                         | Type        | 说明                                                                 |
| -------------- | ----------------------------- | ----------- | --------------------------------------------------------------------------- |
| `model`        | `model_type`                  | `str`       | 固定为 `"BERT"`。 |
|                | `model_checkpoint`            | `str`       | 预训练路径；`null` 表示从零训练。 |
|                | `w2v_model`                   | `str`       | 用于词表 + 嵌入初始化的 Word2Vec 模型；`null` 表示随机初始化。 |
|                | `embedding_file`              | `str`       | 嵌入初始化 CSV；不用则为 `null`。 |
|                | `vocab_size`                  | `int`       | 资产词表大小。 |
|                | `hidden_size`                 | `int`       | BERT 嵌入维度。 |
|                | `num_hidden_layers`           | `int`       | Transformer 编码器层数。 |
|                | `num_attention_heads`         | `int`       | 注意力头数；必须整除 `hidden_size`。 |
|                | `intermediate_size`           | `int`       | FFN 隐层大小，通常为 `hidden_size * 4`。 |
|                | `max_position_embeddings`     | `int`       | 最大序列长度（每个投资者的最大持仓数）。 |
|                | `type_vocab_size`             | `int`       | 类型嵌入维度。设为 1。 |
|                | `freeze_embedding`            | `bool`      | 冻结嵌入层权重。 |
|                | `freeze_encoder`              | `bool`      | 冻结编码器；仅微调嵌入层 + LM 头。 |
| `tokenizer`    | `w2v_model`                   | `str`       | 用于词表/索引映射的 Word2Vec 路径。 |
|                | `vocab_file`                  | `str`       | HuggingFace 格式的词表文件。 |
|                | `embedding_file`              | `str`       | 与 `model.embedding_file` 一致的嵌入初始化 CSV。 |
|                | `alias_file`                  | `str`       | 可选的 `{ticker: name}` 别名映射。 |
| `dataset`      | `data_path`                   | `str` 或 `list[str]` | 输入路径。 |
|                | `data_format`                 | `str`       | `csv` / `json` / `binary`。 |
|                | `id_key`                      | `str`       | 投资者 ID 键。 |
|                | `portfolio_key`               | `str`       | 组合键。 |
|                | `proportion1_key`             | `str`       | 比例（总股本）键。 |
|                | `proportion2_key`             | `str`       | 比例（流通 A 股）键。 |
|                | `include_proportion`          | `bool`      | 每条目输出 `(id, proportion)` 元组。需要 `proportion*_key`。 |
|                | `max_length`                  | `int`       | 每个投资者的最大持仓数；应与 `model.max_position_embeddings` 一致。 |
|                | `mask_prob`                   | `float`     | MLM 遮蔽比例。与 `mask_indices` 互斥。 |
|                | `mask_indices`                | `list[int]` | 固定遮蔽位置。与 `mask_prob` 互斥。 |
|                | `num_repeats`                 | `int`       | 每个 epoch 重复数据集的次数。 |
|                | `cache_size`                  | `int`       | 流式加载器缓存大小。 |
| `dataloader`   | `batch_size`                  | `int`       | 批大小。 |
|                | `shuffle`                     | `bool`      | 是否打乱。 |
|                | `num_workers`                 | `int`       | 加载进程数。 |
|                | `persistent_workers`          | `bool`      | 减少每个 epoch 的 worker 重启。默认：`true`。 |
|                | `pin_memory`                  | `bool`      | 锁页内存以加速 GPU 传输。默认：`true`。 |
|                | `drop_last`                   | `bool`      | 丢弃最后一个不完整批。默认：`false`。 |
| `optimizer`    | `optimizer_type`              | `str`       | `Adam`、`AdamW`、`Adam8bit`、`PagedAdam8bit`、`AdamW8bit`、`PagedAdamW8bit`、`Lion`、`Lion8bit`、`PagedLion8bit`、`DAdaptation`、`SGDNesterov`、`SGD8bit`、`Adafactor`。 |
|                | `optimizer_kwargs`            | `dict`      | 可选附加参数。 |
|                | `learning_rate`               | `float`     | 初始学习率。 |
|                | `lr_scheduler_type`           | `str`       | `constant`、`linear`、`cosine`、`cosine_with_restarts`、`polynomial`、`adafactor`。 |
|                | `lr_scheduler_warmup_steps`   | `int`       | 预热步数。 |
|                | `lr_scheduler_train_steps`    | `int`       | 总步数；`null` 自动计算。 |
|                | `lr_scheduler_num_cycles`     | `float`     | 用于 cosine 类调度器。 |
|                | `lr_scheduler_power`          | `float`     | 用于 polynomial 调度器。 |
| `accelerator`  | `mixed_precision`             | `str`       | `'no'`、`'fp16'`、`'bf16'`、`'auto'`。 |
|                | `gradient_accumulation_steps` | `int`       | 梯度累积。 |
|                | `accelerator_checkpoint`      | `str`       | Accelerator 检查点路径。 |
|                | `use_compile`                 | `bool`      | 启用 `torch.compile`。 |
|                | `compile_backend`             | `str`       | `"inductor"`、`"eager"`、`"aot_eager"`。 |
|                | `compile_mode`                | `str`       | `"default"`、`"reduce-overhead"`、`"max-autotune"`。 |
|                | `compile_fullgraph`           | `bool`      | 要求全图编译。 |
|                | `compile_dynamic`             | `bool`      | 启用动态形状。 |
|                | `compile_cache_dir`           | `str`       | 持久化编译缓存。 |
|                | `allow_tf32`                  | `bool`      | 在 Ampere/Ada 上启用 TF32。 |
| `train`        | `max_epoches`                 | `int`       | 最大训练轮数。 |
|                | `validation_split`            | `float`     | 验证集比例（`null` / 0 表示禁用）。 |
|                | `test_split`                  | `float`     | 测试集比例（`null` / 0 表示禁用）。`validation + test < 1.0`。 |
|                | `split_method`                | `str`       | `"random"` 或 `"sequential"`。 |
|                | `best_metric`                 | `str`       | `"train_loss"` 或 `"val_loss"`。 |
|                | `clip_grad_norm`              | `float`     | L2 梯度裁剪。 |
|                | `clip_grad_value`             | `float`     | 绝对值梯度裁剪。 |
|                | `detect_anomaly`              | `bool`      | 异常检测（调试用）。 |
|                | `check_per_step`              | `int`       | 日志频率（步）。 |
|                | `report_per_epoch`            | `int`       | 中间报告频率。 |
|                | `calculate_contextualized_embeddings` | `bool` | 训练后提取按规模加权的上下文嵌入。需要 `include_proportion=true`。 |
|                | `save_per_epoch`              | `int`       | 保存频率。 |
|                | `keep_best_only`              | `bool`      | 结束时丢弃非最优的检查点/嵌入。 |
|                | `save_folder`                 | `str`       | 保存目录。 |
|                | `save_name`                   | `str`       | 文件名（无扩展名）。 |
|                | `save_format`                 | `str`       | 推荐 `.safetensors`。 |
|                | `seed`                        | `int`       | 随机种子。 |
|                | `early_stop_enabled`          | `bool`      | 启用早停。 |
|                | `early_stop_patience`         | `int`       | 容忍轮数。 |
|                | `early_stop_min_delta`        | `float`     | 改善阈值（若 `early_stop_relative=true` 则为相对值）。 |
|                | `early_stop_mode`             | `str`       | `"min"` / `"max"`。 |
|                | `early_stop_relative`         | `bool`      | 相对改善还是绝对改善。 |
|                | `early_stop_metric`           | `str`       | `"train_loss"` / `"val_loss"`。 |

### 静态嵌入 vs. 上下文嵌入

AssetBERT 提供两种互补的资产表示。

**静态嵌入（static embeddings）**直接从训练好的嵌入层读取。每个词元映射到一个固定向量，类比于标准 BERT 的词嵌入，概括了跨全部训练数据的资产级特征。词表与 AssetW2V 一致（除 BERT 特殊词元外）。

**上下文嵌入（contextualized embeddings）**由 BERT 处理真实投资者持仓序列时的隐状态导出。沿用 Gabaix et al. (2024)，我们对每个资产 $a$、在投资者 $i$ 上计算按规模加权的均值

$$\sum_i S_{i,a} \cdot x_{i,a}$$

其中：

- $x_{i,a}$ 是资产 $a$ 在投资者 $i$ 序列中的 BERT 隐状态，
- $S_{i,a}$ 是该投资者相对流通 A 股的持股比例（即 `proportion2_key` 字段）。

要启用上下文嵌入提取：

1. 设 `dataset.include_proportion=true`。
2. 设 `train.calculate_contextualized_embeddings=true`。
3. 同时配置 `proportion1_key` 与 `proportion2_key`。

两种嵌入都保存为带 `Token` 加 `Embed_1...Embed_N` 列的 CSV。命名上加以区分：

- 静态：`{save_name}_best_embedding.csv`
- 上下文：`{save_name}_best_contextual_embedding.csv`
