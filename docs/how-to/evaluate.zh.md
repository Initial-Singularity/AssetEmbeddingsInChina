# 评估

> 隶属于 [AssetEmbeddingsInChina](../index.md)。另见 [安装](install.md)、[训练](train.md)。

所有评估脚本都接受指向 SQLite 数据库的 `--result_file`（`-r`）。逐期的分数与诊断信息经由结构化的 [记录系统](../explanation/records-system.md) 持久化。每次运行产生：

- 一条 `runs` 行——配置只存一次、来源信息（脚本、git commit、主机名）、状态、耗时，以及 `peak_ram_mb` / `peak_vram_mb` 峰值。
- 一条逐任务的 **结果** 行（`metric`、`value`、所用惩罚、`n_scored`、`n_features`）——`eval_rv_results`、`eval_rc_results` 或 `eval_mp_results`（分组评估每个分组、每期写一行，见 [分组评估](#grouped-evaluation)；[标签探针](#label-probe) 与 [成对相似度](#pairwise-similarity) 有各自的表）；视图 `v_<table>` 附上该运行的配置字段（模型类型、特征、评分方式，...），`v_<table>_summary` 附上跨期均值。
- 每期的 **样本漏斗**（`eval_sample`）：读入的每个文件的行数，以及每次连接与丢弃后剩下的行数。
- 对 MP 脚本，每个 refine epoch 一条 `run_epochs` 行，外加一条 `train_summary` 行。
- 当 `scoring: "holdout"` 时，每期、每个公司切分一条 `eval_ridge_splits` 行（见 [留出评估](#holdout-evaluation)）。

结果按期增量保存，因此脚本中断时已完成的部分会被保留。每张表的完整 schema 见 [记录系统](../explanation/records-system.md)。

所有评估入口共享同一套六参数 CLI（亦见 [CLI 参考](../reference/cli.md#train-eval-shared)）：

```bash
uv run python -m asset_embeddings.scripts.eval.<task> \
    --config CONFIG [--log [LOG ...]] [--result_file RESULT_FILE] \
    [--experiment EXPERIMENT] [--override KEY=VALUE ...] [--verbose]
```

| Flag | Type | 说明 |
|---|---|---|
| `--config`, `-c` | `str` | JSON 配置文件路径（示例见下）。**必填。** |
| `--log`, `-l` | `str`（0+） | 日志文件路径。可选。 |
| `--result_file`, `-r` | `str` | 存放结构化结果的 SQLite 路径（见上）。可选。 |
| `--experiment`, `-e` | `str` | 存入 `runs` 表的实验族（默认由配置路径推导，如 `configs/main/...` 得到 `main`）。runner 会传 `-e "$EXP"`。可选。 |
| `--override`, `-o` | `str`（1+） | 覆盖配置字段，`key=value` 且支持点号嵌套。 |
| `--verbose`, `-v` | flag | DEBUG 级控制台输出（默认 `INFO`）。 |

## RV —— 相对估值

`asset_embeddings.scripts.eval.rv` 评估嵌入向量（如来自 AssetBERT、AssetW2V、AssetRS）能否解释公司之间的**相对估值差异**。它对应论文中的第一项基准：把公司估值残差对资产嵌入做回归。

$$
\begin{gathered}
b_{a,t}=\alpha_{0,t} + \alpha_{1,t} m_{a,t} + b_{a,t}^{\perp}\\
b_{a,t}^{\perp} = \gamma_{0,t} + \boldsymbol{\gamma}_{1,t}^T\boldsymbol{x}_{a,t} + \epsilon_{a,t}\\
\text{RV} = 1 - \frac{1}{T}\sum_{t=1}^{T}\frac{\text{Var}(b_{a,t}^{\perp} - \hat{\gamma}_{0,t} - \boldsymbol{\hat{\gamma}}_{1,t}^T\boldsymbol{x}_{a,t})}{\text{Var}(b_{a,t}^{\perp})}
\end{gathered}
$$

其中 $b_{a,t}$ 与 $m_{a,t}$ 分别是对数账面权益与对数市场权益（基准文件中的 `be` 与 `me` 列）。目标 $b_{a,t}^{\perp}$ 是该期横截面上 $\log(\text{be})$ 对 $\log(\text{me})$ 做 OLS 的残差，分数是该残差对嵌入做 ridge 拟合的横截面解释性 $R^2$。

```bash
uv run python -m asset_embeddings.scripts.eval.rv -c config.json [-r results.db]
```

输出逐期 RV 及其平均值。

### 配置示例

```json
{
    "model_type": "AssetBERT",
    "embedding_path": [
        "checkpoints/finetune/AssetBERT/d4/2023Q2/AssetBERT_d4_2023Q2_best_embedding.csv",
        "checkpoints/finetune/AssetBERT/d4/2023Q3/AssetBERT_d4_2023Q3_best_embedding.csv",
        "checkpoints/finetune/AssetBERT/d4/2023Q4/AssetBERT_d4_2023Q4_best_embedding.csv",
        "checkpoints/finetune/AssetBERT/d4/2024Q1/AssetBERT_d4_2024Q1_best_embedding.csv",
        "checkpoints/finetune/AssetBERT/d4/2024Q2/AssetBERT_d4_2024Q2_best_embedding.csv",
        "checkpoints/finetune/AssetBERT/d4/2024Q3/AssetBERT_d4_2024Q3_best_embedding.csv"
    ],
    "benchmark_path": [
        "data/processed/Benchmark/2023Q2.csv",
        "data/processed/Benchmark/2023Q3.csv",
        "data/processed/Benchmark/2023Q4.csv",
        "data/processed/Benchmark/2024Q1.csv",
        "data/processed/Benchmark/2024Q2.csv",
        "data/processed/Benchmark/2024Q3.csv"
    ],
    "periods": [
        "2023Q2", "2023Q3", "2023Q4",
        "2024Q1", "2024Q2", "2024Q3"
    ],
    "features": ["embedding"],
    "align_chars": true,
    "ridge_alpha": 0.5,
    "seed": 42,
    "embedding_token_col": "Token",
    "embedding_value_pattern": "Embed_*",
    "benchmark_stock_col": "Stkcd",
    "normalization": "standard"
}
```

### 配置字段

每条逐期路径都是一个列表，`periods` 的每一项对应一个文件，顺序相同（只有一期时也接受单个文件）；其他长度会被配置拒绝。文件从不按文件名匹配到期，因为不同路径中文件所属的期不同：预测性设置读取第 t 期的嵌入，与第 t+1 期的基准配对。

| Field            | Type                 | 说明                                                  |
| ---------------- | -------------------- | ------------------------------------------------------------ |
| `model_type`     | `str`                | 被评估模型的标签，随每条结果一起入库，如 `"AssetBERT"`。**必填。** |
| `periods`       | `list[str]`          | 期标签，顺序与各路径列表一致。**必填。** |
| `embedding_path` | `list[str]` / `null` | 每期一个嵌入文件。当且仅当 `features` 含 `"embedding"` 时设置。 |
| `benchmark_path` | `list[str]`          | 每期一个基准文件：提供目标列，未设置 `char_benchmark_path` 时也提供特征列与 `IndustryCode`。**必填。** |
| `char_benchmark_path` | `list[str]` / `null` | 可选，每期一个文件，代替 `benchmark_path` 提供特征列与 `IndustryCode`，按股票代码连接。`null`（默认）：来自 `benchmark_path`。 |
| `features`       | `list[str]`          | 特征块，按列顺序排列；可为以下各项的任意非空组合：`"embedding"`（匹配 `embedding_value_pattern` 的列）、`"placebo"`（每家公司在 [-1, 1] 上均匀抽取的四列）、`"chars"`（传统公司特征）与 `"industry"`（行业哑变量，`drop_first=True`）。不含 `"embedding"` 时，评估在基准文件中的公司上进行：即完整的可用样本，只受其读取的列限制，因此基线的样本可能与嵌入评估的行不同；记录的 `n_scored` 与 `eval_sample` 漏斗给出每次拟合所用的样本。RV 的特征为 beta、资产增长、盈利能力与股利对资产比（`beta_60m`、`at_gr1`、`gp_at`、`div_at`）；RC 使用自己的一组（见下文 RC 一节）。默认：`["embedding"]`。 |
| `align_chars`    | `bool`               | 即使 `"chars"` 不是特征也读取传统公司特征，从而剔除缺任一特征的公司，使样本与加入 `"chars"` 时一致。`features` 含 `"chars"` 时无论如何都会限制公司。`false`：凡具备其他输入的公司都进入。作为结果视图的 `align_chars` 配置列记录。默认：`true`。 |
| `ridge_alpha`    | `float` / `null`     | Ridge 正则强度；`null` 用使 $R^2$ 最大的 10 折交叉验证网格搜索来选。默认：`0.5`。 |
| `scoring`        | `str`                | `"cross_section"` 在该期全部公司上拟合并打分（主结果所用的设定）；`"holdout"` 在 Koijen 式公司切分上打分，见 [留出评估](#holdout-evaluation)。默认：`"cross_section"`。 |
| `holdout_test_size` | `float`           | holdout 打分下每次切分的留出公司占比，实现为 `round(1 / holdout_test_size)` 个哈希折、第 0 折留出。默认：`0.2`。 |
| `holdout_n_splits` | `int`              | holdout 打分下每期公司切分的数量；第 `s` 次切分用种子 `holdout_seed + s`。默认：`20`。 |
| `holdout_seed`   | `int`                | holdout 打分下公司切分的基础种子（独立于 `seed`，后者驱动交叉验证折与 placebo 抽样）。默认：`42`。 |
| `holdout_split_aggregate` | `str`     | holdout 打分下一期取值如何汇总各次公司切分：`"median"` 或 `"mean"`。跨期汇总始终取均值。默认：`"median"`。 |
| `nonpositive_equity` | `str`            | 仅 RV。RV 目标中账面或市场权益非正的公司：`"drop"`（默认）在计算 `log(be) ~ log(me)` 残差与 ridge 之前把它们从该期剔除；`"keep"` 为早期行为，保留以便复现其下记录的结果（此类公司的残差量级为 1e9，会主导任何触及它的拟合）。 |
| `seed`           | `int`                | 所有随机抽样的种子：placebo 列（每期一个生成器，由种子与期标签决定）、惩罚搜索的交叉验证折、bootstrap 与置换检验。默认：`42`。 |
| `embedding_token_col`     | `str`       | 嵌入文件中 Token（股票 ID）列的名称。默认：`"Token"`。 |
| `embedding_value_pattern` | `str`       | 嵌入数值列的通配模式。默认 `"Embed*"`，匹配训练器写出的 `Embed_1 … Embed_N` 列；列名为 `Emb_0 …` 的文本嵌入需要 `"Emb_*"`。 |
| `benchmark_stock_col`     | `str`       | 基准文件中股票 ID 列的名称。默认：`"Stkcd"`。 |
| `normalization`  | `str`                | 特征缩放：`"standard"`（StandardScaler）、`"minmax"`（MinMaxScaler）或 `"none"`。默认：`"standard"`。 |

## RC —— 收益共动

`asset_embeddings.scripts.eval.rc` 评估资产嵌入能否解释未来的收益共动结构——这是检验嵌入是否捕捉资产**系统性风险因子结构**的核心测试。

$$
\begin{gathered}
r_{a,t+1} = c_{t} + \boldsymbol{f}_t^T\boldsymbol{x}_{a,t} + \epsilon_{a,t}\\
\text{RC} = 1 - \frac{1}{T}\sum_{t=1}^{T}\frac{\text{Var}(r_{a,t+1} - \hat{c}_{t} - \boldsymbol{\hat{f}}_t^T\boldsymbol{x}_{a,t})}{\text{Var}(r_{a,t+1})}
\end{gathered}
$$

```bash
uv run python -m asset_embeddings.scripts.eval.rc -c config.json [-r results.db]
```

配置 schema 与 RV 完全一致（见上），只是没有 `nonpositive_equity`：RC 的目标是下一季度收益，而非估值残差。特征集也不同：`"chars"` 与 `align_chars` 使用 beta、规模、账面市值比、资产增长、盈利能力与动量（`beta_60m`、`me`、`be_me`、`at_gr1`、`gp_at`、`ret_12_1`）。

## MP —— 管理型组合

MP 分数衡量**模型在投资者组合的固定位置上预测被遮蔽资产的能力**，反映嵌入对组合构建偏好的刻画程度。

$$
MP = 1 - \frac{\sum_{b_i(k)\in\mathcal{B}}\ln (\text{Pr}(b_i(k)=a_i(k) | \mathcal{A}_i(k)))}{-\lvert\mathcal{B}\rvert \ln V}
$$

其中 $\mathcal{B}$ 是被遮蔽资产的评估集，$\mathcal{A}_i(k)$ 是投资者 $i$ 在位置 $k$ 被遮蔽后的持仓，$a_i(k)$ 是投资者 $i$ 原本在位置 $k$ 持有的资产，$V$ 是分词器词表的大小。每个被遮蔽词元的得分为 $1 + \ln p / \ln V$：在整个词表上均匀猜测得 0，完全确定得 1；报告的分数是测试集上全部被遮蔽词元的均值。该指标取值于 $(-\infty, 1]$，越接近 1 表示预测越好。

由于 W2V 使用固定窗口，它本身不适合这种固定位置的 MLM 任务。对于 RS 与 W2V，嵌入通过一个简单的前馈头（SimplifiedMLM）打分；softmax 在整个词表 $\mathcal{V}$（大小为 $V$）上计算，而不只是在投资者未持有的资产上：

$$
\text{Pr}(b_i(k)=a | \mathcal{A_i}(k)) = \frac{\exp(\boldsymbol{\lambda}_i^T\boldsymbol{x}_a+\delta_a)}{\sum_{b \in \mathcal{V}}\exp(\boldsymbol{\lambda}_i^T\boldsymbol{x}_b+\delta_b)}
$$

为处理时变的市场结构，MP 评估逐期在该期的持仓上 refine 一个模型，持仓被切分为训练、验证与测试集。对于 BERT，refine 从该期微调好的 AssetBERT 检查点（`model.model_checkpoint`，连同编码器与预测头一起加载）及其嵌入文件出发；主网格冻结嵌入层（`freeze_embedding: true`），训练编码器与预测头。对于 W2V 与 RS，则在该期的嵌入文件之上训练一个 SimplifiedMLM 头，嵌入保持冻结。

为 SimplifiedMLM 与 BERT 提供了端到端的训练-验证-测试脚本。

两者都把 `train` 块读作 `MPTrainConfig`：即[训练配置](train.md)中的循环字段加上 `test_split`，去掉仅用于预训练的 `calculate_contextualized_embeddings` 与 `keep_best_only`。`best_metric` 与 `early_stop_metric` 取 `"train_loss"`、`"val_loss"` 或 `"test_loss"`；`"val_loss"` 需要 `validation_split > 0`，`"test_loss"` 需要 `test_split > 0`，在加载配置时检查。每个 epoch 在存在验证集时对其打分，在某个指标读取测试集时对测试集打分。主网格的 MP 模板不设验证集，按 `"test_loss"` 选择 epoch。

> 由于 MP 引入了额外的训练过程，它比 RV 或 RC 耗时明显更长。

### 预训练基线

两个评估脚本都先调用 `prepare()` 再调用 `train()`：`train()` 在任何 refine 之前先于测试集上评估原始模型（从 `model_checkpoint` 加载），把基线写入 `eval_mp_results` 的 `baseline_value` / `baseline_value_std` 列；随后 refine，每当选择损失下降就把 `accelerate` 状态保存到 `<save_name>_accelerator_best`，结束后重新加载该状态并再次评估测试集，从而可直接对比 refine 前后。各 epoch 的状态在结束时删除，只保留最优状态。

### SimplifiedMLM（RS、W2V）

```bash
uv run python -m asset_embeddings.scripts.eval.mp_smlm -c config.json [-r results.db]
```

#### 配置示例

```json
{
    "run": {
        "period": "2023Q2"
    },
    "model": {
        "model_type": "SimplifiedMLM",
        "model_checkpoint": null,
        "embedding_file": "checkpoints/finetune/AssetW2V/d4/2023Q2/AssetW2V_d4_2023Q2_embedding.csv",
        "vocab_size": 4940,
        "hidden_size": 4,
        "max_position_embeddings": 256,
        "type_vocab_size": 1,
        "freeze_embedding": true,
        "freeze_head": false,
        "embedding_model": "AssetW2V"
    },
    "tokenizer": {
        "w2v_model": null,
        "embedding_file": "checkpoints/finetune/AssetW2V/d4/2023Q2/AssetW2V_d4_2023Q2_embedding.csv",
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
        "max_length": 256,
        "mask_prob": null,
        "mask_indices": [2],
        "num_repeats": 1
    },
    "dataloader": {
        "batch_size": 32,
        "shuffle": true,
        "num_workers": 4,
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
        "max_epochs": 30,
        "validation_split": 0.1,
        "test_split": 0.1,
        "split_method": "random",
        "best_metric": "val_loss",
        "clip_grad_norm": null,
        "clip_grad_value": null,
        "detect_anomaly": false,
        "log_every_steps": 200,
        "report_every_epochs": 1,
        "save_every_epochs": 1,
        "save_folder": "checkpoints/refine/SMLM/AssetW2V/d4/2023Q2",
        "save_name": "SMLM_AssetW2V_d4_2023Q2",
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

`tokenizer`、`dataset`、`dataloader`、`optimizer` 与 `accelerator` 块与 AssetBERT 训练相同。`train` 块是 `MPTrainConfig`，即 AssetBERT 的 `train` 块去掉 `calculate_contextualized_embeddings` 与 `keep_best_only`。`run` 与 `model` 块如下：

| Field                     | Type   | 说明                                  |
| ------------------------- | ------ | -------------------------------------------- |
| `run.period`             | `str`  | 写入 `eval_mp_results` 行的期标签，如 `"2023Q2"`。**必填。** |
| `model_type`              | `str`  | 固定为 `"SimplifiedMLM"`。 |
| `model_checkpoint`        | `str`  | 可选的预训练路径。 |
| `embedding_file`          | `str`  | 资产嵌入 CSV。 |
| `vocab_size`              | `int`  | 股票类型数。 |
| `hidden_size`             | `int`  | 隐层维度。 |
| `max_position_embeddings` | `int`  | 最大持仓序列长度。 |
| `freeze_embedding`        | `bool` | 冻结输入嵌入层。 |
| `freeze_head`             | `bool` | 冻结 LM 头。 |
| `embedding_model`         | `str`  | 上游嵌入模型名，如 `"AssetW2V"` 或 `"AssetRS_Binary"`；写入必填列 `eval_mp_results.model_type`，缺少它的运行会在保存结果时失败。 |

### AssetBERT

```bash
uv run python -m asset_embeddings.scripts.eval.mp_bert -c config.json [-r results.db]
```

#### 配置示例

```json
{
    "run": {
        "period": "2023Q2"
    },
    "model": {
        "model_type": "BERT",
        "model_checkpoint": "checkpoints/finetune/AssetBERT/d4/2023Q2/AssetBERT_d4_2023Q2_best/model.safetensors",
        "embedding_file": "checkpoints/finetune/AssetBERT/d4/2023Q2/AssetBERT_d4_2023Q2_best_embedding.csv",
        "vocab_size": 5093,
        "hidden_size": 4,
        "num_hidden_layers": 4,
        "num_attention_heads": 2,
        "intermediate_size": 512,
        "max_position_embeddings": 256,
        "type_vocab_size": 1,
        "freeze_embedding": true,
        "freeze_encoder": false,
        "embedding_model": "AssetBERT"
    },
    "tokenizer": {
        "w2v_model": null,
        "embedding_file": "checkpoints/finetune/AssetBERT/d4/2023Q2/AssetBERT_d4_2023Q2_best_embedding.csv",
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
        "max_length": 256,
        "mask_prob": null,
        "mask_indices": [2],
        "num_repeats": 1
    },
    "dataloader": {
        "batch_size": 32,
        "shuffle": true,
        "num_workers": 4,
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
        "max_epochs": 30,
        "validation_split": 0.1,
        "test_split": 0.1,
        "split_method": "random",
        "best_metric": "val_loss",
        "clip_grad_norm": null,
        "clip_grad_value": null,
        "detect_anomaly": false,
        "log_every_steps": 200,
        "report_every_epochs": 1,
        "save_every_epochs": 1,
        "save_folder": "checkpoints/refine/BERT/AssetBERT/d4/2023Q2",
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

`tokenizer`、`dataset`、`dataloader`、`optimizer` 与 `accelerator` 块与 AssetBERT 训练相同。`train` 块是 `MPTrainConfig`，与 SimplifiedMLM 相同。`run` 与 `model` 块如下：

| Field                     | Type   | 说明                                  |
| ------------------------- | ------ | -------------------------------------------- |
| `run.period`             | `str`  | 写入 `eval_mp_results` 行的期标签，如 `"2023Q2"`。**必填。** |
| `model_type`              | `str`  | 固定为 `"BERT"`。 |
| `model_checkpoint`        | `str`  | refine 的起点检查点（该期微调好的 AssetBERT `model.safetensors`）。可选。 |
| `w2v_model`               | `str`  | 可选的 Word2Vec `.model`，用其向量初始化词元嵌入。 |
| `embedding_file`          | `str`  | 资产嵌入 CSV。 |
| `vocab_size`              | `int`  | 股票类型数（会被分词器的词表大小覆盖）。 |
| `hidden_size`             | `int`  | 隐层维度。 |
| `num_hidden_layers`       | `int`  | Transformer 层数。默认：`4`。 |
| `num_attention_heads`     | `int`  | 注意力头数。默认：`2`。 |
| `intermediate_size`       | `int`  | 前馈层宽度。默认：`512`。 |
| `max_position_embeddings` | `int`  | 最大持仓序列长度。 |
| `type_vocab_size`         | `int`  | token type 词表大小。默认：`1`。 |
| `freeze_embedding`        | `bool` | 冻结输入嵌入层。 |
| `freeze_encoder`          | `bool` | 冻结 Transformer 编码器。 |
| `embedding_model`         | `str`  | 上游嵌入模型名，如 `"AssetBERT"`；写入必填列 `eval_mp_results.model_type`，缺少它的运行会在保存结果时失败。 |

## 留出评估 {#holdout-evaluation}

默认情况下 RV 与 RC 在该期的全部公司上拟合并打分 ridge：RV 报告横截面解释性 $R^2$，RC 报告横截面预测性 $R^2$（其目标是下一季度收益）。设 `scoring: "holdout"` 会把任一任务切换到 Koijen 式的公司切分。对第 $s$ 次切分（种子 `holdout_seed + s`），公司被哈希到 `round(1 / holdout_test_size)` 个折；第 0 折为测试集，其余为训练集。惩罚系数在设置了 `ridge_alpha` 时取该值，否则由训练行上的 10 折 `RidgeCV` 选出。在训练行上拟合的 `Ridge` 预测留出的公司，该次切分的 Koijen $R^2$ 为 $1 - \text{Var}(\text{resid}) / \text{Var}(y_{\text{test}})$；同时记录 sklearn 定义的 $1 - SS_{\text{res}} / SS_{\text{tot}}$。季度分数是 `holdout_n_splits` 次切分的中位数（`holdout_split_aggregate: "mean"` 时取均值），报告的 `ridge_alpha` 是逐切分惩罚的中位数（固定惩罚时即该固定值）。训练行少于 10、测试集为空或测试目标方差为零的切分会直接抛错而不是被跳过，因此分数始终在记录下来的那组公司上计算。默认取中位数，是因为留出 $R^2$ 有上界而无下界：测试折中一家离群公司就能把某次切分拉到远低于零，均值也随之被拉低。

此类运行写入：结果行带 `metric = "r2_holdout"`，以各切分的汇总值作为 `value`、以各切分惩罚的中位数作为 `ridge_alpha`；每期、每次切分一条 `eval_ridge_splits` 行（`value`、`value_sklearn`、`alpha`、`n_train`、`n_test`），用 `load_splits` 读回。各切分间的离散度或另一种汇总，都是对这些行的一次查询。`holdout_split_aggregate` 出现之前写入的运行存的是均值，其视图中的 `holdout_split_aggregate` 列为 NULL。见 [留出切分](../explanation/records-system.md#holdout-splits-eval_ridge_splits)。

`holdout` 实验族以此方式跑主 RV/RC 网格。`templates/eval_holdout_main.json` 生成 `configs/holdout/eval/{fixed,cv}/{RV,RC}/...`——placebo 与特征基线、六个模型 × 七个维度 × `features` 取 `["embedding"]` / `["embedding", "chars"]`，以及全样本（full-universe）变体（`align_chars: false`）——均为 `holdout_test_size: 0.2`、`holdout_n_splits: 20`、`holdout_seed: 42`，分两个惩罚层级：`fixed`（`ridge_alpha: 0.5`，即样本内取值；每个配置数秒）与 `cv`（`ridge_alpha: null`，每个训练切分内做 10 折 `RidgeCV`；约 20 倍耗时）。`scripts/run/eval_holdout_main.sh` 把两个层级都跑进 `results/db/holdout_main.db`，`runs.experiment = "holdout"`；视图中的 `fixed_alpha` 配置列区分两个层级。job 的标记为 `rv` / `rc`、`holdout`、`fixed` / `cv`、模型名、`addchars`、`full` 与 `d{dim}`，因此 CV 层级可按模型分片：

```bash
bash scripts/run/run_notify.sh scripts/run/eval_holdout_main.sh
bash scripts/run/run_notify.sh -k fixed -k cv,chars -k cv,placebo scripts/run/eval_holdout_main.sh
bash scripts/run/run_notify.sh -k cv,assetbert scripts/run/eval_holdout_main.sh
```

## 时序 RC { #temporal-rc }

`asset_embeddings.scripts.eval.rc_temporal` 检验 RC 的映射能否跨期延续。对第一期之后的每一期 $T$，用期 $T-1$ 的目标（期 $T$ 的收益）对 $T-1$ 的特征做 ridge，得到斜率 $\hat{\boldsymbol{f}}_{T-1}$；该斜率不经重估直接作用于期 $T$ 的特征，并以 Koijen $R^2$ 对期 $T$ 的目标（期 $T+1$ 的收益）打分：

$$
\text{RC}^{\text{temporal}}_T = 1 - \frac{\text{Var}(r_{a,T+1} - \boldsymbol{\hat{f}}_{T-1}^T\boldsymbol{x}_{a,T})}{\text{Var}(r_{a,T+1})}
$$

方差为总体方差。在 `slope_window: "expanding"` 下，改用此前每一期 $s < T$ 的斜率，并对它们的预测取平均；坐标系固定时，这相当于把 $\hat{\boldsymbol{f}}_{T-1}$ 换成均值斜率 $\frac{1}{T-1}\sum_{s<T}\hat{\boldsymbol{f}}_s$，即溢价的 Fama–MacBeth 估计。残差被去均值，因此截距在每一期都是自由的：它吸收期 $T$ 的共同收益，而横截面 $R^2$ 本来也不计入这部分。跨期沿用的只有斜率。横截面 RC 中斜率（因子实现值 $\boldsymbol{f}_t$）每期重估；把它固定下来，检验的是这些实现值是否持续，这是一个跨期（out-of-period）的问题，并且要求两期的嵌入坐标含义相同。无关的斜率会增加预测方差，所以负值表示没有预测力，而不是出错。

```bash
uv run python -m asset_embeddings.scripts.eval.rc_temporal -c config.json [-r results.db]
```

配置即 RC 配置（见上），另加两个字段 `align` 与 `slope_window`：

| `align` | 期 $T$ 的嵌入 |
|---|---|
| `"none"`（默认） | 以其自身坐标进入。 |
| `"procrustes"` | 在标准化之前旋转到期 $T-1$ 的坐标系：在两期都出现的公司上，把每期的原始嵌入在这些公司上中心化，求使 $\lVert X_T R - X_{T-1}\rVert_F$ 最小的正交矩阵 $R$（允许旋转与反射，不缩放）；随后期 $T$ 的每家公司都乘以 $R$。这一步只用嵌入，不用收益。 |

| `slope_window` | 作用于期 $T$ 的斜率 |
|---|---|
| `"previous"`（默认） | 期 $T-1$ 上拟合的斜率。 |
| `"expanding"` | 配置中此前每一期上拟合的斜率，各期用各自的惩罚；对它们的预测取平均。在 `align: "procrustes"` 下，期 $T$ 直接分别旋转到每个更早的期（在两期共同的公司上），每个斜率在其自身坐标系中使用。第一个打分期只有一个更早的斜率，两种取值在这一期结果相同。 |

对齐先于逐期标准化，因为除非各列协方差各向同性，按列标准化与旋转不可交换。之后每期在其全样本上按 `normalization` 标准化，与 RC 相同。惩罚取 `ridge_alpha`（fixed 层级），或在 `ridge_alpha: null` 时于每个拟合期的公司上做 10 折 CV，折由 `seed` 决定（cv 层级）。以下限制在加载配置时检查：

- `features` 为 `["embedding"]` 或 `["chars"]`（行业哑变量各期可能不同，placebo 列每期重新抽取）；`align: "procrustes"` 要求 `["embedding"]`。
- `scoring` 为 `"cross_section"`。
- `periods` 为按递增顺序排列的连续季度 `YYYYQn`，从而任何更早一期的目标都不与期 $T$ 的目标重叠。

运行时，相邻两期的特征列必须完全相同（这也涵盖嵌入维度不同的情形）；对齐时，共同公司必须足以确定旋转（数量不少于维度，交叉乘积满秩），否则运行抛错。每个斜率所用的拟合公司数、为期 $T$ 打分的斜率来自哪些期，以及对齐时的共同公司数与交叉乘积的最小奇异值，以 INFO 级别记录在日志中。

运行写入的内容：第一期没有前一期，只写样本漏斗，不写结果行。之后每期在任务 `RC_temporal` 下写一条 `eval_rc_results` 行：`metric` 为 `r2_temporal`（原始坐标）或 `r2_temporal_aligned`，`ridge_alpha` 为所用各斜率惩罚的均值（`"previous"` 下即期 $T-1$ 的惩罚），`n_scored` 为期 $T$ 的公司数。视图把 `slope_window` 作为配置列；该字段出现之前写入的运行在这一列读作 NULL，用的是前一期。整次运行的均值只在打分的各期上计算。`load_results(db, "RC_temporal")` 只读这些运行；`load_results(db, "RC")` 只读横截面与留出运行。

`rc_temporal` 实验族跑两个网格：`templates/eval_rc_temporal.json` 生成 `configs/rc_temporal/{main,timesplit}/{previous,expanding}/{fixed,cv}/...`：每个窗口、斜率窗口与层级包含微调嵌入的六个模型 × 七个维度 × `raw` / `aligned`、特征基线，以及未经微调的预训练 BERT 与 Word2Vec 嵌入（`AssetBERT_static`、`AssetW2V_static`：每期使用同一嵌入，只取 `raw`）的七个维度，各 99 个配置，共 792 个，窗口为 `main`（2023Q2–2024Q3）与 `timesplit_robust`（2020Q4–2024Q3）。`scripts/run/eval_rc_temporal.sh` 把它们写入 `results/db/rc_temporal.db`，`runs.experiment = "rc_temporal"`；job 的标记为 `rc_temporal`、`main` / `timesplit`、`previous` / `expanding`、`fixed` / `cv`、`raw` / `aligned`、`chars`、`static`、模型名与 `d{dim}`。

## 分组评估 { #grouped-evaluation }

`asset_embeddings.scripts.eval.rv_grouped` 与 `asset_embeddings.scripts.eval.rc_grouped` 按分组、逐期重复 RV / RC 的 ridge 拟合，背后驱动的是 `GroupedRVEvaluator` / `GroupedRCEvaluator`（`asset_embeddings.evaluators.grouped`）。公司按逐期分组文件中的某个取值分组：§7.4 中是机构覆盖度（持有该公司的基金数，即 `coverage_count` 列，按分位数或固定分箱分组），§7.1 中是国有属性（`SOE` 列，国有为 1、民营为 0，在 `grouping: "categorical"` 下每个取值一个分组）。每个（分组, 期）单元的分数经 bootstrap 得到置信区间，并用置换检验比较各分组。每个单元是 `eval_grouped_rv_results` / `eval_grouped_rc_results` 中的一行（键为 `period`、`group_label`）；每次检验是 `eval_grouped_rv_tests` / `eval_grouped_rc_tests` 中的一行；该期的漏斗以分组文件结束。

```bash
uv run python -m asset_embeddings.scripts.eval.rv_grouped -c config.json [-r results.db]
uv run python -m asset_embeddings.scripts.eval.rc_grouped -c config.json [-r results.db]
```

`GroupedRVEvalConfig` / `GroupedRCEvalConfig`（`asset_embeddings.configs.eval_configs`）在 RV / RC 配置之上扩展了下表的分组字段，这些字段只在 `GroupingConfig` 中声明一次。设置了 `char_benchmark_path` 会被配置拒绝（特征列取自 `benchmark_path`）。股票数少于 10、或合并后行数少于 10 的（分组, 期）单元会被跳过并给出警告。两个实验族运行这些评估。`coverage`：`templates/eval_coverage.json` 生成 `configs/coverage/eval/...`，`scripts/run/eval_coverage.sh` 把它们跑进 `results/db/coverage.db`，运行时用 `-o grouping=quantile` 或 `-o grouping=fixed bins=...` 选择分组方式。`state_ownership`：`templates/eval_state_ownership.json` 生成 `configs/state_ownership/eval/grouped/{RV,RC}/...`，以 `SOE` 做 `grouping: "categorical"`，`scripts/run/eval_state_ownership.sh` 把它们连同[标签探针](#label-probe)与[成对相似度](#pairwise-similarity)一起跑进 `results/db/state_ownership.db`。

分组在分组文件自身的样本（它列出的每家公司）上划定，而非在数据框中的公司上；该文件以查找方式连接，保留每一行，因为在整个横截面上选择的惩罚（`alpha_scope: "cross_section"`）需要每一行。在 `grouping: "categorical"` 下每个不同取值是一个分组：整数值的数字不带小数标注（`0`、`1`），分组按取值排序，分组取值的统计量（`group_value_mean`、`_median`、`_min`、`_max`）为 null，因为同一分组中每家公司的取值相同。某期的分组即其文件中出现的取值；置换检验在有分数的分组之间进行。

在 `scoring: "holdout"` 下，单元的取值是各次公司切分上留出 R² 按 `holdout_split_aggregate` 的汇总（指标 `r2_holdout`），每次切分一个惩罚：在该切分的单元训练公司上选择（`alpha_scope: "cell"`），或在整期的训练公司上选择（`"cross_section"`），记录其中位数。已打分单元的每次切分写一条 `eval_grouped_ridge_splits` 行（键为 `period`、`group_label`、`split`；用 `load_grouped_splits` 读回）。此时区间是对单元内公司做 bootstrap 得到的该切分汇总值的区间：每个重抽样在单元的惩罚下重新打分，被重复抽到的同一公司落在同一折（折由股票代码哈希决定）。切分无法拟合或打分的单元（训练公司少于 10、没有测试公司，或某次切分的测试公司上目标为常数）会被跳过并给出警告；无法打分的重抽样（某次切分没有测试公司，或测试公司上的目标为常数，小单元中常见）不计入区间，日志给出其个数。每个重抽样都要重算所有切分，因此 bootstrap 每个重抽样约需 `holdout_n_splits` 次 ridge 拟合。置换检验比较相邻分组，分组多于两个时再比较首组与末组；每个 p 值连同其统计量一起入库。

| Field                 | Type          | 说明                                                  |
| --------------------- | ------------- | ------------------------------------------------------------ |
| `group_path`          | `list[str]`   | 每期一个分组文件：提供股票代码与用于分组的取值（覆盖度用 `stock_coverage.csv`，国有属性用 SOE 标签文件）。同一股票列出两次的文件会被拒绝。**必填。** |
| `group_col`           | `str`         | 分组文件中定义分组的列，如 `"coverage_count"` 或 `"SOE"`。**必填。** |
| `group_stock_col`     | `str`         | 分组文件的股票代码列（覆盖度文件中名为 `"Symbol"`）；代码补零到六位。默认：`"Stkcd"`。 |
| `n_quantiles`         | `int`         | `grouping: "quantile"` 下的分位分组数（`G1` = 取值最低）。默认：`4`。 |
| `n_bootstrap`         | `int`         | 置信区间的 bootstrap 重采样次数（`0` 跳过 bootstrap，CI 记为 null）。默认：`1000`。 |
| `n_permutation`       | `int`         | 置换检验迭代次数。默认：`1000`。 |
| `grouping`            | `str`         | `"quantile"`（取值的逐期分位数）、`"fixed"`（按 `bins` 划定的绝对分组）或 `"categorical"`（每个不同取值一个分组；`bins` 与 `bin_labels` 必须不设）。默认：`"quantile"`。 |
| `bins`                | `list` / `null` | `grouping: "fixed"` 下：内部切点（有限、严格递增），如 `[1, 3, 10]` 给出分组 `<=1` / `2-3` / `4-10` / `>10`。 |
| `bin_labels`          | `list` / `null` | 固定分组的标签；长度必须为 `len(bins) + 1`。`null` 时由 `bins` 自动导出。 |
| `winsor_pct`          | `float` / `null` | 目标的双侧缩尾百分位（RV 为 OLS 估值残差，RC 为下一季度收益）：ridge 之前截断到 `[p, 100-p]`，抑制否则会使 CV 惩罚饱和的厚尾。`null` = 不缩尾。 |
| `cv_alpha_min`        | `float` / `null` | CV ridge-alpha 网格的下界，使覆盖良好的组不会正则化不足。`null` = 无下界。RV 与 RC 均适用。 |
| `alpha_scope`         | `str`         | CV ridge-alpha 的解析范围：`"cell"` 在每个（分组, 期）单元内解析；`"cross_section"` 每期在完整横截面上解析一次，并把该惩罚应用到该期的每个分组（特征基线使用，其分组子样本太小、逐单元 CV 不稳定）。`ridge_alpha` 固定时无效。默认：`"cell"`。 |

## 标签探针 { #label-probe }

`asset_embeddings.scripts.eval.label_probe` 考察一个二元公司标签（§7.1 中的国有属性）能在多大程度上从特征中恢复出来，背后驱动的是 `LabelProbeEvaluator`（`asset_embeddings.evaluators.label_probe`）。每期把标签文件内连接到该期的数据框上，因此没有标签的公司离开样本。特征像 RV 与 RC 一样在整个横截面上缩放，划分为 `n_folds` 个分层折（`StratifiedKFold(n_folds, shuffle=True, random_state=seed)`），在训练折上拟合的逻辑回归（`LogisticRegression(max_iter=1000)`，L2，C = 1）以测试折的 ROC AUC 打分。该期的取值是各测试折 AUC 的均值。测试部分或训练部分只含单一类别的折会被跳过。标签只含单一类别、有标签的公司少于折数、或没有任何折可以打分的期不写结果行；其漏斗仍会记录，并有一条警告说明原因。标签取值不只是 0 与 1 时，运行报错停止。

```bash
uv run python -m asset_embeddings.scripts.eval.label_probe -c config.json [-r results.db]
```

`LabelProbeEvalConfig`（`asset_embeddings.configs.eval_configs`）组合了 RV 的数据与特征字段（路径、`periods`、`features`、`align_chars`、`char_benchmark_path`、`normalization`、`seed` 以及列名，同上文 RV 一节的配置字段）与下表的标签字段；它没有惩罚字段，也没有评分方式字段。公司特征为 RV 的四个（beta、资产增长、盈利能力、股利对资产比），因此不设 `embedding_path` 的 `features: ["chars"]` 就是公司特征基线：基准文件中有标签且特征完整的公司。

| Field             | Type        | 说明 |
| ----------------- | ----------- | ----------- |
| `label_path`      | `list[str]` | 每期一个标签文件：股票代码与标签列，每只股票一行（同一股票列出两次会被拒绝）。**必填。** |
| `label_col`       | `str`       | 标签文件中的标签列，如 `"SOE"`。**必填。** |
| `label_stock_col` | `str`       | 标签文件的股票代码列；代码补零到六位。默认：`"Stkcd"`。 |
| `n_folds`         | `int`       | 每期分层划分的折数（2 到 20）；每折的测试 AUC 分别打分，该期取值为其均值。默认：`5`。 |

```json
{
    "model_type": "AssetBERT",
    "embedding_path": [
        "checkpoints/finetune/AssetBERT/d128/2023Q2/AssetBERT_d128_2023Q2_best_embedding.csv",
        "..."
    ],
    "benchmark_path": ["data/processed/Benchmark/2023Q2.csv", "..."],
    "label_path": ["data/processed/FirmLabels/SOE/2023Q2.csv", "..."],
    "label_col": "SOE",
    "periods": [
        "2023Q2", "2023Q3", "2023Q4",
        "2024Q1", "2024Q2", "2024Q3"
    ],
    "features": ["embedding"],
    "align_chars": false,
    "normalization": "standard",
    "embedding_value_pattern": "Embed_*",
    "n_folds": 5,
    "seed": 42
}
```

每个打了分的期写一条 `eval_probe_results` 行（任务 `PROBE`，指标 `auc`，`value` 为各测试折 AUC 的均值，`n_scored` 为有标签的公司数，`n_positive` 为其中标签为 1 的公司数，`n_features`，`n_folds_used`），并为每个打了分的折写一条 `eval_probe_folds` 行（`fold`、`auc`、`n_train`、`n_test`、`n_test_positive`）。漏斗以标签步骤 `label_file`（文件中有标签的行）与 `label_joined`（连接后剩下的行）结束。`load_results(db, "PROBE")` 读取带配置列（`model_type`、`features`、`align_chars`、`label_col`、`n_folds`，...）的结果；`load_details(db, ProbeFold)` 读取各折。

## 成对相似度 { #pairwise-similarity }

`asset_embeddings.scripts.eval.pairwise_similarity` 考察两家公司共享某个标签或行业时，其嵌入相近多少，背后驱动的是 `PairwiseSimilarityEvaluator`（`asset_embeddings.evaluators.pairwise_similarity`）。每期把每一对无序公司的原始嵌入向量（不做特征选择，不缩放）的余弦相似度，用 OLS 回归到一个常数项与每个 `match_on` 列各一个指示变量上（两家公司取值相同时为 1）；`interaction: true` 再加入每个两个及以上指示变量子集的乘积。每个回归元都是该对公司指示模式的函数，因此拟合由 `2**k` 种模式中各自相似度的计数、和与平方和求出，从不构造公司对层面的设计矩阵（一期 4,700 家公司有 1,100 万对，所需内存远低于 1 GB）。最后一期之后，各期系数按 Fama-MacBeth 方式汇总：每一项的取值是其时间序列均值，标准误是其标准差除以期数的平方根，p 值为双侧，取自自由度比期数少一的 t 分布。无法求解的设计（例如某期所有公司共享该标签）会使运行报错停止，并指出是哪一期。

```bash
uv run python -m asset_embeddings.scripts.eval.pairwise_similarity -c config.json [-r results.db]
```

`PairwiseSimilarityEvalConfig` 组合了 RV 的数据字段（路径、`periods`、`seed` 以及列名）、标签字段与下表的两个字段；它没有特征字段，因为回归作用于原始嵌入，且 `embedding_path` 为必填。样本是同时具有嵌入、标签以及每个 `match_on` 列取值的所有公司：标签以外的 `match_on` 列从基准文件读取，缺其中任一列的公司离开样本。

| Field             | Type        | 说明 |
| ----------------- | ----------- | ----------- |
| `label_path`      | `list[str]` | 每期一个标签文件：股票代码与标签列，每只股票一行（同一股票列出两次会被拒绝）。**必填。** |
| `label_col`       | `str`       | 标签文件中的标签列，如 `"SOE"`。**必填。** |
| `label_stock_col` | `str`       | 标签文件的股票代码列；代码补零到六位。默认：`"Stkcd"`。 |
| `match_on`        | `list[str]` | 其共同取值构成公司对同值指示变量的列：标签列，或 `IndustryCode` 这样的基准列；1 到 4 个互不相同的名称，按系数顺序排列。**必填。** |
| `interaction`     | `bool`      | 把每个两个及以上 `match_on` 指示变量子集的乘积作为回归元加入。默认：`false`。 |

```json
{
    "model_type": "AssetBERT",
    "embedding_path": [
        "checkpoints/finetune/AssetBERT/d128/2023Q2/AssetBERT_d128_2023Q2_best_embedding.csv",
        "..."
    ],
    "benchmark_path": ["data/processed/Benchmark/2023Q2.csv", "..."],
    "label_path": ["data/processed/FirmLabels/SOE/2023Q2.csv", "..."],
    "label_col": "SOE",
    "match_on": ["SOE", "IndustryCode"],
    "interaction": false,
    "periods": [
        "2023Q2", "2023Q3", "2023Q4",
        "2024Q1", "2024Q2", "2024Q3"
    ],
    "embedding_value_pattern": "Embed_*",
    "seed": 42
}
```

一次运行写三张表，彼此不重复：每期一条 `eval_pairwise_fits` 行（`r2`、`n_pairs`、`mean_sim` 为全部公司对的平均相似度、`n_firms`），每期每项一条 `eval_pairwise_coefs` 行（`term` 为 `const`、某个 `match_on` 列或 `SOE:IndustryCode` 这样的交互项；`coef`），最后一期之后每项一条 `eval_pairwise_results` 行（任务 `PAIRWISE`，指标 `fm_coef`，`value` 为 Fama-MacBeth 均值，`se`、`t`、`p`、`n_periods`）。漏斗与探针一样以标签步骤结束。相对平均相似度的溢价，是把某一项的取值除以各期 `mean_sim` 的均值：

```python
from asset_embeddings.records import PairwiseFit, load_details, load_results

results = load_results(db, "PAIRWISE")  # one row per run and term, with model_type, match_on, ...
fits = load_details(db, PairwiseFit)
mean_sim = fits.groupby("run_id")["mean_sim"].mean().rename("mean_sim")
results = results.join(mean_sim, on="run_id")
results["premium"] = results["value"] / results["mean_sim"]
```
