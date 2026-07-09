# CLI 参考

每个入口都以模块形式运行：

```bash
uv run python -m asset_embeddings.scripts.<group>.<name> [args...]
```

通用参数：`-c`/`--config`（JSON 配置；训练是配置驱动的）、`-v`/`--verbose`（控制台 DEBUG）、`-l`/`--log`（日志文件路径）。给任意脚本传 `-h` 可看其完整帮助。训练脚本还接受 `--override KEY=VALUE ...`，以在命令行上修补配置字段。

下面的 **data** 脚本是 flag 驱动的。**训练**脚本（`train.rs` / `train.w2v` / `train.bert`）是配置驱动的，并共享一套[通用选项集](#train-eval-shared)；逐模型的配置字段参考见 [训练](../how-to/train.md)。

---

## `data.sharehold` — 持股 → 组合 {#data-sharehold}

把原始 CSMAR 持股记录转为逐期的投资者→股票组合序列。

| 参数 | 类型 | 说明 |
|---|---|---|
| `--input_folder`, `-i` | `str` | 原始 ShareHoldings CSV 所在文件夹。**必填。** |
| `--output_folder`, `-o` | `str` | 处理后组合的输出文件夹。**必填。** |
| `--intermediate_folder`, `-inter` | `str` | 可选，存放逐期有效样本/股票/投资者的中间产物。 |
| `--output_format` | `{csv,json,binary}` | 输出格式。默认：`csv`。 |
| `--frequency`, `-freq` | `{Y,HY,Q,M,N}` | 分组频率：年 / 半年 / 季 / 月 / 不分组。自定义周期映射器需经 Python API。默认：`Q`。 |
| `--date_format` | `str` | 日期解析格式（`frequency=N` 时忽略）。默认：自动探测。 |
| `--aggregation`, `-agg` | `{all,last,first,max,mean}` | 一期内重复 (投资者, 股票) 对的去重策略。`last`/`first` 需要日期列。默认：`all`。 |
| `--include_proportion`, `-p` | `bool` | 在组合之外一并输出两种持仓比例。默认：`False`。 |
| `--investor_lowerbound`, `-il` | `int` | 丢弃持股少于 N 只的投资者。默认：`10`。 |
| `--stock_lowerbound`, `-sl` | `int` | 丢弃被持有少于 N 次的股票。默认：`10`。 |
| `--source_shareholder_id_column` | `str` | 源投资者 ID 列。默认：`ShareHolderID`。 |
| `--source_stock_symbol_column` | `str` | 源股票代码列。默认：`Symbol`。 |
| `--source_end_date_column` | `str` | 源截止日列。默认：`EndDate`。 |
| `--source_proportion1_column` | `str` | 比例 1 的源列（相对总股本）。默认：`HoldProportion`。 |
| `--source_proportion2_column` | `str` | 比例 2 的源列（相对流通 A 股）。默认：`HoldProportion1`。 |
| `--id_key` / `--portfolio_key` | `str` | 输出键。默认：`InvestorID` / `Portfolio`。 |
| `--proportion1_key` / `--proportion2_key` | `str` | 输出比例键。默认：`Proportion1` / `Proportion2`。 |
| `--stats` | `bool` | 随中间数据写出逐期统计（`statistics.json`、`stock_coverage.csv`）。默认：`True`。 |
| `--plot` | `bool` | 随中间数据渲染分布分析图（`distribution_analysis.png`）。默认：`False`。 |
| `--verbose`, `-v` / `--log`, `-l` | `bool` / `str` | 控制台 DEBUG / 日志文件路径。 |

## `data.distributor` — 训练/验证/测试切分 {#data-distributor}

按比例把处理后的数据切分为数据集，可按文件或按行、内存或流式。

| 参数 | 类型 | 说明 |
|---|---|---|
| `--input_path`, `-i` / `--output_path`, `-o` | `str` | 输入（文件或文件夹）/ 输出路径。 |
| `--file_format`, `-f` | `{csv,json,binary}` | 输入/输出格式。 |
| `--split_ratios`, `-r` | `list[float]` | 目标切分比例（和为 1.0）。默认：`[0.8, 0.2]`。 |
| `--dir_names` | `list[str]` | 目标子目录名。默认：`['pretrained', 'finetune']`。 |
| `--keep_file_integrity` | `bool` | 按整文件切分（`True`）还是按行切分（`False`）。默认：`True`。 |
| `--strategy` | `{concat,streaming}` | 按行切分的策略（仅当 `keep_file_integrity=False`）：`concat`（内存内，更快）vs `streaming`（低内存）。默认：`concat`。 |
| `--split_shuffle` | `bool` | 按行切分前是否打乱（仅当 `keep_file_integrity=False`）。默认：`True`。 |
| `--max_lines_per_file` | `int` | 每个输出文件的最大行数（仅按行切分）。默认：`None`。 |
| `--seed` | `int` | 可复现切分的种子。默认：`42`。 |
| `--use_portfolio_encoder` | `bool` | 用 `PortfolioDataEncoder` 解析。默认：`False`。 |
| `--include_proportion` | `bool` | 持股数据包含比例（仅当 `use_portfolio_encoder=True`）。 |
| `--id_key` / `--portfolio_key` / `--proportion1_key` / `--proportion2_key` | `str` | 输出键。默认：`InvestorID` / `Portfolio` / `Proportion1` / `Proportion2`。 |
| `--verbose`, `-v` / `--log`, `-l` | `bool` / `str` | 控制台 DEBUG / 日志文件路径。 |

## `data.text_embedding` — LLM 文本嵌入

通过 OpenAI / Cohere / Voyage / Gemini / 本地 HF（Qwen、BGE），把 CSMAR 公司名称（中/英）编码为资产嵌入，支持原生 + PCA（universal 或 per-period）降维与可续传缓存。各 provider 的速率限制默认值与输出布局：[Text embeddings](../how-to/text-embeddings.md)。

| 参数 | 类型 | 说明 |
|---|---|---|
| `--config`, `-c` | `str` | JSON 配置；其键成为 argparse 默认值（连字符与下划线键均接受），显式 CLI 参数仍会覆盖。 |
| `--mode` | `{names,full}` | `names` 仅运行名称解析阶段；`full` 运行完整嵌入流水线。默认：`full`。 |
| `--csmar-info` | `str` | CSMAR `TRD_Co.csv` 路径。**必填**（经 CLI 或 `--config`）。 |
| `--valid-stocks-dir` | `str` | `{period}/valid_stocks.json` 所在目录（来自 `data.sharehold`）。**必填**（经 CLI 或 `--config`）。 |
| `--periods` | `list[str]` | 要处理的周期子集。默认：`--valid-stocks-dir` 下的全部周期。 |
| `--names-output` | `str` | 名称阶段输出目录（`stock_names.csv`、`missing_stocks.csv`）。默认：`data/text_embeddings/names`。 |
| `--cache-folder` | `str` | JSONL 嵌入缓存目录。默认：`data/text_embeddings/cache`。 |
| `--output-folder` | `str` | 嵌入产物根目录。默认：`data/text_embeddings/embeddings`。 |
| `--provider` | `{openai,cohere,voyage,gemini,local}` | 嵌入 provider。**`--mode full` 时必填。** |
| `--model` | `str` | provider 专属的模型 id。**`--mode full` 时必填。** |
| `--language` | `{zh,zh-short,en}` | `zh`=Conme，`zh-short`=Stknme，`en`=Conme_en。默认：`zh`。 |
| `--transforms` | `{native,pca-universal,pca-perperiod}` | 要生成的降维策略（可组合）。默认：`native pca-universal`。 |
| `--target-dims` | `list[int]` | 目标嵌入维度。默认：`4 8 10 16 32 64 128`。 |
| `--seed` | `int` | PCA 的 `random_state`。默认：`42`。 |
| `--batch-size` | `int` | 每请求的文本数（在线 provider）与 GPU mini-batch（本地）；会被裁剪到各 provider 的上限。默认：`100`。 |
| `--max-retries` | `int` | 每批最大重试次数。默认：`6`。 |
| `--api-key` | `str` | 字面量 API key；优先级高于 `--api-key-env` 与 provider 默认环境变量。 |
| `--api-key-env` | `str` | 用于读取 key 的环境变量名（仅当 `--api-key` 未设置时使用）。 |
| `--device` | `str` | 本地 HF 模型所用设备（在线 provider 忽略）。默认：`cuda:0`。 |
| `--hf-dtype` | `{bfloat16,float16,float32,int8,int4}` | 本地模型 dtype；`int8`/`int4` 尚未实现（抛 `NotImplementedError`）。默认：`bfloat16`。 |
| `--missing-name-fallback` | `{skip,akshare,error}` | 在 valid_stocks 中但 CSMAR 缺失的股票（full 模式）：`skip` 丢弃并告警、`error` 中止、`akshare` 保留待实现。默认：`skip`。 |
| `--verbose`, `-v` / `--log`, `-l` | `bool` / `str` | 控制台 DEBUG / 日志文件路径。 |

## 训练（共享选项） {#train-eval-shared}

训练（`train.rs`、`train.w2v`、`train.bert`）是配置驱动的：每个模型与运行参数都在 JSON 配置中。三个脚本都暴露**同一套**五个参数——没有逐脚本的专属选项。

| 参数 | 类型 | 说明 |
|---|---|---|
| `--config`, `-c` | `str` | JSON 配置文件路径。**必填。** |
| `--log`, `-l` | `str`（0+） | 日志文件路径。默认：不写文件日志。 |
| `--result_file`, `-r` | `str` | 存放结构化训练记录的 SQLite 路径。默认：`None`。 |
| `--override`, `-o` | `str`（1+） | 在命令行上以 `key=value` 修补配置字段，嵌套键用点号（如 `train.max_epoches=10 optimizer.learning_rate=0.001`）。 |
| `--verbose`, `-v` | flag | 控制台 DEBUG 输出。默认：`INFO`。 |

逐模型的**配置字段**参考见 [训练](../how-to/train.md)。
