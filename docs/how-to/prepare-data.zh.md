# 准备数据

两种数据形态：用于生产/论文复现的 **CSMAR**，以及随仓库提交、供演示用的**合成**数据集（无需任何外部数据）。把原始数据转成训练输入的逐脚本 CLI 参数，参见 [CLI 参考](../reference/cli.md)。

!!! tip "只想要 embedding？"
    如果你只需要训练好的 asset 与 text embedding（而非从原始数据重跑流水线），可从 Zenodo 下载：
    [10.5281/zenodo.20640781](https://doi.org/10.5281/zenodo.20640781)（CC-BY-4.0）。这些是派生的表示向量、
    并非 CSMAR 源数据——无需 CSMAR 访问权限。只有从零重新生成 embedding 时才需要 CSMAR，详见下文。

## CSMAR（生产数据）

生产运行需要 CSMAR 的 A 股数据——机构持股记录，若还要运行 text-embedding 流水线则另需公司名称。CSMAR 需付费订阅；机构访问通常可通过高校图书馆获得。CSMAR 的许可不允许再分发源数据或其任何派生切片，因此本仓库不提供下载镜像——请直接从 CSMAR 获取这些表。

### 持股面板

机构持股记录（沪 / 深 / 北交易所 + 公募基金定期报告），按年份拆分为 UTF-8 CSV。关键字段：

| 字段 | 说明 |
|---|---|
| `InstitutionID` | 上市公司记录代码（CSMAR 内部编码，**与 `Symbol` 一一对应**）。它**不是**投资者 ID——投资者 ID 是 `ShareHolderID`。参见 [数据流水线](../explanation/data-pipeline.md)。 |
| `Symbol` | 股票代码。 |
| `EndDate` | 报告期截止日。 |
| `ShareHolderID` / `ShareHolderName` | 持有机构 ID / 名称（真实的投资者身份）。 |
| `SystematicsID` / `CategoryCode` | 机构分类代码（基金 / QFII / 券商 / 保险 / 社保 / 信托 / 银行 / …）。 |
| `Holdshares` | 持股数量。 |
| `HoldProportion` | 持股占**总**股本的比例。 |
| `HoldProportion1` | 持股占**流通 A 股**的比例。 |
| `Price` | 最接近 `EndDate` 的股价。 |
| `IndustryCode` / `IndustryName` | `EndDate` 时点的证监会 2012 行业分类。 |

## 合成数据（演示）

一份随仓库提交的数据集——复现演示无需任何外部数据：

- `examples/data/`——合成组合数据，供 `examples/quickstart.{ipynb,py}` 中端到端的 W2V-PT → W2V-FT → BERT-PT → BERT-FT 走查使用。schema 与生成理由：`examples/data/README.md`。

## 处理流水线

原始数据由 `asset_embeddings.scripts.data.*` 工具转为训练输入——`sharehold`（持股 → 组合）、`distributor`（训练切分，如 80/20 pretrained/finetune）、`text_embedding`（公司名称 → LLM 文本嵌入，见 [文本嵌入](text-embeddings.md)）。每个都是 flag 驱动的；完整参数表见 [CLI 参考](../reference/cli.md)。基于预设的批量路径见 [安装 → 用预设快速上手](install.md)。
