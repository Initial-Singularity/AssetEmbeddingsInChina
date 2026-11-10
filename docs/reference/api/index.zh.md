# API 参考

由源码 docstring 自动生成，并经过整理，只保留**公开、受支持的接口**：从 `asset_embeddings` 及其子包再导出的符号，外加这些符号返回或在文档中引用的辅助类与函数（例如评估器的 ridge 拟合结果）。内部管线被略去。概念性指南见 [Explanation](../../explanation/config-system.md)；命令行用法见 [CLI 参考](../cli.md)。

| 页面 | 内容 |
|---|---|
| [配置](configs.md) | `Config`、`ConfigField`、`ConfigContainer` 与约束框架。 |
| [数据集](datasets.md) | 组合 CSV 格式、组合语料以及掩码语言模型数据集与 collator。 |
| [Preparers](preparers.md) | 配置驱动的组件构建。 |
| [训练器](trainers.md) | 掩码语言模型训练循环、AssetBERT 与 managed-portfolio 训练器，以及 Word2Vec 与 RS 训练类。 |
| [评估器](evaluators.md) | RV 与 RC 两个 ridge 基准、它们的分组变体、标签探针、成对相似度、managed-portfolio 分数，以及数据层与估计层。 |
| [模型](models.md) | `SimplifiedMLM` 模块及其 `BertEmbeddings` 层。 |
| [记录](records.md) | 记录类型、`RecordStore`、`Run` 对象、来源信息与查询层。 |
| [工具](utils.md) | 随机种子、累加器、资源探针与日志入口。 |

各具体配置类（`LoggerConfig`、`AssetBERTTrainConfig`、`RVEvalConfig`，…）在你用到它们的地方逐字段记录——见 [训练](../../how-to/train.md) 与 [评估](../../how-to/evaluate.md)。
