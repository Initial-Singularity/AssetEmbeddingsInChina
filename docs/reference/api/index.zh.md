# API 参考

由源码 docstring 自动生成，并经过整理，只保留**公开、受支持的接口**（即从 `asset_embeddings` 及其子包再导出的符号）。内部管线被略去。概念性指南见 [Explanation](../../explanation/config-system.md)；命令行用法见 [CLI 参考](../cli.md)。

| 页面 | 内容 |
|---|---|
| [配置](configs.md) | `Config`、`ConfigField`、`ConfigContainer` 与约束框架。 |
| [数据集](datasets.md) | 组合编码与训练数据集。 |
| [Preparers](preparers.md) | 配置驱动的组件构建。 |
| [模型](models.md) | `BertEmbeddings` 模块。 |
| [记录](records.md) | 结构化结果记录与 `RecordStore`。 |
| [工具](utils.md) | 随机种子、累加器与辅助函数。 |

各具体配置类（`LoggerConfig`、`AssetBERTTrainConfig`、`AssetW2VTrainConfig`，…）在你用到它们的地方逐字段记录——见 [训练](../../how-to/train.md)。
