# API reference

Auto-generated from the source docstrings, curated to the **public, supported surface**: the symbols re-exported from `asset_embeddings` and its subpackages, plus the helper classes and functions those symbols return or document (for example the ridge fit results of the evaluators). Internal plumbing is omitted. For the conceptual guides, see [Explanation](../../explanation/config-system.md); for command-line usage, see the [CLI reference](../cli.md).

| Page | Contents |
|---|---|
| [Configuration](configs.md) | `Config`, `ConfigField`, `ConfigContainer`, and the constraint framework. |
| [Datasets](datasets.md) | The portfolio CSV format, the portfolio corpus and the masked-LM dataset and collator. |
| [Preparers](preparers.md) | Config-driven component construction. |
| [Trainers](trainers.md) | The masked-LM training loop, the AssetBERT and managed-portfolio trainers, and the Word2Vec and RS training classes. |
| [Evaluators](evaluators.md) | The RV and RC ridge benchmarks, their grouped variants, the label probe, the pairwise similarity, the managed-portfolio score, and the data and estimation layers. |
| [Models](models.md) | The `SimplifiedMLM` module and its `BertEmbeddings` layer. |
| [Records](records.md) | Record types, the `RecordStore`, the `Run` object, provenance and the query layer. |
| [Utilities](utils.md) | Seeding, accumulators, the resource probe and the logging entry points. |

The concrete config classes (`LoggerConfig`, `AssetBERTTrainConfig`, `RVEvalConfig`, …) are
documented field-by-field where you use them — see [Train](../../how-to/train.md) and
[Evaluate](../../how-to/evaluate.md).
