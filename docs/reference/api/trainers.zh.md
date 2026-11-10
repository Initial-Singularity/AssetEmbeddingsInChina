# 训练器 API

`asset_embeddings.scripts.train` 各 CLI 背后的训练类。脚本的 `main()` 构造一个，挂上 [`Run`](records.md)，加载配置，再调用 `prepare()` 与 `train()`；命令行与配置见[训练](../../how-to/train.md)。

::: asset_embeddings.trainers.BaseTrainer

::: asset_embeddings.trainers.mlm.BaseMLMTrainer

::: asset_embeddings.trainers.mlm.split_dataset

::: asset_embeddings.trainers.mlm.FitSummary

::: asset_embeddings.trainers.AssetBERTTrainer

::: asset_embeddings.trainers.mp.MPTrainer

::: asset_embeddings.trainers.mp.MPBertTrainer

::: asset_embeddings.trainers.mp.MPSimplifiedMLMTrainer

::: asset_embeddings.trainers.Word2Vec_Trainer

::: asset_embeddings.trainers.AssetRS_Trainer
