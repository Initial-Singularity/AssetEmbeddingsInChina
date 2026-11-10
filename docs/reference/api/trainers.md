# Trainers API

The training classes behind the `asset_embeddings.scripts.train` CLIs. A script's `main()` constructs one,
attaches the [`Run`](records.md), loads the config and calls `prepare()` then `train()`; see
[Train](../../how-to/train.md) for the command lines and configs.

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
