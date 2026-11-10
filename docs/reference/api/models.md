# Models API

The PyTorch modules under `asset_embeddings.modules`: the attention-free `SimplifiedMLM` baseline used by the managed-portfolio benchmark, and its `BertEmbeddings` layer, a standalone copy of BERT's embedding block with the same parameter names, so an AssetBERT checkpoint loads into it key for key.

::: asset_embeddings.modules.SimplifiedMLM.SimplifiedMLM

::: asset_embeddings.modules.BertEmbeddings.BertEmbeddings
