"""Training record types for BERT, W2V, and RS models."""

from .base import RecordField, Recording


class BERTTrainRecording(Recording):
    """BERT per-run training summary."""

    task: str = RecordField(required=True, doc="pretrain_BERT / finetune_BERT")
    model: str = RecordField(required=True, doc="AssetBERT")
    seed: int = RecordField()

    total_epochs: int = RecordField(doc="Config max_epoches")
    actual_epochs: int = RecordField(doc="Epochs actually completed (incremental)")
    best_epoch: int = RecordField()
    best_train_loss: float = RecordField()
    final_train_loss: float = RecordField()
    final_val_loss: float = RecordField(doc="None if no validation")

    early_stopped: bool = RecordField()
    early_stop_epoch: int = RecordField()

    trainable_params: int = RecordField()
    duration_sec: float = RecordField()
    tensorboard_dir: str = RecordField(doc="results/tensorboard/... path")
    checkpoint_path: str = RecordField(doc="Best checkpoint directory")
    embedding_path: str = RecordField(doc="Best embedding CSV path")


class W2VTrainRecording(Recording):
    """Word2Vec (gensim) per-run training summary."""

    task: str = RecordField(required=True, doc="pretrain_W2V / finetune_W2V")
    model: str = RecordField(required=True, doc="AssetW2V")
    seed: int = RecordField()

    total_epochs: int = RecordField()
    actual_epochs: int = RecordField()
    vocab_size: int = RecordField()
    corpus_count: int = RecordField()
    embedding_dim: int = RecordField()

    final_loss: float = RecordField(doc="gensim reported loss (known issue: always 0.0)")

    duration_sec: float = RecordField()
    embedding_path: str = RecordField()


class RSTrainRecording(Recording):
    """PCA/ICA (AssetRS) single-pass fit summary."""

    task: str = RecordField(required=True, doc="fit_RS")
    model: str = RecordField(required=True, doc="RS_Binary / RS_Ranks / RS_Level0 / RS_LevelMin")
    seed: int = RecordField()

    n_components: int = RecordField()
    matrix_shape: str = RecordField(doc="'(n_investors, n_assets)' as string")
    explained_variance_ratio: str = RecordField(doc="JSON array of variance ratios")
    cumulative_variance: float = RecordField(doc="Sum of explained_variance_ratio")

    duration_sec: float = RecordField()
    embedding_path: str = RecordField()
