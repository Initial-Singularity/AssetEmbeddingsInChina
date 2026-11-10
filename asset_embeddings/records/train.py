"""Training records: the per-epoch losses, the masked-LM loop summary (AssetBERT and MP), Word2Vec and RS fits."""

from .base import Detail, RecordField


class Epoch(Detail):
    """One epoch of a masked-LM loop: the mean loss of each pass that ran."""

    __table__ = "run_epochs"
    __key__ = ("epoch",)

    epoch: int = RecordField(required=True, doc="1-based epoch index")
    train_loss: float = RecordField(doc="Mean training loss")
    val_loss: float = RecordField(doc="Mean validation loss; NULL when there is no validation split")
    test_loss: float = RecordField(doc="Mean test loss; NULL unless a selection metric reads the test split")


class TrainingSummary(Detail):
    """What the masked-LM loop produced (AssetBERT pretraining and finetuning, both MP refines); one row per run."""

    __table__ = "train_summary"

    epochs_run: int = RecordField(doc="Epochs completed (fewer than max_epochs after an early stop)")
    best_epoch: int = RecordField(doc="Epoch selected by best_metric, 1-based")
    best_loss: float = RecordField(doc="The best_metric loss at best_epoch")
    final_train_loss: float = RecordField(doc="Mean training loss of the last epoch")
    final_val_loss: float = RecordField(doc="Mean validation loss of the last epoch")
    final_test_loss: float = RecordField(doc="Mean test loss of the last epoch")
    early_stop_epoch: int = RecordField(doc="Epoch at which early stopping ended the loop; NULL otherwise")
    n_parameters: int = RecordField(doc="Trainable parameters")
    tensorboard_dir: str = RecordField(doc="TensorBoard directory of the run")
    checkpoint_path: str = RecordField(doc="Best checkpoint (AssetBERT: the HF model; MP: the accelerate state)")
    embedding_path: str = RecordField(doc="Best static embedding CSV (AssetBERT)")


class W2VTraining(Detail):
    """A Word2Vec run: the corpus and vocabulary it saw and where the embedding went; one row per run."""

    __table__ = "train_w2v_summary"

    epochs_run: int = RecordField(doc="Epochs completed")
    n_vocab: int = RecordField(doc="Tokens in the vocabulary")
    n_sentences: int = RecordField(doc="Portfolios in the corpus")
    embedding_path: str = RecordField(doc="Embedding CSV")


class RSFit(Detail):
    """A recommender-system (PCA) fit: the holdings matrix and where the embedding went; one row per run."""

    __table__ = "train_rs_summary"

    n_investors: int = RecordField(doc="Rows of the holdings matrix")
    n_assets: int = RecordField(doc="Columns of the holdings matrix")
    embedding_path: str = RecordField(doc="Embedding CSV")


class RSComponent(Detail):
    """One principal component of an RS fit."""

    __table__ = "train_rs_components"
    __key__ = ("component",)

    component: int = RecordField(required=True, doc="Component index, from 1")
    explained_variance_ratio: float = RecordField(required=True, doc="Share of variance the component explains")
