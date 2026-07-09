"""Record system for structured experiment results and diagnostics.

Usage:
    from asset_embeddings.records import BERTTrainRecording, RecordStore, generate_run_id
"""

from .base import (
    SQLITE_TYPE_AFFINITY,
    RecordField,
    Record,
    Result,
    Recording,
    generate_run_id,
)
from .train import (
    BERTTrainRecording,
    W2VTrainRecording,
    RSTrainRecording,
)
from .store import RecordStore

__all__ = [
    "SQLITE_TYPE_AFFINITY",
    "RecordField",
    "Record",
    "Result",
    "Recording",
    "generate_run_id",
    "BERTTrainRecording",
    "W2VTrainRecording",
    "RSTrainRecording",
    "RecordStore",
]
