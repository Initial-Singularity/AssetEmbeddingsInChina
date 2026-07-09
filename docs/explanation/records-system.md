# Records system

This document describes the AssetEmbeddings experiment records system: the `Record` base class, the `RecordField` descriptor, the training record types, and the `RecordStore` persistence layer.

## Contents

- [Overview](#overview)
- [The RecordField descriptor](#the-recordfield-descriptor)
- [The Record class hierarchy](#the-record-class-hierarchy)
- [Training record types](#training-record-types)
- [RecordStore persistence](#recordstore-persistence)
- [Database file layout](#database-file-layout)
- [Error-recovery strategy](#error-recovery-strategy)
- [Extending the system](#extending-the-system)

---

## Overview

The records system lives under `asset_embeddings/records/` and provides structured result records and diagnostic-data persistence for training runs.

### Design principles

- **Lightweight descriptor** — `RecordField` has no constraint system; it provides only type checking and documentation, making it simpler than `ConfigField`.
- **SQLite affinity** — field types map automatically to SQLite types (`int→INTEGER`, `float→REAL`, `str→TEXT`, `bool→INTEGER`).
- **Incremental saving** — training scripts update their record after every epoch; already-saved data is not lost on interruption.
- **Tables created on demand** — a DB file contains only the tables for tasks that were actually run.
- **Separation of concerns** — `Record` handles field definitions and serialization; `RecordStore` handles database operations.

### File layout

```
asset_embeddings/records/
├── __init__.py         # exports all public names
├── base.py             # RecordField, Record, Result, Recording, generate_run_id
├── train.py            # BERTTrainRecording, W2VTrainRecording, RSTrainRecording
└── store.py            # RecordStore (save, save_many, save_or_replace)
```

### Relationship to the CLI

Every training script takes a SQLite database path via the `--result_file` (`-r`) argument. When it is omitted, nothing is recorded. When it is set, the script creates a `RecordStore` instance internally and writes records incrementally as it runs.

```bash
# Record to a database during training
uv run python -m asset_embeddings.scripts.train.bert -c configs/finetune/bert.json -r results/db/main.db
```

---

## The RecordField descriptor

`RecordField` is the field descriptor for Record classes, providing type checking and default-value support.

```python
class RecordField:
    def __init__(
        self,
        type_hint: Optional[Type] = None,  # explicit type; inferred from the annotation if None
        default: Any = None,               # default value
        required: bool = False,            # whether the field is required
        doc: str = "",                     # documentation string
    )
```

### Differences from ConfigField

| Feature | RecordField | ConfigField |
|------|-------------|-------------|
| Constraint system | None | RangeConstraint, ChoiceConstraint, etc. |
| Type checking | Runtime check on assignment | Runtime + constraint validation |
| Freeze support | None | `freeze()` |
| SQLite mapping | Built-in `SQLITE_TYPE_AFFINITY` | None |
| Design goal | Data records | Configuration management |

### Type inference

When `type_hint=None`, `RecordField` infers the type automatically from the class's type annotations:

```python
class MyRecord(Record):
    score: float = RecordField(doc="Primary metric")  # inferred as float
    name: str = RecordField(required=True)             # inferred as str
```

### SQLite type mapping

```python
SQLITE_TYPE_AFFINITY = {
    int: "INTEGER",
    float: "REAL",
    str: "TEXT",
    bool: "INTEGER",   # bool is stored as 0/1
    bytes: "BLOB",
}
```

An `Optional[X]` type resolves to the SQLite type of `X`. Types that cannot be mapped default to `TEXT`.

---

## The Record class hierarchy

### Class hierarchy

```
Record (base class)
│   core abilities: field management, to_dict(), get_table_schema()
│
├── Result (experiment-result base class)
│   fields: run_id, recorded_at, config_file, config_content, status, error_message
│   (no built-in subclasses; available for score-style records in custom pipelines)
│
└── Recording (runtime-diagnostic base class)
    fields: run_id, recorded_at, config_file, config_content, status, error_message
    ├── BERTTrainRecording      (BERT per-run training summary)
    ├── W2VTrainRecording       (W2V per-run training summary)
    └── RSTrainRecording        (RS per-run training summary)
```

### Result vs Recording

| | Result | Recording |
|---|--------|-----------|
| Semantics | An experiment's core metric (score) | Runtime diagnostic data (training process, run summary) |
| Granularity | One row per metric observation. The shipped training scripts write no Result. | One row per training run |
| Table | One table per task (`<task>_results`) | One table per type (`train_bert_recordings`, …) |
| run_id | Identifies one complete run | Links rows of the same run together via run_id |

### The Record base class

```python
class Record:
    """Structured-record base class, providing SQLite compatibility."""

    # Collects every ancestor's RecordField descriptors via the MRO
    _core_fields: Dict[str, RecordField]

    def __init__(self, **kwargs):
        # Core fields are assigned through descriptors (with type checking)
        # Unknown kwargs are stored as dynamic fields (no predefinition needed; columns are created automatically)
        ...

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a flat dict. Skips None core fields; converts bool to int."""

    @classmethod
    def get_table_schema(cls) -> Dict[str, str]:
        """Return {field_name: sqlite_type} for CREATE TABLE."""

    @classmethod
    def get_core_field_names(cls) -> List[str]:
        """Return the list of all core field names."""

    def validate_sqlite_compatible(self) -> bool:
        """Check that every field value is a SQLite-compatible type."""
```

### Dynamic fields

`Record` supports passing fields that are not predefined at construction time (dynamic fields). These fields are stored automatically and trigger an `ALTER TABLE ADD COLUMN` when written to the database:

```python
recording = BERTTrainRecording(
    run_id="abc", recorded_at="2024-01-01",
    task="finetune_BERT", model="AssetBERT", seed=42,
    experiment_tag="robustness",  # dynamic field: not predefined on BERTTrainRecording
)
# The experiment_tag column is created automatically when written to the database
```

### generate_run_id()

```python
def generate_run_id() -> str:
    """Generate a run_id in UUID4 format."""
    return str(uuid.uuid4())
```

Called once at the start of each training run; all record rows of the same run share the same `run_id`.

---

## Training record types

Training records all inherit from `Recording`, kept separate per model type (the three training kinds differ substantially, and a unified table would produce many NULLs). Every class adds its fields on top of the base `Recording` fields (`run_id`, `recorded_at`, `config_file`, `config_content`, `status`, `error_message`).

### BERTTrainRecording

Run summary for each BERT training run (`train_bert_recordings`).

| Field | Type | Description |
|------|------|------|
| `task` | `str` | `"pretrain_BERT"` / `"finetune_BERT"` |
| `model` | `str` | `"AssetBERT"` |
| `seed` | `int` | Random seed |
| `total_epochs` | `int` | Configured max_epoches |
| `actual_epochs` | `int` | Number of epochs actually completed (updated incrementally) |
| `best_epoch` | `int` | Best epoch |
| `best_train_loss` | `float` | Training loss at the best epoch |
| `final_train_loss` | `float` | Training loss at the final epoch |
| `final_val_loss` | `float` | Validation loss at the final epoch (None when there is no validation set) |
| `early_stopped` | `bool` | Whether early stopping was triggered |
| `early_stop_epoch` | `int` | The epoch at which early stopping triggered |
| `trainable_params` | `int` | Number of trainable parameters |
| `duration_sec` | `float` | Training time (seconds) |
| `tensorboard_dir` | `str` | TensorBoard log path |
| `checkpoint_path` | `str` | Best checkpoint directory |
| `embedding_path` | `str` | Best embedding CSV path |

### W2VTrainRecording

Run summary for each Word2Vec (gensim) training run (`train_w2v_recordings`).

| Field | Type | Description |
|------|------|------|
| `task` | `str` | `"pretrain_W2V"` / `"finetune_W2V"` |
| `model` | `str` | `"AssetW2V"` |
| `seed` | `int` | Random seed |
| `total_epochs` | `int` | Configured number of epochs |
| `actual_epochs` | `int` | Number of epochs actually completed |
| `vocab_size` | `int` | Vocabulary size |
| `corpus_count` | `int` | Number of corpus samples |
| `embedding_dim` | `int` | Embedding dimension |
| `final_loss` | `float` | Loss reported by gensim (known issue: always 0.0) |
| `duration_sec` | `float` | Training time (seconds) |
| `embedding_path` | `str` | Output embedding path |

### RSTrainRecording

Summary of a single PCA/ICA (AssetRS) fit (`train_rs_recordings`).

| Field | Type | Description |
|------|------|------|
| `task` | `str` | `"fit_RS"` |
| `model` | `str` | `"RS_Binary"` / `"RS_Ranks"` / `"RS_Level0"` / `"RS_LevelMin"` |
| `seed` | `int` | Random seed |
| `n_components` | `int` | Target dimension for dimensionality reduction |
| `matrix_shape` | `str` | A `"(n_investors, n_assets)"`-format string |
| `explained_variance_ratio` | `str` | Explained-variance ratios (JSON array) |
| `cumulative_variance` | `float` | Cumulative explained-variance ratio |
| `duration_sec` | `float` | fit time (seconds) |
| `embedding_path` | `str` | Output embedding path |

---

## RecordStore persistence

`RecordStore` is the SQLite persistence layer, responsible for connection management, table creation/migration, and write operations. It is separated from Record to keep a single responsibility.

### Initialization

```python
store = RecordStore(db_path="results/db/main.db", logger=logger)
```

- Creates the parent directory automatically (`results/db/`).
- Supports `:memory:` mode for testing (reuses a single connection).
- In file mode, opens and closes the connection on each operation.

### API

#### save()

Save a single record. Creates the table or migrates it (adds new columns) automatically.

```python
store.save(recording, "train_bert_recordings")     # → True / False
```

#### save_many()

Save multiple records in a single transaction.

```python
store.save_many(recordings_list, "train_bert_recordings")
```

#### save_or_replace()

Delete an existing row by key field (default `run_id`), then insert a new one. Used for the incremental-update pattern in training scripts.

```python
# Update the same row after each epoch
store.save_or_replace(recording, "train_bert_recordings")
```

### Automatic table migration

When a Record class gains a new field, or a dynamic field is passed, `RecordStore` automatically performs an `ALTER TABLE ADD COLUMN`:

```python
# v1: without config_content
store.save(recording_v1, "train_bert_recordings")

# v2: adds the config_content field
store.save(recording_v2, "train_bert_recordings")
# → automatic ALTER TABLE train_bert_recordings ADD COLUMN config_content TEXT
```

The new field's value is `NULL` for older rows.

---

## Database file layout

```
results/
└── db/
    ├── main.db                          ← train_main.sh
    │   ├── train_bert_recordings
    │   ├── train_w2v_recordings
    │   └── train_rs_recordings
    │
    ├── semiannual_robust.db             ← train_semiannual_robust.sh
    ├── timesplit_robust.db              ← train_timesplit_robust.sh
    └── ...
```

Tables are created on demand — if a DB only ran BERT training, it contains just `train_bert_recordings`.

Different experiment pipelines use separate DB files to avoid mixing data. Shell scripts manage the path uniformly through the `DB` variable:

```bash
DB="results/db/main.db"
try_run -m bert -m pretrain -m d64 "BERT pretrain d64" \
    uv run python -m asset_embeddings.scripts.train.bert -c configs/main/pretrained/AssetBERT/AssetBERT_d64_pretrained.json -r "$DB"
```

---

## Error-recovery strategy

### The `status` and `error_message` fields

Every `Result` / `Recording` inherits two fields from the base class:

- `status: str` (default `"completed"`): `"running"` / `"completed"` / `"error"`
- `error_message: str` (default `None`): the exception traceback string

Each script's `main(args)` entry point **proactively** fills these two fields into the DB when an exception occurs, so that failed runs can be located directly via SQL:

```sql
-- Find all failed runs in one experiment
SELECT run_id, task, error_message FROM train_bert_recordings
WHERE status = 'error';
```

### Training scripts (single-run incremental update)

The `main(args)` flow of the BERT/W2V/RS training scripts:

```
main() entry
  → generate run_id and assemble _record_store (before prepare())
  → save_or_replace(status="running") inside prepare()  ← initial row in DB
  → save_or_replace(actual_epochs=k, ...) after epoch k
  → save_or_replace(status="completed") on normal completion

Exception path (caught by main()'s try/except):
  case A — prepare() fails before _recording is constructed:
    fall back to constructing a minimal _recording (required fields + status="error" + error_message)
    → save_or_replace
    → reraise
  case B — train() fails when _recording already exists:
    update _recording.status="error", error_message
    → save_or_replace
    → reraise
```

The `task` derivation in the fallback branch **must stay consistent with the logic inside `prepare()`** (there is a `must mirror prepare()'s logic` comment in the code).

### Failure modes that do not write to the DB

The following scenarios **do not write to the DB**; they are only written by `@log_exceptions_inclass` to `logs/error/<Class>_<ts>.log`:

- Script startup: failures in `set_logger` / `load_configs` / `load_overrides`
- Any failure before a `run_id` is generated

The reasoning: at these points run_id is not yet ready, so a valid Recording row cannot be constructed; they are also errors that are immediately visible at startup, where the console and logs are enough to diagnose.

---

## Extending the system

### Adding a new training record type

1. Define a new Recording subclass in `asset_embeddings/records/train.py`:

```python
class NewTrainRecording(Recording):
    """New model per-run training summary."""
    task: str = RecordField(required=True, doc="e.g., pretrain_NewModel")
    model: str = RecordField(required=True, doc="Model name")
    total_epochs: int = RecordField(doc="Configured number of epochs")
    actual_epochs: int = RecordField(doc="Epochs actually completed")
    duration_sec: float = RecordField(doc="Training time (seconds)")
```

2. Export the new class in `__init__.py`.

3. Use `save_or_replace()` for incremental updates in the training script:

```python
from asset_embeddings.records import NewTrainRecording, RecordStore, generate_run_id

run_id = generate_run_id()
store = RecordStore(db_path, logger=self.logger) if db_path else None

recording = NewTrainRecording(
    run_id=run_id,
    recorded_at=datetime.now().isoformat(),
    task="pretrain_NewModel",
    model="NewModel",
    status="running",
)
if store:
    store.save_or_replace(recording, "train_newmodel_recordings")

for epoch in range(max_epochs):
    train_one_epoch()
    recording.actual_epochs = epoch + 1
    if store:
        store.save_or_replace(recording, "train_newmodel_recordings")

recording.status = "completed"
if store:
    store.save_or_replace(recording, "train_newmodel_recordings")
```

4. Make sure the `status` field correctly reflects the run state (`"running"` → `"completed"` / `"error"`).

### Adding a score-style record type

For pipelines that produce a per-observation metric, subclass `Result` instead and write with `save()` / `save_many()`:

```python
class NewTaskResult(Result):
    """New task score (one row per quarter)."""
    task: str = RecordField(required=True, doc="e.g., NewTask")
    model_type: str = RecordField(required=True, doc="Model name")
    quarter: str = RecordField(required=True, doc="e.g., 2023Q2")
    metric_name: str = RecordField(required=True, doc="e.g., r_squared")
    score: float = RecordField(required=True, doc="Metric value")
```

By convention, score tables are named `<task>_results` and diagnostic tables `<task>_recordings`.

### Notes

- **Table-naming convention**: training writes one Recording table per model type (`train_{model}_recordings`); score-style records go to `{task}_results` tables.
- **bool fields**: `to_dict()` converts `bool` to `int` (0/1) automatically, compatible with SQLite `INTEGER`.
- **JSON fields**: complex data (such as `explained_variance_ratio`) is serialized to a JSON string and stored in a `TEXT` column.
- **Dynamic fields**: you can pass arbitrary extra kwargs at construction time, and the database columns are created automatically. This suits experimental annotations (such as `experiment_tag`), but core fields should be predefined as `RecordField`.
