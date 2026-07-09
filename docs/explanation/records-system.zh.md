# 记录系统

本文档介绍 AssetEmbeddings 的实验记录系统：`Record` 基类、`RecordField` 描述符、训练记录类型，以及 `RecordStore` 持久化层。

## 目录

- [概述](#概述)
- [RecordField 描述符](#recordfield-描述符)
- [Record 类层级](#record-类层级)
- [训练记录类型](#训练记录类型)
- [RecordStore 持久化](#recordstore-持久化)
- [数据库文件布局](#数据库文件布局)
- [错误恢复策略](#错误恢复策略)
- [扩展系统](#扩展系统)

---

## 概述

记录系统位于 `asset_embeddings/records/`，为训练运行提供结构化结果记录与诊断数据持久化。

### 设计原则

- **轻量描述符**——`RecordField` 没有约束系统；只提供类型检查与文档，比 `ConfigField` 更简单。
- **SQLite 亲和**——字段类型自动映射到 SQLite 类型（`int→INTEGER`、`float→REAL`、`str→TEXT`、`bool→INTEGER`）。
- **增量保存**——训练脚本在每个 epoch 后更新记录；中断时已保存的数据不丢失。
- **按需建表**——一个 DB 文件只包含实际跑过的任务对应的表。
- **关注点分离**——`Record` 负责字段定义与序列化；`RecordStore` 负责数据库操作。

### 文件结构

```
asset_embeddings/records/
├── __init__.py         # 导出所有公开名称
├── base.py             # RecordField, Record, Result, Recording, generate_run_id
├── train.py            # BERTTrainRecording, W2VTrainRecording, RSTrainRecording
└── store.py            # RecordStore（save, save_many, save_or_replace）
```

### 与 CLI 的关系

每个训练脚本都通过 `--result_file`（`-r`）参数接受一个 SQLite 数据库路径。省略时不记录任何内容。设置时，脚本在内部创建 `RecordStore` 实例，并在运行中增量写入记录。

```bash
# 训练时记录到数据库
uv run python -m asset_embeddings.scripts.train.bert -c configs/finetune/bert.json -r results/db/main.db
```

---

## RecordField 描述符

`RecordField` 是 Record 类的字段描述符，提供类型检查与默认值支持。

```python
class RecordField:
    def __init__(
        self,
        type_hint: Optional[Type] = None,  # 显式类型；为 None 时从注解推断
        default: Any = None,               # 默认值
        required: bool = False,            # 是否必填
        doc: str = "",                     # 文档串
    )
```

### 与 ConfigField 的差异

| 特性 | RecordField | ConfigField |
|------|-------------|-------------|
| 约束系统 | 无 | RangeConstraint、ChoiceConstraint 等 |
| 类型检查 | 赋值时运行时检查 | 运行时 + 约束验证 |
| 冻结支持 | 无 | `freeze()` |
| SQLite 映射 | 内置 `SQLITE_TYPE_AFFINITY` | 无 |
| 设计目标 | 数据记录 | 配置管理 |

### 类型推断

当 `type_hint=None` 时，`RecordField` 从类的类型注解自动推断类型：

```python
class MyRecord(Record):
    score: float = RecordField(doc="Primary metric")  # 推断为 float
    name: str = RecordField(required=True)             # 推断为 str
```

### SQLite 类型映射

```python
SQLITE_TYPE_AFFINITY = {
    int: "INTEGER",
    float: "REAL",
    str: "TEXT",
    bool: "INTEGER",   # bool 以 0/1 存储
    bytes: "BLOB",
}
```

`Optional[X]` 类型解析为 `X` 的 SQLite 类型。无法映射的类型默认为 `TEXT`。

---

## Record 类层级

### 类层级

```
Record (base class)
│   核心能力：字段管理、to_dict()、get_table_schema()
│
├── Result (实验结果基类)
│   字段：run_id, recorded_at, config_file, config_content, status, error_message
│   （无内置子类；供自定义流水线的分数型记录使用）
│
└── Recording (运行时诊断基类)
    字段：run_id, recorded_at, config_file, config_content, status, error_message
    ├── BERTTrainRecording      (BERT 逐运行训练摘要)
    ├── W2VTrainRecording       (W2V 逐运行训练摘要)
    └── RSTrainRecording        (RS 逐运行训练摘要)
```

### Result vs Recording

| | Result | Recording |
|---|--------|-----------|
| 语义 | 实验的核心指标（分数） | 运行时诊断数据（训练过程、运行摘要） |
| 粒度 | 每个指标观测一行。随框架发布的训练脚本不写 Result。 | 每次训练运行一行 |
| 表 | 每个任务一张表（`<task>_results`） | 每种类型一张表（`train_bert_recordings`，…） |
| run_id | 标识一次完整运行 | 经 run_id 把同一次运行的各行关联起来 |

### Record 基类

```python
class Record:
    """Structured-record base class, providing SQLite compatibility."""

    # 经 MRO 收集每个祖先的 RecordField 描述符
    _core_fields: Dict[str, RecordField]

    def __init__(self, **kwargs):
        # 核心字段通过描述符赋值（带类型检查）
        # 未知 kwargs 作为动态字段存储（无需预定义；列自动创建）
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

### 动态字段

`Record` 支持在构造时传入未预定义的字段（动态字段）。这些字段被自动存储，并在写入数据库时触发 `ALTER TABLE ADD COLUMN`：

```python
recording = BERTTrainRecording(
    run_id="abc", recorded_at="2024-01-01",
    task="finetune_BERT", model="AssetBERT", seed=42,
    experiment_tag="robustness",  # 动态字段：未在 BERTTrainRecording 上预定义
)
# 写入数据库时自动创建 experiment_tag 列
```

### generate_run_id()

```python
def generate_run_id() -> str:
    """Generate a run_id in UUID4 format."""
    return str(uuid.uuid4())
```

在每次训练运行开始时调用一次；同一次运行的所有记录行共享同一 `run_id`。

---

## 训练记录类型

训练记录都继承自 `Recording`，按模型类型分开保存（三种训练差异很大，统一成一张表会产生大量 NULL）。每个类都在基础 `Recording` 字段（`run_id`、`recorded_at`、`config_file`、`config_content`、`status`、`error_message`）之上添加自己的字段。

### BERTTrainRecording

每次 BERT 训练运行的摘要（`train_bert_recordings`）。

| 字段 | 类型 | 说明 |
|------|------|------|
| `task` | `str` | `"pretrain_BERT"` / `"finetune_BERT"` |
| `model` | `str` | `"AssetBERT"` |
| `seed` | `int` | 随机种子 |
| `total_epochs` | `int` | 配置的 max_epoches |
| `actual_epochs` | `int` | 实际完成的 epoch 数（增量更新） |
| `best_epoch` | `int` | 最优 epoch |
| `best_train_loss` | `float` | 最优 epoch 的训练 loss |
| `final_train_loss` | `float` | 末轮训练 loss |
| `final_val_loss` | `float` | 末轮验证 loss（无验证集时为 None） |
| `early_stopped` | `bool` | 是否触发早停 |
| `early_stop_epoch` | `int` | 触发早停的 epoch |
| `trainable_params` | `int` | 可训练参数量 |
| `duration_sec` | `float` | 训练时长（秒） |
| `tensorboard_dir` | `str` | TensorBoard 日志路径 |
| `checkpoint_path` | `str` | 最优检查点目录 |
| `embedding_path` | `str` | 最优嵌入 CSV 路径 |

### W2VTrainRecording

每次 Word2Vec（gensim）训练运行的摘要（`train_w2v_recordings`）。

| 字段 | 类型 | 说明 |
|------|------|------|
| `task` | `str` | `"pretrain_W2V"` / `"finetune_W2V"` |
| `model` | `str` | `"AssetW2V"` |
| `seed` | `int` | 随机种子 |
| `total_epochs` | `int` | 配置的 epoch 数 |
| `actual_epochs` | `int` | 实际完成的 epoch 数 |
| `vocab_size` | `int` | 词表大小 |
| `corpus_count` | `int` | 语料样本数 |
| `embedding_dim` | `int` | 嵌入维度 |
| `final_loss` | `float` | gensim 报告的 loss（已知问题：始终为 0.0） |
| `duration_sec` | `float` | 训练时长（秒） |
| `embedding_path` | `str` | 输出嵌入路径 |

### RSTrainRecording

单次 PCA/ICA（AssetRS）拟合的摘要（`train_rs_recordings`）。

| 字段 | 类型 | 说明 |
|------|------|------|
| `task` | `str` | `"fit_RS"` |
| `model` | `str` | `"RS_Binary"` / `"RS_Ranks"` / `"RS_Level0"` / `"RS_LevelMin"` |
| `seed` | `int` | 随机种子 |
| `n_components` | `int` | 降维目标维度 |
| `matrix_shape` | `str` | `"(n_investors, n_assets)"` 格式的字符串 |
| `explained_variance_ratio` | `str` | 各成分解释方差比（JSON 数组） |
| `cumulative_variance` | `float` | 累计解释方差比 |
| `duration_sec` | `float` | fit 时长（秒） |
| `embedding_path` | `str` | 输出嵌入路径 |

---

## RecordStore 持久化

`RecordStore` 是 SQLite 持久化层，负责连接管理、建表/迁移以及写操作。它与 Record 分离，以保持单一职责。

### 初始化

```python
store = RecordStore(db_path="results/db/main.db", logger=logger)
```

- 自动创建父目录（`results/db/`）。
- 支持 `:memory:` 模式用于测试（复用单一连接）。
- 文件模式下，每次操作开启并关闭连接。

### API

#### save()

保存单条记录。自动建表或迁移（新增列）。

```python
store.save(recording, "train_bert_recordings")     # → True / False
```

#### save_many()

在单个事务中保存多条记录。

```python
store.save_many(recordings_list, "train_bert_recordings")
```

#### save_or_replace()

按键字段（默认 `run_id`）删除已有行，再插入新行。用于训练脚本的增量更新模式。

```python
# 每个 epoch 后更新同一行
store.save_or_replace(recording, "train_bert_recordings")
```

### 自动表迁移

当 Record 类新增字段、或传入动态字段时，`RecordStore` 自动执行 `ALTER TABLE ADD COLUMN`：

```python
# v1：没有 config_content
store.save(recording_v1, "train_bert_recordings")

# v2：新增 config_content 字段
store.save(recording_v2, "train_bert_recordings")
# → 自动 ALTER TABLE train_bert_recordings ADD COLUMN config_content TEXT
```

旧行中新字段的值为 `NULL`。

---

## 数据库文件布局

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

表按需创建——若某个 DB 只跑了 BERT 训练，它就只含 `train_bert_recordings`。

不同实验流水线使用各自的 DB 文件以避免数据混杂。Shell 脚本通过 `DB` 变量统一管理路径：

```bash
DB="results/db/main.db"
try_run -m bert -m pretrain -m d64 "BERT pretrain d64" \
    uv run python -m asset_embeddings.scripts.train.bert -c configs/main/pretrained/AssetBERT/AssetBERT_d64_pretrained.json -r "$DB"
```

---

## 错误恢复策略

### `status` 与 `error_message` 字段

每个 `Result` / `Recording` 都从基类继承两个字段：

- `status: str`（默认 `"completed"`）：`"running"` / `"completed"` / `"error"`
- `error_message: str`（默认 `None`）：异常 traceback 字符串

每个脚本的 `main(args)` 入口在发生异常时**主动**把这两个字段填入 DB，从而可直接用 SQL 定位失败的运行：

```sql
-- 找出某个实验中所有失败的运行
SELECT run_id, task, error_message FROM train_bert_recordings
WHERE status = 'error';
```

### 训练脚本（单次运行增量更新）

BERT/W2V/RS 训练脚本的 `main(args)` 流程：

```
main() 入口
  → 生成 run_id 并装配 _record_store（在 prepare() 之前）
  → prepare() 内 save_or_replace(status="running")  ← DB 中的初始行
  → 第 k 个 epoch 后 save_or_replace(actual_epochs=k, ...)
  → 正常完成时 save_or_replace(status="completed")

异常路径（由 main() 的 try/except 捕获）：
  情形 A —— prepare() 在 _recording 构造前失败：
    退回构造一个最小 _recording（必填字段 + status="error" + error_message）
    → save_or_replace
    → reraise
  情形 B —— train() 在 _recording 已存在时失败：
    更新 _recording.status="error"、error_message
    → save_or_replace
    → reraise
```

退回分支中的 `task` 推导**必须与 `prepare()` 内部的逻辑保持一致**（代码中有 `must mirror prepare()'s logic` 注释）。

### 不写入 DB 的失败模式

以下场景**不写入 DB**；它们只由 `@log_exceptions_inclass` 写入 `logs/error/<Class>_<ts>.log`：

- 脚本启动：`set_logger` / `load_configs` / `load_overrides` 失败
- 任何在 `run_id` 生成之前的失败

理由：此时 run_id 尚未就绪，无法构造有效的 Recording 行；它们也是启动时立即可见的错误，控制台与日志足以诊断。

---

## 扩展系统

### 新增训练记录类型

1. 在 `asset_embeddings/records/train.py` 中定义新的 Recording 子类：

```python
class NewTrainRecording(Recording):
    """New model per-run training summary."""
    task: str = RecordField(required=True, doc="e.g., pretrain_NewModel")
    model: str = RecordField(required=True, doc="Model name")
    total_epochs: int = RecordField(doc="Configured number of epochs")
    actual_epochs: int = RecordField(doc="Epochs actually completed")
    duration_sec: float = RecordField(doc="Training time (seconds)")
```

2. 在 `__init__.py` 中导出新类。

3. 在训练脚本中用 `save_or_replace()` 做增量更新：

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

4. 确保 `status` 字段正确反映运行状态（`"running"` → `"completed"` / `"error"`）。

### 新增分数型记录类型

对于产出逐观测指标的流水线，改为继承 `Result`，并用 `save()` / `save_many()` 写入：

```python
class NewTaskResult(Result):
    """New task score (one row per quarter)."""
    task: str = RecordField(required=True, doc="e.g., NewTask")
    model_type: str = RecordField(required=True, doc="Model name")
    quarter: str = RecordField(required=True, doc="e.g., 2023Q2")
    metric_name: str = RecordField(required=True, doc="e.g., r_squared")
    score: float = RecordField(required=True, doc="Metric value")
```

按约定，分数表命名为 `<task>_results`，诊断表命名为 `<task>_recordings`。

### 说明

- **表命名约定**：训练按模型类型各写一张 Recording 表（`train_{model}_recordings`）；分数型记录写入 `{task}_results` 表。
- **bool 字段**：`to_dict()` 自动把 `bool` 转为 `int`（0/1），兼容 SQLite `INTEGER`。
- **JSON 字段**：复杂数据（如 `explained_variance_ratio`）序列化为 JSON 字符串存入 `TEXT` 列。
- **动态字段**：构造时可传任意额外 kwargs，数据库列自动创建。这适合实验性注记（如 `experiment_tag`），但核心字段应预定义为 `RecordField`。
