"""End-to-end asset-embeddings training pipeline on synthetic data.

Mirrors `quickstart.ipynb` cell-for-cell. Runs the project's canonical chain
on CPU in roughly one minute:

    portfolio_pretrained.csv          portfolio_finetune.csv
              |                                |
              v                                |
    +-----------------+                        |
    |  Step 1 - W2V   |                        |
    |  10 ep, d=16    |                        |
    +-----------------+                        |
              | init                           |
              v                                |
    +-----------------+                        |
    | Step 2 - BERT-PT|                        |
    | 8 ep, 2 layers  |                        |
    +-----------------+                        |
              | init                           |
              v                                |
    +-----------------+                        |
    | Step 3 - BERT-FT| <----------------------+
    | 5 ep            |
    +-----------------+
              |
              v
      3 embedding CSVs (W2V, BERT-PT, BERT-FT)

Each model produces its own embedding CSV — the deliverable of the pipeline.
BERT's embedding layer is warm-started from the trained W2V embeddings
(W2V -> BERT), and each fine-tune is initialized from the pretrained model
(PT -> FT), which keeps the quarterly embeddings comparable over time.

    uv run python examples/quickstart.py
"""

import json
import random
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from gensim.models import Word2Vec


def section(title: str) -> None:
    print(f"\n=== {title} ===")


def run_step(cmd: list[str], cwd: Path, *, step: str) -> None:
    t0 = time.time()
    result = subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True, timeout=600)
    elapsed = time.time() - t0
    print(f"[{step}] returncode={result.returncode}  ({elapsed:.1f}s)")
    if result.returncode != 0:
        print("--- stderr (tail) ---")
        print(result.stderr[-1800:])
        raise RuntimeError(f"{step} failed")


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


# ----------------------------------------------------------------------------
# Setup
# ----------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

DATA = PROJECT_ROOT / "examples" / "data"
WORK = PROJECT_ROOT / "examples" / "_output"
shutil.rmtree(WORK, ignore_errors=True)
WORK.mkdir(parents=True, exist_ok=True)

print(f"Project root: {PROJECT_ROOT}")
print(f"Data dir    : {DATA}")
print(f"Workdir     : {WORK}")
print(f"Seed        : {SEED}")

# ----------------------------------------------------------------------------
# Step 1 — W2V pretrain on PT data
# ----------------------------------------------------------------------------
section("Step 1 - W2V pretrain on PT data")

w2v_folder = WORK / "w2v"
w2v_name = "w2v_d16"

w2v_config = {
    "embedding_dim": 16,
    "epochs": 10,
    "window": 5,
    "min_count": 1,
    "sg": 1,
    "sample": 1e-3,
    "negative_sample": 3,
    "data_path": str(DATA / "portfolio_pretrained.csv"),
    "data_format": "csv",
    "workers": 1,
    "save_folder": str(w2v_folder),
    "save_name": w2v_name,
    "save_format": ".model",
    "seed": SEED,
}
w2v_config_path = WORK / "w2v_config.json"
write_json(w2v_config_path, w2v_config)

run_step(
    [sys.executable, "-m", "asset_embeddings.scripts.train.w2v", "-c", str(w2v_config_path)],
    cwd=PROJECT_ROOT,
    step="W2V",
)

w2v_model_path = w2v_folder / f"{w2v_name}.model"
w2v_embedding_csv = w2v_folder / f"{w2v_name}_embedding.csv"

# Infer BERT vocab size from the trained W2V model (W2V vocab + 3 special tokens).
w2v = Word2Vec.load(str(w2v_model_path))
BERT_VOCAB = len(w2v.wv) + 3
print(f"W2V vocab: {len(w2v.wv)}   ->   BERT vocab (incl. [MASK]/[PAD]/[UNK]): {BERT_VOCAB}")

# ----------------------------------------------------------------------------
# Step 2 — BERT pretrain on PT data, init from W2V
# ----------------------------------------------------------------------------
section("Step 2 - BERT pretrain on PT data (init from W2V)")

bert_pt_folder = WORK / "bert_pt"
bert_pt_name = "bert_pt_d16"

bert_pt_config = {
    "model": {
        "model_type": "BERT",
        "model_checkpoint": None,
        "w2v_model": str(w2v_model_path),
        "embedding_file": None,
        "vocab_size": BERT_VOCAB,
        "hidden_size": 16,
        "num_hidden_layers": 2,
        "num_attention_heads": 2,
        "intermediate_size": 64,
        "max_position_embeddings": 64,
        "type_vocab_size": 1,
        "freeze_embedding": False,
        "freeze_encoder": False,
    },
    "tokenizer": {
        "w2v_model": str(w2v_model_path),
        "vocab_file": None,
        "pretrained_tokenizer_file": None,
        "alias_file": None,
    },
    "dataset": {
        "data_path": str(DATA / "portfolio_pretrained.csv"),
        "data_format": "csv",
        "id_key": "InvestorID",
        "portfolio_key": "Portfolio",
        "proportion1_key": "Proportion1",
        "proportion2_key": "Proportion2",
        "include_proportion": False,
        "max_length": 64,
        "num_repeats": 1,
        "mask_prob": 0.15,
        "mask_indices": None,
        "cache_size": 100,
    },
    "dataloader": {
        # CPU-safe overrides vs. the production configs:
        #   - num_workers=0 avoids Windows + Jupyter dataloader deadlocks.
        #   - smaller batch keeps a single-process loader responsive.
        "batch_size": 64,
        "shuffle": True,
        "num_workers": 0,
        "persistent_workers": False,
        "pin_memory": False,
        "drop_last": False,
    },
    "optimizer": {
        # AdamW instead of the default PagedAdam8bit -- the 8-bit optimizer
        # requires bitsandbytes + CUDA.
        "optimizer_type": "AdamW",
        "optimizer_kwargs": None,
        "learning_rate": 0.001,
        "lr_scheduler_type": "cosine",
        "lr_scheduler_warmup_steps": 0,
        "lr_scheduler_train_steps": None,
        "lr_scheduler_num_cycles": 0.5,
        "lr_scheduler_power": 0,
    },
    "train": {
        "max_epoches": 8,
        "accelerator_checkpoint": None,
        "validation_split": 0.2,
        "split_method": "random",
        "best_metric": "val_loss",
        "clip_grad_norm": None,
        "clip_grad_value": None,
        "gradient_accumulation_steps": 1,
        "detect_anomaly": False,
        "mixed_precision": "no",  # explicit; AMP only helps on CUDA
        "check_per_step": 200,
        "report_per_epoch": 1,
        "calculate_contextualized_embeddings": True,
        "save_per_epoch": 1,
        "save_folder": str(bert_pt_folder),
        "save_name": bert_pt_name,
        "save_format": ".safetensors",
        "seed": SEED,
    },
}
bert_pt_config_path = WORK / "bert_pt_config.json"
write_json(bert_pt_config_path, bert_pt_config)

run_step(
    [sys.executable, "-m", "asset_embeddings.scripts.train.bert", "-c", str(bert_pt_config_path)],
    cwd=PROJECT_ROOT,
    step="BERT-PT",
)

bert_pt_checkpoint = bert_pt_folder / f"{bert_pt_name}_best" / "model.safetensors"
bert_pt_embedding_csv = bert_pt_folder / f"{bert_pt_name}_best_contextual_embedding.csv"
assert bert_pt_checkpoint.exists() and bert_pt_embedding_csv.exists()

# ----------------------------------------------------------------------------
# Step 3 — BERT fine-tune on FT data, init from BERT-PT
# ----------------------------------------------------------------------------
section("Step 3 - BERT fine-tune on FT data (init from BERT-PT)")

bert_ft_folder = WORK / "bert_ft"
bert_ft_name = "bert_ft_d16"

# FT differs from PT in four places: checkpoint, data path, proportions on,
# and a gentler training schedule (lower LR, smaller batch, fewer epochs).
bert_ft_config = dict(bert_pt_config)
bert_ft_config["model"] = {**bert_pt_config["model"], "model_checkpoint": str(bert_pt_checkpoint)}
bert_ft_config["dataset"] = {
    **bert_pt_config["dataset"],
    "data_path": str(DATA / "portfolio_finetune.csv"),
    "include_proportion": True,
}
bert_ft_config["dataloader"] = {**bert_pt_config["dataloader"], "batch_size": 32}
bert_ft_config["optimizer"] = {**bert_pt_config["optimizer"], "learning_rate": 5e-4}
bert_ft_config["train"] = {
    **bert_pt_config["train"],
    "max_epoches": 5,
    "save_folder": str(bert_ft_folder),
    "save_name": bert_ft_name,
}

bert_ft_config_path = WORK / "bert_ft_config.json"
write_json(bert_ft_config_path, bert_ft_config)

run_step(
    [sys.executable, "-m", "asset_embeddings.scripts.train.bert", "-c", str(bert_ft_config_path)],
    cwd=PROJECT_ROOT,
    step="BERT-FT",
)

bert_ft_embedding_csv = bert_ft_folder / f"{bert_ft_name}_best_contextual_embedding.csv"
assert bert_ft_embedding_csv.exists()

# ----------------------------------------------------------------------------
# Results — the three embedding artifacts
# ----------------------------------------------------------------------------
section("Results - embedding artifacts")

artifacts = {
    "W2V": (w2v_embedding_csv, "PT data"),
    "BERT-PT": (bert_pt_embedding_csv, "PT data"),
    "BERT-FT": (bert_ft_embedding_csv, "FT data"),
}

print(f"\n{'Model':<10}{'Trained on':<12}{'Shape':<14}Path")
print("-" * 90)
embeddings = {}
for label, (csv_path, trained) in artifacts.items():
    df = pd.read_csv(csv_path, dtype={"Token": str})
    embeddings[label] = df.set_index("Token")
    print(f"{label:<10}{trained:<12}{f'{df.shape[0]} x {df.shape[1] - 1}':<14}{csv_path.relative_to(PROJECT_ROOT)}")

# A quick look at what the embeddings encode: nearest neighbours of one stock
# in each space. The synthetic universe has latent cluster structure, so
# neighbours of a stock should stay within its cluster. The token column holds
# special tokens like [CLS] alongside stock codes, so skip those when probing.
probe = next(t for t in embeddings["W2V"].index if not t.startswith("["))
print(f"\nTop-5 cosine neighbours of stock {probe}:")
for label, emb in embeddings.items():
    x = emb.to_numpy()
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    sims = x @ x[emb.index.get_loc(probe)]
    order = np.argsort(-sims)
    neighbours = [emb.index[i] for i in order if emb.index[i] != probe and not emb.index[i].startswith("[")][:5]
    print(f"  {label:8s}  {', '.join(neighbours)}")

print(
    "\nInterpretation:\n"
    "  Each CSV is a (Token, Embed_1..Embed_d) matrix -- the asset-embedding\n"
    "  deliverable. BERT's embedding layer is warm-started from the trained\n"
    "  W2V embeddings, and BERT-FT is initialized from BERT-PT, so the FT\n"
    "  embedding moves with the fine-tune period's holdings while staying\n"
    "  comparable to its pretrained starting point."
)
