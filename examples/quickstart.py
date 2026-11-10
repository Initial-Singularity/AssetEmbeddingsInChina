"""End-to-end asset-embeddings pipeline on synthetic data.

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
              |  3 embeddings collected (W2V, BERT-PT, BERT-FT)
              v
    +-----------------+
    | Step 4 - RV x3  | <----- benchmark.csv (2024Q2)
    +-----------------+

Each model produces its own embedding CSV; all three are scored against the
same FT-aligned benchmark via Relative Valuation. The benchmark is generated
from FT-period factors, so PT-trained models (W2V, BERT-PT) plateau together
while BERT-FT picks up the FT-period drift and lifts R^2.

    uv run python examples/quickstart.py
"""

import json
import os
import random
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from gensim.models import Word2Vec


# Child processes write UTF-8 and the parent decodes UTF-8, whatever the console code page (e.g. GBK on Windows).
CHILD_ENV = {**os.environ, "PYTHONUTF8": "1"}


def section(title: str) -> None:
    print(f"\n=== {title} ===")


def run_step(cmd: list[str], cwd: Path, *, step: str) -> None:
    t0 = time.time()
    result = subprocess.run(
        cmd, cwd=str(cwd), capture_output=True, encoding="utf-8", errors="replace", env=CHILD_ENV, timeout=600
    )
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
    "accelerator": {"mixed_precision": "no"},  # explicit; AMP only helps on CUDA
    "train": {
        "max_epochs": 8,
        "validation_split": 0.2,
        "split_method": "random",
        "best_metric": "val_loss",
        "clip_grad_norm": None,
        "clip_grad_value": None,
        "detect_anomaly": False,
        "log_every_steps": 200,
        "report_every_epochs": 1,
        "calculate_contextualized_embeddings": True,
        "save_every_epochs": 1,
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
    "max_epochs": 5,
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
# Step 4 — Relative Valuation × 3
# ----------------------------------------------------------------------------
section("Step 4 - RV evaluation x 3 (W2V vs BERT-PT vs BERT-FT)")

benchmark_dir = WORK / "benchmark"
benchmark_dir.mkdir(exist_ok=True)
shutil.copy(DATA / "benchmark.csv", benchmark_dir / "2024Q2.csv")


def rv_eval(embedding_csv: Path, model_type: str, label: str) -> float:
    cfg = {
        "model_type": model_type,
        "embedding_path": [str(embedding_csv)],
        "benchmark_path": str(benchmark_dir / "2024Q2.csv"),
        "periods": ["2024Q2"],
        "features": ["embedding"],
        "align_chars": True,
        "normalization": "minmax",
        "seed": SEED,
    }
    cfg_path = WORK / f"rv_{embedding_csv.stem}.json"
    write_json(cfg_path, cfg)
    res = subprocess.run(
        [sys.executable, "-m", "asset_embeddings.scripts.eval.rv", "-c", str(cfg_path)],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        encoding="utf-8",
        errors="replace",
        env=CHILD_ENV,
        timeout=120,
    )
    if res.returncode != 0:
        print(res.stderr[-1500:])
        raise RuntimeError(f"RV for {label} failed")
    for line in (res.stderr + res.stdout).splitlines():
        if "RV Score for period" in line:
            return float(line.rsplit(":", 1)[1].strip())
    raise RuntimeError(f"Could not extract RV score for {label}")


scores = {
    "W2V": rv_eval(w2v_embedding_csv, "AssetW2V", "W2V"),
    "BERT-PT": rv_eval(bert_pt_embedding_csv, "AssetBERT", "BERT-PT"),
    "BERT-FT": rv_eval(bert_ft_embedding_csv, "AssetBERT", "BERT-FT"),
}
for label, s in scores.items():
    print(f"  {label:8s}  RV = {s:.4f}")

# ----------------------------------------------------------------------------
# Results
# ----------------------------------------------------------------------------
section("Results")

baseline = scores["W2V"]
notes = {
    "W2V": "baseline (PT-trained)",
    "BERT-PT": "matches W2V (same training data)",
    "BERT-FT": "+{:.3f} from FT lift".format(scores["BERT-FT"] - baseline),
}

print(f"\n{'Model':<10}{'Trained on':<14}{'RV @ 2024Q2':>14}    Note")
print("-" * 70)
for label, score in scores.items():
    trained = "FT data" if label == "BERT-FT" else "PT data"
    print(f"{label:<10}{trained:<14}{score:>14.4f}    {notes[label]}")

print(
    "\nInterpretation:\n"
    "  W2V and BERT-PT are both PT-trained, so they recover the PT-period\n"
    "  cluster structure but miss the cluster-level drift baked into the\n"
    "  FT-aligned benchmark -- their R^2 plateau together. BERT-FT, fine-\n"
    "  tuned on FT-period portfolios, picks up the drift and posts a clear\n"
    "  R^2 lift.\n"
    "\n"
    "On real CSMAR data the lift tends to be larger, since real markets\n"
    "carry richer time-varying structure than this cluster-drift toy model."
)
