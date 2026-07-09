#!/bin/bash
# ── Text Embedding Main Grid ────────────────────────────
# Runs the 18-config text embedding grid: 9 (provider, model) pairs x {zh, en}.
# Configs are produced from `templates/text_embedding.json` via `python -m scripts.tools.fast_generator`.
# zh-short ablation arm is invoked manually, not by this script.
#
# Markers: text_embed | openai cohere voyage gemini local | zh en
#
# Usage:
#   # Step 0 — (Re)generate configs:
#   uv run python -m scripts.tools.fast_generator config --from-preset templates/text_embedding.json
#
#   # Step 1 — Run all 18:
#   bash scripts/run/run_notify.sh scripts/run/data_text_embeddings.sh
#
# Examples:
#   bash scripts/run/run_notify.sh -k openai            scripts/run/data_text_embeddings.sh
#   bash scripts/run/run_notify.sh -k local,zh          scripts/run/data_text_embeddings.sh
#   bash scripts/run/run_notify.sh --exclude local      scripts/run/data_text_embeddings.sh

source scripts/run/run_helper.sh

mkdir -p logs/text_embedding

echo "=== Text Embedding Main Grid ==="

# OpenAI
try_run -m text_embed -m openai -m zh "OpenAI text-embedding-3-small zh" \
    uv run python -m asset_embeddings.scripts.data.text_embedding \
        -c configs/text_embedding/openai_text-embedding-3-small_zh.json \
        -l logs/text_embedding/openai_text-embedding-3-small_zh.log
try_run -m text_embed -m openai -m en "OpenAI text-embedding-3-small en" \
    uv run python -m asset_embeddings.scripts.data.text_embedding \
        -c configs/text_embedding/openai_text-embedding-3-small_en.json \
        -l logs/text_embedding/openai_text-embedding-3-small_en.log
try_run -m text_embed -m openai -m zh "OpenAI text-embedding-3-large zh" \
    uv run python -m asset_embeddings.scripts.data.text_embedding \
        -c configs/text_embedding/openai_text-embedding-3-large_zh.json \
        -l logs/text_embedding/openai_text-embedding-3-large_zh.log
try_run -m text_embed -m openai -m en "OpenAI text-embedding-3-large en" \
    uv run python -m asset_embeddings.scripts.data.text_embedding \
        -c configs/text_embedding/openai_text-embedding-3-large_en.json \
        -l logs/text_embedding/openai_text-embedding-3-large_en.log

# Cohere
try_run -m text_embed -m cohere -m zh "Cohere embed-v4.0 zh" \
    uv run python -m asset_embeddings.scripts.data.text_embedding \
        -c configs/text_embedding/cohere_embed-v4.0_zh.json \
        -l logs/text_embedding/cohere_embed-v4.0_zh.log
try_run -m text_embed -m cohere -m en "Cohere embed-v4.0 en" \
    uv run python -m asset_embeddings.scripts.data.text_embedding \
        -c configs/text_embedding/cohere_embed-v4.0_en.json \
        -l logs/text_embedding/cohere_embed-v4.0_en.log

# Voyage
try_run -m text_embed -m voyage -m zh "Voyage voyage-3-large zh" \
    uv run python -m asset_embeddings.scripts.data.text_embedding \
        -c configs/text_embedding/voyage_voyage-3-large_zh.json \
        -l logs/text_embedding/voyage_voyage-3-large_zh.log
try_run -m text_embed -m voyage -m en "Voyage voyage-3-large en" \
    uv run python -m asset_embeddings.scripts.data.text_embedding \
        -c configs/text_embedding/voyage_voyage-3-large_en.json \
        -l logs/text_embedding/voyage_voyage-3-large_en.log

# Gemini
try_run -m text_embed -m gemini -m zh "Gemini gemini-embedding-001 zh" \
    uv run python -m asset_embeddings.scripts.data.text_embedding \
        -c configs/text_embedding/gemini_gemini-embedding-001_zh.json \
        -l logs/text_embedding/gemini_gemini-embedding-001_zh.log
try_run -m text_embed -m gemini -m en "Gemini gemini-embedding-001 en" \
    uv run python -m asset_embeddings.scripts.data.text_embedding \
        -c configs/text_embedding/gemini_gemini-embedding-001_en.json \
        -l logs/text_embedding/gemini_gemini-embedding-001_en.log

# Local — Qwen3-Embedding
try_run -m text_embed -m local -m zh "Local Qwen3-Embedding-0.6B zh" \
    uv run python -m asset_embeddings.scripts.data.text_embedding \
        -c configs/text_embedding/local_Qwen-Qwen3-Embedding-0.6B_zh.json \
        -l logs/text_embedding/local_Qwen-Qwen3-Embedding-0.6B_zh.log
try_run -m text_embed -m local -m en "Local Qwen3-Embedding-0.6B en" \
    uv run python -m asset_embeddings.scripts.data.text_embedding \
        -c configs/text_embedding/local_Qwen-Qwen3-Embedding-0.6B_en.json \
        -l logs/text_embedding/local_Qwen-Qwen3-Embedding-0.6B_en.log
try_run -m text_embed -m local -m zh "Local Qwen3-Embedding-4B zh" \
    uv run python -m asset_embeddings.scripts.data.text_embedding \
        -c configs/text_embedding/local_Qwen-Qwen3-Embedding-4B_zh.json \
        -l logs/text_embedding/local_Qwen-Qwen3-Embedding-4B_zh.log
try_run -m text_embed -m local -m en "Local Qwen3-Embedding-4B en" \
    uv run python -m asset_embeddings.scripts.data.text_embedding \
        -c configs/text_embedding/local_Qwen-Qwen3-Embedding-4B_en.json \
        -l logs/text_embedding/local_Qwen-Qwen3-Embedding-4B_en.log
try_run -m text_embed -m local -m zh "Local Qwen3-Embedding-8B zh" \
    uv run python -m asset_embeddings.scripts.data.text_embedding \
        -c configs/text_embedding/local_Qwen-Qwen3-Embedding-8B_zh.json \
        -l logs/text_embedding/local_Qwen-Qwen3-Embedding-8B_zh.log
try_run -m text_embed -m local -m en "Local Qwen3-Embedding-8B en" \
    uv run python -m asset_embeddings.scripts.data.text_embedding \
        -c configs/text_embedding/local_Qwen-Qwen3-Embedding-8B_en.json \
        -l logs/text_embedding/local_Qwen-Qwen3-Embedding-8B_en.log

# Local — BGE-m3
try_run -m text_embed -m local -m zh "Local BGE-m3 zh" \
    uv run python -m asset_embeddings.scripts.data.text_embedding \
        -c configs/text_embedding/local_BAAI-bge-m3_zh.json \
        -l logs/text_embedding/local_BAAI-bge-m3_zh.log
try_run -m text_embed -m local -m en "Local BGE-m3 en" \
    uv run python -m asset_embeddings.scripts.data.text_embedding \
        -c configs/text_embedding/local_BAAI-bge-m3_en.json \
        -l logs/text_embedding/local_BAAI-bge-m3_en.log

print_summary
