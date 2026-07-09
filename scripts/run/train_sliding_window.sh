#!/bin/bash
# ── Train: Sliding Window Pretrain Strategy ─────────────
# For each window size N, retrain W2V + BERT on most recent N quarters,
# then finetune on current quarter data.
#
# N=8 (2yr), N=20 (5yr), N=40 (10yr)
# Per N: 7×6 W2V pretrain + 7×6 BERT pretrain + 7×6 finetune = 126 runs
# Total: 3 × 126 = 378 runs
#
# Markers: w2v bert pretrain finetune sliding_window pt_strategy_ablation sw{N} | d{4..128} | {quarter}
#
# Usage:
#   bash scripts/run/run_notify.sh -a results/db/sliding_window.db scripts/run/train_sliding_window.sh
#
# Examples:
#   bash scripts/run/run_notify.sh -k sw20,d64 scripts/run/train_sliding_window.sh
#   bash scripts/run/run_notify.sh -k finetune,sw8 scripts/run/train_sliding_window.sh

source scripts/run/run_helper.sh

DIMS="4 8 10 16 32 64 128"
QUARTERS="2023Q2 2023Q3 2023Q4 2024Q1 2024Q2 2024Q3"
WINDOW_SIZES="8 20 40"
DB="results/db/sliding_window.db"

for N in $WINDOW_SIZES; do
    METHOD="sw${N}"
    CFG="configs/pt_strategy_ablation/sliding_window/N${N}"

    # ========== W2V Pretrain ==========
    echo "=== Sliding Window N=${N}: W2V pretrain ==="
    for dim in $DIMS; do
        for quarter in $QUARTERS; do
            try_run -m w2v -m pretrain -m sliding_window -m pt_strategy_ablation -m "${METHOD}" -m "d${dim}" -m "${quarter,,}" \
                "W2V ${METHOD} pretrain d${dim} ${quarter}" \
                uv run python -m asset_embeddings.scripts.train.w2v \
                    --config "${CFG}/pretrained/AssetW2V/d${dim}/AssetW2V_${METHOD}_d${dim}_${quarter}.json" \
                    --log "checkpoints/pt_strategy_ablation/sliding_window/N${N}/pretrained/AssetW2V/d${dim}/${quarter}/train.log" \
                          "logs/pt_strategy_ablation/sliding_window/N${N}/pretrain/AssetW2V/d${dim}/${quarter}.log" \
                    -r "$DB"
        done
    done

    # ========== BERT Pretrain ==========
    echo "=== Sliding Window N=${N}: BERT pretrain ==="
    for dim in $DIMS; do
        for quarter in $QUARTERS; do
            try_run -m bert -m pretrain -m sliding_window -m pt_strategy_ablation -m "${METHOD}" -m "d${dim}" -m "${quarter,,}" \
                "BERT ${METHOD} pretrain d${dim} ${quarter}" \
                uv run python -m asset_embeddings.scripts.train.bert \
                    --config "${CFG}/pretrained/AssetBERT/d${dim}/AssetBERT_${METHOD}_d${dim}_${quarter}_pretrained.json" \
                    --log "checkpoints/pt_strategy_ablation/sliding_window/N${N}/pretrained/AssetBERT/d${dim}/${quarter}/train.log" \
                          "logs/pt_strategy_ablation/sliding_window/N${N}/pretrain/AssetBERT/d${dim}/${quarter}.log" \
                    -r "$DB"
        done
    done

    # ========== BERT Finetune ==========
    echo "=== Sliding Window N=${N}: BERT finetune ==="
    for dim in $DIMS; do
        for quarter in $QUARTERS; do
            try_run -m bert -m finetune -m sliding_window -m pt_strategy_ablation -m "${METHOD}" -m "d${dim}" -m "${quarter,,}" \
                "BERT ${METHOD} finetune d${dim} ${quarter}" \
                uv run python -m asset_embeddings.scripts.train.bert \
                    --config "${CFG}/finetune/AssetBERT/d${dim}/AssetBERT_${METHOD}_d${dim}_${quarter}.json" \
                    --log "checkpoints/pt_strategy_ablation/sliding_window/N${N}/finetune/AssetBERT/d${dim}/${quarter}/train.log" \
                          "logs/pt_strategy_ablation/sliding_window/N${N}/finetune/AssetBERT/d${dim}/${quarter}.log" \
                    -r "$DB"
        done
    done
done

print_summary
