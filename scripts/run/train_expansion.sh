#!/bin/bash
# ── Train: Expansion Pretrain Strategy ──────────────────
# AE: Treatment of sparse holdings and quarterly fine-tuning
# Reviewer 2 Key Issues #i: Show how quarterly fine-tuning addresses specific issues
#
# Expansion: for each quarter t, retrain W2V + BERT from scratch on
# cumulative data up to t-1, then finetune on quarter t.
# Q2023Q2 reuses base pretrain (data scope identical).
#
# Total: 5×7 W2V pretrain + 5×7 BERT pretrain + 6×7 finetune = 112 runs
#
# Markers: w2v bert pretrain finetune expansion pt_strategy_ablation | d{4..128} | {quarter}
#
# Usage:
#   bash scripts/run/run_notify.sh -a results/db/expansion.db scripts/run/train_expansion.sh
#
# Examples:
#   bash scripts/run/run_notify.sh -k pretrain,d64 scripts/run/train_expansion.sh
#   bash scripts/run/run_notify.sh -k finetune scripts/run/train_expansion.sh

source scripts/run/run_helper.sh

DIMS="4 8 10 16 32 64 128"
QUARTERS="2023Q2 2023Q3 2023Q4 2024Q1 2024Q2 2024Q3"
DB="results/db/expansion.db"
CFG="configs/pt_strategy_ablation/expansion"

# ========== W2V Pretrain (skip Q2023Q2 — reuses base) ==========
echo "=== Expansion: W2V pretrain ==="
for dim in $DIMS; do
    for quarter in $QUARTERS; do
        [ "$quarter" = "2023Q2" ] && continue
        try_run -m w2v -m pretrain -m expansion -m pt_strategy_ablation -m "d${dim}" -m "${quarter,,}" \
            "W2V expansion pretrain d${dim} ${quarter}" \
            uv run python -m asset_embeddings.scripts.train.w2v \
                --config "${CFG}/pretrained/AssetW2V/d${dim}/AssetW2V_expansion_d${dim}_${quarter}.json" \
                --log "checkpoints/pt_strategy_ablation/expansion/pretrained/AssetW2V/d${dim}/${quarter}/train.log" \
                      "logs/pt_strategy_ablation/expansion/pretrain/AssetW2V/d${dim}/${quarter}.log" \
                -r "$DB"
    done
done

# ========== BERT Pretrain (skip Q2023Q2) ==========
echo "=== Expansion: BERT pretrain ==="
for dim in $DIMS; do
    for quarter in $QUARTERS; do
        [ "$quarter" = "2023Q2" ] && continue
        try_run -m bert -m pretrain -m expansion -m pt_strategy_ablation -m "d${dim}" -m "${quarter,,}" \
            "BERT expansion pretrain d${dim} ${quarter}" \
            uv run python -m asset_embeddings.scripts.train.bert \
                --config "${CFG}/pretrained/AssetBERT/d${dim}/AssetBERT_expansion_d${dim}_${quarter}_pretrained.json" \
                --log "checkpoints/pt_strategy_ablation/expansion/pretrained/AssetBERT/d${dim}/${quarter}/train.log" \
                      "logs/pt_strategy_ablation/expansion/pretrain/AssetBERT/d${dim}/${quarter}.log" \
                -r "$DB"
    done
done

# ========== BERT Finetune (all quarters) ==========
echo "=== Expansion: BERT finetune ==="
for dim in $DIMS; do
    for quarter in $QUARTERS; do
        try_run -m bert -m finetune -m expansion -m pt_strategy_ablation -m "d${dim}" -m "${quarter,,}" \
            "BERT expansion finetune d${dim} ${quarter}" \
            uv run python -m asset_embeddings.scripts.train.bert \
                --config "${CFG}/finetune/AssetBERT/d${dim}/AssetBERT_expansion_d${dim}_${quarter}.json" \
                --log "checkpoints/pt_strategy_ablation/expansion/finetune/AssetBERT/d${dim}/${quarter}/train.log" \
                      "logs/pt_strategy_ablation/expansion/finetune/AssetBERT/d${dim}/${quarter}.log" \
                -r "$DB"
    done
done

print_summary
