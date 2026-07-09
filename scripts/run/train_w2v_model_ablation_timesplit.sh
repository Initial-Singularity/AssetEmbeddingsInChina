#!/bin/bash
# ── W2V Model Ablation TimeSplit (CBOW, 50/50 split, 16 quarters) Training ──
# 16-quarter robustness version of w2v_model_ablation. Data uses
# MutualFundShareHoldings_TimeSplit/ (50/50 PT/FT split).
#
# Pretrain (7 dims each): AssetW2V_cbow, AssetBERT_cbow
# Finetune (7 dims × 16 quarters each): AssetW2V_cbow, AssetBERT_cbow
# Total: 2×7 + 2×7×16 = 238 runs
#
# Markers: w2v bert w2v_model_ts pretrain finetune | d{4..128} | {quarter}
#
# Usage:
#   bash scripts/run/run_notify.sh -a results/db/w2v_model_ablation_timesplit.db scripts/run/train_w2v_model_ablation_timesplit.sh

source scripts/run/run_helper.sh

DB="results/db/w2v_model_ablation_timesplit.db"
DIMS="4 8 10 16 32 64 128"
QUARTERS="2020Q4 2021Q1 2021Q2 2021Q3 2021Q4 2022Q1 2022Q2 2022Q3 2022Q4 2023Q1 2023Q2 2023Q3 2023Q4 2024Q1 2024Q2 2024Q3"

# === Pretrain ===
echo "=== Pretrain: W2V (CBOW, timesplit) ==="
for dim in $DIMS; do
    try_run -m w2v -m w2v_model_ts -m pretrain -m "d${dim}" "W2V_cbow pretrain d${dim}" \
        uv run python -m asset_embeddings.scripts.train.w2v \
            --config "configs/w2v_model_ablation_timesplit/pretrained/AssetW2V_cbow/AssetW2V_cbow_d${dim}_pretrained.json" \
            --log "checkpoints/w2v_model_ablation_timesplit/pretrained/AssetW2V_cbow/d${dim}/train.log" \
                  "logs/w2v_model_ablation_timesplit/train/pretrained/AssetW2V_cbow/d${dim}.log" \
            -r "$DB"
done

echo "=== Pretrain: BERT (CBOW init, timesplit) ==="
for dim in $DIMS; do
    try_run -m bert -m w2v_model_ts -m pretrain -m "d${dim}" "BERT_cbow pretrain d${dim}" \
        uv run python -m asset_embeddings.scripts.train.bert \
            --config "configs/w2v_model_ablation_timesplit/pretrained/AssetBERT_cbow/AssetBERT_cbow_d${dim}_pretrained.json" \
            --log "checkpoints/w2v_model_ablation_timesplit/pretrained/AssetBERT_cbow/d${dim}/train.log" \
                  "logs/w2v_model_ablation_timesplit/train/pretrained/AssetBERT_cbow/d${dim}.log" \
            -r "$DB"
done

# === Finetune ===
echo "=== Finetune: W2V (CBOW, timesplit) ==="
for dim in $DIMS; do
    for quarter in $QUARTERS; do
        try_run -m w2v -m w2v_model_ts -m finetune -m "d${dim}" -m "${quarter,,}" "W2V_cbow finetune d${dim} ${quarter}" \
            uv run python -m asset_embeddings.scripts.train.w2v \
                --config "configs/w2v_model_ablation_timesplit/finetune/AssetW2V_cbow/d${dim}/AssetW2V_cbow_d${dim}_${quarter}.json" \
                --log "checkpoints/w2v_model_ablation_timesplit/finetune/AssetW2V_cbow/d${dim}/${quarter}/train.log" \
                      "logs/w2v_model_ablation_timesplit/train/finetune/AssetW2V_cbow/d${dim}/${quarter}.log" \
                -r "$DB"
    done
done

echo "=== Finetune: BERT (CBOW-init pretrain → FT, timesplit) ==="
for dim in $DIMS; do
    for quarter in $QUARTERS; do
        try_run -m bert -m w2v_model_ts -m finetune -m "d${dim}" -m "${quarter,,}" "BERT_cbow finetune d${dim} ${quarter}" \
            uv run python -m asset_embeddings.scripts.train.bert \
                --config "configs/w2v_model_ablation_timesplit/finetune/AssetBERT_cbow/d${dim}/AssetBERT_cbow_d${dim}_${quarter}.json" \
                --log "checkpoints/w2v_model_ablation_timesplit/finetune/AssetBERT_cbow/d${dim}/${quarter}/train.log" \
                      "logs/w2v_model_ablation_timesplit/train/finetune/AssetBERT_cbow/d${dim}/${quarter}.log" \
                -r "$DB"
    done
done

print_summary
