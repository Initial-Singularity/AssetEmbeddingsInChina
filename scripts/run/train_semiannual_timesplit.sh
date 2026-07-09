#!/bin/bash
# ── Semi-Annual x Time-Split Robustness Training ──────────
# Re-train all models on the Q2/Q4 subset of the 50/50 time-split
# (semi-annual restriction applied on top of timesplit_robust).
# Mirrors train_semiannual_robust.sh structure with configs under
# configs/semiannual_timesplit/{pretrained,finetune}/.
#
# Pretrain (7 dims each): AssetW2V, AssetBERT
# Finetune (7 dims x 8 quarters each): AssetW2V, AssetBERT
# (RS skipped: no pretrain stage, so semi-annual x timesplit results would be
#  identical to timesplit_robust on the Q2/Q4 subset — finetune CSVs are
#  byte-identical via `cp` from MutualFundShareHoldings_TimeSplit/.)
# Total: 2x7 + (1+1)x7x8 = 126 runs
#
# Markers: w2v bert pretrain finetune semiannual_timesplit | d{4..128} | {quarter}
#
# Usage:
#   bash scripts/run/run_notify.sh -a results/db/semiannual_timesplit.db scripts/run/train_semiannual_timesplit.sh
#
# Examples:
#   bash scripts/run/run_notify.sh -k bert,pretrain scripts/run/train_semiannual_timesplit.sh
#   bash scripts/run/run_notify.sh -k finetune,d64 scripts/run/train_semiannual_timesplit.sh

source scripts/run/run_helper.sh

DB="results/db/semiannual_timesplit.db"
DIMS="4 8 10 16 32 64 128"
QUARTERS="2020Q4 2021Q2 2021Q4 2022Q2 2022Q4 2023Q2 2023Q4 2024Q2"
PT_PREFIX="configs/semiannual_timesplit/pretrained"
FT_PREFIX="configs/semiannual_timesplit/finetune"

# === Pretrain ===
echo "=== Pretrain: W2V ==="
for dim in $DIMS; do
    try_run -m w2v -m pretrain -m semiannual_timesplit -m "d${dim}" "W2V pretrain d${dim}" \
        uv run python -m asset_embeddings.scripts.train.w2v \
            --config "${PT_PREFIX}/AssetW2V/AssetW2V_d${dim}_pretrained.json" \
            --log "checkpoints/semiannual_timesplit/pretrained/AssetW2V/d${dim}/train.log" \
                  "logs/semiannual_timesplit/train/pretrained/AssetW2V/d${dim}.log" \
            -r "$DB"
done

echo "=== Pretrain: BERT ==="
for dim in $DIMS; do
    try_run -m bert -m pretrain -m semiannual_timesplit -m "d${dim}" "BERT pretrain d${dim}" \
        uv run python -m asset_embeddings.scripts.train.bert \
            --config "${PT_PREFIX}/AssetBERT/AssetBERT_d${dim}_pretrained.json" \
            --log "checkpoints/semiannual_timesplit/pretrained/AssetBERT/d${dim}/train.log" \
                  "logs/semiannual_timesplit/train/pretrained/AssetBERT/d${dim}.log" \
            -r "$DB"
done

# === Finetune ===
echo "=== Finetune: W2V ==="
for dim in $DIMS; do
    for quarter in $QUARTERS; do
        try_run -m w2v -m finetune -m semiannual_timesplit -m "d${dim}" -m "${quarter,,}" "W2V finetune d${dim} ${quarter}" \
            uv run python -m asset_embeddings.scripts.train.w2v \
                --config "${FT_PREFIX}/AssetW2V/d${dim}/AssetW2V_d${dim}_${quarter}.json" \
                --log "checkpoints/semiannual_timesplit/finetune/AssetW2V/d${dim}/${quarter}/train.log" \
                      "logs/semiannual_timesplit/train/finetune/AssetW2V/d${dim}/${quarter}.log" \
                -r "$DB"
    done
done

echo "=== Finetune: BERT ==="
for dim in $DIMS; do
    for quarter in $QUARTERS; do
        try_run -m bert -m finetune -m semiannual_timesplit -m "d${dim}" -m "${quarter,,}" "BERT finetune d${dim} ${quarter}" \
            uv run python -m asset_embeddings.scripts.train.bert \
                --config "${FT_PREFIX}/AssetBERT/d${dim}/AssetBERT_d${dim}_${quarter}.json" \
                --log "checkpoints/semiannual_timesplit/finetune/AssetBERT/d${dim}/${quarter}/train.log" \
                      "logs/semiannual_timesplit/train/finetune/AssetBERT/d${dim}/${quarter}.log" \
                -r "$DB"
    done
done

print_summary
