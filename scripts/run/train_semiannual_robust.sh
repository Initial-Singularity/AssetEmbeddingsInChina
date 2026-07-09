#!/bin/bash
# ── Semi-Annual Robustness Training ─────────────────────
# Re-train all models on semi-annual (Q2/Q4 only) data.
# Mirrors train.sh structure with configs under configs/semiannual_robust/{pretrained,finetune}/.
#
# Pretrain (7 dims each): AssetW2V, AssetBERT
# Finetune (7 dims × 3 quarters each): AssetW2V, AssetBERT
# (RS skipped: no pretrain stage, so SemiAnnual results would be identical to
#  quarterly Q2/Q4 — SemiAnnual data is a `cp` of those files.)
# Total: 2×7 + (1+1)×7×3 = 56 runs
#
# Markers: w2v bert pretrain finetune semiannual | d{4..128} | {quarter}
#
# Usage:
#   bash scripts/run/run_notify.sh -a results/db/semiannual_robust.db scripts/run/train_semiannual_robust.sh
#
# Examples:
#   bash scripts/run/run_notify.sh -k bert,pretrain scripts/run/train_semiannual_robust.sh
#   bash scripts/run/run_notify.sh -k finetune,d64 scripts/run/train_semiannual_robust.sh

source scripts/run/run_helper.sh

DB="results/db/semiannual_robust.db"
DIMS="4 8 10 16 32 64 128"
QUARTERS="2023Q2 2023Q4 2024Q2"
PT_PREFIX="configs/semiannual_robust/pretrained"
FT_PREFIX="configs/semiannual_robust/finetune"

# === Pretrain ===
echo "=== Pretrain: W2V ==="
for dim in $DIMS; do
    try_run -m w2v -m pretrain -m semiannual -m "d${dim}" "W2V pretrain d${dim}" \
        uv run python -m asset_embeddings.scripts.train.w2v \
            --config "${PT_PREFIX}/AssetW2V/AssetW2V_d${dim}_pretrained.json" \
            --log "checkpoints/semiannual_robust/pretrained/AssetW2V/d${dim}/train.log" \
                  "logs/semiannual_robust/train/pretrained/AssetW2V/d${dim}.log" \
            -r "$DB"
done

echo "=== Pretrain: BERT ==="
for dim in $DIMS; do
    try_run -m bert -m pretrain -m semiannual -m "d${dim}" "BERT pretrain d${dim}" \
        uv run python -m asset_embeddings.scripts.train.bert \
            --config "${PT_PREFIX}/AssetBERT/AssetBERT_d${dim}_pretrained.json" \
            --log "checkpoints/semiannual_robust/pretrained/AssetBERT/d${dim}/train.log" \
                  "logs/semiannual_robust/train/pretrained/AssetBERT/d${dim}.log" \
            -r "$DB"
done

# === Finetune ===
echo "=== Finetune: W2V ==="
for dim in $DIMS; do
    for quarter in $QUARTERS; do
        try_run -m w2v -m finetune -m semiannual -m "d${dim}" -m "${quarter,,}" "W2V finetune d${dim} ${quarter}" \
            uv run python -m asset_embeddings.scripts.train.w2v \
                --config "${FT_PREFIX}/AssetW2V/d${dim}/AssetW2V_d${dim}_${quarter}.json" \
                --log "checkpoints/semiannual_robust/finetune/AssetW2V/d${dim}/${quarter}/train.log" \
                      "logs/semiannual_robust/train/finetune/AssetW2V/d${dim}/${quarter}.log" \
                -r "$DB"
    done
done

echo "=== Finetune: BERT ==="
for dim in $DIMS; do
    for quarter in $QUARTERS; do
        try_run -m bert -m finetune -m semiannual -m "d${dim}" -m "${quarter,,}" "BERT finetune d${dim} ${quarter}" \
            uv run python -m asset_embeddings.scripts.train.bert \
                --config "${FT_PREFIX}/AssetBERT/d${dim}/AssetBERT_d${dim}_${quarter}.json" \
                --log "checkpoints/semiannual_robust/finetune/AssetBERT/d${dim}/${quarter}/train.log" \
                      "logs/semiannual_robust/train/finetune/AssetBERT/d${dim}/${quarter}.log" \
                -r "$DB"
    done
done

print_summary
