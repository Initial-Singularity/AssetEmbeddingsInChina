#!/bin/bash
# ── 50/50 Time-Split Robustness Training ──────────────────
# Re-train all models with 50% pretrain / 50% finetune split
# (baseline uses ~92% pretrain / ~8% finetune).
# Mirrors train.sh structure with configs under configs/timesplit_robust/{pretrained,finetune}/.
#
# Pretrain (7 dims each): AssetW2V, AssetBERT
# Finetune (7 dims x 16 quarters each):
#   AssetRS (4 variants), AssetW2V, AssetBERT
# Total: 2x7 + (4+1+1)x7x16 = 686 runs
#
# Markers: w2v bert rs rs_binary rs_ranks rs_level0 rs_levelmin
#          pretrain finetune timesplit | d{4..128} | {quarter}
#
# Usage:
#   bash scripts/run/run_notify.sh -a results/db/timesplit_robust.db scripts/run/train_timesplit_robust.sh
#
# Examples:
#   bash scripts/run/run_notify.sh -k bert,pretrain scripts/run/train_timesplit_robust.sh
#   bash scripts/run/run_notify.sh -k finetune,d64 --exclude rs_binary,rs_ranks scripts/run/train_timesplit_robust.sh

source scripts/run/run_helper.sh

DB="results/db/timesplit_robust.db"
DIMS="4 8 10 16 32 64 128"
QUARTERS="2020Q4 2021Q1 2021Q2 2021Q3 2021Q4 2022Q1 2022Q2 2022Q3 2022Q4 2023Q1 2023Q2 2023Q3 2023Q4 2024Q1 2024Q2 2024Q3"
RS_VARIANTS="Binary Ranks Level0 LevelMin"
PT_PREFIX="configs/timesplit_robust/pretrained"
FT_PREFIX="configs/timesplit_robust/finetune"

# === Pretrain ===
echo "=== Pretrain: W2V ==="
for dim in $DIMS; do
    try_run -m w2v -m pretrain -m timesplit -m "d${dim}" "W2V pretrain d${dim}" \
        uv run python -m asset_embeddings.scripts.train.w2v \
            --config "${PT_PREFIX}/AssetW2V/AssetW2V_d${dim}_pretrained.json" \
            --log "checkpoints/timesplit_robust/pretrained/AssetW2V/d${dim}/train.log" \
                  "logs/timesplit_robust/train/pretrained/AssetW2V/d${dim}.log" \
            -r "$DB"
done

echo "=== Pretrain: BERT ==="
for dim in $DIMS; do
    try_run -m bert -m pretrain -m timesplit -m "d${dim}" "BERT pretrain d${dim}" \
        uv run python -m asset_embeddings.scripts.train.bert \
            --config "${PT_PREFIX}/AssetBERT/AssetBERT_d${dim}_pretrained.json" \
            --log "checkpoints/timesplit_robust/pretrained/AssetBERT/d${dim}/train.log" \
                  "logs/timesplit_robust/train/pretrained/AssetBERT/d${dim}.log" \
            -r "$DB"
done

# === Finetune ===
echo "=== Finetune: RS ==="
for variant in $RS_VARIANTS; do
    for dim in $DIMS; do
        for quarter in $QUARTERS; do
            try_run -m "rs_${variant,,}" -m rs -m finetune -m timesplit -m "d${dim}" -m "${quarter,,}" "RS_${variant} finetune d${dim} ${quarter}" \
                uv run python -m asset_embeddings.scripts.train.rs \
                    --config "${FT_PREFIX}/AssetRS/RS_${variant}/d${dim}/AssetRS_${variant}_d${dim}_${quarter}.json" \
                    --log "checkpoints/timesplit_robust/finetune/AssetRS/RS_${variant}/d${dim}/${quarter}/train.log" \
                          "logs/timesplit_robust/train/finetune/AssetRS/RS_${variant}/d${dim}/${quarter}.log" \
                    -r "$DB"
        done
    done
done

echo "=== Finetune: W2V ==="
for dim in $DIMS; do
    for quarter in $QUARTERS; do
        try_run -m w2v -m finetune -m timesplit -m "d${dim}" -m "${quarter,,}" "W2V finetune d${dim} ${quarter}" \
            uv run python -m asset_embeddings.scripts.train.w2v \
                --config "${FT_PREFIX}/AssetW2V/d${dim}/AssetW2V_d${dim}_${quarter}.json" \
                --log "checkpoints/timesplit_robust/finetune/AssetW2V/d${dim}/${quarter}/train.log" \
                      "logs/timesplit_robust/train/finetune/AssetW2V/d${dim}/${quarter}.log" \
                -r "$DB"
    done
done

echo "=== Finetune: BERT ==="
for dim in $DIMS; do
    for quarter in $QUARTERS; do
        try_run -m bert -m finetune -m timesplit -m "d${dim}" -m "${quarter,,}" "BERT finetune d${dim} ${quarter}" \
            uv run python -m asset_embeddings.scripts.train.bert \
                --config "${FT_PREFIX}/AssetBERT/d${dim}/AssetBERT_d${dim}_${quarter}.json" \
                --log "checkpoints/timesplit_robust/finetune/AssetBERT/d${dim}/${quarter}/train.log" \
                      "logs/timesplit_robust/train/finetune/AssetBERT/d${dim}/${quarter}.log" \
                -r "$DB"
    done
done

print_summary
