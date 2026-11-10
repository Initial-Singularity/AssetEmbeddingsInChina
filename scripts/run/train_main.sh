#!/bin/bash
# ── Main Training Pipeline ──────────────────────────────
# Paper Table 2-4 / Figure 3-5: core asset embedding models
#
# Pretrain (7 dims each):
#   AssetW2V   — Word2Vec (Skip-gram)
#   AssetBERT  — Transformer MLM
# Finetune (7 dims × 6 quarters each):
#   AssetRS    — 4 variants (Binary, Ranks, Level0, LevelMin)
#   AssetW2V   — Word2Vec fine-tuned per quarter
#   AssetBERT  — BERT fine-tuned per quarter
# Total: 2×7 + (4+1+1)×7×6 = 266 runs
#
# Markers: w2v bert rs rs_binary rs_ranks rs_level0 rs_levelmin
#          pretrain finetune | d{4..128} | {quarter}
#
# Usage:
#   bash scripts/run/run_notify.sh -a results/db/main.db scripts/run/train_main.sh
#
# Examples:
#   bash scripts/run/run_notify.sh -k bert,pretrain scripts/run/train_main.sh
#   bash scripts/run/run_notify.sh -k finetune,d64 --exclude rs_binary,rs_ranks scripts/run/train_main.sh

source scripts/run/run_helper.sh

DB="results/db/main.db"
EXP="main"  # runs.experiment; the --experiment flag of every script below
DIMS="4 8 10 16 32 64 128"
QUARTERS="2023Q2 2023Q3 2023Q4 2024Q1 2024Q2 2024Q3"
RS_VARIANTS="Binary Ranks Level0 LevelMin"

# === Pretrain ===
echo "=== Pretrain: W2V ==="
for dim in $DIMS; do
    cfg="configs/main/pretrained/AssetW2V/AssetW2V_d${dim}_pretrained.json"
    job -m w2v -m pretrain -m "d${dim}" -m "$cfg" "W2V pretrain d${dim}" \
        uv run python -m asset_embeddings.scripts.train.w2v \
            --config "$cfg" \
            --log "checkpoints/pretrained/AssetW2V/d${dim}/train.log" \
                  "logs/main/train/pretrained/AssetW2V/d${dim}.log" \
            -r "$DB" -e "$EXP"
done

echo "=== Pretrain: BERT ==="
for dim in $DIMS; do
    cfg="configs/main/pretrained/AssetBERT/AssetBERT_d${dim}_pretrained.json"
    job -m bert -m pretrain -m "d${dim}" -m "$cfg" "BERT pretrain d${dim}" \
        uv run python -m asset_embeddings.scripts.train.bert \
            --config "$cfg" \
            --log "checkpoints/pretrained/AssetBERT/d${dim}/train.log" \
                  "logs/main/train/pretrained/AssetBERT/d${dim}.log" \
            -r "$DB" -e "$EXP"
done

# === Finetune ===
echo "=== Finetune: RS ==="
for variant in $RS_VARIANTS; do
    for dim in $DIMS; do
        for quarter in $QUARTERS; do
            cfg="configs/main/finetune/AssetRS/RS_${variant}/d${dim}/AssetRS_${variant}_d${dim}_${quarter}.json"
            job -m "rs_${variant,,}" -m rs -m finetune -m "d${dim}" -m "${quarter,,}" -m "$cfg" \
                "RS_${variant} finetune d${dim} ${quarter}" \
                uv run python -m asset_embeddings.scripts.train.rs \
                    --config "$cfg" \
                    --log "checkpoints/finetune/AssetRS/RS_${variant}/d${dim}/${quarter}/train.log" \
                          "logs/main/train/finetune/AssetRS/RS_${variant}/d${dim}/${quarter}.log" \
                    -r "$DB" -e "$EXP"
        done
    done
done

echo "=== Finetune: W2V ==="
for dim in $DIMS; do
    for quarter in $QUARTERS; do
        cfg="configs/main/finetune/AssetW2V/d${dim}/AssetW2V_d${dim}_${quarter}.json"
        job -m w2v -m finetune -m "d${dim}" -m "${quarter,,}" -m "$cfg" "W2V finetune d${dim} ${quarter}" \
            uv run python -m asset_embeddings.scripts.train.w2v \
                --config "$cfg" \
                --log "checkpoints/finetune/AssetW2V/d${dim}/${quarter}/train.log" \
                      "logs/main/train/finetune/AssetW2V/d${dim}/${quarter}.log" \
                -r "$DB" -e "$EXP"
    done
done

echo "=== Finetune: BERT ==="
for dim in $DIMS; do
    for quarter in $QUARTERS; do
        cfg="configs/main/finetune/AssetBERT/d${dim}/AssetBERT_d${dim}_${quarter}.json"
        job -m bert -m finetune -m "d${dim}" -m "${quarter,,}" -m "$cfg" "BERT finetune d${dim} ${quarter}" \
            uv run python -m asset_embeddings.scripts.train.bert \
                --config "$cfg" \
                --log "checkpoints/finetune/AssetBERT/d${dim}/${quarter}/train.log" \
                      "logs/main/train/finetune/AssetBERT/d${dim}/${quarter}.log" \
                -r "$DB" -e "$EXP"
    done
done

print_summary
