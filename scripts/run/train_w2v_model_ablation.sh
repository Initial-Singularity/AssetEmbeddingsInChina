#!/bin/bash
# ── W2V Model Ablation (CBOW) Training ─────────────────────
# Replace Skip-gram W2V (sg=1) with CBOW W2V (sg=0) and cascade
# through BERT pretrain & finetune. All other hyperparameters
# match templates/train_main.json.
#
# Pretrain (7 dims each): AssetW2V_cbow, AssetBERT_cbow
# Finetune (7 dims × 6 quarters each): AssetW2V_cbow, AssetBERT_cbow
# Total: 2×7 + 2×7×6 = 98 runs
#
# Markers: w2v bert w2v_model pretrain finetune | d{4..128} | {quarter}
#
# Usage:
#   bash scripts/run/run_notify.sh -a results/db/w2v_model_ablation.db scripts/run/train_w2v_model_ablation.sh

source scripts/run/run_helper.sh

DB="results/db/w2v_model_ablation.db"
EXP="w2v_model_ablation"  # runs.experiment; the --experiment flag of every script below
DIMS="4 8 10 16 32 64 128"
QUARTERS="2023Q2 2023Q3 2023Q4 2024Q1 2024Q2 2024Q3"

# === Pretrain ===
echo "=== Pretrain: W2V (CBOW) ==="
for dim in $DIMS; do
    cfg="configs/w2v_model_ablation/pretrained/AssetW2V_cbow/AssetW2V_cbow_d${dim}_pretrained.json"
    job -m w2v -m w2v_model -m pretrain -m "d${dim}" -m "$cfg" "W2V_cbow pretrain d${dim}" \
        uv run python -m asset_embeddings.scripts.train.w2v \
            --config "$cfg" \
            --log "checkpoints/w2v_model_ablation/pretrained/AssetW2V_cbow/d${dim}/train.log" \
                  "logs/w2v_model_ablation/train/pretrained/AssetW2V_cbow/d${dim}.log" \
            -r "$DB" -e "$EXP"
done

echo "=== Pretrain: BERT (CBOW init) ==="
for dim in $DIMS; do
    cfg="configs/w2v_model_ablation/pretrained/AssetBERT_cbow/AssetBERT_cbow_d${dim}_pretrained.json"
    job -m bert -m w2v_model -m pretrain -m "d${dim}" -m "$cfg" "BERT_cbow pretrain d${dim}" \
        uv run python -m asset_embeddings.scripts.train.bert \
            --config "$cfg" \
            --log "checkpoints/w2v_model_ablation/pretrained/AssetBERT_cbow/d${dim}/train.log" \
                  "logs/w2v_model_ablation/train/pretrained/AssetBERT_cbow/d${dim}.log" \
            -r "$DB" -e "$EXP"
done

# === Finetune ===
echo "=== Finetune: W2V (CBOW) ==="
for dim in $DIMS; do
    for quarter in $QUARTERS; do
        cfg="configs/w2v_model_ablation/finetune/AssetW2V_cbow/d${dim}/AssetW2V_cbow_d${dim}_${quarter}.json"
        job -m w2v -m w2v_model -m finetune -m "d${dim}" -m "${quarter,,}" -m "$cfg" "W2V_cbow finetune d${dim} ${quarter}" \
            uv run python -m asset_embeddings.scripts.train.w2v \
                --config "$cfg" \
                --log "checkpoints/w2v_model_ablation/finetune/AssetW2V_cbow/d${dim}/${quarter}/train.log" \
                      "logs/w2v_model_ablation/train/finetune/AssetW2V_cbow/d${dim}/${quarter}.log" \
                -r "$DB" -e "$EXP"
    done
done

echo "=== Finetune: BERT (CBOW-init pretrain → FT) ==="
for dim in $DIMS; do
    for quarter in $QUARTERS; do
        cfg="configs/w2v_model_ablation/finetune/AssetBERT_cbow/d${dim}/AssetBERT_cbow_d${dim}_${quarter}.json"
        job -m bert -m w2v_model -m finetune -m "d${dim}" -m "${quarter,,}" -m "$cfg" "BERT_cbow finetune d${dim} ${quarter}" \
            uv run python -m asset_embeddings.scripts.train.bert \
                --config "$cfg" \
                --log "checkpoints/w2v_model_ablation/finetune/AssetBERT_cbow/d${dim}/${quarter}/train.log" \
                      "logs/w2v_model_ablation/train/finetune/AssetBERT_cbow/d${dim}/${quarter}.log" \
                -r "$DB" -e "$EXP"
    done
done

print_summary
