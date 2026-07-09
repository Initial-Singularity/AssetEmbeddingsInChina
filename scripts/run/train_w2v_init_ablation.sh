#!/bin/bash
# ── Exp: W2V Initialization 2x2 Ablation — Training ────
# 2x2 factorial design isolating W2V contribution at pretrain vs finetune:
#
#   |                     | FT: W2V reinit    | FT: No reinit     |
#   |---------------------|-------------------|-------------------|
#   | PT: W2V init        | AssetBERT (base)  | AssetBERT_w2v_pt  |
#   | PT: Random init     | AssetBERT_w2v_ft  | AssetBERT_no_w2v  |
#
# AssetBERT (base) is already trained in the main pipeline.
# This script trains the other 3 conditions.
#   Phase 1: Pretrain w2v_ft (7 runs)
#   Phase 2: Finetune all 3 conditions (3×42 = 126 runs)
# Total: 133 runs
#
# Markers: bert pretrain finetune w2v_init
#          assetbert_w2v_ft assetbert_no_w2v assetbert_w2v_pt
#          d{4..128} | {quarter}
#
# Usage:
#   bash scripts/run/run_notify.sh -a results/db/w2v_init_ablation.db scripts/run/train_w2v_init_ablation.sh
#
# Examples:
#   bash scripts/run/run_notify.sh -k pretrain scripts/run/train_w2v_init_ablation.sh
#   bash scripts/run/run_notify.sh -k finetune,assetbert_no_w2v scripts/run/train_w2v_init_ablation.sh

source scripts/run/run_helper.sh

DIMS="4 8 10 16 32 64 128"
QUARTERS="2023Q2 2023Q3 2023Q4 2024Q1 2024Q2 2024Q3"
DB="results/db/w2v_init_ablation.db"

# 3 ablation models (AssetBERT baseline already exists)
MODELS="AssetBERT_w2v_ft AssetBERT_no_w2v AssetBERT_w2v_pt"

# ========== Phase 1: Pretrain (random init) ==========
# Shared checkpoint for w2v_ft and no_w2v
echo "=== Phase 1: Pretrain w2v_ft (random init, shared by w2v_ft & no_w2v) ==="
for dim in $DIMS; do
    try_run -m bert -m pretrain -m w2v_init -m "d${dim}" \
        "w2v_ft pretrain d${dim}" \
        uv run python -m asset_embeddings.scripts.train.bert \
            --config "configs/w2v_init_ablation/pretrained/AssetBERT_w2v_ft/AssetBERT_w2v_ft_d${dim}_pretrained.json" \
            --log "checkpoints/pretrained/AssetBERT_w2v_ft/d${dim}/train.log" \
                  "logs/w2v_init_ablation/train/pretrained/AssetBERT_w2v_ft/d${dim}.log" \
            -r "$DB"
done

# ========== Phase 2: Finetune all 3 conditions ==========
for model in $MODELS; do
    echo "=== Phase 2: Finetune ${model} ==="
    for dim in $DIMS; do
        for quarter in $QUARTERS; do
            try_run -m bert -m finetune -m w2v_init -m "${model,,}" -m "d${dim}" -m "${quarter,,}" \
                "${model} finetune d${dim} ${quarter}" \
                uv run python -m asset_embeddings.scripts.train.bert \
                    --config "configs/w2v_init_ablation/finetune/${model}/d${dim}/${model}_d${dim}_${quarter}.json" \
                    --log "checkpoints/finetune/${model}/d${dim}/${quarter}/train.log" \
                          "logs/w2v_init_ablation/train/finetune/${model}/d${dim}/${quarter}.log" \
                    -r "$DB"
        done
    done
done

print_summary
