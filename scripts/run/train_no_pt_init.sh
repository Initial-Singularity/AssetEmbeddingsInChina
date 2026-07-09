#!/bin/bash
# ── pt_strategy_ablation / no_pt_init Training ─────────────
# Direct AssetBERT finetune from random init on each quarter
# (no AssetBERT pretrain). Reuses timesplit_robust's per-quarter
# AssetW2V .model as tokenizer/vocab source only — embedding
# layer is randomly initialised.
#
# 7 dims × 16 quarters = 112 runs.
#
# Markers: bert finetune no_pt_init pt_strategy_ablation
#          | d{4..128} | {quarter}
#
# Usage:
#   bash scripts/run/run_notify.sh -a results/db/no_pt_init.db scripts/run/train_no_pt_init.sh
#
# Examples:
#   bash scripts/run/run_notify.sh -k no_pt_init,d64 scripts/run/train_no_pt_init.sh
#   bash scripts/run/run_notify.sh -k pt_strategy_ablation,2023q4 scripts/run/train_no_pt_init.sh

source scripts/run/run_helper.sh

DB="results/db/no_pt_init.db"
DIMS="4 8 10 16 32 64 128"
QUARTERS="2020Q4 2021Q1 2021Q2 2021Q3 2021Q4 2022Q1 2022Q2 2022Q3 2022Q4 2023Q1 2023Q2 2023Q3 2023Q4 2024Q1 2024Q2 2024Q3"
FT_PREFIX="configs/pt_strategy_ablation/no_pt_init/finetune"

echo "=== Finetune: BERT (no_pt_init) ==="
for dim in $DIMS; do
    for quarter in $QUARTERS; do
        try_run -m bert -m finetune -m no_pt_init -m pt_strategy_ablation -m "d${dim}" -m "${quarter,,}" "BERT no_pt_init d${dim} ${quarter}" \
            uv run python -m asset_embeddings.scripts.train.bert \
                --config "${FT_PREFIX}/AssetBERT/d${dim}/AssetBERT_no_pt_init_d${dim}_${quarter}.json" \
                --log "checkpoints/pt_strategy_ablation/no_pt_init/finetune/AssetBERT/d${dim}/${quarter}/train.log" \
                      "logs/pt_strategy_ablation/no_pt_init/train/finetune/AssetBERT/d${dim}/${quarter}.log" \
                -r "$DB"
    done
done

print_summary
