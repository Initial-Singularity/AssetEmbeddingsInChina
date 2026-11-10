#!/bin/bash
# ── MP BERT Random-Encoder Ablation (main, 6Q) ─────────────
# AssetBERT embedding + random-init BERT encoder (model_checkpoint=null).
# Closes the (embedding × encoder-init) 2×2 against main.db MP cells.
#
# Quarters: 2023Q2 - 2024Q3 (6 quarters, matches main pipeline)
# Runs: 7 dims × 6 quarters = 42
#
# Markers: mp eval mp_bert_random_encoder bert assetbert | d{4..128} | {quarter}
#
# Usage:
#   bash scripts/run/run_notify.sh -a results/db/mp_bert_random_encoder.db \
#       scripts/run/eval_mp_bert_random_encoder.sh

source scripts/run/run_helper.sh

DIMS="4 8 10 16 32 64 128"
QUARTERS="2023Q2 2023Q3 2023Q4 2024Q1 2024Q2 2024Q3"
PREFIX="configs/mp_bert_random_encoder/eval"
DB="results/db/mp_bert_random_encoder.db"
EXP="mp_bert_random_encoder"  # runs.experiment; the --experiment flag of every script below

echo "=== Eval: MP (AssetBERT random encoder) ==="
for dim in $DIMS; do
    for quarter in $QUARTERS; do
        cfg="${PREFIX}/MP/BERT/AssetBERT/d${dim}/AssetBERT_d${dim}_${quarter}.json"
        job -m mp -m eval -m mp_bert_random_encoder -m bert -m assetbert -m "$cfg" \
                -m "d${dim}" -m "${quarter,,}" \
            "MP randEnc AssetBERT d${dim} ${quarter}" \
            uv run python -m asset_embeddings.scripts.eval.mp_bert \
                --config "$cfg" \
                --log "checkpoints/mp_bert_random_encoder/refine/BERT/AssetBERT/d${dim}/${quarter}/refine.log" \
                      "logs/mp_bert_random_encoder/eval/MP/BERT/AssetBERT/d${dim}/${quarter}.log" \
                -r "$DB" -e "$EXP"
    done
done

print_summary
