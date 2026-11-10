#!/bin/bash
# ── MP BERT Random-Encoder Ablation (TimeSplit, 16Q) ───────
# AssetBERT embedding + random-init BERT encoder, on the 50/50 time-split
# robustness setting. Closes the (embedding × encoder-init) 2×2 against
# timesplit_robust.db MP cells.
#
# Quarters: 2020Q4 - 2024Q3 (16 quarters)
# Runs: 7 dims × 16 quarters = 112
#
# Markers: mp eval mp_bert_random_encoder timesplit bert assetbert
#          | d{4..128} | {quarter}
#
# Usage:
#   bash scripts/run/run_notify.sh -a results/db/mp_bert_random_encoder_timesplit.db \
#       scripts/run/eval_mp_bert_random_encoder_timesplit.sh

source scripts/run/run_helper.sh

DIMS="4 8 10 16 32 64 128"
QUARTERS="2020Q4 2021Q1 2021Q2 2021Q3 2021Q4 2022Q1 2022Q2 2022Q3 2022Q4 2023Q1 2023Q2 2023Q3 2023Q4 2024Q1 2024Q2 2024Q3"
PREFIX="configs/mp_bert_random_encoder_timesplit/eval"
DB="results/db/mp_bert_random_encoder_timesplit.db"
EXP="mp_bert_random_encoder_timesplit"  # runs.experiment; the --experiment flag of every script below

echo "=== Eval: MP (AssetBERT random encoder, TimeSplit) ==="
for dim in $DIMS; do
    for quarter in $QUARTERS; do
        cfg="${PREFIX}/MP/BERT/AssetBERT/d${dim}/AssetBERT_d${dim}_${quarter}.json"
        job -m mp -m eval -m mp_bert_random_encoder -m timesplit -m bert -m assetbert -m "$cfg" \
                -m "d${dim}" -m "${quarter,,}" \
            "MP randEnc ts AssetBERT d${dim} ${quarter}" \
            uv run python -m asset_embeddings.scripts.eval.mp_bert \
                --config "$cfg" \
                --log "checkpoints/mp_bert_random_encoder_timesplit/refine/BERT/AssetBERT/d${dim}/${quarter}/refine.log" \
                      "logs/mp_bert_random_encoder_timesplit/eval/MP/BERT/AssetBERT/d${dim}/${quarter}.log" \
                -r "$DB" -e "$EXP"
    done
done

print_summary
