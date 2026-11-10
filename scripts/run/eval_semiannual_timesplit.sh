#!/bin/bash
# ── Semi-Annual x Time-Split Robustness Evaluation ────────
# Re-run full eval pipeline on the Q2/Q4 subset of the 50/50 time-split
# (semi-annual restriction applied on top of timesplit_robust) to test
# joint robustness against pretrain-corpus reduction along both axes
# (frequency: Q2/Q4 only, time: 50% earlier history).
#
# Quarters: 2020Q4 2021Q2 2021Q4 2022Q2 2022Q4 2023Q2 2023Q4 2024Q2
# RV/RC: 2 models x 7 dims x 2 suffixes x 2 tasks = 56 runs
# MP:    (1 BERT + 1 SMLM + 1 BERT-head) x 7 dims x 8 quarters = 168 runs
# Total: 224 runs
# (RS variants and Placebo/Chars baselines skipped: results identical to
#  timesplit_robust on the Q2/Q4 subset since finetune data is a `cp` of
#  MutualFundShareHoldings_TimeSplit/finetune/.)
#
# Markers: rv rc mp eval semiannual_timesplit | assetbert assetw2v
#          bert smlm bert_head addchars | d{4..128} | {quarter}
#
# Usage:
#   bash scripts/run/run_notify.sh -a results/db/semiannual_timesplit.db scripts/run/eval_semiannual_timesplit.sh
#
# Examples:
#   bash scripts/run/run_notify.sh -k mp,smlm,d64 scripts/run/eval_semiannual_timesplit.sh
#   bash scripts/run/run_notify.sh -k rv,assetbert scripts/run/eval_semiannual_timesplit.sh

source scripts/run/run_helper.sh

DIMS="4 8 10 16 32 64 128"
QUARTERS="2020Q4 2021Q2 2021Q4 2022Q2 2022Q4 2023Q2 2023Q4 2024Q2"
EVAL_MODELS="AssetW2V AssetBERT"
BERT_MODELS="AssetBERT"
PREFIX="configs/semiannual_timesplit/eval"
DB="results/db/semiannual_timesplit.db"
EXP="semiannual_timesplit"  # runs.experiment; the --experiment flag of every script below

SMLM_MODELS=(
    "AssetW2V|AssetW2V"
)

# ========== RV ==========
echo "=== Eval: RV (models) ==="
for model in $EVAL_MODELS; do
    for dim in $DIMS; do
        for suffix in "" "_addChars"; do
            local_markers=(-m rv -m eval -m semiannual_timesplit -m "${model,,}" -m "d${dim}")
            [ -n "$suffix" ] && local_markers+=(-m addchars)
            cfg="${PREFIX}/RV/d${dim}/${model}_d${dim}${suffix}.json"
            job "${local_markers[@]}" -m "$cfg" "RV semi_ts ${model} d${dim}${suffix}" \
                uv run python -m asset_embeddings.scripts.eval.rv \
                    --config "$cfg" \
                    --log "logs/semiannual_timesplit/eval/RV/${model}${suffix}.log" \
                    -r "$DB" -e "$EXP"
        done
    done
done

# ========== RC ==========
echo "=== Eval: RC (models) ==="
for model in $EVAL_MODELS; do
    for dim in $DIMS; do
        for suffix in "" "_addChars"; do
            local_markers=(-m rc -m eval -m semiannual_timesplit -m "${model,,}" -m "d${dim}")
            [ -n "$suffix" ] && local_markers+=(-m addchars)
            cfg="${PREFIX}/RC/d${dim}/${model}_d${dim}${suffix}.json"
            job "${local_markers[@]}" -m "$cfg" "RC semi_ts ${model} d${dim}${suffix}" \
                uv run python -m asset_embeddings.scripts.eval.rc \
                    --config "$cfg" \
                    --log "logs/semiannual_timesplit/eval/RC/${model}${suffix}.log" \
                    -r "$DB" -e "$EXP"
        done
    done
done

# ========== MP: BERT ==========
echo "=== Eval: MP (BERT) ==="
for model in $BERT_MODELS; do
    for dim in $DIMS; do
        for quarter in $QUARTERS; do
            cfg="${PREFIX}/MP/BERT/${model}/d${dim}/${model}_d${dim}_${quarter}.json"
            job -m mp -m eval -m semiannual_timesplit -m bert -m "${model,,}" -m "d${dim}" -m "${quarter,,}" -m "$cfg" "MP semi_ts ${model} d${dim} ${quarter}" \
                uv run python -m asset_embeddings.scripts.eval.mp_bert \
                    --config "$cfg" \
                    --log "checkpoints/semiannual_timesplit/refine/BERT/${model}/d${dim}/${quarter}/refine.log" \
                          "logs/semiannual_timesplit/eval/MP/BERT/${model}/d${dim}/${quarter}.log" \
                    -r "$DB" -e "$EXP"
        done
    done
done

# ========== MP: SMLM (embedding models) ==========
echo "=== Eval: MP SMLM (embedding models) ==="
for entry in "${SMLM_MODELS[@]}"; do
    IFS='|' read -r model_type model_path <<< "$entry"
    for dim in $DIMS; do
        for quarter in $QUARTERS; do
            cfg="${PREFIX}/MP/SMLM/${model_path}/d${dim}/SMLM_${model_type}_d${dim}_${quarter}.json"
            job -m mp -m eval -m semiannual_timesplit -m smlm -m "${model_type,,}" -m "d${dim}" -m "${quarter,,}" -m "$cfg" "MP semi_ts SMLM ${model_type} d${dim} ${quarter}" \
                uv run python -m asset_embeddings.scripts.eval.mp_smlm \
                    --config "$cfg" \
                    --log "checkpoints/semiannual_timesplit/refine/SMLM/${model_path}/d${dim}/${quarter}/refine.log" \
                          "logs/semiannual_timesplit/eval/MP/SMLM/${model_path}/d${dim}/${quarter}.log" \
                    -r "$DB" -e "$EXP"
        done
    done
done

# ========== MP BERT-head: embedding models via BERT head ==========
echo "=== Eval: MP BERT-head (embedding models) ==="
for entry in "${SMLM_MODELS[@]}"; do
    IFS='|' read -r model_type model_path <<< "$entry"
    for dim in $DIMS; do
        for quarter in $QUARTERS; do
            cfg="${PREFIX}/MP/BERT/${model_path}/d${dim}/BERT_${model_type}_d${dim}_${quarter}.json"
            job -m mp -m eval -m semiannual_timesplit -m bert_head -m "${model_type,,}" -m "d${dim}" -m "${quarter,,}" -m "$cfg" "MP_BERT semi_ts ${model_type} d${dim} ${quarter}" \
                uv run python -m asset_embeddings.scripts.eval.mp_bert \
                    --config "$cfg" \
                    --log "checkpoints/semiannual_timesplit/refine/BERT/${model_path}/d${dim}/${quarter}/refine.log" \
                          "logs/semiannual_timesplit/eval/MP/BERT/${model_path}/d${dim}/${quarter}.log" \
                    -r "$DB" -e "$EXP"
        done
    done
done

print_summary
