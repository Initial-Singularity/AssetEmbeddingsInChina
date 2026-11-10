#!/bin/bash
# ── Semi-Annual Robustness Evaluation ───────────────────
# Re-run full eval pipeline on semi-annual (Q2/Q4 only) data to test
# robustness against disclosure-rule variation.
#
# Quarters: 2023Q2 2023Q4 2024Q2 (semi-annual subset)
# RV/RC: 2 models × 7 dims × 2 suffixes × 2 tasks = 56 runs
# MP:    (1 BERT + 1 SMLM + 1 BERT-head) × 7 dims × 3 quarters = 63 runs
# Total: 119 runs
# (RS variants and Placebo/Chars baselines skipped: results identical to
#  quarterly Q2/Q4 since SemiAnnual data is a `cp` of those files.)
#
# Markers: rv rc mp eval semiannual | assetbert assetw2v
#          bert smlm bert_head addchars | d{4..128} | {quarter}
#
# Usage:
#   bash scripts/run/run_notify.sh -a results/db/semiannual_robust.db scripts/run/eval_semiannual_robust.sh
#
# Examples:
#   bash scripts/run/run_notify.sh -k mp,smlm,d64 scripts/run/eval_semiannual_robust.sh
#   bash scripts/run/run_notify.sh -k rv,assetbert scripts/run/eval_semiannual_robust.sh

source scripts/run/run_helper.sh

DIMS="4 8 10 16 32 64 128"
QUARTERS="2023Q2 2023Q4 2024Q2"
EVAL_MODELS="AssetW2V AssetBERT"
BERT_MODELS="AssetBERT"
PREFIX="configs/semiannual_robust/eval"
DB="results/db/semiannual_robust.db"
EXP="semiannual_robust"  # runs.experiment; the --experiment flag of every script below

SMLM_MODELS=(
    "AssetW2V|AssetW2V"
)

# ========== RV ==========
echo "=== Eval: RV (models) ==="
for model in $EVAL_MODELS; do
    for dim in $DIMS; do
        for suffix in "" "_addChars"; do
            local_markers=(-m rv -m eval -m semiannual -m "${model,,}" -m "d${dim}")
            [ -n "$suffix" ] && local_markers+=(-m addchars)
            cfg="${PREFIX}/RV/d${dim}/${model}_d${dim}${suffix}.json"
            job "${local_markers[@]}" -m "$cfg" "RV semi ${model} d${dim}${suffix}" \
                uv run python -m asset_embeddings.scripts.eval.rv \
                    --config "$cfg" \
                    --log "logs/semiannual_robust/eval/RV/${model}${suffix}.log" \
                    -r "$DB" -e "$EXP"
        done
    done
done

# ========== RC ==========
echo "=== Eval: RC (models) ==="
for model in $EVAL_MODELS; do
    for dim in $DIMS; do
        for suffix in "" "_addChars"; do
            local_markers=(-m rc -m eval -m semiannual -m "${model,,}" -m "d${dim}")
            [ -n "$suffix" ] && local_markers+=(-m addchars)
            cfg="${PREFIX}/RC/d${dim}/${model}_d${dim}${suffix}.json"
            job "${local_markers[@]}" -m "$cfg" "RC semi ${model} d${dim}${suffix}" \
                uv run python -m asset_embeddings.scripts.eval.rc \
                    --config "$cfg" \
                    --log "logs/semiannual_robust/eval/RC/${model}${suffix}.log" \
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
            job -m mp -m eval -m semiannual -m bert -m "${model,,}" -m "d${dim}" -m "${quarter,,}" -m "$cfg" "MP semi ${model} d${dim} ${quarter}" \
                uv run python -m asset_embeddings.scripts.eval.mp_bert \
                    --config "$cfg" \
                    --log "checkpoints/semiannual_robust/refine/BERT/${model}/d${dim}/${quarter}/refine.log" \
                          "logs/semiannual_robust/eval/MP/BERT/${model}/d${dim}/${quarter}.log" \
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
            job -m mp -m eval -m semiannual -m smlm -m "${model_type,,}" -m "d${dim}" -m "${quarter,,}" -m "$cfg" "MP semi SMLM ${model_type} d${dim} ${quarter}" \
                uv run python -m asset_embeddings.scripts.eval.mp_smlm \
                    --config "$cfg" \
                    --log "checkpoints/semiannual_robust/refine/SMLM/${model_path}/d${dim}/${quarter}/refine.log" \
                          "logs/semiannual_robust/eval/MP/SMLM/${model_path}/d${dim}/${quarter}.log" \
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
            job -m mp -m eval -m semiannual -m bert_head -m "${model_type,,}" -m "d${dim}" -m "${quarter,,}" -m "$cfg" "MP_BERT semi ${model_type} d${dim} ${quarter}" \
                uv run python -m asset_embeddings.scripts.eval.mp_bert \
                    --config "$cfg" \
                    --log "checkpoints/semiannual_robust/refine/BERT/${model_path}/d${dim}/${quarter}/refine.log" \
                          "logs/semiannual_robust/eval/MP/BERT/${model_path}/d${dim}/${quarter}.log" \
                    -r "$DB" -e "$EXP"
        done
    done
done

print_summary
