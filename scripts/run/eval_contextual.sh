#!/bin/bash
# ── Contextual Embedding Evaluation ─────────────────────
# Evaluate contextual (sequence-level) BERT embeddings on RV + RC + MP.
#
# Only AssetBERT — contextual embeddings are BERT-specific.
# RV/RC: 1 model × 7 dims × 2 suffixes × 2 tasks = 28 eval runs
# MP:    1 model × 7 dims × 6 quarters = 42 eval runs
# Total: 70 runs
#
# Markers: rv rc mp eval contextual assetbert addchars | d{4..128} | {quarter}
#
# Usage:
#   bash scripts/run/run_notify.sh -a results/db/contextual.db scripts/run/eval_contextual.sh
#
# Examples:
#   bash scripts/run/run_notify.sh -k mp,d64 scripts/run/eval_contextual.sh
#   bash scripts/run/run_notify.sh -k rv,addchars scripts/run/eval_contextual.sh

source scripts/run/run_helper.sh

DIMS="4 8 10 16 32 64 128"
QUARTERS="2023Q2 2023Q3 2023Q4 2024Q1 2024Q2 2024Q3"
BERT_MODELS="AssetBERT"
DB="results/db/contextual.db"
EXP="contextual"  # runs.experiment; the --experiment flag of every script below

echo "=== Eval: Contextual RV ==="
for model in $BERT_MODELS; do
    for dim in $DIMS; do
        for suffix in "" "_addChars"; do
            local_markers=(-m rv -m eval -m contextual -m "${model,,}" -m "d${dim}")
            [ -n "$suffix" ] && local_markers+=(-m addchars)
            cfg="configs/contextual/eval/RV/d${dim}/${model}_d${dim}${suffix}.json"
            job "${local_markers[@]}" -m "$cfg" "RV ctx ${model} d${dim}${suffix}" \
                uv run python -m asset_embeddings.scripts.eval.rv \
                    --config "$cfg" \
                    --log "logs/contextual/eval/RV/${model}${suffix}.log" \
                    -r "$DB" -e "$EXP"
        done
    done
done

echo "=== Eval: Contextual RC ==="
for model in $BERT_MODELS; do
    for dim in $DIMS; do
        for suffix in "" "_addChars"; do
            local_markers=(-m rc -m eval -m contextual -m "${model,,}" -m "d${dim}")
            [ -n "$suffix" ] && local_markers+=(-m addchars)
            cfg="configs/contextual/eval/RC/d${dim}/${model}_d${dim}${suffix}.json"
            job "${local_markers[@]}" -m "$cfg" "RC ctx ${model} d${dim}${suffix}" \
                uv run python -m asset_embeddings.scripts.eval.rc \
                    --config "$cfg" \
                    --log "logs/contextual/eval/RC/${model}${suffix}.log" \
                    -r "$DB" -e "$EXP"
        done
    done
done

echo "=== Eval: Contextual MP ==="
for model in $BERT_MODELS; do
    for dim in $DIMS; do
        for quarter in $QUARTERS; do
            cfg="configs/contextual/eval/MP/BERT/${model}/d${dim}/${model}_d${dim}_${quarter}.json"
            job -m mp -m eval -m contextual -m bert -m "${model,,}" -m "d${dim}" -m "${quarter,,}" -m "$cfg" "MP ctx ${model} d${dim} ${quarter}" \
                uv run python -m asset_embeddings.scripts.eval.mp_bert \
                    --config "$cfg" \
                    --log "checkpoints/contextual/refine/BERT/${model}/d${dim}/${quarter}/refine.log" \
                          "logs/contextual/eval/MP/BERT/${model}/d${dim}/${quarter}.log" \
                    -r "$DB" -e "$EXP"
        done
    done
done

print_summary
