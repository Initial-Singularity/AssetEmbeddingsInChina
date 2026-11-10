#!/bin/bash
# ── Eval: Expansion Pretrain Strategy ───────────────────
# RV + RC (3 variants each) + MP = 7×3×2 + 7×6 = 84 eval runs
#
# Markers: rv rc mp eval expansion pt_strategy_ablation bert addchars nonaligned | d{4..128} | {quarter}
#
# Usage:
#   bash scripts/run/run_notify.sh -a results/db/expansion.db scripts/run/eval_expansion.sh
#
# Examples:
#   bash scripts/run/run_notify.sh -k rv,d64 scripts/run/eval_expansion.sh
#   bash scripts/run/run_notify.sh -k mp scripts/run/eval_expansion.sh

source scripts/run/run_helper.sh

DIMS="4 8 10 16 32 64 128"
QUARTERS="2023Q2 2023Q3 2023Q4 2024Q1 2024Q2 2024Q3"
DB="results/db/expansion.db"
EXP="expansion"  # runs.experiment; the --experiment flag of every script below
CFG="configs/pt_strategy_ablation/expansion/eval"

# ========== RV ==========
echo "=== Expansion eval: RV ==="
for dim in $DIMS; do
    for suffix in "" "_nonaligned" "_addChars"; do
        local_markers=(-m rv -m eval -m expansion -m pt_strategy_ablation -m bert -m "d${dim}")
        [ "$suffix" = "_addChars" ] && local_markers+=(-m addchars)
        [ "$suffix" = "_nonaligned" ] && local_markers+=(-m nonaligned)
        cfg="${CFG}/RV/d${dim}/AssetBERT_expansion_d${dim}${suffix}.json"
        job "${local_markers[@]}" -m "$cfg" "RV expansion d${dim}${suffix}" \
            uv run python -m asset_embeddings.scripts.eval.rv \
                --config "$cfg" \
                --log "logs/pt_strategy_ablation/expansion/eval/RV/AssetBERT_expansion${suffix}.log" \
                -r "$DB" -e "$EXP"
    done
done

# ========== RC ==========
echo "=== Expansion eval: RC ==="
for dim in $DIMS; do
    for suffix in "" "_nonaligned" "_addChars"; do
        local_markers=(-m rc -m eval -m expansion -m pt_strategy_ablation -m bert -m "d${dim}")
        [ "$suffix" = "_addChars" ] && local_markers+=(-m addchars)
        [ "$suffix" = "_nonaligned" ] && local_markers+=(-m nonaligned)
        cfg="${CFG}/RC/d${dim}/AssetBERT_expansion_d${dim}${suffix}.json"
        job "${local_markers[@]}" -m "$cfg" "RC expansion d${dim}${suffix}" \
            uv run python -m asset_embeddings.scripts.eval.rc \
                --config "$cfg" \
                --log "logs/pt_strategy_ablation/expansion/eval/RC/AssetBERT_expansion${suffix}.log" \
                -r "$DB" -e "$EXP"
    done
done

# ========== MP (BERT head) ==========
echo "=== Expansion eval: MP ==="
for dim in $DIMS; do
    for quarter in $QUARTERS; do
        cfg="${CFG}/MP/BERT/AssetBERT_expansion/d${dim}/AssetBERT_expansion_d${dim}_${quarter}.json"
        job -m mp -m eval -m expansion -m pt_strategy_ablation -m bert -m "d${dim}" -m "${quarter,,}" -m "$cfg" \
            "MP expansion d${dim} ${quarter}" \
            uv run python -m asset_embeddings.scripts.eval.mp_bert \
                --config "$cfg" \
                --log "checkpoints/refine/BERT/AssetBERT_expansion/d${dim}/${quarter}/refine.log" \
                      "logs/pt_strategy_ablation/expansion/eval/MP/BERT/AssetBERT_expansion/d${dim}/${quarter}.log" \
                -r "$DB" -e "$EXP"
    done
done

print_summary
