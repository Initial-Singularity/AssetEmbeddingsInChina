#!/bin/bash
# ── Eval: Sliding Window Pretrain Strategy ──────────────
# Per N: 7×3 RV + 7×3 RC + 7×6 MP = 84 eval runs
# Total: 3 × 84 = 252 eval runs
#
# Markers: rv rc mp eval sliding_window pt_strategy_ablation sw{N} bert addchars nonaligned | d{4..128} | {quarter}
#
# Usage:
#   bash scripts/run/run_notify.sh -a results/db/sliding_window.db scripts/run/eval_sliding_window.sh
#
# Examples:
#   bash scripts/run/run_notify.sh -k sw20,rv scripts/run/eval_sliding_window.sh
#   bash scripts/run/run_notify.sh -k mp,sw8 scripts/run/eval_sliding_window.sh

source scripts/run/run_helper.sh

DIMS="4 8 10 16 32 64 128"
QUARTERS="2023Q2 2023Q3 2023Q4 2024Q1 2024Q2 2024Q3"
WINDOW_SIZES="8 20 40"
DB="results/db/sliding_window.db"

for N in $WINDOW_SIZES; do
    METHOD="sw${N}"
    EXP="sliding_window/N${N}"  # runs.experiment, one family per window length (matches sliding_window.db)
    CFG="configs/pt_strategy_ablation/sliding_window/N${N}/eval"

    # ========== RV ==========
    echo "=== Sliding Window N=${N} eval: RV ==="
    for dim in $DIMS; do
        for suffix in "" "_nonaligned" "_addChars"; do
            local_markers=(-m rv -m eval -m sliding_window -m pt_strategy_ablation -m "${METHOD}" -m bert -m "d${dim}")
            [ "$suffix" = "_addChars" ] && local_markers+=(-m addchars)
            [ "$suffix" = "_nonaligned" ] && local_markers+=(-m nonaligned)
            cfg="${CFG}/RV/d${dim}/AssetBERT_${METHOD}_d${dim}${suffix}.json"
            job "${local_markers[@]}" -m "$cfg" "RV ${METHOD} d${dim}${suffix}" \
                uv run python -m asset_embeddings.scripts.eval.rv \
                    --config "$cfg" \
                    --log "logs/pt_strategy_ablation/sliding_window/N${N}/eval/RV/AssetBERT_${METHOD}${suffix}.log" \
                    -r "$DB" -e "$EXP"
        done
    done

    # ========== RC ==========
    echo "=== Sliding Window N=${N} eval: RC ==="
    for dim in $DIMS; do
        for suffix in "" "_nonaligned" "_addChars"; do
            local_markers=(-m rc -m eval -m sliding_window -m pt_strategy_ablation -m "${METHOD}" -m bert -m "d${dim}")
            [ "$suffix" = "_addChars" ] && local_markers+=(-m addchars)
            [ "$suffix" = "_nonaligned" ] && local_markers+=(-m nonaligned)
            cfg="${CFG}/RC/d${dim}/AssetBERT_${METHOD}_d${dim}${suffix}.json"
            job "${local_markers[@]}" -m "$cfg" "RC ${METHOD} d${dim}${suffix}" \
                uv run python -m asset_embeddings.scripts.eval.rc \
                    --config "$cfg" \
                    --log "logs/pt_strategy_ablation/sliding_window/N${N}/eval/RC/AssetBERT_${METHOD}${suffix}.log" \
                    -r "$DB" -e "$EXP"
        done
    done

    # ========== MP (BERT head) ==========
    echo "=== Sliding Window N=${N} eval: MP ==="
    for dim in $DIMS; do
        for quarter in $QUARTERS; do
            cfg="${CFG}/MP/BERT/AssetBERT_${METHOD}/d${dim}/AssetBERT_${METHOD}_d${dim}_${quarter}.json"
            job -m mp -m eval -m sliding_window -m pt_strategy_ablation -m "${METHOD}" -m bert -m "d${dim}" -m "${quarter,,}" -m "$cfg" \
                "MP ${METHOD} d${dim} ${quarter}" \
                uv run python -m asset_embeddings.scripts.eval.mp_bert \
                    --config "$cfg" \
                    --log "checkpoints/refine/BERT/AssetBERT_${METHOD}/d${dim}/${quarter}/refine.log" \
                          "logs/pt_strategy_ablation/sliding_window/N${N}/eval/MP/BERT/AssetBERT_${METHOD}/d${dim}/${quarter}.log" \
                    -r "$DB" -e "$EXP"
        done
    done
done

print_summary
