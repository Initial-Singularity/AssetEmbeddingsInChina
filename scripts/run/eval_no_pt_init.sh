#!/bin/bash
# ── pt_strategy_ablation / no_pt_init Evaluation ──────────
# RV/RC/MP-BERT for AssetBERT_no_pt_init across 7 dims × 16 quarters.
# Baselines (Placebo/Chars) and other-model rows live in
# eval_timesplit_robust.sh and are not duplicated here.
#
# RV: 14, RC: 14, MP-BERT: 112 → 140 runs total.
#
# Markers: rv rc mp eval no_pt_init pt_strategy_ablation
#          assetbert addchars bert | d{4..128} | {quarter}
#
# Usage:
#   bash scripts/run/run_notify.sh -a results/db/no_pt_init.db scripts/run/eval_no_pt_init.sh
#
# Examples:
#   bash scripts/run/run_notify.sh -k mp,no_pt_init,d64 scripts/run/eval_no_pt_init.sh
#   bash scripts/run/run_notify.sh -k rv,pt_strategy_ablation scripts/run/eval_no_pt_init.sh

source scripts/run/run_helper.sh

DIMS="4 8 10 16 32 64 128"
QUARTERS="2020Q4 2021Q1 2021Q2 2021Q3 2021Q4 2022Q1 2022Q2 2022Q3 2022Q4 2023Q1 2023Q2 2023Q3 2023Q4 2024Q1 2024Q2 2024Q3"
PREFIX="configs/pt_strategy_ablation/no_pt_init/eval"
DB="results/db/no_pt_init.db"
EXP="pt_strategy_ablation/no_pt_init"  # runs.experiment (matches the derived label in no_pt_init.db)

# ========== RV ==========
echo "=== Eval: RV (AssetBERT no_pt_init) ==="
for dim in $DIMS; do
    for suffix in "" "_addChars"; do
        local_markers=(-m rv -m eval -m no_pt_init -m pt_strategy_ablation -m bert -m assetbert -m "d${dim}")
        [ -n "$suffix" ] && local_markers+=(-m addchars)
        cfg="${PREFIX}/RV/d${dim}/AssetBERT_no_pt_init_d${dim}${suffix}.json"
        job "${local_markers[@]}" -m "$cfg" "RV no_pt_init AssetBERT d${dim}${suffix}" \
            uv run python -m asset_embeddings.scripts.eval.rv \
                --config "$cfg" \
                --log "logs/pt_strategy_ablation/no_pt_init/eval/RV/AssetBERT_no_pt_init${suffix}.log" \
                -r "$DB" -e "$EXP"
    done
done

# ========== RC ==========
echo "=== Eval: RC (AssetBERT no_pt_init) ==="
for dim in $DIMS; do
    for suffix in "" "_addChars"; do
        local_markers=(-m rc -m eval -m no_pt_init -m pt_strategy_ablation -m bert -m assetbert -m "d${dim}")
        [ -n "$suffix" ] && local_markers+=(-m addchars)
        cfg="${PREFIX}/RC/d${dim}/AssetBERT_no_pt_init_d${dim}${suffix}.json"
        job "${local_markers[@]}" -m "$cfg" "RC no_pt_init AssetBERT d${dim}${suffix}" \
            uv run python -m asset_embeddings.scripts.eval.rc \
                --config "$cfg" \
                --log "logs/pt_strategy_ablation/no_pt_init/eval/RC/AssetBERT_no_pt_init${suffix}.log" \
                -r "$DB" -e "$EXP"
    done
done

# ========== MP: BERT head ==========
echo "=== Eval: MP (AssetBERT no_pt_init via BERT head) ==="
for dim in $DIMS; do
    for quarter in $QUARTERS; do
        cfg="${PREFIX}/MP/BERT/AssetBERT_no_pt_init/d${dim}/AssetBERT_no_pt_init_d${dim}_${quarter}.json"
        job -m mp -m eval -m no_pt_init -m pt_strategy_ablation -m bert -m assetbert -m "d${dim}" -m "${quarter,,}" -m "$cfg" "MP no_pt_init AssetBERT d${dim} ${quarter}" \
            uv run python -m asset_embeddings.scripts.eval.mp_bert \
                --config "$cfg" \
                --log "checkpoints/pt_strategy_ablation/no_pt_init/refine/BERT/AssetBERT_no_pt_init/d${dim}/${quarter}/refine.log" \
                      "logs/pt_strategy_ablation/no_pt_init/eval/MP/BERT/AssetBERT_no_pt_init/d${dim}/${quarter}.log" \
                -r "$DB" -e "$EXP"
    done
done

print_summary
