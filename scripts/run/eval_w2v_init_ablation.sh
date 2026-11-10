#!/bin/bash
# ── Exp: W2V Initialization 2x2 Ablation — Evaluation ──
# 2x2 factorial design isolating W2V contribution at pretrain vs finetune:
#
#   |                     | FT: W2V reinit    | FT: No reinit     |
#   |---------------------|-------------------|-------------------|
#   | PT: W2V init        | AssetBERT (base)  | AssetBERT_w2v_pt  |
#   | PT: Random init     | AssetBERT_w2v_ft  | AssetBERT_no_w2v  |
#
# AssetBERT (base) is already evaluated in the main pipeline.
# This script evaluates the other 3 conditions.
#   Phase 1: Eval RV+RC (3×42 = 126 runs)
#   Phase 2: Eval MP    (3×42 = 126 runs)
# Total: 252 runs
#
# Markers: eval w2v_init rv rc mp
#          assetbert_w2v_ft assetbert_no_w2v assetbert_w2v_pt
#          bert addchars nonaligned | d{4..128} | {quarter}
#
# Usage:
#   bash scripts/run/run_notify.sh -a results/db/w2v_init_ablation.db scripts/run/eval_w2v_init_ablation.sh
#
# Examples:
#   bash scripts/run/run_notify.sh -k rv,assetbert_no_w2v scripts/run/eval_w2v_init_ablation.sh
#   bash scripts/run/run_notify.sh -k mp,d64 scripts/run/eval_w2v_init_ablation.sh

source scripts/run/run_helper.sh

DIMS="4 8 10 16 32 64 128"
QUARTERS="2023Q2 2023Q3 2023Q4 2024Q1 2024Q2 2024Q3"
DB="results/db/w2v_init_ablation.db"
EXP="w2v_init_ablation"  # runs.experiment; the --experiment flag of every script below

# 3 ablation models (AssetBERT baseline already exists)
MODELS="AssetBERT_w2v_ft AssetBERT_no_w2v AssetBERT_w2v_pt"

# ========== Phase 1: Eval RV + RC ==========
for model in $MODELS; do
    echo "=== Phase 1a: Eval RV ${model} ==="
    for dim in $DIMS; do
        for suffix in "" "_nonaligned" "_addChars"; do
            local_markers=(-m rv -m eval -m w2v_init -m "${model,,}" -m "d${dim}")
            [ "$suffix" = "_addChars" ] && local_markers+=(-m addchars)
            [ "$suffix" = "_nonaligned" ] && local_markers+=(-m nonaligned)
            cfg="configs/w2v_init_ablation/eval/RV/d${dim}/${model}_d${dim}${suffix}.json"
            job "${local_markers[@]}" -m "$cfg" "RV ${model} d${dim}${suffix}" \
                uv run python -m asset_embeddings.scripts.eval.rv \
                    --config "$cfg" \
                    --log "logs/w2v_init_ablation/eval/RV/${model}${suffix}.log" \
                    -r "$DB" -e "$EXP"
        done
    done

    echo "=== Phase 1b: Eval RC ${model} ==="
    for dim in $DIMS; do
        for suffix in "" "_nonaligned" "_addChars"; do
            local_markers=(-m rc -m eval -m w2v_init -m "${model,,}" -m "d${dim}")
            [ "$suffix" = "_addChars" ] && local_markers+=(-m addchars)
            [ "$suffix" = "_nonaligned" ] && local_markers+=(-m nonaligned)
            cfg="configs/w2v_init_ablation/eval/RC/d${dim}/${model}_d${dim}${suffix}.json"
            job "${local_markers[@]}" -m "$cfg" "RC ${model} d${dim}${suffix}" \
                uv run python -m asset_embeddings.scripts.eval.rc \
                    --config "$cfg" \
                    --log "logs/w2v_init_ablation/eval/RC/${model}${suffix}.log" \
                    -r "$DB" -e "$EXP"
        done
    done
done

# ========== Phase 2: Eval MP ==========
for model in $MODELS; do
    echo "=== Phase 2: Eval MP ${model} ==="
    for dim in $DIMS; do
        for quarter in $QUARTERS; do
            cfg="configs/w2v_init_ablation/eval/MP/BERT/${model}/d${dim}/${model}_d${dim}_${quarter}.json"
            job -m mp -m eval -m w2v_init -m bert -m "${model,,}" -m "d${dim}" -m "${quarter,,}" -m "$cfg" \
                "MP ${model} d${dim} ${quarter}" \
                uv run python -m asset_embeddings.scripts.eval.mp_bert \
                    --config "$cfg" \
                    --log "checkpoints/refine/BERT/${model}/d${dim}/${quarter}/refine.log" \
                          "logs/w2v_init_ablation/eval/MP/BERT/${model}/d${dim}/${quarter}.log" \
                    -r "$DB" -e "$EXP"
        done
    done
done

print_summary
