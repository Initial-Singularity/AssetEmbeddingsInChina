#!/bin/bash
# ── Ridge Penalty Robustness Evaluation ─────────────────
# Sweep ridge alpha across 10 values to verify RV/RC stability.
#
# Penalties: 0.0 0.1 0.25 0.5 0.75 1.0 2.0 5.0 10.0 None (CV)
# RV/RC per penalty: 2 baselines + 6 models × 7 dims × 2 suffixes = 86
# Total: 10 penalties × 86 × 2 tasks = 1720 runs
#
# Markers: rv rc eval ridge_robust | ridge{penalty}
#          placebo chars assetrs_binary assetrs_ranks assetrs_level0
#          assetrs_levelmin assetw2v assetbert addchars | d{4..128}
#
# Usage:
#   bash scripts/run/run_notify.sh -a results/db/ridge_robust.db scripts/run/eval_ridge_robust.sh
#
# Examples:
#   bash scripts/run/run_notify.sh -k rv,ridge0.5 scripts/run/eval_ridge_robust.sh
#   bash scripts/run/run_notify.sh -k assetbert,d64 scripts/run/eval_ridge_robust.sh

source scripts/run/run_helper.sh

DIMS="4 8 10 16 32 64 128"
EVAL_MODELS="AssetRS_Binary AssetRS_Ranks AssetRS_Level0 AssetRS_LevelMin AssetW2V AssetBERT"
RIDGE_PENALTIES="0.0 0.1 0.25 0.5 0.75 1.0 2.0 5.0 10.0 None"
DB="results/db/ridge_robust.db"
EXP="ridge_robust"  # runs.experiment; the --experiment flag of every script below

# ========== RV ==========
echo "=== Eval: Ridge RV (Placebo/Chars) ==="
for penalty in $RIDGE_PENALTIES; do
    for model in Placebo Chars; do
        cfg="configs/ridge_robust/eval/ridge${penalty}/RV/${model}.json"
        job -m rv -m eval -m ridge_robust -m "ridge${penalty}" -m "${model,,}" -m "$cfg" "RV ridge${penalty} ${model}" \
            uv run python -m asset_embeddings.scripts.eval.rv \
                --config "$cfg" \
                --log "logs/ridge_robust/eval/ridge${penalty}/RV/${model}.log" \
                -r "$DB" -e "$EXP"
    done
done

echo "=== Eval: Ridge RV (models) ==="
for penalty in $RIDGE_PENALTIES; do
    for model in $EVAL_MODELS; do
        for dim in $DIMS; do
            for suffix in "" "_addChars"; do
                local_markers=(-m rv -m eval -m ridge_robust -m "ridge${penalty}" -m "${model,,}" -m "d${dim}")
                [ -n "$suffix" ] && local_markers+=(-m addchars)
                cfg="configs/ridge_robust/eval/ridge${penalty}/RV/d${dim}/${model}_d${dim}${suffix}.json"
                job "${local_markers[@]}" -m "$cfg" "RV ridge${penalty} ${model} d${dim}${suffix}" \
                    uv run python -m asset_embeddings.scripts.eval.rv \
                        --config "$cfg" \
                        --log "logs/ridge_robust/eval/ridge${penalty}/RV/${model}${suffix}.log" \
                        -r "$DB" -e "$EXP"
            done
        done
    done
done

# ========== RC ==========
echo "=== Eval: Ridge RC (Placebo/Chars) ==="
for penalty in $RIDGE_PENALTIES; do
    for model in Placebo Chars; do
        cfg="configs/ridge_robust/eval/ridge${penalty}/RC/${model}.json"
        job -m rc -m eval -m ridge_robust -m "ridge${penalty}" -m "${model,,}" -m "$cfg" "RC ridge${penalty} ${model}" \
            uv run python -m asset_embeddings.scripts.eval.rc \
                --config "$cfg" \
                --log "logs/ridge_robust/eval/ridge${penalty}/RC/${model}.log" \
                -r "$DB" -e "$EXP"
    done
done

echo "=== Eval: Ridge RC (models) ==="
for penalty in $RIDGE_PENALTIES; do
    for model in $EVAL_MODELS; do
        for dim in $DIMS; do
            for suffix in "" "_addChars"; do
                local_markers=(-m rc -m eval -m ridge_robust -m "ridge${penalty}" -m "${model,,}" -m "d${dim}")
                [ -n "$suffix" ] && local_markers+=(-m addchars)
                cfg="configs/ridge_robust/eval/ridge${penalty}/RC/d${dim}/${model}_d${dim}${suffix}.json"
                job "${local_markers[@]}" -m "$cfg" "RC ridge${penalty} ${model} d${dim}${suffix}" \
                    uv run python -m asset_embeddings.scripts.eval.rc \
                        --config "$cfg" \
                        --log "logs/ridge_robust/eval/ridge${penalty}/RC/${model}${suffix}.log" \
                        -r "$DB" -e "$EXP"
            done
        done
    done
done

print_summary
