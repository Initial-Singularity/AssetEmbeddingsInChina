#!/bin/bash
# ── Main Evaluation Pipeline ────────────────────────────
# Paper Table 2-4 / Figure 3-5: asset embedding evaluation across three tasks
#
# Models (6 embedding + 2 baseline):
#   Baselines: Placebo, Chars
#   AssetRS  — 4 variants (Binary, Ranks, Level0, LevelMin)
#   AssetW2V, AssetBERT
# RV/RC: 2 baselines + 6 models × 7 dims × 2 suffixes = 170 eval runs
# MP:    BERT (1×7×6) + SMLM (5×7×6) + BERT-head (5×7×6) = 462 eval runs
# Total: ~632 runs
#
# Markers: rv rc mp eval | placebo chars assetbert assetw2v
#          assetrs_binary assetrs_ranks assetrs_level0 assetrs_levelmin
#          bert smlm bert_head addchars | d{4..128} | {quarter}
#
# Usage:
#   bash scripts/run/run_notify.sh -a results/db/main.db scripts/run/eval_main.sh
#
# Examples:
#   bash scripts/run/run_notify.sh -k rv,assetbert scripts/run/eval_main.sh
#   bash scripts/run/run_notify.sh -k mp,smlm,d64 scripts/run/eval_main.sh

source scripts/run/run_helper.sh

DIMS="4 8 10 16 32 64 128"
QUARTERS="2023Q2 2023Q3 2023Q4 2024Q1 2024Q2 2024Q3"
EVAL_MODELS="AssetRS_Binary AssetRS_Ranks AssetRS_Level0 AssetRS_LevelMin AssetW2V AssetBERT"
BERT_MODELS="AssetBERT"
DB="${DB:-results/db/main.db}"  # override with DB=... in the environment
EXP="${EXP:-main}"  # runs.experiment; the --experiment flag of every script below (override with EXP=...)

# SMLM model_type|model_path pairs
SMLM_MODELS=(
    "AssetRS_Binary|AssetRS/RS_Binary"
    "AssetRS_Ranks|AssetRS/RS_Ranks"
    "AssetRS_Level0|AssetRS/RS_Level0"
    "AssetRS_LevelMin|AssetRS/RS_LevelMin"
    "AssetW2V|AssetW2V"
)

# ========== RV ==========
echo "=== Eval: RV (Placebo/Chars) ==="
for model in Placebo Chars; do
    cfg="configs/main/eval/RV/${model}.json"
    job -m rv -m eval -m "${model,,}" -m "$cfg" "RV ${model}" \
        uv run python -m asset_embeddings.scripts.eval.rv \
            --config "$cfg" \
            --log "logs/main/eval/RV/${model}.log" \
            -r "$DB" -e "$EXP"
done

echo "=== Eval: RV (models) ==="
for model in $EVAL_MODELS; do
    for dim in $DIMS; do
        for suffix in "" "_addChars"; do
            local_markers=(-m rv -m eval -m "${model,,}" -m "d${dim}")
            [ -n "$suffix" ] && local_markers+=(-m addchars)
            cfg="configs/main/eval/RV/d${dim}/${model}_d${dim}${suffix}.json"
            job "${local_markers[@]}" -m "$cfg" "RV ${model} d${dim}${suffix}" \
                uv run python -m asset_embeddings.scripts.eval.rv \
                    --config "$cfg" \
                    --log "logs/main/eval/RV/${model}${suffix}.log" \
                    -r "$DB" -e "$EXP"
        done
    done
done

# ========== RC ==========
echo "=== Eval: RC (Placebo/Chars) ==="
for model in Placebo Chars; do
    cfg="configs/main/eval/RC/${model}.json"
    job -m rc -m eval -m "${model,,}" -m "$cfg" "RC ${model}" \
        uv run python -m asset_embeddings.scripts.eval.rc \
            --config "$cfg" \
            --log "logs/main/eval/RC/${model}.log" \
            -r "$DB" -e "$EXP"
done

echo "=== Eval: RC (models) ==="
for model in $EVAL_MODELS; do
    for dim in $DIMS; do
        for suffix in "" "_addChars"; do
            local_markers=(-m rc -m eval -m "${model,,}" -m "d${dim}")
            [ -n "$suffix" ] && local_markers+=(-m addchars)
            cfg="configs/main/eval/RC/d${dim}/${model}_d${dim}${suffix}.json"
            job "${local_markers[@]}" -m "$cfg" "RC ${model} d${dim}${suffix}" \
                uv run python -m asset_embeddings.scripts.eval.rc \
                    --config "$cfg" \
                    --log "logs/main/eval/RC/${model}${suffix}.log" \
                    -r "$DB" -e "$EXP"
        done
    done
done

# ========== MP: BERT ==========
echo "=== Eval: MP (BERT) ==="
for model in $BERT_MODELS; do
    for dim in $DIMS; do
        for quarter in $QUARTERS; do
            cfg="configs/main/eval/MP/BERT/${model}/d${dim}/${model}_d${dim}_${quarter}.json"
            job -m mp -m eval -m bert -m "${model,,}" -m "d${dim}" -m "${quarter,,}" -m "$cfg" \
                "MP ${model} d${dim} ${quarter}" \
                uv run python -m asset_embeddings.scripts.eval.mp_bert \
                    --config "$cfg" \
                    --log "checkpoints/refine/BERT/${model}/d${dim}/${quarter}/refine.log" \
                          "logs/main/eval/MP/BERT/${model}/d${dim}/${quarter}.log" \
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
            cfg="configs/main/eval/MP/SMLM/${model_path}/d${dim}/SMLM_${model_type}_d${dim}_${quarter}.json"
            job -m mp -m eval -m smlm -m "${model_type,,}" -m "d${dim}" -m "${quarter,,}" -m "$cfg" \
                "MP SMLM ${model_type} d${dim} ${quarter}" \
                uv run python -m asset_embeddings.scripts.eval.mp_smlm \
                    --config "$cfg" \
                    --log "checkpoints/refine/SMLM/${model_path}/d${dim}/${quarter}/refine.log" \
                          "logs/main/eval/MP/SMLM/${model_path}/d${dim}/${quarter}.log" \
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
            cfg="configs/main/eval/MP/BERT/${model_path}/d${dim}/BERT_${model_type}_d${dim}_${quarter}.json"
            job -m mp -m eval -m bert_head -m "${model_type,,}" -m "d${dim}" -m "${quarter,,}" -m "$cfg" \
                "MP_BERT ${model_type} d${dim} ${quarter}" \
                uv run python -m asset_embeddings.scripts.eval.mp_bert \
                    --config "$cfg" \
                    --log "checkpoints/refine/BERT/${model_path}/d${dim}/${quarter}/refine.log" \
                          "logs/main/eval/MP/BERT/${model_path}/d${dim}/${quarter}.log" \
                    -r "$DB" -e "$EXP"
        done
    done
done

print_summary
