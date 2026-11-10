#!/bin/bash
# ── 50/50 Time-Split Robustness Evaluation ─────────────────
# Re-run full eval pipeline on models trained with 50% pretrain / 50%
# finetune split to test robustness against training period allocation.
#
# Quarters: 2020Q4 - 2024Q3 (16 quarters)
# RV/RC: 2 baselines + 6 models x 7 dims x 2 suffixes, x 2 tasks = ~172 runs
# MP:    (1 BERT + 5 SMLM + 5 BERT-head) x 7 dims x 16 quarters = 1232 runs
# Total: ~1404 runs
#
# Markers: rv rc mp eval timesplit | placebo chars assetbert assetw2v
#          assetrs_binary assetrs_ranks assetrs_level0 assetrs_levelmin
#          bert smlm bert_head addchars | d{4..128} | {quarter}
#
# Usage:
#   bash scripts/run/run_notify.sh -a results/db/timesplit_robust.db scripts/run/eval_timesplit_robust.sh
#
# Examples:
#   bash scripts/run/run_notify.sh -k mp,smlm,d64 scripts/run/eval_timesplit_robust.sh
#   bash scripts/run/run_notify.sh -k rv,assetbert scripts/run/eval_timesplit_robust.sh

source scripts/run/run_helper.sh

DIMS="4 8 10 16 32 64 128"
QUARTERS="2020Q4 2021Q1 2021Q2 2021Q3 2021Q4 2022Q1 2022Q2 2022Q3 2022Q4 2023Q1 2023Q2 2023Q3 2023Q4 2024Q1 2024Q2 2024Q3"
EVAL_MODELS="AssetRS_Binary AssetRS_Ranks AssetRS_Level0 AssetRS_LevelMin AssetW2V AssetBERT"
BERT_MODELS="AssetBERT"
PREFIX="configs/timesplit_robust/eval"
DB="results/db/timesplit_robust.db"
EXP="timesplit_robust"  # runs.experiment; the --experiment flag of every script below

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
    cfg="${PREFIX}/RV/${model}.json"
    job -m rv -m eval -m timesplit -m "${model,,}" -m "$cfg" "RV ts ${model}" \
        uv run python -m asset_embeddings.scripts.eval.rv \
            --config "$cfg" \
            --log "logs/timesplit_robust/eval/RV/${model}.log" \
            -r "$DB" -e "$EXP"
done

echo "=== Eval: RV (models) ==="
for model in $EVAL_MODELS; do
    for dim in $DIMS; do
        for suffix in "" "_addChars"; do
            local_markers=(-m rv -m eval -m timesplit -m "${model,,}" -m "d${dim}")
            [ -n "$suffix" ] && local_markers+=(-m addchars)
            cfg="${PREFIX}/RV/d${dim}/${model}_d${dim}${suffix}.json"
            job "${local_markers[@]}" -m "$cfg" "RV ts ${model} d${dim}${suffix}" \
                uv run python -m asset_embeddings.scripts.eval.rv \
                    --config "$cfg" \
                    --log "logs/timesplit_robust/eval/RV/${model}${suffix}.log" \
                    -r "$DB" -e "$EXP"
        done
    done
done

# ========== RC ==========
echo "=== Eval: RC (Placebo/Chars) ==="
for model in Placebo Chars; do
    cfg="${PREFIX}/RC/${model}.json"
    job -m rc -m eval -m timesplit -m "${model,,}" -m "$cfg" "RC ts ${model}" \
        uv run python -m asset_embeddings.scripts.eval.rc \
            --config "$cfg" \
            --log "logs/timesplit_robust/eval/RC/${model}.log" \
            -r "$DB" -e "$EXP"
done

echo "=== Eval: RC (models) ==="
for model in $EVAL_MODELS; do
    for dim in $DIMS; do
        for suffix in "" "_addChars"; do
            local_markers=(-m rc -m eval -m timesplit -m "${model,,}" -m "d${dim}")
            [ -n "$suffix" ] && local_markers+=(-m addchars)
            cfg="${PREFIX}/RC/d${dim}/${model}_d${dim}${suffix}.json"
            job "${local_markers[@]}" -m "$cfg" "RC ts ${model} d${dim}${suffix}" \
                uv run python -m asset_embeddings.scripts.eval.rc \
                    --config "$cfg" \
                    --log "logs/timesplit_robust/eval/RC/${model}${suffix}.log" \
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
            job -m mp -m eval -m timesplit -m bert -m "${model,,}" -m "d${dim}" -m "${quarter,,}" -m "$cfg" "MP ts ${model} d${dim} ${quarter}" \
                uv run python -m asset_embeddings.scripts.eval.mp_bert \
                    --config "$cfg" \
                    --log "checkpoints/timesplit_robust/refine/BERT/${model}/d${dim}/${quarter}/refine.log" \
                          "logs/timesplit_robust/eval/MP/BERT/${model}/d${dim}/${quarter}.log" \
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
            job -m mp -m eval -m timesplit -m smlm -m "${model_type,,}" -m "d${dim}" -m "${quarter,,}" -m "$cfg" "MP ts SMLM ${model_type} d${dim} ${quarter}" \
                uv run python -m asset_embeddings.scripts.eval.mp_smlm \
                    --config "$cfg" \
                    --log "checkpoints/timesplit_robust/refine/SMLM/${model_path}/d${dim}/${quarter}/refine.log" \
                          "logs/timesplit_robust/eval/MP/SMLM/${model_path}/d${dim}/${quarter}.log" \
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
            job -m mp -m eval -m timesplit -m bert_head -m "${model_type,,}" -m "d${dim}" -m "${quarter,,}" -m "$cfg" "MP_BERT ts ${model_type} d${dim} ${quarter}" \
                uv run python -m asset_embeddings.scripts.eval.mp_bert \
                    --config "$cfg" \
                    --log "checkpoints/timesplit_robust/refine/BERT/${model_path}/d${dim}/${quarter}/refine.log" \
                          "logs/timesplit_robust/eval/MP/BERT/${model_path}/d${dim}/${quarter}.log" \
                    -r "$DB" -e "$EXP"
        done
    done
done

print_summary
