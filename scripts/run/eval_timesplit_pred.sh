#!/bin/bash
# ── Predictive Evaluation Pipeline (TimeSplit, Pred-RV + Pred-MP) ─
# Same two prediction tasks as eval_pred.sh but on the 50/50 timesplit
# models with 16 finetune quarters (2020Q4-2024Q3) → 15 prediction pairs.
#
# Pred-RV:  2 baselines + 6 models × 7 dims × 2 suffixes = 86 runs
# Pred-MP:  BERT (1×7×15) + SMLM (5×7×15) + BERT-head (5×7×15) = 1155 runs
# Total: ~1241 runs
#
# Markers: pred_rv pred_mp eval timesplit | placebo chars
#          assetbert assetw2v assetrs_binary assetrs_ranks
#          assetrs_level0 assetrs_levelmin
#          bert smlm bert_head addchars | d{4..128} | {quarter_filename}
#
# Usage:
#   bash scripts/run/run_notify.sh -a results/db/timesplit_pred.db scripts/run/eval_timesplit_pred.sh
#
# Examples:
#   bash scripts/run/run_notify.sh -k pred_rv,assetbert scripts/run/eval_timesplit_pred.sh
#   bash scripts/run/run_notify.sh -k pred_mp,smlm scripts/run/eval_timesplit_pred.sh

source scripts/run/run_helper.sh

DIMS="4 8 10 16 32 64 128"
EVAL_MODELS="AssetRS_Binary AssetRS_Ranks AssetRS_Level0 AssetRS_LevelMin AssetW2V AssetBERT"
BERT_MODELS="AssetBERT"
PREFIX="configs/timesplit_robust/eval"
DB="results/db/timesplit_pred.db"
EXP="timesplit_pred"  # runs.experiment; the --experiment flag of every script below

SMLM_MODELS=(
    "AssetRS_Binary|AssetRS/RS_Binary"
    "AssetRS_Ranks|AssetRS/RS_Ranks"
    "AssetRS_Level0|AssetRS/RS_Level0"
    "AssetRS_LevelMin|AssetRS/RS_LevelMin"
    "AssetW2V|AssetW2V"
)

# 15 quarter pairs: t -> t+1 (2020Q4..2024Q2 -> 2021Q1..2024Q3)
QUARTER_T=(    2020Q4 2021Q1 2021Q2 2021Q3 2021Q4 2022Q1 2022Q2 2022Q3 2022Q4 2023Q1 2023Q2 2023Q3 2023Q4 2024Q1 2024Q2)
QUARTER_NEXT=( 2021Q1 2021Q2 2021Q3 2021Q4 2022Q1 2022Q2 2022Q3 2022Q4 2023Q1 2023Q2 2023Q3 2023Q4 2024Q1 2024Q2 2024Q3)
QUARTER_FILE=(2020Q4_to_2021Q1 2021Q1_to_2021Q2 2021Q2_to_2021Q3 2021Q3_to_2021Q4 2021Q4_to_2022Q1 2022Q1_to_2022Q2 2022Q2_to_2022Q3 2022Q3_to_2022Q4 2022Q4_to_2023Q1 2023Q1_to_2023Q2 2023Q2_to_2023Q3 2023Q3_to_2023Q4 2023Q4_to_2024Q1 2024Q1_to_2024Q2 2024Q2_to_2024Q3)

# ========== Pred-RV ==========
echo "=== Eval: Pred-RV TimeSplit (Placebo/Chars) ==="
for model in Placebo Chars; do
    cfg="${PREFIX}/Pred-RV/${model}.json"
    job -m pred_rv -m eval -m timesplit -m "${model,,}" -m "$cfg" "Pred-RV ts ${model}" \
        uv run python -m asset_embeddings.scripts.eval.rv \
            --config "$cfg" \
            --log "logs/timesplit_robust/eval/Pred-RV/${model}.log" \
            -r "$DB" -e "$EXP"
done

echo "=== Eval: Pred-RV TimeSplit (models) ==="
for model in $EVAL_MODELS; do
    for dim in $DIMS; do
        for suffix in "" "_addChars"; do
            local_markers=(-m pred_rv -m eval -m timesplit -m "${model,,}" -m "d${dim}")
            [ -n "$suffix" ] && local_markers+=(-m addchars)
            cfg="${PREFIX}/Pred-RV/d${dim}/${model}_d${dim}${suffix}.json"
            job "${local_markers[@]}" -m "$cfg" "Pred-RV ts ${model} d${dim}${suffix}" \
                uv run python -m asset_embeddings.scripts.eval.rv \
                    --config "$cfg" \
                    --log "logs/timesplit_robust/eval/Pred-RV/${model}_d${dim}${suffix}.log" \
                    -r "$DB" -e "$EXP"
        done
    done
done

# ========== Pred-MP: BERT head (AssetBERT models) ==========
echo "=== Eval: Pred-MP TimeSplit BERT (AssetBERT) ==="
for model in $BERT_MODELS; do
    for dim in $DIMS; do
        for i in "${!QUARTER_T[@]}"; do
            qt="${QUARTER_T[$i]}"
            qn="${QUARTER_NEXT[$i]}"
            qf="${QUARTER_FILE[$i]}"
            cfg="${PREFIX}/Pred-MP/BERT/${model}/d${dim}/${model}_d${dim}_${qf}.json"
            job -m pred_mp -m eval -m timesplit -m bert -m "${model,,}" -m "d${dim}" -m "${qf,,}" -m "$cfg" \
                "Pred-MP ts BERT ${model} d${dim} ${qf}" \
                uv run python -m asset_embeddings.scripts.eval.mp_bert \
                    --config "$cfg" \
                    --log "checkpoints/timesplit_robust/pred_refine/BERT/${model}/d${dim}/${qt}_to_${qn}/refine.log" \
                          "logs/timesplit_robust/eval/Pred-MP/BERT/${model}/d${dim}/${qf}.log" \
                    -r "$DB" -e "$EXP"
        done
    done
done

# ========== Pred-MP: SMLM head (RS/W2V) ==========
echo "=== Eval: Pred-MP TimeSplit SMLM (RS/W2V) ==="
for entry in "${SMLM_MODELS[@]}"; do
    IFS='|' read -r model_type model_path <<< "$entry"
    for dim in $DIMS; do
        for i in "${!QUARTER_T[@]}"; do
            qt="${QUARTER_T[$i]}"
            qn="${QUARTER_NEXT[$i]}"
            qf="${QUARTER_FILE[$i]}"
            cfg="${PREFIX}/Pred-MP/SMLM/${model_path}/d${dim}/SMLM_${model_type}_d${dim}_${qf}.json"
            job -m pred_mp -m eval -m timesplit -m smlm -m "${model_type,,}" -m "d${dim}" -m "${qf,,}" -m "$cfg" \
                "Pred-MP ts SMLM ${model_type} d${dim} ${qf}" \
                uv run python -m asset_embeddings.scripts.eval.mp_smlm \
                    --config "$cfg" \
                    --log "checkpoints/timesplit_robust/pred_refine/SMLM/${model_path}/d${dim}/${qt}_to_${qn}/refine.log" \
                          "logs/timesplit_robust/eval/Pred-MP/SMLM/${model_path}/d${dim}/${qf}.log" \
                    -r "$DB" -e "$EXP"
        done
    done
done

# ========== Pred-MP: BERT-head (RS/W2V via BERT head) ==========
echo "=== Eval: Pred-MP TimeSplit BERT-head (RS/W2V) ==="
for entry in "${SMLM_MODELS[@]}"; do
    IFS='|' read -r model_type model_path <<< "$entry"
    for dim in $DIMS; do
        for i in "${!QUARTER_T[@]}"; do
            qt="${QUARTER_T[$i]}"
            qn="${QUARTER_NEXT[$i]}"
            qf="${QUARTER_FILE[$i]}"
            cfg="${PREFIX}/Pred-MP/BERT/${model_path}/d${dim}/BERT_${model_type}_d${dim}_${qf}.json"
            job -m pred_mp -m eval -m timesplit -m bert_head -m "${model_type,,}" -m "d${dim}" -m "${qf,,}" -m "$cfg" \
                "Pred-MP ts BERT ${model_type} d${dim} ${qf}" \
                uv run python -m asset_embeddings.scripts.eval.mp_bert \
                    --config "$cfg" \
                    --log "checkpoints/timesplit_robust/pred_refine/BERT/${model_path}/d${dim}/${qt}_to_${qn}/refine.log" \
                          "logs/timesplit_robust/eval/Pred-MP/BERT/${model_path}/d${dim}/${qf}.log" \
                    -r "$DB" -e "$EXP"
        done
    done
done

print_summary
