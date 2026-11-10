#!/bin/bash
# ── W2V Model Ablation (CBOW) Evaluation ────────────────────
# RV/RC/MP eval for CBOW-trained W2V and BERT models. Compare
# with main.db (Skip-gram column) by joining on (task, dim, quarter).
#
# Models: AssetW2V_cbow, AssetBERT_cbow
# RV/RC: 2 models × 7 dims × 2 suffixes × 2 tasks = 56 runs
# MP:    BERT (1×7×6) + SMLM (1×7×6) + BERT-head (1×7×6) = 126 runs
# Total: 182 runs
#
# Markers: rv rc mp eval w2v_model | assetw2v_cbow assetbert_cbow
#          bert smlm bert_head addchars | d{4..128} | {quarter}
#
# Usage:
#   bash scripts/run/run_notify.sh -a results/db/w2v_model_ablation.db scripts/run/eval_w2v_model_ablation.sh
#
# Examples:
#   bash scripts/run/run_notify.sh -k rv,assetbert_cbow scripts/run/eval_w2v_model_ablation.sh
#   bash scripts/run/run_notify.sh -k mp,smlm,d64 scripts/run/eval_w2v_model_ablation.sh

source scripts/run/run_helper.sh

DIMS="4 8 10 16 32 64 128"
QUARTERS="2023Q2 2023Q3 2023Q4 2024Q1 2024Q2 2024Q3"
EVAL_MODELS="AssetW2V_cbow AssetBERT_cbow"
BERT_MODELS="AssetBERT_cbow"
DB="results/db/w2v_model_ablation.db"
EXP="w2v_model_ablation"  # runs.experiment; the --experiment flag of every script below

# SMLM model_type|model_path pairs
SMLM_MODELS=(
    "AssetW2V_cbow|AssetW2V_cbow"
)

# ========== RV ==========
echo "=== Eval: RV (models) ==="
for model in $EVAL_MODELS; do
    for dim in $DIMS; do
        for suffix in "" "_addChars"; do
            local_markers=(-m rv -m eval -m w2v_model -m "${model,,}" -m "d${dim}")
            [ -n "$suffix" ] && local_markers+=(-m addchars)
            cfg="configs/w2v_model_ablation/eval/RV/d${dim}/${model}_d${dim}${suffix}.json"
            job "${local_markers[@]}" -m "$cfg" "RV ${model} d${dim}${suffix}" \
                uv run python -m asset_embeddings.scripts.eval.rv \
                    --config "$cfg" \
                    --log "logs/w2v_model_ablation/eval/RV/${model}${suffix}.log" \
                    -r "$DB" -e "$EXP"
        done
    done
done

# ========== RC ==========
echo "=== Eval: RC (models) ==="
for model in $EVAL_MODELS; do
    for dim in $DIMS; do
        for suffix in "" "_addChars"; do
            local_markers=(-m rc -m eval -m w2v_model -m "${model,,}" -m "d${dim}")
            [ -n "$suffix" ] && local_markers+=(-m addchars)
            cfg="configs/w2v_model_ablation/eval/RC/d${dim}/${model}_d${dim}${suffix}.json"
            job "${local_markers[@]}" -m "$cfg" "RC ${model} d${dim}${suffix}" \
                uv run python -m asset_embeddings.scripts.eval.rc \
                    --config "$cfg" \
                    --log "logs/w2v_model_ablation/eval/RC/${model}${suffix}.log" \
                    -r "$DB" -e "$EXP"
        done
    done
done

# ========== MP: BERT ==========
echo "=== Eval: MP (BERT) ==="
for model in $BERT_MODELS; do
    for dim in $DIMS; do
        for quarter in $QUARTERS; do
            cfg="configs/w2v_model_ablation/eval/MP/BERT/${model}/d${dim}/${model}_d${dim}_${quarter}.json"
            job -m mp -m eval -m w2v_model -m bert -m "${model,,}" -m "d${dim}" -m "${quarter,,}" -m "$cfg" \
                "MP ${model} d${dim} ${quarter}" \
                uv run python -m asset_embeddings.scripts.eval.mp_bert \
                    --config "$cfg" \
                    --log "checkpoints/w2v_model_ablation/refine/BERT/${model}/d${dim}/${quarter}/refine.log" \
                          "logs/w2v_model_ablation/eval/MP/BERT/${model}/d${dim}/${quarter}.log" \
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
            cfg="configs/w2v_model_ablation/eval/MP/SMLM/${model_path}/d${dim}/SMLM_${model_type}_d${dim}_${quarter}.json"
            job -m mp -m eval -m w2v_model -m smlm -m "${model_type,,}" -m "d${dim}" -m "${quarter,,}" -m "$cfg" \
                "MP SMLM ${model_type} d${dim} ${quarter}" \
                uv run python -m asset_embeddings.scripts.eval.mp_smlm \
                    --config "$cfg" \
                    --log "checkpoints/w2v_model_ablation/refine/SMLM/${model_path}/d${dim}/${quarter}/refine.log" \
                          "logs/w2v_model_ablation/eval/MP/SMLM/${model_path}/d${dim}/${quarter}.log" \
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
            cfg="configs/w2v_model_ablation/eval/MP/BERT/${model_path}/d${dim}/BERT_${model_type}_d${dim}_${quarter}.json"
            job -m mp -m eval -m w2v_model -m bert_head -m "${model_type,,}" -m "d${dim}" -m "${quarter,,}" -m "$cfg" \
                "MP_BERT ${model_type} d${dim} ${quarter}" \
                uv run python -m asset_embeddings.scripts.eval.mp_bert \
                    --config "$cfg" \
                    --log "checkpoints/w2v_model_ablation/refine/BERT/${model_path}/d${dim}/${quarter}/refine.log" \
                          "logs/w2v_model_ablation/eval/MP/BERT/${model_path}/d${dim}/${quarter}.log" \
                    -r "$DB" -e "$EXP"
        done
    done
done

print_summary
