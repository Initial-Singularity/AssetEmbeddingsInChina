#!/bin/bash
# ── Eval: Mask Position Sensitivity ────────────────────
# AE Key Issues #1: Clarify and justify the 'mask one position' design
# Reviewer 1 Minor #3: Assess sensitivity to rank chosen
#
# All embedding types × both evaluation heads × 5 mask positions:
#   Embeddings: AssetBERT, AssetRS (4 variants), AssetW2V
#   Heads:      BERT (asset_embeddings.scripts.eval.mp_bert), SMLM (asset_embeddings.scripts.eval.mp_smlm)
#   Positions:  k ∈ {1, 2, 3, 5, 10}
# Total: 6 models × 2 heads × 5 positions × 7 dims × 6 quarters = 2520 runs
#
# Markers: mp eval mask_position | bert bert_head smlm
#          assetbert assetrs_binary assetrs_ranks assetrs_level0
#          assetrs_levelmin assetw2v
#          mask1 mask2 mask3 mask5 mask10 | d{4..128} | {quarter}
#
# Usage:
#   bash scripts/run/run_notify.sh -a results/db/mask_position.db scripts/run/eval_mask_position.sh
#
# Examples:
#   bash scripts/run/run_notify.sh -k smlm,mask3 scripts/run/eval_mask_position.sh
#   bash scripts/run/run_notify.sh -k assetbert,d64 scripts/run/eval_mask_position.sh

source scripts/run/run_helper.sh

DIMS="4 8 10 16 32 64 128"
QUARTERS="2023Q2 2023Q3 2023Q4 2024Q1 2024Q2 2024Q3"
MASK_POSITIONS="1 2 3 5 10"
DB="results/db/mask_position.db"
EXP="mask_position"  # runs.experiment; the --experiment flag of every script below

# ========== Part 1: AssetBERT via BERT head ==========
echo "=== Mask Position Part 1: AssetBERT + BERT head ==="
for k in $MASK_POSITIONS; do
    for dim in $DIMS; do
        for quarter in $QUARTERS; do
            cfg="configs/mask_position/eval/MP_mask${k}/BERT/AssetBERT/d${dim}/AssetBERT_d${dim}_${quarter}.json"
            job -m mp -m eval -m mask_position -m bert -m assetbert -m "mask${k}" -m "d${dim}" -m "${quarter,,}" -m "$cfg" \
                "MP mask${k} AssetBERT BERT d${dim} ${quarter}" \
                uv run python -m asset_embeddings.scripts.eval.mp_bert \
                    --config "$cfg" \
                    --log "checkpoints/refine_mask${k}/BERT/AssetBERT/d${dim}/${quarter}/refine.log" \
                          "logs/mask_position/eval/MP_mask${k}/BERT/AssetBERT/d${dim}/${quarter}.log" \
                    -r "$DB" -e "$EXP"
        done
    done
done

# ========== Part 2: RS/W2V via BERT head (allbert) ==========
echo "=== Mask Position Part 2: RS/W2V + BERT head ==="

declare -A MODEL_PATHS
MODEL_PATHS=(
    ["AssetRS_Binary"]="AssetRS/RS_Binary"
    ["AssetRS_Ranks"]="AssetRS/RS_Ranks"
    ["AssetRS_Level0"]="AssetRS/RS_Level0"
    ["AssetRS_LevelMin"]="AssetRS/RS_LevelMin"
    ["AssetW2V"]="AssetW2V"
)

for k in $MASK_POSITIONS; do
    for model_type in AssetRS_Binary AssetRS_Ranks AssetRS_Level0 AssetRS_LevelMin AssetW2V; do
        model_path=${MODEL_PATHS[$model_type]}
        for dim in $DIMS; do
            for quarter in $QUARTERS; do
                cfg="configs/mask_position/eval/MP_mask${k}/BERT/${model_path}/d${dim}/BERT_${model_type}_d${dim}_${quarter}.json"
                job -m mp -m eval -m mask_position -m bert_head -m "${model_type,,}" -m "mask${k}" -m "d${dim}" -m "${quarter,,}" -m "$cfg" \
                    "MP mask${k} ${model_type} BERT d${dim} ${quarter}" \
                    uv run python -m asset_embeddings.scripts.eval.mp_bert \
                        --config "$cfg" \
                        --log "checkpoints/refine_mask${k}/BERT/${model_path}/d${dim}/${quarter}/refine.log" \
                              "logs/mask_position/eval/MP_mask${k}/BERT/${model_path}/d${dim}/${quarter}.log" \
                        -r "$DB" -e "$EXP"
            done
        done
    done
done

# ========== Part 3: RS/W2V via SMLM head ==========
echo "=== Mask Position Part 3: RS/W2V + SMLM head ==="
for k in $MASK_POSITIONS; do
    for model_type in AssetRS_Binary AssetRS_Ranks AssetRS_Level0 AssetRS_LevelMin AssetW2V; do
        model_path=${MODEL_PATHS[$model_type]}
        for dim in $DIMS; do
            for quarter in $QUARTERS; do
                cfg="configs/mask_position/eval/MP_mask${k}/SMLM/${model_path}/d${dim}/SMLM_${model_type}_d${dim}_${quarter}.json"
                job -m mp -m eval -m mask_position -m smlm -m "${model_type,,}" -m "mask${k}" -m "d${dim}" -m "${quarter,,}" -m "$cfg" \
                    "MP mask${k} ${model_type} SMLM d${dim} ${quarter}" \
                    uv run python -m asset_embeddings.scripts.eval.mp_smlm \
                        --config "$cfg" \
                        --log "checkpoints/refine_mask${k}/SMLM/${model_path}/d${dim}/${quarter}/refine.log" \
                              "logs/mask_position/eval/MP_mask${k}/SMLM/${model_path}/d${dim}/${quarter}.log" \
                        -r "$DB" -e "$EXP"
            done
        done
    done
done

# ========== Part 4: AssetBERT via SMLM head ==========
echo "=== Mask Position Part 4: AssetBERT + SMLM head ==="
for k in $MASK_POSITIONS; do
    for dim in $DIMS; do
        for quarter in $QUARTERS; do
            cfg="configs/mask_position/eval/MP_mask${k}/SMLM/AssetBERT/d${dim}/SMLM_AssetBERT_d${dim}_${quarter}.json"
            job -m mp -m eval -m mask_position -m smlm -m assetbert -m "mask${k}" -m "d${dim}" -m "${quarter,,}" -m "$cfg" \
                "MP mask${k} AssetBERT SMLM d${dim} ${quarter}" \
                uv run python -m asset_embeddings.scripts.eval.mp_smlm \
                    --config "$cfg" \
                    --log "checkpoints/refine_mask${k}/SMLM/AssetBERT/d${dim}/${quarter}/refine.log" \
                          "logs/mask_position/eval/MP_mask${k}/SMLM/AssetBERT/d${dim}/${quarter}.log" \
                    -r "$DB" -e "$EXP"
        done
    done
done

print_summary
