#!/bin/bash
# Temporal RC of the main (6Q) and timesplit (16Q) grids: RC slopes fitted on earlier periods and applied unchanged
# to period T, in raw coordinates and after an orthogonal Procrustes alignment of T onto each earlier period.
#
# Per window, tier and slope window: 6 models x 7 dims x {raw, aligned} + Chars + 2 Without FT models x 7 dims (raw)
# = 99 configs; 2 windows x 2 tiers x 2 slope windows = 792 runs.
#   fixed: ridge_alpha 0.5    cv: ridge_alpha null, 10-fold CV inside each fitted period
#   previous: the slope of T-1    expanding: the mean over the slopes of every earlier period
# Configs: configs/rc_temporal/{main,timesplit}/{previous,expanding}/{fixed,cv}/...   (templates/eval_rc_temporal.json)
# DB:      results/db/rc_temporal.db, runs.experiment = "rc_temporal"
#
# Markers: rc_temporal eval main timesplit previous expanding fixed cv raw aligned chars static
#          | model (lowercase) | d{4..128}
#
# Usage:
#   bash scripts/run/run_notify.sh scripts/run/eval_rc_temporal.sh
#   bash scripts/run/run_notify.sh -k fixed -k expanding scripts/run/eval_rc_temporal.sh

source scripts/run/run_helper.sh

DIMS="4 8 10 16 32 64 128"
EVAL_MODELS="AssetRS_Binary AssetRS_Ranks AssetRS_Level0 AssetRS_LevelMin AssetW2V AssetBERT"
STATIC_MODELS="AssetBERT_static AssetW2V_static"  # pretrained, without finetuning
DB="${DB:-results/db/rc_temporal.db}"  # override with DB=... in the environment
EXP="${EXP:-rc_temporal}"  # runs.experiment; the --experiment flag of every script below (override with EXP=...)

for WINDOW in main timesplit; do
    for SLOPE in previous expanding; do
        for TIER in fixed cv; do
            dir="configs/rc_temporal/${WINDOW}/${SLOPE}/${TIER}"
            logdir="logs/rc_temporal/${WINDOW}/${SLOPE}/${TIER}"
            tags=(-m rc_temporal -m eval -m "$WINDOW" -m "$SLOPE" -m "$TIER")

            echo "=== Eval: RC temporal (Chars) [${WINDOW}, ${SLOPE}, ${TIER}] ==="
            cfg="${dir}/Chars.json"
            job "${tags[@]}" -m chars -m raw -m "$cfg" "RC temporal Chars (${WINDOW}, ${SLOPE}, ${TIER})"                 uv run python -m asset_embeddings.scripts.eval.rc_temporal                     --config "$cfg"                     --log "${logdir}/Chars.log"                     -r "$DB" -e "$EXP"

            echo "=== Eval: RC temporal (models) [${WINDOW}, ${SLOPE}, ${TIER}] ==="
            for model in $EVAL_MODELS; do
                for dim in $DIMS; do
                    for align in raw aligned; do
                        cfg="${dir}/d${dim}/${model}_d${dim}_${align}.json"
                        job "${tags[@]}" -m "${model,,}" -m "d${dim}" -m "$align" -m "$cfg"                             "RC temporal ${model} d${dim} ${align} (${WINDOW}, ${SLOPE}, ${TIER})"                             uv run python -m asset_embeddings.scripts.eval.rc_temporal                                 --config "$cfg"                                 --log "${logdir}/${model}_${align}.log"                                 -r "$DB" -e "$EXP"
                    done
                done
            done

            echo "=== Eval: RC temporal (Without FT) [${WINDOW}, ${SLOPE}, ${TIER}] ==="
            for model in $STATIC_MODELS; do
                for dim in $DIMS; do
                    cfg="${dir}/d${dim}/${model}_d${dim}_raw.json"
                    job "${tags[@]}" -m static -m "${model,,}" -m "d${dim}" -m raw -m "$cfg"                         "RC temporal ${model} d${dim} raw (${WINDOW}, ${SLOPE}, ${TIER})"                         uv run python -m asset_embeddings.scripts.eval.rc_temporal                             --config "$cfg"                             --log "${logdir}/${model}_raw.log"                             -r "$DB" -e "$EXP"
                done
            done
        done
    done
done

print_summary
