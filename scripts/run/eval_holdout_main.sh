#!/bin/bash
# ── Holdout RV/RC evaluation of the §4 main grid ──────────────────
# Koijen-style firm splits (config scoring "holdout"),
# 20 seeded 80/20 splits per quarter, two penalty tiers run back to back:
#   fixed — ridge_alpha 0.5 (the §4 in-sample value; seconds per config)
#   cv    — ridge_alpha null, 10-fold RidgeCV inside each training split (Koijen;
#           ~20x the in-sample CV cost: 15–65 s per quarter, run it sharded, see below)
#
# Per tier and task: 2 baselines + 6 models × 7 dims × {plain, +Chars} + 6 × 7 full-universe
#                    = 2 + 84 + 42 = 128 configs; 256 per tier, 512 runs in total, 6 quarters each.
# Configs: configs/holdout/eval/{fixed,cv}/{RV,RC}/...   (templates/eval_holdout_main.json)
# DB:      results/db/holdout_main.db, runs.experiment = "holdout" (both tiers; the views' fixed_alpha column tells them apart)
#
# Markers: rv rc eval holdout fixed cv | placebo chars assetbert assetw2v
#          assetrs_binary assetrs_ranks assetrs_level0 assetrs_levelmin
#          addchars full | d{4..128}
#
# Usage:
#   bash scripts/run/run_notify.sh scripts/run/eval_holdout_main.sh
#
# Sharding the CV tier across tmux sessions (one per model; the fixed tier and the
# CV baselines fit in one more):
#   bash scripts/run/run_notify.sh -k fixed -k cv,chars -k cv,placebo scripts/run/eval_holdout_main.sh
#   bash scripts/run/run_notify.sh -k cv,assetbert scripts/run/eval_holdout_main.sh
#   bash scripts/run/run_notify.sh -k cv,assetw2v scripts/run/eval_holdout_main.sh
#   ... (assetrs_binary, assetrs_ranks, assetrs_level0, assetrs_levelmin)

source scripts/run/run_helper.sh

DIMS="4 8 10 16 32 64 128"
EVAL_MODELS="AssetRS_Binary AssetRS_Ranks AssetRS_Level0 AssetRS_LevelMin AssetW2V AssetBERT"
DB="${DB:-results/db/holdout_main.db}"  # override with DB=... in the environment
EXP="${EXP:-holdout}"  # runs.experiment; the --experiment flag of every script below (override with EXP=...)

for TIER in fixed cv; do
    # ========== RV ==========
    echo "=== Eval: RV (Placebo/Chars) [${TIER}] ==="
    for model in Placebo Chars; do
        cfg="configs/holdout/eval/${TIER}/RV/${model}.json"
        job -m rv -m eval -m holdout -m "$TIER" -m "${model,,}" -m "$cfg" "RV ${model} (${TIER})" \
            uv run python -m asset_embeddings.scripts.eval.rv \
                --config "$cfg" \
                --log "logs/holdout/eval/${TIER}/RV/${model}.log" \
                -r "$DB" -e "$EXP"
    done

    echo "=== Eval: RV (models) [${TIER}] ==="
    for model in $EVAL_MODELS; do
        for dim in $DIMS; do
            for suffix in "" "_addChars"; do
                local_markers=(-m rv -m eval -m holdout -m "$TIER" -m "${model,,}" -m "d${dim}")
                [ -n "$suffix" ] && local_markers+=(-m addchars)
                cfg="configs/holdout/eval/${TIER}/RV/d${dim}/${model}_d${dim}${suffix}.json"
                job "${local_markers[@]}" -m "$cfg" "RV ${model} d${dim}${suffix} (${TIER})" \
                    uv run python -m asset_embeddings.scripts.eval.rv \
                        --config "$cfg" \
                        --log "logs/holdout/eval/${TIER}/RV/${model}${suffix}.log" \
                        -r "$DB" -e "$EXP"
            done
        done
    done

    echo "=== Eval: RV (models, full universe) [${TIER}] ==="
    for model in $EVAL_MODELS; do
        for dim in $DIMS; do
            cfg="configs/holdout/eval/${TIER}/RV/d${dim}/${model}_d${dim}_full.json"
            job -m rv -m eval -m holdout -m "$TIER" -m full -m "${model,,}" -m "d${dim}" -m "$cfg" "RV ${model} d${dim} full (${TIER})" \
                uv run python -m asset_embeddings.scripts.eval.rv \
                    --config "$cfg" \
                    --log "logs/holdout/eval/${TIER}/RV/${model}_full.log" \
                    -r "$DB" -e "$EXP"
        done
    done

    # ========== RC ==========
    echo "=== Eval: RC (Placebo/Chars) [${TIER}] ==="
    for model in Placebo Chars; do
        cfg="configs/holdout/eval/${TIER}/RC/${model}.json"
        job -m rc -m eval -m holdout -m "$TIER" -m "${model,,}" -m "$cfg" "RC ${model} (${TIER})" \
            uv run python -m asset_embeddings.scripts.eval.rc \
                --config "$cfg" \
                --log "logs/holdout/eval/${TIER}/RC/${model}.log" \
                -r "$DB" -e "$EXP"
    done

    echo "=== Eval: RC (models) [${TIER}] ==="
    for model in $EVAL_MODELS; do
        for dim in $DIMS; do
            for suffix in "" "_addChars"; do
                local_markers=(-m rc -m eval -m holdout -m "$TIER" -m "${model,,}" -m "d${dim}")
                [ -n "$suffix" ] && local_markers+=(-m addchars)
                cfg="configs/holdout/eval/${TIER}/RC/d${dim}/${model}_d${dim}${suffix}.json"
                job "${local_markers[@]}" -m "$cfg" "RC ${model} d${dim}${suffix} (${TIER})" \
                    uv run python -m asset_embeddings.scripts.eval.rc \
                        --config "$cfg" \
                        --log "logs/holdout/eval/${TIER}/RC/${model}${suffix}.log" \
                        -r "$DB" -e "$EXP"
            done
        done
    done

    echo "=== Eval: RC (models, full universe) [${TIER}] ==="
    for model in $EVAL_MODELS; do
        for dim in $DIMS; do
            cfg="configs/holdout/eval/${TIER}/RC/d${dim}/${model}_d${dim}_full.json"
            job -m rc -m eval -m holdout -m "$TIER" -m full -m "${model,,}" -m "d${dim}" -m "$cfg" "RC ${model} d${dim} full (${TIER})" \
                uv run python -m asset_embeddings.scripts.eval.rc \
                    --config "$cfg" \
                    --log "logs/holdout/eval/${TIER}/RC/${model}_full.log" \
                    -r "$DB" -e "$EXP"
        done
    done
done

print_summary
