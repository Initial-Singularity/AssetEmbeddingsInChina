#!/bin/bash
# ── Eval: Coverage Gradient (§7.4) ───────────
# Cross-sectional coverage gradient of embedding informativeness (RV + RC),
# run through the grouped scripts (rv_grouped / rc_grouped) with groups formed
# from institutional coverage, across the 6-model panel (BERT + W2V + 4 RS variants), 7 dims, in BOTH
# binning modes (quantile + fixed absolute bins), plus the model-independent
# characteristics baseline (features ["chars"]).
#
# Grid: 6 models × 7 dims × {RV,RC} × {quantile,fixed} = 168 embedding runs
#       + 2 tasks × {quantile,fixed} char baselines            = 4 char runs
#       Total = 172 runs.
#
# Binning mode is selected at run time via -o overrides (configs are
# binning-agnostic, default quantile). Fixed bins default to COVERAGE_BINS
# below; a distribution-fit alternative is e.g. [1,5,20,60] (see design §6).
# The chosen bins are persisted in each row's config_content.
#
# RV target regularization (set in the configs, RV-only): the OLS valuation
# residual is winsorized (winsor_pct=0.1) and the CV ridge-alpha grid is
# floored (cv_alpha_min=1000) so the well-covered groups are not under-
# regularized. RC embedding runs are left untouched (winsor_pct/cv_alpha_min
# = null). The characteristics baseline carries its own caliber (char
# subsamples are far smaller; the design matrix is 4-6 columns, not 128):
# RV char fixes the penalty at ridge_alpha=700 with the residual target
# winsorized at 1% (largest penalty-relaxation keeping every embedding
# model above the char row in every group); RC char
# winsorizes the return target at 0.5% and selects the CV penalty once on
# the period's whole cross section (alpha_scope=cross_section).
#
# Markers: eval coverage rv rc quantile fixed char | assetbert assetw2v
#          assetrs_binary assetrs_ranks assetrs_level0 assetrs_levelmin | d{4..128}
#
# Usage:
#   bash scripts/run/run_notify.sh -a results/db/coverage.db scripts/run/eval_coverage.sh
#
# Examples:
#   bash scripts/run/run_notify.sh -k rv,assetw2v,fixed,d128 scripts/run/eval_coverage.sh
#   bash scripts/run/run_notify.sh -k char scripts/run/eval_coverage.sh
#   bash scripts/run/run_notify.sh --exclude fixed scripts/run/eval_coverage.sh   # quantile only

source scripts/run/run_helper.sh

MODELS="AssetBERT AssetW2V AssetRS_Binary AssetRS_Ranks AssetRS_Level0 AssetRS_LevelMin"
DIMS="4 8 10 16 32 64 128"
MODES="quantile fixed"
TASKS="RV RC"
COVERAGE_BINS="[1,3,10]"   # fixed-mode inner cut points -> groups 1 / 2-3 / 4-10 / >10
DB="results/db/coverage.db"
EXP="coverage"  # runs.experiment; the --experiment flag of every script below

# Populate the OVERRIDES array for a binning mode. An array (not command
# substitution) keeps bins=[1,3,10] one literal argv token — no
# word-splitting, no glob expansion of the brackets.
OVERRIDES=()
build_overrides() {
    local mode="$1"
    if [ "$mode" = "fixed" ]; then
        OVERRIDES=(-o grouping=fixed "bins=${COVERAGE_BINS}")
    else
        OVERRIDES=(-o grouping=quantile)
    fi
}

# ── Embedding grid: 6 models × 7 dims × {RV,RC} × {quantile,fixed} ──
for task in $TASKS; do
    module="asset_embeddings.scripts.eval.${task,,}_grouped"
    for model in $MODELS; do
        for mode in $MODES; do
            build_overrides "$mode"
            for dim in $DIMS; do
                echo "=== Coverage ${task} ${model} d${dim} ${mode} ==="
                cfg="configs/coverage/eval/${task}/d${dim}/${model}_d${dim}.json"
                job -m "${task,,}" -m eval -m coverage -m "${model,,}" -m "d${dim}" -m "${mode}" -m "$cfg" \
                    "Coverage ${task} ${model} d${dim} ${mode}" \
                    uv run python -m "$module" \
                        --config "$cfg" \
                        "${OVERRIDES[@]}" \
                        --log "logs/coverage/${task}/${model}_d${dim}_${mode}.log" \
                        -r "$DB" -e "$EXP"
            done
        done
    done
done

# ── Characteristics baseline: {RV,RC} × {quantile,fixed} (model/dim-independent) ──
for task in $TASKS; do
    module="asset_embeddings.scripts.eval.${task,,}_grouped"
    for mode in $MODES; do
        build_overrides "$mode"
        echo "=== Coverage ${task} char baseline ${mode} ==="
        cfg="configs/coverage/eval/${task}/char/char_baseline.json"
        job -m "${task,,}" -m eval -m coverage -m char -m "${mode}" -m "$cfg" \
            "Coverage ${task} char baseline ${mode}" \
            uv run python -m "$module" \
                --config "$cfg" \
                "${OVERRIDES[@]}" \
                --log "logs/coverage/${task}/char_${mode}.log" \
                -r "$DB" -e "$EXP"
    done
done

print_summary
