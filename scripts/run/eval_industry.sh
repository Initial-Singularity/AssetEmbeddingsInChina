#!/bin/bash
# ── Eval: Industry Dummy Variable Evaluation ───────────
# Reviewer 1 (mandatory): Include industry dummies alongside stock characteristics
# AE Key Issues #2: Upgrade characteristic-based comparators with industry dummies
#
# Variants (each × aligned/non-aligned × 7 dims × RV/RC):
#   IndustryDummy              — industry dummies only
#   Chars_IndustryDummy        — chars + industry dummies
#   AssetBERT_IndustryDummy    — BERT + industry dummies
#   AssetBERT_addChars_IndustryDummy — BERT + chars + industry dummies
# Total: 98 eval runs
#
# Markers: rv rc eval industry | bert industrydummy chars_industrydummy
#          industrydummy_nonaligned addchars_industrydummy nonaligned
#          d{4..128}
#
# Usage:
#   bash scripts/run/run_notify.sh -a results/db/industry.db scripts/run/eval_industry.sh
#
# Examples:
#   bash scripts/run/run_notify.sh -k rv,bert scripts/run/eval_industry.sh
#   bash scripts/run/run_notify.sh -k rc,nonaligned scripts/run/eval_industry.sh

source scripts/run/run_helper.sh

DIMS="4 8 10 16 32 64 128"
DB="results/db/industry.db"
EXP="industry"  # runs.experiment; the --experiment flag of every script below

# ========== RV: No-embedding variants ==========
echo "=== Eval: RV Industry (no-embedding variants) ==="
for dim in $DIMS; do
    for variant in IndustryDummy Chars_IndustryDummy; do
        for na in "" "_nonaligned"; do
            local_markers=(-m rv -m eval -m industry -m "${variant,,}" -m "d${dim}")
            [ -n "$na" ] && local_markers+=(-m nonaligned)
            cfg="configs/industry/eval/RV/d${dim}/${variant}_d${dim}${na}.json"
            job "${local_markers[@]}" -m "$cfg" "RV ${variant} d${dim}${na}" \
                uv run python -m asset_embeddings.scripts.eval.rv \
                    --config "$cfg" \
                    --log "logs/industry/eval/RV/${variant}${na}.log" \
                    -r "$DB" -e "$EXP"
        done
    done
done

# ========== RV: BERT + Industry variants ==========
echo "=== Eval: RV Industry (BERT variants) ==="
for dim in $DIMS; do
    for combo in "IndustryDummy" "IndustryDummy_nonaligned" "addChars_IndustryDummy"; do
        cfg="configs/industry/eval/RV/d${dim}/AssetBERT_d${dim}_${combo}.json"
        job -m rv -m eval -m industry -m bert -m "${combo,,}" -m "d${dim}" -m "$cfg" \
            "RV AssetBERT_${combo} d${dim}" \
            uv run python -m asset_embeddings.scripts.eval.rv \
                --config "$cfg" \
                --log "logs/industry/eval/RV/AssetBERT_${combo}.log" \
                -r "$DB" -e "$EXP"
    done
done

# ========== RC: No-embedding variants ==========
echo "=== Eval: RC Industry (no-embedding variants) ==="
for dim in $DIMS; do
    for variant in IndustryDummy Chars_IndustryDummy; do
        for na in "" "_nonaligned"; do
            local_markers=(-m rc -m eval -m industry -m "${variant,,}" -m "d${dim}")
            [ -n "$na" ] && local_markers+=(-m nonaligned)
            cfg="configs/industry/eval/RC/d${dim}/${variant}_d${dim}${na}.json"
            job "${local_markers[@]}" -m "$cfg" "RC ${variant} d${dim}${na}" \
                uv run python -m asset_embeddings.scripts.eval.rc \
                    --config "$cfg" \
                    --log "logs/industry/eval/RC/${variant}${na}.log" \
                    -r "$DB" -e "$EXP"
        done
    done
done

# ========== RC: BERT + Industry variants ==========
echo "=== Eval: RC Industry (BERT variants) ==="
for dim in $DIMS; do
    for combo in "IndustryDummy" "IndustryDummy_nonaligned" "addChars_IndustryDummy"; do
        cfg="configs/industry/eval/RC/d${dim}/AssetBERT_d${dim}_${combo}.json"
        job -m rc -m eval -m industry -m bert -m "${combo,,}" -m "d${dim}" -m "$cfg" \
            "RC AssetBERT_${combo} d${dim}" \
            uv run python -m asset_embeddings.scripts.eval.rc \
                --config "$cfg" \
                --log "logs/industry/eval/RC/AssetBERT_${combo}.log" \
                -r "$DB" -e "$EXP"
    done
done

print_summary
