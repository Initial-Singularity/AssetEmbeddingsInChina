#!/bin/bash
# ── Eval: Static vs Dynamic on TimeSplit Robust (50/50) ──────────
# Static ablation on the 50/50 time-split data.
# Uses pretrained (non-finetuned) embeddings from timesplit_robust pretrain.
# Compare against dynamic results already in timesplit_robust.db.
#
# BERT static:
#   RV/RC: 7 dims × 3 suffixes = 21 eval runs each (42 total)
#   MP:    7 dims × 16 quarters = 112 eval runs
# W2V static:
#   RV/RC: 7 dims × 3 suffixes = 21 eval runs each (42 total)
#   MP SMLM: 7 dims × 16 quarters = 112 eval runs
#   MP BERT: 7 dims × 16 quarters = 112 eval runs
# Total: 420 runs
#
# Markers: rv rc mp eval static bert w2v smlm addchars nonaligned timesplit | d{4..128} | {quarter}
#
# Usage:
#   bash scripts/run/run_notify.sh -a results/db/timesplit_static.db scripts/run/eval_timesplit_static.sh
#
# Examples:
#   bash scripts/run/run_notify.sh -k rv,bert scripts/run/eval_timesplit_static.sh
#   bash scripts/run/run_notify.sh -k mp,d64 scripts/run/eval_timesplit_static.sh

source scripts/run/run_helper.sh

DIMS="4 8 10 16 32 64 128"
QUARTERS="2020Q4 2021Q1 2021Q2 2021Q3 2021Q4 2022Q1 2022Q2 2022Q3 2022Q4 2023Q1 2023Q2 2023Q3 2023Q4 2024Q1 2024Q2 2024Q3"
DB="results/db/timesplit_static.db"
EXP="timesplit_static"  # runs.experiment; the --experiment flag of every script below
PREFIX="configs/timesplit_static/eval"

# ========== BERT Static: RV ==========
echo "=== Eval: Static BERT RV ==="
for dim in $DIMS; do
    for suffix in "" "_nonaligned" "_addChars"; do
        local_markers=(-m rv -m eval -m static -m bert -m "d${dim}")
        [ "$suffix" = "_addChars" ] && local_markers+=(-m addchars)
        [ "$suffix" = "_nonaligned" ] && local_markers+=(-m nonaligned)
        cfg="${PREFIX}/RV/d${dim}/AssetBERT_static_d${dim}${suffix}.json"
        job "${local_markers[@]}" -m "$cfg" "RV static BERT d${dim}${suffix}" \
            uv run python -m asset_embeddings.scripts.eval.rv \
                --config "$cfg" \
                --log "logs/timesplit_static/eval/RV/AssetBERT_static${suffix}.log" \
                -r "$DB" -e "$EXP"
    done
done

# ========== BERT Static: RC ==========
echo "=== Eval: Static BERT RC ==="
for dim in $DIMS; do
    for suffix in "" "_nonaligned" "_addChars"; do
        local_markers=(-m rc -m eval -m static -m bert -m "d${dim}")
        [ "$suffix" = "_addChars" ] && local_markers+=(-m addchars)
        [ "$suffix" = "_nonaligned" ] && local_markers+=(-m nonaligned)
        cfg="${PREFIX}/RC/d${dim}/AssetBERT_static_d${dim}${suffix}.json"
        job "${local_markers[@]}" -m "$cfg" "RC static BERT d${dim}${suffix}" \
            uv run python -m asset_embeddings.scripts.eval.rc \
                --config "$cfg" \
                --log "logs/timesplit_static/eval/RC/AssetBERT_static${suffix}.log" \
                -r "$DB" -e "$EXP"
    done
done

# ========== BERT Static: MP (BERT head) ==========
echo "=== Eval: Static BERT MP ==="
for dim in $DIMS; do
    for quarter in $QUARTERS; do
        cfg="${PREFIX}/MP/BERT/AssetBERT_static/d${dim}/AssetBERT_static_d${dim}_${quarter}.json"
        job -m mp -m eval -m static -m bert -m "d${dim}" -m "${quarter,,}" -m "$cfg" \
            "MP static BERT d${dim} ${quarter}" \
            uv run python -m asset_embeddings.scripts.eval.mp_bert \
                --config "$cfg" \
                --log "checkpoints/timesplit_robust/refine/BERT/AssetBERT_static/d${dim}/${quarter}/refine.log" \
                      "logs/timesplit_static/eval/MP/BERT/AssetBERT_static/d${dim}/${quarter}.log" \
                -r "$DB" -e "$EXP"
    done
done

# ========== W2V Static: RV ==========
echo "=== Eval: Static W2V RV ==="
for dim in $DIMS; do
    for suffix in "" "_nonaligned" "_addChars"; do
        local_markers=(-m rv -m eval -m static -m w2v -m "d${dim}")
        [ "$suffix" = "_addChars" ] && local_markers+=(-m addchars)
        [ "$suffix" = "_nonaligned" ] && local_markers+=(-m nonaligned)
        cfg="${PREFIX}/RV/d${dim}/AssetW2V_static_d${dim}${suffix}.json"
        job "${local_markers[@]}" -m "$cfg" "RV static W2V d${dim}${suffix}" \
            uv run python -m asset_embeddings.scripts.eval.rv \
                --config "$cfg" \
                --log "logs/timesplit_static/eval/RV/AssetW2V_static${suffix}.log" \
                -r "$DB" -e "$EXP"
    done
done

# ========== W2V Static: RC ==========
echo "=== Eval: Static W2V RC ==="
for dim in $DIMS; do
    for suffix in "" "_nonaligned" "_addChars"; do
        local_markers=(-m rc -m eval -m static -m w2v -m "d${dim}")
        [ "$suffix" = "_addChars" ] && local_markers+=(-m addchars)
        [ "$suffix" = "_nonaligned" ] && local_markers+=(-m nonaligned)
        cfg="${PREFIX}/RC/d${dim}/AssetW2V_static_d${dim}${suffix}.json"
        job "${local_markers[@]}" -m "$cfg" "RC static W2V d${dim}${suffix}" \
            uv run python -m asset_embeddings.scripts.eval.rc \
                --config "$cfg" \
                --log "logs/timesplit_static/eval/RC/AssetW2V_static${suffix}.log" \
                -r "$DB" -e "$EXP"
    done
done

# ========== W2V Static: MP (SMLM head) ==========
echo "=== Eval: Static W2V MP (SMLM) ==="
for dim in $DIMS; do
    for quarter in $QUARTERS; do
        cfg="${PREFIX}/MP/SMLM/AssetW2V_static/d${dim}/SMLM_AssetW2V_static_d${dim}_${quarter}.json"
        job -m mp -m eval -m static -m w2v -m smlm -m "d${dim}" -m "${quarter,,}" -m "$cfg" \
            "MP static W2V SMLM d${dim} ${quarter}" \
            uv run python -m asset_embeddings.scripts.eval.mp_smlm \
                --config "$cfg" \
                --log "checkpoints/timesplit_robust/refine/SMLM/AssetW2V_static/d${dim}/${quarter}/refine.log" \
                      "logs/timesplit_static/eval/MP/SMLM/AssetW2V_static/d${dim}/${quarter}.log" \
                -r "$DB" -e "$EXP"
    done
done

# ========== W2V Static: MP (BERT head) ==========
echo "=== Eval: Static W2V MP (BERT head) ==="
for dim in $DIMS; do
    for quarter in $QUARTERS; do
        cfg="${PREFIX}/MP/BERT/AssetW2V_static/d${dim}/BERT_AssetW2V_static_d${dim}_${quarter}.json"
        job -m mp -m eval -m static -m w2v -m bert -m "d${dim}" -m "${quarter,,}" -m "$cfg" \
            "MP static W2V BERT d${dim} ${quarter}" \
            uv run python -m asset_embeddings.scripts.eval.mp_bert \
                --config "$cfg" \
                --log "checkpoints/timesplit_robust/refine/BERT/AssetW2V_static/d${dim}/${quarter}/refine.log" \
                      "logs/timesplit_static/eval/MP/BERT/AssetW2V_static/d${dim}/${quarter}.log" \
                -r "$DB" -e "$EXP"
    done
done

print_summary
