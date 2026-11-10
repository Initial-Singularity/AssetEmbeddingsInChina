#!/bin/bash
# ── Eval: Static vs Dynamic Embedding ──────────────────
# AE Key Issues #1: Quantify incremental benefit of quarterly fine-tuning
# Reviewer 1 & 2: Does dynamic tuning materially improve performance?
#
# Static = pretrained encoder/embedding, never fine-tuned on quarterly data.
# Compare against dynamic AssetBERT/AssetW2V (already in base eval results).
#
# BERT static:
#   RV/RC: 7 dims × 3 suffixes × 2 benchmarks = 42 eval runs
#   MP:    7 dims × 6 quarters = 42 eval runs
# W2V static:
#   RV/RC: 7 dims × 3 suffixes × 2 benchmarks = 42 eval runs
#   MP SMLM: 7 dims × 6 quarters = 42 eval runs
#   MP BERT: 7 dims × 6 quarters = 42 eval runs
# Total: 210 runs
#
# Markers: rv rc mp eval static bert w2v smlm addchars nonaligned | d{4..128} | {quarter}
#
# Usage:
#   bash scripts/run/run_notify.sh -a results/db/static.db scripts/run/eval_static.sh
#
# Examples:
#   bash scripts/run/run_notify.sh -k mp,d64 scripts/run/eval_static.sh
#   bash scripts/run/run_notify.sh -k rv,w2v scripts/run/eval_static.sh
#   bash scripts/run/run_notify.sh -k mp,smlm scripts/run/eval_static.sh

source scripts/run/run_helper.sh

DIMS="4 8 10 16 32 64 128"
QUARTERS="2023Q2 2023Q3 2023Q4 2024Q1 2024Q2 2024Q3"
DB="results/db/static.db"
EXP="static"  # runs.experiment; the --experiment flag of every script below

# ========== BERT Static: RV ==========
echo "=== Eval: Static BERT RV ==="
for dim in $DIMS; do
    for suffix in "" "_nonaligned" "_addChars"; do
        local_markers=(-m rv -m eval -m static -m bert -m "d${dim}")
        [ "$suffix" = "_addChars" ] && local_markers+=(-m addchars)
        [ "$suffix" = "_nonaligned" ] && local_markers+=(-m nonaligned)
        cfg="configs/static/eval/RV/d${dim}/AssetBERT_static_d${dim}${suffix}.json"
        job "${local_markers[@]}" -m "$cfg" "RV static BERT d${dim}${suffix}" \
            uv run python -m asset_embeddings.scripts.eval.rv \
                --config "$cfg" \
                --log "logs/static/eval/RV/AssetBERT_static${suffix}.log" \
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
        cfg="configs/static/eval/RC/d${dim}/AssetBERT_static_d${dim}${suffix}.json"
        job "${local_markers[@]}" -m "$cfg" "RC static BERT d${dim}${suffix}" \
            uv run python -m asset_embeddings.scripts.eval.rc \
                --config "$cfg" \
                --log "logs/static/eval/RC/AssetBERT_static${suffix}.log" \
                -r "$DB" -e "$EXP"
    done
done

# ========== BERT Static: MP (BERT head) ==========
echo "=== Eval: Static BERT MP ==="
for dim in $DIMS; do
    for quarter in $QUARTERS; do
        cfg="configs/static/eval/MP/BERT/AssetBERT_static/d${dim}/AssetBERT_static_d${dim}_${quarter}.json"
        job -m mp -m eval -m static -m bert -m "d${dim}" -m "${quarter,,}" -m "$cfg" \
            "MP static BERT d${dim} ${quarter}" \
            uv run python -m asset_embeddings.scripts.eval.mp_bert \
                --config "$cfg" \
                --log "checkpoints/refine/BERT/AssetBERT_static/d${dim}/${quarter}/refine.log" \
                      "logs/static/eval/MP/BERT/AssetBERT_static/d${dim}/${quarter}.log" \
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
        cfg="configs/static/eval/RV/d${dim}/AssetW2V_static_d${dim}${suffix}.json"
        job "${local_markers[@]}" -m "$cfg" "RV static W2V d${dim}${suffix}" \
            uv run python -m asset_embeddings.scripts.eval.rv \
                --config "$cfg" \
                --log "logs/static/eval/RV/AssetW2V_static${suffix}.log" \
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
        cfg="configs/static/eval/RC/d${dim}/AssetW2V_static_d${dim}${suffix}.json"
        job "${local_markers[@]}" -m "$cfg" "RC static W2V d${dim}${suffix}" \
            uv run python -m asset_embeddings.scripts.eval.rc \
                --config "$cfg" \
                --log "logs/static/eval/RC/AssetW2V_static${suffix}.log" \
                -r "$DB" -e "$EXP"
    done
done

# ========== W2V Static: MP (SMLM head) ==========
echo "=== Eval: Static W2V MP (SMLM) ==="
for dim in $DIMS; do
    for quarter in $QUARTERS; do
        cfg="configs/static/eval/MP/SMLM/AssetW2V_static/d${dim}/SMLM_AssetW2V_static_d${dim}_${quarter}.json"
        job -m mp -m eval -m static -m w2v -m smlm -m "d${dim}" -m "${quarter,,}" -m "$cfg" \
            "MP static W2V SMLM d${dim} ${quarter}" \
            uv run python -m asset_embeddings.scripts.eval.mp_smlm \
                --config "$cfg" \
                --log "checkpoints/refine/SMLM/AssetW2V_static/d${dim}/${quarter}/refine.log" \
                      "logs/static/eval/MP/SMLM/AssetW2V_static/d${dim}/${quarter}.log" \
                -r "$DB" -e "$EXP"
    done
done

# ========== W2V Static: MP (BERT head) ==========
echo "=== Eval: Static W2V MP (BERT head) ==="
for dim in $DIMS; do
    for quarter in $QUARTERS; do
        cfg="configs/static/eval/MP/BERT/AssetW2V_static/d${dim}/BERT_AssetW2V_static_d${dim}_${quarter}.json"
        job -m mp -m eval -m static -m w2v -m bert -m "d${dim}" -m "${quarter,,}" -m "$cfg" \
            "MP static W2V BERT d${dim} ${quarter}" \
            uv run python -m asset_embeddings.scripts.eval.mp_bert \
                --config "$cfg" \
                --log "checkpoints/refine/BERT/AssetW2V_static/d${dim}/${quarter}/refine.log" \
                      "logs/static/eval/MP/BERT/AssetW2V_static/d${dim}/${quarter}.log" \
                -r "$DB" -e "$EXP"
    done
done

print_summary
