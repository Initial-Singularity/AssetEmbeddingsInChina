#!/bin/bash
# ── Text Embedding Evaluation Pipeline ────────────────────────────
# §4.3 Text Embedding Comparison: evaluates 9 LLM-based asset embeddings
# against §3 main results (main_rerun) on RV, RC, MP-BERT.
#
# Coverage:
#   9 (provider, model) × 2 langs × {2-3} transforms × 7 dims
#   = 24 model×transform tuples × 2 langs × 7 dims = 336 embedding sets
#   cohere/voyage/gemini lack native (target_dims not in supported_native_dims).
#
# Per embedding set:
#   RV plain + RV +Chars (2)
#   RC plain + RC +Chars (2)
#   MP-BERT × 6 quarters  (6)
#   = 10 eval runs / set → ~3,360 total
#
# MP uses BERT head (model_checkpoint=null, freeze_embedding=true, freeze_encoder=false),
# matching main_rerun's recipe for non-BERT embeddings (AssetW2V/RS). SMLM is
# intentionally excluded — main_rerun showed SMLM cannot distinguish quality across
# non-BERT embeddings (all variants converge to identical scores).
#
# Markers: text_embedding rv rc mp eval bert_head addchars
#          openai cohere voyage gemini local
#          openai-3-small openai-3-large cohere-v4 voyage-3-large gemini-001
#          qwen-0.6b qwen-4b qwen-8b bge-m3
#          zh en | native pca pca-universal pca-perperiod
#          d{4..128} | 2023q2 2023q3 2023q4 2024q1 2024q2 2024q3
#
# Prerequisites:
#   1. Generate configs:
#        uv run python -m scripts.tools.fast_generator config \
#            --from-preset templates/eval_text_embedding.json
#   2. Embeddings produced under data/text_embeddings/embeddings/{model_slug}/...
#      (prepared beforehand).
#
# Usage:
#   bash scripts/run/run_notify.sh -a results/db/text_embedding.db \
#       scripts/run/eval_text_embedding.sh
#
# Examples:
#   # RV/RC only (skip the slow MP step)
#   bash scripts/run/run_notify.sh --exclude mp scripts/run/eval_text_embedding.sh
#   # OpenAI-3-small zh native only
#   bash scripts/run/run_notify.sh -k openai-3-small,zh,native scripts/run/eval_text_embedding.sh
#   # All PCA results across providers, RV only
#   bash scripts/run/run_notify.sh -k pca,rv scripts/run/eval_text_embedding.sh
#   # Hosted models only (skip local Qwen/BGE — useful while local is still embedding)
#   bash scripts/run/run_notify.sh --exclude local scripts/run/eval_text_embedding.sh

source scripts/run/run_helper.sh

DIMS="4 8 10 16 32 64 128"
LANGS="zh en"
TRANSFORMS_ALL="native pca-universal pca-perperiod"
TRANSFORMS_PCA_ONLY="pca-universal pca-perperiod"
QUARTERS="2023Q2 2023Q3 2023Q4 2024Q1 2024Q2 2024Q3"
DB="results/db/text_embedding.db"
EXP="text_embedding"  # runs.experiment; the --experiment flag of every script below

# (provider | model_id | model_slug | has_native | short_marker)
TEXT_MODELS=(
    "openai|text-embedding-3-small|openai_text-embedding-3-small|true|openai-3-small"
    "openai|text-embedding-3-large|openai_text-embedding-3-large|true|openai-3-large"
    "cohere|embed-v4.0|cohere_embed-v4.0|false|cohere-v4"
    "voyage|voyage-3-large|voyage_voyage-3-large|false|voyage-3-large"
    "gemini|gemini-embedding-001|gemini_gemini-embedding-001|false|gemini-001"
    "local|Qwen/Qwen3-Embedding-0.6B|local_Qwen-Qwen3-Embedding-0.6B|true|qwen-0.6b"
    "local|Qwen/Qwen3-Embedding-4B|local_Qwen-Qwen3-Embedding-4B|true|qwen-4b"
    "local|Qwen/Qwen3-Embedding-8B|local_Qwen-Qwen3-Embedding-8B|true|qwen-8b"
    "local|BAAI/bge-m3|local_BAAI-bge-m3|true|bge-m3"
)

# Helper: emit transform markers; pca-* variants also tagged with bare "pca"
# Usage: read into an array, e.g. transform_markers=( $(_transform_markers "$transform") )
_transform_markers() {
    case "$1" in
        native)        echo "native" ;;
        pca-universal) echo "pca pca-universal" ;;
        pca-perperiod) echo "pca pca-perperiod" ;;
        *) echo "$1" ;;
    esac
}

# ========== RV ==========
echo "=== Eval: RV (text embedding) ==="
for entry in "${TEXT_MODELS[@]}"; do
    IFS='|' read -r provider model_id model_slug has_native short_marker <<< "$entry"
    if [ "$has_native" = "true" ]; then
        transforms="$TRANSFORMS_ALL"
    else
        transforms="$TRANSFORMS_PCA_ONLY"
    fi
    for lang in $LANGS; do
        for transform in $transforms; do
            t_markers=()
            for t in $(_transform_markers "$transform"); do t_markers+=(-m "$t"); done
            for dim in $DIMS; do
                for suffix in "" "_addChars"; do
                    local_markers=(-m text_embedding -m rv -m eval -m "$provider" -m "$short_marker" -m "$lang" "${t_markers[@]}" -m "d${dim}")
                    [ -n "$suffix" ] && local_markers+=(-m addchars)
                    cfg="configs/text_embedding/eval/RV/d${dim}/${model_slug}_${lang}_${transform}_d${dim}${suffix}.json"
                    job "${local_markers[@]}" -m "$cfg" \
                        "RV ${short_marker} ${lang} ${transform} d${dim}${suffix}" \
                        uv run python -m asset_embeddings.scripts.eval.rv \
                            --config "$cfg" \
                            --log "logs/text_embedding/eval/RV/${model_slug}_${lang}_${transform}${suffix}.log" \
                            -r "$DB" -e "$EXP"
                done
            done
        done
    done
done

# ========== RC ==========
echo "=== Eval: RC (text embedding) ==="
for entry in "${TEXT_MODELS[@]}"; do
    IFS='|' read -r provider model_id model_slug has_native short_marker <<< "$entry"
    if [ "$has_native" = "true" ]; then
        transforms="$TRANSFORMS_ALL"
    else
        transforms="$TRANSFORMS_PCA_ONLY"
    fi
    for lang in $LANGS; do
        for transform in $transforms; do
            t_markers=()
            for t in $(_transform_markers "$transform"); do t_markers+=(-m "$t"); done
            for dim in $DIMS; do
                for suffix in "" "_addChars"; do
                    local_markers=(-m text_embedding -m rc -m eval -m "$provider" -m "$short_marker" -m "$lang" "${t_markers[@]}" -m "d${dim}")
                    [ -n "$suffix" ] && local_markers+=(-m addchars)
                    cfg="configs/text_embedding/eval/RC/d${dim}/${model_slug}_${lang}_${transform}_d${dim}${suffix}.json"
                    job "${local_markers[@]}" -m "$cfg" \
                        "RC ${short_marker} ${lang} ${transform} d${dim}${suffix}" \
                        uv run python -m asset_embeddings.scripts.eval.rc \
                            --config "$cfg" \
                            --log "logs/text_embedding/eval/RC/${model_slug}_${lang}_${transform}${suffix}.log" \
                            -r "$DB" -e "$EXP"
                done
            done
        done
    done
done

# ========== MP via BERT head ==========
echo "=== Eval: MP (text embedding via BERT head) ==="
for entry in "${TEXT_MODELS[@]}"; do
    IFS='|' read -r provider model_id model_slug has_native short_marker <<< "$entry"
    if [ "$has_native" = "true" ]; then
        transforms="$TRANSFORMS_ALL"
    else
        transforms="$TRANSFORMS_PCA_ONLY"
    fi
    for lang in $LANGS; do
        for transform in $transforms; do
            t_markers=()
            for t in $(_transform_markers "$transform"); do t_markers+=(-m "$t"); done
            for dim in $DIMS; do
                for quarter in $QUARTERS; do
                    cfg="configs/text_embedding/eval/MP/BERT/${model_slug}/${lang}/${transform}/d${dim}/${quarter}.json"
                    job -m text_embedding -m mp -m eval -m bert_head -m "$cfg" \
                        -m "$provider" -m "$short_marker" -m "$lang" "${t_markers[@]}" \
                        -m "d${dim}" -m "${quarter,,}" \
                        "MP ${short_marker} ${lang} ${transform} d${dim} ${quarter}" \
                        uv run python -m asset_embeddings.scripts.eval.mp_bert \
                            --config "$cfg" \
                            --log "checkpoints/refine/text_embedding/BERT/${model_slug}/${lang}/${transform}/d${dim}/${quarter}/refine.log" \
                                  "logs/text_embedding/eval/MP/BERT/${model_slug}/${lang}/${transform}/d${dim}/${quarter}.log" \
                            -r "$DB" -e "$EXP"
                done
            done
        done
    done
done

print_summary
