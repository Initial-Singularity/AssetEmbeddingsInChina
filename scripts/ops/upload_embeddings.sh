#!/bin/bash
# upload_embeddings.sh — Upload collected embedding CSVs to Aliyun Drive.
#
# Uploads embeddings organized by experiment/model to the specified remote
# directory using aliyunpan CLI.  Uses --skip by default for incremental
# uploads (skips existing same-name files).
#
# Expected local structure:
#   {input}/{experiment}/{Model}/d{dim}/{quarter}/*_embedding.csv
#
# Remote structure (mirrors local):
#   {remote}/{experiment}/{Model}/d{dim}/{quarter}/*_embedding.csv
#
# Usage:
#   # Upload everything
#   bash scripts/ops/upload_embeddings.sh -i data/embeddings -r /AssetEmbedding_latest
#
#   # Upload a single experiment
#   bash scripts/ops/upload_embeddings.sh -i data/embeddings -r /AssetEmbedding_latest -e main
#
#   # Upload a single experiment + model
#   bash scripts/ops/upload_embeddings.sh -i data/embeddings -r /AssetEmbedding_latest -e main -m AssetBERT
#
#   # Overwrite existing files
#   bash scripts/ops/upload_embeddings.sh -i data/embeddings -r /AssetEmbedding_latest --overwrite
#
#   # Dry run (preview only)
#   bash scripts/ops/upload_embeddings.sh -i data/embeddings -r /AssetEmbedding_latest --dry-run
#
# Collect + Upload workflow:
#   # 1. Collect embeddings from checkpoints (experiment-first)
#   bash scripts/collect_embeddings.sh -i checkpoints/finetune -o data/embeddings/main
#   bash scripts/collect_embeddings.sh -i checkpoints/timesplit_robust/finetune -o data/embeddings/timesplit_robust
#   bash scripts/collect_embeddings.sh -i checkpoints/semiannual_robust/finetune -o data/embeddings/semiannual_robust
#
#   # 2. Upload all collected embeddings
#   bash scripts/ops/upload_embeddings.sh -i data/embeddings -r /AssetEmbedding_latest

set -euo pipefail

# ── Defaults ──────────────────────────────────────────────────────────────────
INPUT_DIR=""
REMOTE_DIR=""
EXPERIMENT=""
MODEL=""
DRY_RUN=false
OVERWRITE=false
PARALLEL=3

# ── Counters ──────────────────────────────────────────────────────────────────
_UPLOADED=0
_SKIPPED=0

# ── Argument parsing ─────────────────────────────────────────────────────────
usage() {
    cat <<'EOF'
Usage: upload_embeddings.sh -i <input_dir> -r <remote_dir> [options]

Required:
  -i, --input       Local embeddings root (e.g. data/embeddings)
  -r, --remote      Remote target directory (e.g. /AssetEmbedding_latest)

Filters:
  -e, --experiment  Upload only this experiment (e.g. main, timesplit_robust)
  -m, --model       Upload only this model (e.g. AssetBERT, AssetRS)

Options:
  -p, --parallel    Parallel upload count (default: 3)
  --overwrite       Overwrite existing files (moved to trash on cloud)
  --skip            Skip existing same-name files (default behavior)
  --dry-run         Preview operations without uploading
  -h, --help        Show this help

Examples:
  bash scripts/ops/upload_embeddings.sh -i data/embeddings -r /AssetEmbedding_latest
  bash scripts/ops/upload_embeddings.sh -i data/embeddings -r /AssetEmbedding_latest -e main
  bash scripts/ops/upload_embeddings.sh -i data/embeddings -r /AssetEmbedding_latest -e main -m AssetBERT
  bash scripts/ops/upload_embeddings.sh -i data/embeddings -r /AssetEmbedding_latest --dry-run
EOF
    exit "${1:-0}"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        -i|--input)      INPUT_DIR="$2";    shift 2 ;;
        -r|--remote)     REMOTE_DIR="$2";   shift 2 ;;
        -e|--experiment) EXPERIMENT="$2";    shift 2 ;;
        -m|--model)      MODEL="$2";        shift 2 ;;
        -p|--parallel)   PARALLEL="$2";     shift 2 ;;
        --overwrite)     OVERWRITE=true;     shift ;;
        --skip)          shift ;;  # default, no-op
        --dry-run)       DRY_RUN=true;       shift ;;
        -h|--help)       usage 0 ;;
        *)               echo "Unknown option: $1" >&2; usage 1 ;;
    esac
done

if [[ -z "$INPUT_DIR" || -z "$REMOTE_DIR" ]]; then
    echo "Error: -i and -r are required." >&2
    usage 1
fi

if [[ ! -d "$INPUT_DIR" ]]; then
    echo "Error: Input directory does not exist: $INPUT_DIR" >&2
    exit 1
fi

if ! command -v aliyunpan &>/dev/null; then
    echo "Error: aliyunpan not found. Install from https://github.com/tickstep/aliyunpan" >&2
    exit 1
fi

# Normalize
INPUT_DIR="${INPUT_DIR%/}"
REMOTE_DIR="${REMOTE_DIR%/}"

# ── Build upload flags ───────────────────────────────────────────────────────
UPLOAD_FLAGS="-p $PARALLEL"
if $OVERWRITE; then
    UPLOAD_FLAGS="$UPLOAD_FLAGS --ow"
else
    UPLOAD_FLAGS="$UPLOAD_FLAGS --skip"
fi

# ── Main ─────────────────────────────────────────────────────────────────────
echo "Source:  $INPUT_DIR"
echo "Remote:  $REMOTE_DIR"
[[ -n "$EXPERIMENT" ]] && echo "Filter:  experiment=$EXPERIMENT"
[[ -n "$MODEL" ]]      && echo "Filter:  model=$MODEL"
$DRY_RUN && echo "(dry-run mode)"
echo ""

for exp_dir in "$INPUT_DIR"/*/; do
    [[ -d "$exp_dir" ]] || continue
    exp="$(basename "$exp_dir")"

    # Filter by experiment
    if [[ -n "$EXPERIMENT" && "$exp" != "$EXPERIMENT" ]]; then
        continue
    fi

    for model_dir in "$exp_dir"/*/; do
        [[ -d "$model_dir" ]] || continue
        model="$(basename "$model_dir")"

        # Filter by model
        if [[ -n "$MODEL" && "$model" != "$MODEL" ]]; then
            continue
        fi

        remote_path="$REMOTE_DIR/$exp"

        if $DRY_RUN; then
            echo "  [dry-run] aliyunpan upload $UPLOAD_FLAGS $model_dir $remote_path"
            _SKIPPED=$((_SKIPPED + 1))
        else
            echo "  UPLOAD  $exp/$model -> $remote_path/$model"
            # aliyunpan upload uploads the directory INTO the target,
            # so model_dir -> remote_path creates remote_path/Model/d{dim}/...
            aliyunpan upload $UPLOAD_FLAGS "$model_dir" "$remote_path"
            _UPLOADED=$((_UPLOADED + 1))
        fi
    done
done

# ── Summary ──────────────────────────────────────────────────────────────────
echo ""
echo "======================================"
if $DRY_RUN; then
    echo "Previewed: $_SKIPPED model(s)  (dry-run — nothing uploaded)"
else
    echo "Uploaded: $_UPLOADED model(s)"
fi
echo "======================================"
