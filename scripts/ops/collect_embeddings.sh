#!/bin/bash
# collect_embeddings.sh — Collect embedding CSVs from checkpoint finetune directories.
#
# Scans the checkpoint tree for *_embedding.csv files and copies them into a
# clean output directory, preserving the relative directory structure.
# Epoch-specific embeddings (*_epoch*_embedding.csv) are skipped by default.
#
# Directory structure (input):
#   {input}/{ModelType}/d{dim}/{quarter}/*_embedding.csv
#   {input}/{ModelType}/{Variant}/d{dim}/{quarter}/*_embedding.csv   (e.g. AssetRS)
#
# Output mirrors the relative structure:
#   {output}/{ModelType}/d{dim}/{quarter}/*_embedding.csv
#   {output}/{ModelType}/{Variant}/d{dim}/{quarter}/*_embedding.csv
#
# Usage:
#   bash scripts/collect_embeddings.sh -i checkpoints/finetune -o data/embeddings
#   bash scripts/collect_embeddings.sh -i checkpoints/finetune -o data/embeddings --dry-run
#   bash scripts/collect_embeddings.sh -i checkpoints/finetune -o data/embeddings --overwrite

set -euo pipefail

# ── Defaults ──────────────────────────────────────────────────────────────────
INPUT_DIR=""
OUTPUT_DIR=""
DRY_RUN=false
OVERWRITE=false

# ── Counters ──────────────────────────────────────────────────────────────────
_COPIED=0
_SKIPPED=0
_WARNED=0
_TOTAL=0

# ── Validation regex ──────────────────────────────────────────────────────────
# Expected filename: Asset*_d{N}_*_embedding.csv  (not epoch)
FILENAME_RE='^Asset.+_d[0-9]+_.+_embedding\.csv$'
# Expected path must contain a d{N}/ directory and a quarter-like directory
DIM_DIR_RE='/d[0-9]+/'
QUARTER_DIR_RE='/[0-9]{4}Q[1-4]/'

# ── Argument parsing ─────────────────────────────────────────────────────────
usage() {
    cat <<'EOF'
Usage: collect_embeddings.sh -i <input_dir> -o <output_dir> [options]

Options:
  -i, --input       Source directory (checkpoint finetune root)
  -o, --output      Output directory (e.g. data/embeddings)
  --dry-run         Preview operations without copying
  --overwrite       Overwrite existing files in output
  -h, --help        Show this help

Examples:
  bash scripts/collect_embeddings.sh -i checkpoints/finetune -o data/embeddings
  bash scripts/collect_embeddings.sh -i checkpoints/finetune -o data/embeddings --dry-run
EOF
    exit "${1:-0}"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        -i|--input)    INPUT_DIR="$2";  shift 2 ;;
        -o|--output)   OUTPUT_DIR="$2"; shift 2 ;;
        --dry-run)     DRY_RUN=true;    shift ;;
        --overwrite)   OVERWRITE=true;  shift ;;
        -h|--help)     usage 0 ;;
        *)             echo "Unknown option: $1" >&2; usage 1 ;;
    esac
done

if [[ -z "$INPUT_DIR" || -z "$OUTPUT_DIR" ]]; then
    echo "Error: -i and -o are required." >&2
    usage 1
fi

if [[ ! -d "$INPUT_DIR" ]]; then
    echo "Error: Input directory does not exist: $INPUT_DIR" >&2
    exit 1
fi

# Normalize to remove trailing slash
INPUT_DIR="${INPUT_DIR%/}"
OUTPUT_DIR="${OUTPUT_DIR%/}"

# ── Main ─────────────────────────────────────────────────────────────────────
echo "Source:  $INPUT_DIR"
echo "Output:  $OUTPUT_DIR"
$DRY_RUN && echo "(dry-run mode)"
echo ""

while IFS= read -r src; do
    _TOTAL=$((_TOTAL + 1))
    filename="$(basename "$src")"

    # Skip epoch-specific embeddings
    if [[ "$filename" == *_epoch*_embedding.csv ]]; then
        _SKIPPED=$((_SKIPPED + 1))
        continue
    fi

    # Compute relative path from input root
    rel="${src#"$INPUT_DIR"/}"
    dest="$OUTPUT_DIR/$rel"

    # ── Validation ────────────────────────────────────────────────────────
    valid=true

    if ! [[ "$filename" =~ $FILENAME_RE ]]; then
        echo "  WARN  unexpected filename: $rel"
        valid=false
    fi

    if ! [[ "/$rel" =~ $DIM_DIR_RE ]]; then
        echo "  WARN  missing d{N} directory: $rel"
        valid=false
    fi

    if ! [[ "/$rel" =~ $QUARTER_DIR_RE ]]; then
        echo "  WARN  missing quarter directory: $rel"
        valid=false
    fi

    if ! $valid; then
        _WARNED=$((_WARNED + 1))
        # Still copy, but warn — the user can review
    fi

    # ── Copy ──────────────────────────────────────────────────────────────
    if [[ -f "$dest" ]] && ! $OVERWRITE; then
        echo "  SKIP  $rel  (exists)"
        _SKIPPED=$((_SKIPPED + 1))
        continue
    fi

    if $DRY_RUN; then
        echo "  COPY  $rel"
    else
        mkdir -p "$(dirname "$dest")"
        cp "$src" "$dest"
        echo "  COPY  $rel"
    fi
    _COPIED=$((_COPIED + 1))

done < <(find "$INPUT_DIR" -name '*_embedding.csv' -type f | sort)

# ── Summary ──────────────────────────────────────────────────────────────────
echo ""
echo "======================================"
echo "Collected: $_COPIED  Skipped: $_SKIPPED  Warnings: $_WARNED  (Total scanned: $_TOTAL)"
$DRY_RUN && echo "(dry-run — no files were actually copied)"
echo "======================================"
