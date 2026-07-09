#!/bin/bash
# Wrapper: run any experiment script with email notification on completion/failure.
#
# Email tags are determined automatically by exit code:
#   exit 0  → [SUCCESS]  All commands passed.
#   exit 1  → [PARTIAL]  Ran to completion, but some commands failed.
#   exit 2  → [FAILED]   Aborted early by -x or -M (max-errors reached).
#
# Default behavior is continue-on-error (all commands run even if some fail).
# Use -x to stop on first failure instead.
#
# Reads NOTIFY_EMAIL from .env (default: ccarzit@gmail.com).

set -uo pipefail
# Note: intentionally no -e; we capture the exit code ourselves.

usage() {
    cat <<'EOF'
Usage: run_notify.sh [OPTIONS] <script.sh> [SCRIPT_ARGS...]

Run an experiment script with email notification on completion/failure.

Arguments:
  <script.sh>         Path to the experiment script to run
  [SCRIPT_ARGS...]    Additional arguments passed to the script

Options:
  -x                       Stop on first failure (alias for -M 1)
  -k, --filter <markers>   Include filter: comma-separated AND group (repeatable for OR).
                             e.g., -k bert,pretrain -k w2v,pretrain
                             → run commands tagged (bert AND pretrain) OR (w2v AND pretrain)
  -e, --exclude <markers>  Exclude filter: comma-separated tags to skip (repeatable).
                             e.g., -e rs_binary,rs_ranks
                             → skip commands tagged rs_binary or rs_ranks
  -v, --verbose            Verbose: show full command lines
  -q, --quiet              Quiet: only show PASS/FAIL/SKIP one-liners
  -a, --attach <file>      Attach a file to the notification email (repeatable)
  -T, --throttle <seconds> Min interval between per-error emails in
                             run_helper.sh (0=no throttle, default: 0)
  -M, --max-errors <N>     Abort batch after N errors in run_helper.sh
                             (0=unlimited, default: 0)
  -h, --help               Show this help message and exit

Email tags (automatic, based on exit code):
  [SUCCESS]   exit 0  — all commands passed
  [PARTIAL]   exit 1  — ran to completion, some commands failed
  [FAILED]    exit 2  — aborted early by -x / -M (max-errors reached)

Environment:
  NOTIFY_EMAIL        Recipient email (reads from .env, default: ccarzit@gmail.com)
  AE_ERROR_THROTTLE   Same as --throttle (CLI flag takes precedence)
  AE_MAX_ERRORS       Same as --max-errors (CLI flag takes precedence)
  AE_VERBOSE          Same as -v/-q (0=quiet, 1=normal, 2=verbose)
  AE_FILTER_INCLUDE   Same as -k (|-separated OR groups, ,-separated AND)
  AE_FILTER_EXCLUDE   Same as --exclude (,-separated)

Examples:
  # Run full pipeline (continue on error → [PARTIAL] if any fail)
  bash scripts/run/run_notify.sh scripts/run/train_main.sh

  # Stop on first failure → [FAILED]
  bash scripts/run/run_notify.sh -x scripts/run/train_main.sh

  # Only BERT pretrain tasks
  bash scripts/run/run_notify.sh -k bert,pretrain scripts/run/train_main.sh

  # BERT or W2V finetune, stop on first failure
  bash scripts/run/run_notify.sh -x -k bert,finetune -k w2v,finetune scripts/run/train_main.sh

  # Everything except RS variants
  bash scripts/run/run_notify.sh --exclude rs_binary,rs_ranks,rs_level0,rs_levelmin scripts/run/train_main.sh

  # Verbose mode with throttled errors, abort after 5
  bash scripts/run/run_notify.sh -v -T 300 -M 5 scripts/run/train_main.sh

  # Attach result DB to notification
  bash scripts/run/run_notify.sh -a results/db/main.db scripts/run/train_main.sh
EOF
}

# Parse flags
EXTRA_ATTACHMENTS=()
CLI_THROTTLE=""
CLI_MAX_ERRORS=""
CLI_VERBOSE=""
INCLUDE_GROUPS=()
EXCLUDE_TAGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help)
            usage
            exit 0
            ;;
        -x)
            CLI_MAX_ERRORS=1
            shift
            ;;
        -v|--verbose)
            CLI_VERBOSE=2
            shift
            ;;
        -q|--quiet)
            CLI_VERBOSE=0
            shift
            ;;
        -k|--filter)
            if [[ $# -lt 2 ]]; then
                echo "Error: -k/--filter requires a markers argument (e.g., -k bert,pretrain)" >&2
                exit 1
            fi
            INCLUDE_GROUPS+=("${2,,}")  # lowercase
            shift 2
            ;;
        -e|--exclude)
            if [[ $# -lt 2 ]]; then
                echo "Error: -e/--exclude requires a markers argument" >&2
                exit 1
            fi
            EXCLUDE_TAGS+=("${2,,}")  # lowercase
            shift 2
            ;;
        -a|--attach)
            if [[ $# -lt 2 ]]; then
                echo "Error: -a requires a file path argument" >&2
                exit 1
            fi
            EXTRA_ATTACHMENTS+=("$2")
            shift 2
            ;;
        -T|--throttle)
            if [[ $# -lt 2 ]]; then
                echo "Error: -T requires a number (seconds)" >&2
                exit 1
            fi
            CLI_THROTTLE="$2"
            shift 2
            ;;
        -M|--max-errors)
            if [[ $# -lt 2 ]]; then
                echo "Error: -M requires a number" >&2
                exit 1
            fi
            CLI_MAX_ERRORS="$2"
            shift 2
            ;;
        -*)
            echo "Unknown flag: $1" >&2
            echo "Run 'run_notify.sh --help' for usage." >&2
            exit 1
            ;;
        *)
            break
            ;;
    esac
done

SCRIPT="${1:?Error: missing <script.sh>. Run 'run_notify.sh --help' for usage.}"
shift
SCRIPT_ARGS=("$@")

# CLI flags take precedence over env vars
if [ -n "$CLI_THROTTLE" ]; then
    export AE_ERROR_THROTTLE="$CLI_THROTTLE"
fi
if [ -n "$CLI_MAX_ERRORS" ]; then
    export AE_MAX_ERRORS="$CLI_MAX_ERRORS"
fi
if [ -n "$CLI_VERBOSE" ]; then
    export AE_VERBOSE="$CLI_VERBOSE"
fi

# Build filter env vars from -k and --exclude flags
if [ ${#INCLUDE_GROUPS[@]} -gt 0 ]; then
    # Join groups with | (OR separator)
    _include_val=""
    for _g in "${INCLUDE_GROUPS[@]}"; do
        if [ -n "$_include_val" ]; then
            _include_val="${_include_val}|${_g}"
        else
            _include_val="$_g"
        fi
    done
    export AE_FILTER_INCLUDE="$_include_val"
fi

if [ ${#EXCLUDE_TAGS[@]} -gt 0 ]; then
    # Join all exclude tags with comma
    _exclude_val=""
    for _e in "${EXCLUDE_TAGS[@]}"; do
        if [ -n "$_exclude_val" ]; then
            _exclude_val="${_exclude_val},${_e}"
        else
            _exclude_val="$_e"
        fi
    done
    export AE_FILTER_EXCLUDE="$_exclude_val"
fi

SCRIPT_NAME=$(basename "$SCRIPT" .sh)

# Read NOTIFY_EMAIL from .env if available
if [ -f .env ]; then
    NOTIFY_EMAIL=$(grep -E '^NOTIFY_EMAIL=' .env | cut -d= -f2- | tr -d ' "'"'"'')
fi
NOTIFY_EMAIL="${NOTIFY_EMAIL:-ccarzit@gmail.com}"

# Create temp files
MARKER=$(mktemp)                    # to find error logs created during this run
export AE_SUMMARY_FILE=$(mktemp)    # print_summary writes structured data here

echo "=== run_notify: ${SCRIPT_NAME} ==="
echo "Notify: ${NOTIFY_EMAIL}"
if [ -n "${AE_FILTER_INCLUDE:-}" ]; then
    echo "Filter include: ${AE_FILTER_INCLUDE}"
fi
if [ -n "${AE_FILTER_EXCLUDE:-}" ]; then
    echo "Filter exclude: ${AE_FILTER_EXCLUDE}"
fi

# Run the experiment
START=$(date +%s)
bash "$SCRIPT" "${SCRIPT_ARGS[@]+"${SCRIPT_ARGS[@]}"}"
EXIT_CODE=$?
END=$(date +%s)

ELAPSED=$((END - START))
HOURS=$((ELAPSED / 3600))
MINUTES=$(( (ELAPSED % 3600) / 60 ))
SECONDS=$((ELAPSED % 60))
DURATION="${HOURS}h ${MINUTES}m ${SECONDS}s"

# --- Read summary data from print_summary ---
PASS_COUNT=0; FAIL_COUNT=0; SKIP_COUNT=0; TOTAL_COUNT=0
SUMMARY_LINE=""; FAILED_LIST=""
if [ -s "$AE_SUMMARY_FILE" ]; then
    # Single-line fields via grep
    PASS_COUNT=$(grep '^PASS_COUNT=' "$AE_SUMMARY_FILE" | cut -d= -f2-)
    FAIL_COUNT=$(grep '^FAIL_COUNT=' "$AE_SUMMARY_FILE" | cut -d= -f2-)
    SKIP_COUNT=$(grep '^SKIP_COUNT=' "$AE_SUMMARY_FILE" | cut -d= -f2-)
    TOTAL_COUNT=$(grep '^TOTAL_COUNT=' "$AE_SUMMARY_FILE" | cut -d= -f2-)
    SUMMARY_LINE=$(grep '^SUMMARY_LINE=' "$AE_SUMMARY_FILE" | cut -d= -f2-)
    # Multi-line: everything after the FAILED_LIST= marker
    FAILED_LIST=$(sed -n '/^FAILED_LIST=/{ s/^FAILED_LIST=//; p; :a; n; p; ba }' "$AE_SUMMARY_FILE")
    # Defaults for missing values
    PASS_COUNT="${PASS_COUNT:-0}"
    FAIL_COUNT="${FAIL_COUNT:-0}"
    SKIP_COUNT="${SKIP_COUNT:-0}"
    TOTAL_COUNT="${TOTAL_COUNT:-0}"
fi
rm -f "$AE_SUMMARY_FILE"

# --- Determine email tag from exit code ---
#   0 → SUCCESS   (all passed)
#   2 → FAILED    (aborted by -x / -M max-errors)
#   * → PARTIAL   (ran to completion, some failures)
if [ $EXIT_CODE -eq 0 ]; then
    TAG="SUCCESS"
elif [ $EXIT_CODE -eq 2 ]; then
    TAG="FAILED"
else
    TAG="PARTIAL"
fi

# --- Build subject line ---
# [SUCCESS] train — 266 passed (2h 15m 30s)
# [PARTIAL] train — 260 passed, 6 failed (2h 15m 30s)
# [FAILED]  train — 3 passed, 1 failed (aborted) (5m 30s)
COUNTS="${PASS_COUNT} passed"
if [ "$FAIL_COUNT" -gt 0 ]; then
    COUNTS="${COUNTS}, ${FAIL_COUNT} failed"
fi
ABORT_TAG=""
if [ "$TAG" = "FAILED" ]; then
    ABORT_TAG=" (aborted)"
fi
SUBJECT="[${TAG}] ${SCRIPT_NAME} — ${COUNTS}${ABORT_TAG} (${DURATION})"

# --- Build body ---
BODY="Experiment: ${SCRIPT_NAME}
Status:     ${TAG}
Duration:   ${DURATION}"

# Append run config lines (only when non-default)
if [ -n "${AE_FILTER_INCLUDE:-}" ]; then
    BODY="${BODY}
Filter:     include=[${AE_FILTER_INCLUDE}]"
fi
if [ -n "${AE_FILTER_EXCLUDE:-}" ]; then
    BODY="${BODY}
Filter:     exclude=[${AE_FILTER_EXCLUDE}]"
fi
if [ -n "${AE_MAX_ERRORS:-}" ] && [ "${AE_MAX_ERRORS}" != "0" ]; then
    BODY="${BODY}
Max errors: ${AE_MAX_ERRORS}"
fi

BODY="${BODY}

${SUMMARY_LINE}"

if [ "$FAIL_COUNT" -gt 0 ] && [ -n "$FAILED_LIST" ]; then
    BODY="${BODY}

Failed commands:
${FAILED_LIST}"
fi

# --- Collect attachments ---
ATTACH_ARGS=()
_ZIP_FILE=""

# Collect error logs created during this run
_ERR_LOGS=()
if [ -d logs/error ]; then
    while IFS= read -r err_log; do
        _ERR_LOGS+=("$err_log")
    done < <(find logs/error -name "*.log" -newer "$MARKER" 2>/dev/null)
fi
rm -f "$MARKER"

# Zip error logs if multiple, attach single file directly
if [ ${#_ERR_LOGS[@]} -eq 1 ]; then
    ATTACH_ARGS+=("-a" "${_ERR_LOGS[0]}")
elif [ ${#_ERR_LOGS[@]} -gt 1 ]; then
    _ZIP_FILE="$(mktemp).zip"
    zip -j -q "$_ZIP_FILE" "${_ERR_LOGS[@]}" 2>/dev/null
    if [ $? -eq 0 ]; then
        ATTACH_ARGS+=("-a" "$_ZIP_FILE")
        echo "Zipped ${#_ERR_LOGS[@]} error logs into $(basename "$_ZIP_FILE")"
    else
        echo "[Warning] zip failed, attaching files individually"
        for el in "${_ERR_LOGS[@]}"; do
            ATTACH_ARGS+=("-a" "$el")
        done
        _ZIP_FILE=""
    fi
fi

# Attach user-specified files
for f in "${EXTRA_ATTACHMENTS[@]+"${EXTRA_ATTACHMENTS[@]}"}"; do
    if [ -f "$f" ]; then
        ATTACH_ARGS+=("-a" "$f")
    else
        echo "[Warning] Attachment not found: $f"
    fi
done

# --- Send notification ---
echo "--- Sending notification to ${NOTIFY_EMAIL} ---"
uv run python -m scripts.tools.smtp -t "$NOTIFY_EMAIL" -s "$SUBJECT" -b "$BODY" "${ATTACH_ARGS[@]+"${ATTACH_ARGS[@]}"}" || \
    echo "[Warning] Email notification failed (SMTP error)"

# Cleanup
[ -n "$_ZIP_FILE" ] && rm -f "$_ZIP_FILE"

exit $EXIT_CODE
