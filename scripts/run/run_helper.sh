#!/bin/bash
# run_helper.sh — Error-tolerant run helper for exploration batches.
#
# Source this file from exploration scripts. Provides:
#   job [-m tag]... "description" command args...
#   print_summary
#
# Marker-based filtering (set via run_notify.sh -k / --exclude / --filter-file / --exclude-file):
#   job -m bert -m pretrain -m d64 -m "$cfg" "BERT pretrain d64" uv run python ... --config "$cfg"
#   Each -m adds a tag for filtering; by convention the last tag is the config path exactly as passed to
#   --config (repo-relative, forward slashes), which is the key shared with runs.config_file. Inputs:
#     AE_FILTER_INCLUDE="bert,pretrain|w2v"  — |-separated OR groups, comma-separated AND terms
#     AE_FILTER_EXCLUDE="rs_binary,rs_ranks" — comma-separated, any match → skip (checked before include)
#     AE_FILTER_INCLUDE_FILE=path            — one AND-group per line (comma-separated); lines are OR-ed
#     AE_FILTER_EXCLUDE_FILE=path            — one term per line
#   Blank lines and lines starting with # are ignored. Terms and tags are lowercased. A term containing
#   * or ? is a bash glob matched against each tag ("configs/main/finetune/*"); every other term is an
#   exact set lookup, so lists of thousands of config paths cost nothing. Filters are parsed once when
#   this file is sourced. Commands without -m flags are always executed (untagged = unconditional).
#   Requires bash >= 4.3.
#
# Error email throttling (set via AE_ERROR_THROTTLE env var or run_notify.sh --throttle):
#   When throttle > 0, per-error emails are buffered and sent in batches no more
#   frequently than every <throttle> seconds. Remaining buffered errors are flushed
#   in print_summary.
#
# Max errors (set via AE_MAX_ERRORS env var or run_notify.sh --max-errors):
#   When max_errors > 0, job calls print_summary and exits the entire script
#   after the Nth failure, aborting the batch.
#
# Verbosity (set via AE_VERBOSE env var or run_notify.sh -v / -q):
#   0 = quiet: only PASS/FAIL/SKIP one-liners
#   1 = normal (default): description + PASS/FAIL/SKIP
#   2 = verbose: description + full command + PASS/FAIL/SKIP
#
# Usage:
#   source scripts/run/run_helper.sh
#   cfg="configs/main/pretrained/AssetBERT/AssetBERT_d64_pretrained.json"
#   job -m bert -m pretrain -m d64 -m "$cfg" "BERT pretrain d64" uv run python -m asset_embeddings.scripts.train.bert --config "$cfg"
#   cfg="configs/main/finetune/AssetW2V/d32/AssetW2V_d32_2023Q2.json"
#   job -m w2v  -m finetune -m d32 -m "$cfg" "W2V finetune d32"  uv run python -m asset_embeddings.scripts.train.w2v --config "$cfg"
#   print_summary

# --- Internal state ---
_RH_PASS_COUNT=0
_RH_FAIL_COUNT=0
_RH_SKIP_COUNT=0
_RH_TOTAL_COUNT=0
_RH_FAILED_COMMANDS=""

# Config (from env, set by run_notify.sh)
_RH_THROTTLE="${AE_ERROR_THROTTLE:-0}"
_RH_MAX_ERRORS="${AE_MAX_ERRORS:-0}"
_RH_VERBOSE="${AE_VERBOSE:-1}"

# Throttle state: buffer file and last-sent timestamp
_RH_ERROR_BUFFER=$(mktemp)
_RH_ATTACH_BUFFER=$(mktemp)
# Initialize to current time so the first error also respects the throttle window
_RH_LAST_SENT=$(date +%s)
_RH_BUFFERED_COUNT=0
_RH_ZIP_FILE=""

# Recipient: NOTIFY_EMAIL from the environment, else from .env (read once at source time); no default,
# and no failure mail is sent when neither is set
if [ -f .env ]; then
    _RH_NOTIFY_EMAIL=$(grep -E '^NOTIFY_EMAIL=' .env | cut -d= -f2- | tr -d ' "'"'"'')
fi
_RH_NOTIFY_EMAIL="${NOTIFY_EMAIL:-${_RH_NOTIFY_EMAIL:-}}"

# ---------------------------------------------------------------------------
# Filters (parsed ONCE at source time; see the header for the four inputs and the file format).
# ---------------------------------------------------------------------------
if (( BASH_VERSINFO[0] < 4 || (BASH_VERSINFO[0] == 4 && BASH_VERSINFO[1] < 3) )); then
    echo "run_helper.sh: bash >= 4.3 required (associative arrays, [[ -v ]]); found ${BASH_VERSION}" >&2
    exit 1
fi
declare -A _RH_EX_EXACT=()     # exact exclude terms → 1
_RH_EX_GLOB=()                 # exclude terms containing * or ?
_RH_IN_GROUPS=()               # include AND-groups, each a comma-joined string
_RH_FILTER_ACTIVE=0            # 1 once any term was loaded

_rh_trim() {  # sets _RH_T to $1 without leading/trailing whitespace (no subshell: forks are slow on Windows)
    _RH_T="$1"
    _RH_T="${_RH_T#"${_RH_T%%[![:space:]]*}"}"
    _RH_T="${_RH_T%"${_RH_T##*[![:space:]]}"}"
}
_rh_is_glob() { [[ "$1" == *[\*\?]* ]]; }

_rh_add_exclude() {  # _rh_add_exclude <term>
    _rh_trim "${1,,}"; local t="$_RH_T"
    [ -z "$t" ] && return 0
    if _rh_is_glob "$t"; then _RH_EX_GLOB+=("$t"); else _RH_EX_EXACT["$t"]=1; fi
    _RH_FILTER_ACTIVE=1
}
_rh_add_include_group() {  # _rh_add_include_group "a,b"
    _rh_trim "${1,,}"; local g="$_RH_T"
    [ -z "$g" ] && return 0
    _RH_IN_GROUPS+=("$g")
    _RH_FILTER_ACTIVE=1
}
_rh_read_filter_file() {  # _rh_read_filter_file <path> <adder>; skips blank and # lines
    local path="$1" adder="$2" line
    if [ ! -r "$path" ]; then
        echo "run_helper.sh: cannot read filter file '$path' (AE_FILTER_INCLUDE_FILE / AE_FILTER_EXCLUDE_FILE)" >&2
        exit 1
    fi
    while IFS= read -r line || [ -n "$line" ]; do
        _rh_trim "$line"; line="$_RH_T"
        [ -z "$line" ] && continue
        [[ "$line" == \#* ]] && continue
        "$adder" "$line"
    done < "$path"
}
_rh_load_filters() {
    local item
    if [ -n "${AE_FILTER_EXCLUDE:-}" ]; then
        IFS=',' read -ra _rh_items <<< "$AE_FILTER_EXCLUDE"
        for item in "${_rh_items[@]}"; do _rh_add_exclude "$item"; done
    fi
    [ -n "${AE_FILTER_EXCLUDE_FILE:-}" ] && _rh_read_filter_file "$AE_FILTER_EXCLUDE_FILE" _rh_add_exclude
    if [ -n "${AE_FILTER_INCLUDE:-}" ]; then
        IFS='|' read -ra _rh_items <<< "$AE_FILTER_INCLUDE"
        for item in "${_rh_items[@]}"; do _rh_add_include_group "$item"; done
    fi
    [ -n "${AE_FILTER_INCLUDE_FILE:-}" ] && _rh_read_filter_file "$AE_FILTER_INCLUDE_FILE" _rh_add_include_group
    return 0
}
_rh_load_filters

# _rh_match_filter tag...  → 0 = execute, 1 = skip
#   untagged command → execute; no term loaded → execute; exclude checked before include;
#   include: some OR-group has all its terms present (exact: set lookup; glob: pattern over the tags).
_rh_match_filter() {
    local markers=("$@")
    [ ${#markers[@]} -eq 0 ] && return 0
    [ "$_RH_FILTER_ACTIVE" -eq 0 ] && return 0
    local m p
    local -A have=()
    for m in "${markers[@]}"; do have["$m"]=1; done
    for m in "${markers[@]}"; do
        [[ -v _RH_EX_EXACT["$m"] ]] && return 1
    done
    for p in "${_RH_EX_GLOB[@]+"${_RH_EX_GLOB[@]}"}"; do
        for m in "${markers[@]}"; do [[ "$m" == $p ]] && return 1; done
    done
    [ ${#_RH_IN_GROUPS[@]} -eq 0 ] && return 0
    local group term ok found
    for group in "${_RH_IN_GROUPS[@]}"; do
        ok=1
        IFS=',' read -ra _rh_terms <<< "$group"
        for term in "${_rh_terms[@]}"; do
            _rh_trim "$term"; term="$_RH_T"
            [ -z "$term" ] && continue
            if _rh_is_glob "$term"; then
                found=0
                for m in "${markers[@]}"; do [[ "$m" == $term ]] && { found=1; break; }; done
                [ $found -eq 1 ] || { ok=0; break; }
            else
                [[ -v have["$term"] ]] || { ok=0; break; }
            fi
        done
        [ $ok -eq 1 ] && return 0
    done
    return 1
}

# _rh_flush_buffer — send all buffered errors as a single summary email, then clear buffer.
# Called when throttle window expires or from print_summary.
_rh_flush_buffer() {
    if [ "$_RH_BUFFERED_COUNT" -eq 0 ]; then
        return
    fi

    local subject body
    if [ "$_RH_BUFFERED_COUNT" -eq 1 ]; then
        subject="[ERROR] 1 failure buffered"
    else
        subject="[ERROR] ${_RH_BUFFERED_COUNT} failures buffered"
    fi

    body="$(cat "$_RH_ERROR_BUFFER")

Pass so far: ${_RH_PASS_COUNT}
Fail so far: ${_RH_FAIL_COUNT}"

    # Collect unique attachment paths and zip if multiple
    local attach_args=()
    _RH_ZIP_FILE=""
    if [ -s "$_RH_ATTACH_BUFFER" ]; then
        local att_files=()
        while IFS= read -r att; do
            if [ -f "$att" ]; then
                att_files+=("$att")
            fi
        done < <(sort -u "$_RH_ATTACH_BUFFER")

        if [ ${#att_files[@]} -eq 1 ]; then
            attach_args+=("-a" "${att_files[0]}")
        elif [ ${#att_files[@]} -gt 1 ]; then
            _RH_ZIP_FILE="$(mktemp).zip"
            zip -j -q "$_RH_ZIP_FILE" "${att_files[@]}" 2>/dev/null
            if [ $? -eq 0 ]; then
                attach_args+=("-a" "$_RH_ZIP_FILE")
            else
                # zip failed, fall back to individual attachments
                echo ">>> [Warning] zip failed, attaching files individually"
                for af in "${att_files[@]}"; do
                    attach_args+=("-a" "$af")
                done
                _RH_ZIP_FILE=""
            fi
        fi
    fi

    if [ -n "$_RH_NOTIFY_EMAIL" ]; then
        uv run python -m scripts.tools.smtp \
            -t "$_RH_NOTIFY_EMAIL" \
            -s "$subject" \
            -b "$body" \
            "${attach_args[@]+"${attach_args[@]}"}" 2>/dev/null || \
            echo ">>> [Warning] Buffered error email failed (SMTP error)"
    fi

    # Clean up zip temp file
    if [ -n "$_RH_ZIP_FILE" ]; then
        rm -f "$_RH_ZIP_FILE"
        _RH_ZIP_FILE=""
    fi

    # Clear buffer
    : > "$_RH_ERROR_BUFFER"
    : > "$_RH_ATTACH_BUFFER"
    _RH_BUFFERED_COUNT=0
    _RH_LAST_SENT=$(date +%s)
}

# _rh_cleanup — remove temp files on exit
_rh_cleanup() {
    rm -f "$_RH_ERROR_BUFFER" "$_RH_ATTACH_BUFFER"
    [ -n "$_RH_ZIP_FILE" ] && rm -f "$_RH_ZIP_FILE"
}
trap _rh_cleanup EXIT

# job [-m tag]... "description" command args...
#
# Runs the command. On success, increments pass count.
# On failure, increments fail count, collects error logs, and either sends
# an immediate email (throttle=0) or buffers for batched sending.
#
# Returns 0 normally (so calling script continues).
# Exits the script (exit 2) when max-errors is reached.
# Exit codes: 0=success, 1=ran to completion with failures, 2=aborted by max-errors.
job() {
    # --- Parse -m flags ---
    local markers=()
    while [[ "${1:-}" == "-m" ]]; do
        if [[ -z "${2:-}" || "${2}" == -* ]]; then
            echo "job: -m flag requires a non-empty tag argument" >&2
            return 1
        fi
        markers+=("${2,,}")  # lowercase
        shift 2
    done

    local desc="$1"
    shift

    _RH_TOTAL_COUNT=$((_RH_TOTAL_COUNT + 1))

    # --- Filter check ---
    if ! _rh_match_filter "${markers[@]+"${markers[@]}"}"; then
        _RH_SKIP_COUNT=$((_RH_SKIP_COUNT + 1))
        echo ">>> [SKIP] ${desc}"
        return 0
    fi

    # --- Print header (verbosity-dependent) ---
    if [ "$_RH_VERBOSE" -ge 2 ]; then
        echo ""
        echo ">>> [job] ${desc}"
        echo ">>> Command: $*"
    elif [ "$_RH_VERBOSE" -ge 1 ]; then
        echo ""
        echo ">>> [job] ${desc}"
    fi

    # Create a temp file to identify error logs from this command
    local ts_marker
    ts_marker=$(mktemp)

    # Run the command (don't exit on failure)
    "$@"
    local rc=$?

    if [ $rc -eq 0 ]; then
        _RH_PASS_COUNT=$((_RH_PASS_COUNT + 1))
        echo ">>> [PASS] ${desc}"
        rm -f "$ts_marker"
        return 0
    fi

    # --- Failure path ---
    _RH_FAIL_COUNT=$((_RH_FAIL_COUNT + 1))
    _RH_FAILED_COMMANDS="${_RH_FAILED_COMMANDS}\n  - ${desc} (exit ${rc})"
    echo ">>> [FAIL] ${desc} (exit code ${rc})"

    # Collect error logs created during this command
    local err_logs=()
    if [ -d logs/error ]; then
        while IFS= read -r err_log; do
            err_logs+=("$err_log")
        done < <(find logs/error -name "*.log" -newer "$ts_marker" 2>/dev/null)
    fi
    rm -f "$ts_marker"

    if [ "$_RH_THROTTLE" -le 0 ]; then
        # --- No throttle: send immediately (original behavior) ---
        local attach_args=()
        for el in "${err_logs[@]+"${err_logs[@]}"}"; do
            attach_args+=("-a" "$el")
        done

        local subject="[ERROR] ${desc} — exit code ${rc}"
        local body="Command failed: $*
Exit code: ${rc}
Description: ${desc}
Pass so far: ${_RH_PASS_COUNT}
Fail so far: ${_RH_FAIL_COUNT}"

        if [ -n "$_RH_NOTIFY_EMAIL" ]; then
            uv run python -m scripts.tools.smtp \
                -t "$_RH_NOTIFY_EMAIL" \
                -s "$subject" \
                -b "$body" \
                "${attach_args[@]+"${attach_args[@]}"}" 2>/dev/null || \
                echo ">>> [Warning] Per-error email failed (SMTP error)"
        fi
    else
        # --- Throttled: buffer the error ---
        _RH_BUFFERED_COUNT=$((_RH_BUFFERED_COUNT + 1))
        cat >> "$_RH_ERROR_BUFFER" <<EOF
--- Failure ${_RH_BUFFERED_COUNT} ---
Description: ${desc}
Command: $*
Exit code: ${rc}

EOF
        # Append attachment paths to buffer
        for el in "${err_logs[@]+"${err_logs[@]}"}"; do
            echo "$el" >> "$_RH_ATTACH_BUFFER"
        done

        # Check if throttle window has elapsed
        local now
        now=$(date +%s)
        local elapsed=$(( now - _RH_LAST_SENT ))
        if [ "$elapsed" -ge "$_RH_THROTTLE" ]; then
            _rh_flush_buffer
        else
            echo ">>> [Throttled] Error buffered (${_RH_BUFFERED_COUNT} pending, next send in $((  _RH_THROTTLE - elapsed ))s)"
        fi
    fi

    # Check max-errors limit
    if [ "$_RH_MAX_ERRORS" -gt 0 ] && [ "$_RH_FAIL_COUNT" -ge "$_RH_MAX_ERRORS" ]; then
        echo ">>> [ABORT] Max errors reached (${_RH_FAIL_COUNT}/${_RH_MAX_ERRORS})"
        # Flush remaining errors, print summary, then exit 2 to signal abort.
        # Exit 2 distinguishes "aborted" from "ran to completion with failures" (exit 1).
        _rh_flush_buffer
        print_summary
        exit 2
    fi

    return 0
}

# print_summary — print pass/fail/skip counts, flush buffered errors,
# write structured summary to AE_SUMMARY_FILE (if set), return non-zero if any failures.
print_summary() {
    # Flush any remaining buffered errors
    _rh_flush_buffer

    local summary="Run Summary: ${_RH_PASS_COUNT} passed, ${_RH_FAIL_COUNT} failed, ${_RH_SKIP_COUNT} skipped (${_RH_TOTAL_COUNT} total)"

    echo ""
    echo "======================================"
    echo "$summary"
    if [ $_RH_FAIL_COUNT -gt 0 ]; then
        echo "Failed commands:"
        echo -e "${_RH_FAILED_COMMANDS}"
    fi
    echo "======================================"

    # Write structured data for run_notify.sh to read
    if [ -n "${AE_SUMMARY_FILE:-}" ]; then
        cat > "$AE_SUMMARY_FILE" <<EOF
PASS_COUNT=${_RH_PASS_COUNT}
FAIL_COUNT=${_RH_FAIL_COUNT}
SKIP_COUNT=${_RH_SKIP_COUNT}
TOTAL_COUNT=${_RH_TOTAL_COUNT}
SUMMARY_LINE=${summary}
EOF
        # Append failed commands list (may contain newlines)
        echo -n "FAILED_LIST=" >> "$AE_SUMMARY_FILE"
        echo -e "${_RH_FAILED_COMMANDS}" >> "$AE_SUMMARY_FILE"
    fi

    [ $_RH_FAIL_COUNT -eq 0 ]
}
