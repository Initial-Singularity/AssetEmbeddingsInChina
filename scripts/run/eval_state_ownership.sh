#!/bin/bash
# ── Eval: State Ownership (§7.1) ───────────
# Does the embedding, learned from holdings alone, separate state-controlled
# firms from private ones? Three parts on the finetuned embeddings of the
# 6-model panel (BERT + W2V + 4 RS variants), 7 dims, periods 2023Q2-2024Q3,
# with the SOE label of data/processed/FirmLabels/SOE (prepared beforehand):
#
#   probe    — cross-validated logit AUC of SOE on the embedding (Table 1),
#              plus the four-characteristic baseline (features ["chars"])
#   grouped  — RV and RC within SOE and non-SOE firms (Table 2): rv_grouped /
#              rc_grouped with grouping=categorical on SOE, per-cell CV
#              penalty, chars-aligned pool; plus the characteristics baseline
#   pairwise — pair cosine similarity on same-ownership and same-industry
#              indicators, Fama-MacBeth over the six periods (Table 3)
#
# Grid: probe    6 models × 7 dims + 1 char            = 43 runs
#       grouped  6 models × 7 dims × {RV,RC} + 2 char  = 86 runs
#       pairwise 6 models × 7 dims                     = 42 runs
#       Total = 171 runs.
#
# Markers: eval state_ownership probe grouped pairwise rv rc char |
#          assetbert assetw2v assetrs_binary assetrs_ranks assetrs_level0
#          assetrs_levelmin | d{4..128}
#
# Usage:
#   bash scripts/run/run_notify.sh -a results/db/state_ownership.db scripts/run/eval_state_ownership.sh
#
# Examples:
#   bash scripts/run/run_notify.sh -k probe scripts/run/eval_state_ownership.sh
#   bash scripts/run/run_notify.sh -k assetbert,d128 scripts/run/eval_state_ownership.sh
#   bash scripts/run/run_notify.sh --exclude pairwise scripts/run/eval_state_ownership.sh

source scripts/run/run_helper.sh

MODELS="AssetBERT AssetW2V AssetRS_Binary AssetRS_Ranks AssetRS_Level0 AssetRS_LevelMin"
DIMS="4 8 10 16 32 64 128"
DB="results/db/state_ownership.db"
EXP="state_ownership"  # runs.experiment; the --experiment flag of every script below
ROOT="configs/state_ownership/eval"

# ── Probe: 6 models × 7 dims + char baseline ──
for model in $MODELS; do
    for dim in $DIMS; do
        echo "=== State ownership probe ${model} d${dim} ==="
        cfg="${ROOT}/probe/d${dim}/${model}_d${dim}.json"
        job -m eval -m state_ownership -m probe -m "${model,,}" -m "d${dim}" -m "$cfg" \
            "State ownership probe ${model} d${dim}" \
            uv run python -m asset_embeddings.scripts.eval.label_probe \
                --config "$cfg" \
                --log "logs/state_ownership/probe/${model}_d${dim}.log" \
                -r "$DB" -e "$EXP"
    done
done

echo "=== State ownership probe char baseline ==="
cfg="${ROOT}/probe/char/char_baseline.json"
job -m eval -m state_ownership -m probe -m char -m "$cfg" \
    "State ownership probe char baseline" \
    uv run python -m asset_embeddings.scripts.eval.label_probe \
        --config "$cfg" \
        --log "logs/state_ownership/probe/char.log" \
        -r "$DB" -e "$EXP"

# ── Grouped: {RV,RC} × (6 models × 7 dims + char baseline) ──
for task in RV RC; do
    module="asset_embeddings.scripts.eval.${task,,}_grouped"
    for model in $MODELS; do
        for dim in $DIMS; do
            echo "=== State ownership grouped ${task} ${model} d${dim} ==="
            cfg="${ROOT}/grouped/${task}/d${dim}/${model}_d${dim}.json"
            job -m eval -m state_ownership -m grouped -m "${task,,}" -m "${model,,}" -m "d${dim}" -m "$cfg" \
                "State ownership grouped ${task} ${model} d${dim}" \
                uv run python -m "$module" \
                    --config "$cfg" \
                    --log "logs/state_ownership/grouped/${task}/${model}_d${dim}.log" \
                    -r "$DB" -e "$EXP"
        done
    done

    echo "=== State ownership grouped ${task} char baseline ==="
    cfg="${ROOT}/grouped/${task}/char/char_baseline.json"
    job -m eval -m state_ownership -m grouped -m "${task,,}" -m char -m "$cfg" \
        "State ownership grouped ${task} char baseline" \
        uv run python -m "$module" \
            --config "$cfg" \
            --log "logs/state_ownership/grouped/${task}/char.log" \
            -r "$DB" -e "$EXP"
done

# ── Pairwise: 6 models × 7 dims ──
for model in $MODELS; do
    for dim in $DIMS; do
        echo "=== State ownership pairwise ${model} d${dim} ==="
        cfg="${ROOT}/pairwise/d${dim}/${model}_d${dim}.json"
        job -m eval -m state_ownership -m pairwise -m "${model,,}" -m "d${dim}" -m "$cfg" \
            "State ownership pairwise ${model} d${dim}" \
            uv run python -m asset_embeddings.scripts.eval.pairwise_similarity \
                --config "$cfg" \
                --log "logs/state_ownership/pairwise/${model}_d${dim}.log" \
                -r "$DB" -e "$EXP"
    done
done

print_summary
