#!/bin/bash
# ── Eval: disclosure-regime chains (§7.5) ──
# Chars once (panel-independent), then per chain:
#   RV / RC x {AssetBERT, AssetW2V} x {ft, static}          = 8 jobs (16 quarters each; configs {model}_d{dim}_{ft,static}.json)
#   MP via BERT-head refine x {ft, static} x 16 quarters     = 32 jobs
# Depth ladder: 9 levels x training seeds 42/43/44; breadth and random ladders: 6 shares x 3 draws, seed 42.
# 63 chains x 40 + 2 Chars = 2,522 jobs into results/db/disclosure_regime${SUFFIX}.db, with
# runs.experiment = disclosure_regime/{level}/s{seed} (Chars: disclosure_regime/chars).
# §7.5 reports RV and RC; exclude MP at invocation with --exclude mp. This runner only enumerates the grid; what runs
# is decided by the filters of run_notify.sh, and resumption lists come from scripts.tools.runs_query.
#
# Markers: chars | depth breadth random | {level} s{seed} | rv rc mp | assetbert assetw2v | ft static | {quarter} | {config path}
#
# Usage:
#   bash scripts/run/run_notify.sh --exclude mp scripts/run/eval_disclosure_regime.sh                 # everything but MP
#   bash scripts/run/run_notify.sh -k chars scripts/run/eval_disclosure_regime.sh                      # Chars only
#   bash scripts/run/run_notify.sh -k full,s42 --exclude mp scripts/run/eval_disclosure_regime.sh      # one chain
#   bash scripts/run/run_notify.sh -k random,rv scripts/run/eval_disclosure_regime.sh                  # RV of the random ladder
#   # resume: whatever has no completed run
#   uv run python -m scripts.tools.runs_query -d results/db/disclosure_regime.db missing --root configs/disclosure_regime --compress > /tmp/todo.txt
#   bash scripts/run/run_notify.sh --filter-file /tmp/todo.txt --exclude mp scripts/run/eval_disclosure_regime.sh
#   DIM=128 bash scripts/run/run_notify.sh -k full,s42 --exclude mp scripts/run/eval_disclosure_regime.sh   # the d128 grid

source scripts/run/run_helper.sh

DIM="${DIM:-16}"                                   # embedding dimension (DIM=128 for the d128 grid)
SUFFIX=""; [ "$DIM" = 16 ] || SUFFIX="_d${DIM}"   # d16 keeps the original names; other dims get their own DB and log tree
DB="results/db/disclosure_regime${SUFFIX}.db"
QUARTERS="2020Q4 2021Q1 2021Q2 2021Q3 2021Q4 2022Q1 2022Q2 2022Q3 2022Q4 2023Q1 2023Q2 2023Q3 2023Q4 2024Q1 2024Q2 2024Q3"
DEPTH_LEVELS="full top100 top50 top20 top10 top5 top3 top1 top0"
SHARES="70 50 30 10 05 02"
DRAWS="1 2 3"

# The 40 jobs of one chain. Arguments: ladder, level, training seed.
chain() {
    local ladder="$1" level="$2" seed="$3"
    local EXP="disclosure_regime/${level}/s${seed}"
    local CFG="configs/disclosure_regime/${level}/s${seed}"
    local LOG="logs/disclosure_regime${SUFFIX}/${level}_s${seed}"
    local task model kind cfg q

    for task in RV RC; do
        for model in AssetBERT AssetW2V; do
            for kind in ft static; do
                cfg="${CFG}/eval/${task}/${model}_d${DIM}_${kind}.json"
                job -m "$ladder" -m "$level" -m "s${seed}" -m "${task,,}" -m "${model,,}" -m "$kind" -m "$cfg" \
                    "${task} ${model} ${kind} ${level} s${seed}" \
                    uv run python -m asset_embeddings.scripts.eval."${task,,}" --config "$cfg" \
                        --log "${LOG}/eval_${task}_${model}_${kind}.log" -r "$DB" -e "$EXP"
            done
        done
    done

    for kind in ft static; do
        model="AssetBERT"; [ "$kind" = static ] && model="AssetBERT_static"
        for q in $QUARTERS; do
            cfg="${CFG}/eval/MP/${model}/AssetBERT_d${DIM}_${q}.json"
            job -m "$ladder" -m "$level" -m "s${seed}" -m mp -m assetbert -m "$kind" -m "${q,,}" -m "$cfg" \
                "MP ${model} ${q} ${level} s${seed}" \
                uv run python -m asset_embeddings.scripts.eval.mp_bert --config "$cfg" \
                    --log "${LOG}/eval_MP_${model}_${q}.log" -r "$DB" -e "$EXP"
        done
    done
}

# ── Chars (once) ──
for task in RV RC; do
    cfg="configs/disclosure_regime/chars/eval/${task}/Chars.json"
    job -m chars -m "${task,,}" -m "$cfg" "Chars ${task}" \
        uv run python -m asset_embeddings.scripts.eval."${task,,}" --config "$cfg" \
            --log "logs/disclosure_regime${SUFFIX}/chars/${task}.log" -r "$DB" -e "disclosure_regime/chars"
done

# ── Depth ladder ──
for level in $DEPTH_LEVELS; do
    for seed in 42 43 44; do
        chain depth "$level" "$seed"
    done
done

# ── Breadth ladder ──
for r in $SHARES; do
    for s in $DRAWS; do
        chain breadth "keep${r}_s${s}" 42
    done
done

# ── Random control ladder ──
for r in $SHARES; do
    for s in $DRAWS; do
        chain random "rand${r}_s${s}" 42
    done
done

print_summary
