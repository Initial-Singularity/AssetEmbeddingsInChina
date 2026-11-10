#!/bin/bash
# ── Train: disclosure-regime chains (§7.5) ──
# One chain = one panel level x one training seed:
#   W2V pretrain -> BERT pretrain (validation-based selection) -> 16 x (W2V finetune, BERT finetune) = 34 jobs,
#   at d16 by default (DIM=128 for the d128 grid: results/db/disclosure_regime_d128.db, logs/disclosure_regime_d128/).
# Depth ladder: 9 levels x training seeds 42/43/44; breadth and random ladders: 6 shares x 3 draws, seed 42.
# 63 chains = 2,142 jobs. This runner only enumerates the grid; what runs is decided by the filters of run_notify.sh,
# and resumption lists come from scripts.tools.runs_query.
#
# Markers: depth breadth random | {level} s{seed} | pretrain finetune | w2v bert | {quarter} | {config path}
#
# Usage:
#   bash scripts/run/run_notify.sh scripts/run/train_disclosure_regime.sh                     # everything
#   bash scripts/run/run_notify.sh -k full,s42 scripts/run/train_disclosure_regime.sh         # one chain
#   bash scripts/run/run_notify.sh -k breadth,pretrain scripts/run/train_disclosure_regime.sh # every breadth pretrain
#   bash scripts/run/run_notify.sh -k 'configs/disclosure_regime/top20/*' scripts/run/train_disclosure_regime.sh
#   # resume: whatever has no completed run
#   uv run python -m scripts.tools.runs_query -d results/db/disclosure_regime.db missing --root configs/disclosure_regime --compress > /tmp/todo.txt
#   bash scripts/run/run_notify.sh --filter-file /tmp/todo.txt scripts/run/train_disclosure_regime.sh
#   DIM=128 bash scripts/run/run_notify.sh -k full,s42 scripts/run/train_disclosure_regime.sh   # the d128 grid

source scripts/run/run_helper.sh

DIM="${DIM:-16}"                                   # embedding dimension (DIM=128 for the d128 grid)
SUFFIX=""; [ "$DIM" = 16 ] || SUFFIX="_d${DIM}"   # d16 keeps the original names; other dims get their own DB and log tree
DB="results/db/disclosure_regime${SUFFIX}.db"
QUARTERS="2020Q4 2021Q1 2021Q2 2021Q3 2021Q4 2022Q1 2022Q2 2022Q3 2022Q4 2023Q1 2023Q2 2023Q3 2023Q4 2024Q1 2024Q2 2024Q3"
DEPTH_LEVELS="full top100 top50 top20 top10 top5 top3 top1 top0"
SHARES="70 50 30 10 05 02"
DRAWS="1 2 3"

# The 34 jobs of one chain. Arguments: ladder, level, training seed.
chain() {
    local ladder="$1" level="$2" seed="$3"
    local EXP="disclosure_regime/${level}/s${seed}"
    local CFG="configs/disclosure_regime/${level}/s${seed}"
    local LOG="logs/disclosure_regime${SUFFIX}/${level}_s${seed}"
    local cfg q

    cfg="${CFG}/pretrained/AssetW2V/AssetW2V_d${DIM}_pretrained.json"
    job -m "$ladder" -m "$level" -m "s${seed}" -m pretrain -m w2v -m "$cfg" "W2V pretrain ${level} s${seed}" \
        uv run python -m asset_embeddings.scripts.train.w2v --config "$cfg" \
            --log "${LOG}/pretrain_w2v.log" -r "$DB" -e "$EXP"

    cfg="${CFG}/pretrained/AssetBERT/AssetBERT_d${DIM}_pretrained.json"
    job -m "$ladder" -m "$level" -m "s${seed}" -m pretrain -m bert -m "$cfg" "BERT pretrain ${level} s${seed}" \
        uv run python -m asset_embeddings.scripts.train.bert --config "$cfg" \
            --log "${LOG}/pretrain_bert.log" -r "$DB" -e "$EXP"

    for q in $QUARTERS; do
        cfg="${CFG}/finetune/AssetW2V/AssetW2V_d${DIM}_${q}.json"
        job -m "$ladder" -m "$level" -m "s${seed}" -m finetune -m w2v -m "${q,,}" -m "$cfg" \
            "W2V finetune ${level} s${seed} ${q}" \
            uv run python -m asset_embeddings.scripts.train.w2v --config "$cfg" \
                --log "${LOG}/finetune_w2v_${q}.log" -r "$DB" -e "$EXP"

        cfg="${CFG}/finetune/AssetBERT/AssetBERT_d${DIM}_${q}.json"
        job -m "$ladder" -m "$level" -m "s${seed}" -m finetune -m bert -m "${q,,}" -m "$cfg" \
            "BERT finetune ${level} s${seed} ${q}" \
            uv run python -m asset_embeddings.scripts.train.bert --config "$cfg" \
                --log "${LOG}/finetune_bert_${q}.log" -r "$DB" -e "$EXP"
    done
}

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
