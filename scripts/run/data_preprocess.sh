#!/bin/bash
# ── Data Preprocessing Pipeline ─────────────────────────
# Converts raw CSV dumps into model-ready portfolio datasets.
#
# Steps:
#   1. Shareholding portfolios  — quarterly CSVs with proportions
#   2. Data distributor         — split into pretrained (80%) / finetune (20%)
#   3. Data distributor         — 50/50 time-split for robustness test
#   4. Semi-annual copy         — Q2/Q4 subset for robustness test
#   5. Semi-annual × time-split — Q2/Q4 subset of the 50/50 time-split
# Total: 3 runs (steps 4 & 5 are plain cp, not try_run)
#
# Markers: data | sharehold distributor
#
# Usage:
#   bash scripts/run/run_notify.sh scripts/run/data_preprocess.sh
#
# Examples:
#   bash scripts/run/run_notify.sh -k data,sharehold scripts/run/data_preprocess.sh

source scripts/run/run_helper.sh

echo "=== Data Preprocessing ==="

try_run -m data -m sharehold "Shareholding data" \
    uv run python -m asset_embeddings.scripts.data.sharehold \
        -i data/raw/ShareHolding \
        -o data/processed/ShareHoldingsPortfolio \
        --intermediate_folder data/processed/ShareHoldingsIntermediate \
        --output_format csv --frequency Q --date_format %Y-%m-%d \
        -il 10 -sl 10 --include_proportion True \
        --stats True \
        -v True -l logs/data_preprocess/shareholdings.log

try_run -m data -m distributor "Data distributor" \
    uv run python -m asset_embeddings.scripts.data.distributor \
        -i data/processed/ShareHoldingsPortfolio \
        -o data/processed/MutualFundShareHoldings \
        -f csv -r 0.8 0.2 --dir_names pretrained finetune \
        -l logs/data_preprocess/distributor.log

# Remove incomplete quarter
rm -f data/processed/MutualFundShareHoldings/finetune/2024Q4.csv

# 50/50 time-split for robustness test
try_run -m data -m distributor "Data distributor (50/50 time-split)" \
    uv run python -m asset_embeddings.scripts.data.distributor \
        -i data/processed/ShareHoldingsPortfolio \
        -o data/processed/MutualFundShareHoldings_TimeSplit \
        -f csv -r 0.5 0.5 --dir_names pretrained finetune \
        -l logs/data_preprocess/distributor_timesplit.log

# Remove incomplete quarter from time-split
rm -f data/processed/MutualFundShareHoldings_TimeSplit/finetune/2024Q4.csv

# Copy Q2/Q4 files for semi-annual robustness test
SRC="data/processed/MutualFundShareHoldings"
DST="data/processed/MutualFundShareHoldings_SemiAnnual"

if [ -d "$SRC" ]; then
    echo "Copying Q2/Q4 files to ${DST}..."
    mkdir -p "${DST}/pretrained" "${DST}/finetune"
    for subdir in pretrained finetune; do
        if [ -d "${SRC}/${subdir}" ]; then
            find "${SRC}/${subdir}" -type f \( -name "*Q2.*" -o -name "*Q4.*" \) \
                -exec cp {} "${DST}/${subdir}/" \;
        fi
    done
    echo "Semi-annual data copy done."
else
    echo "[Warning] ${SRC} not found, skipping semi-annual copy."
fi

# Copy Q2/Q4 files from TimeSplit for semi-annual × time-split robustness test
SRC_TS="data/processed/MutualFundShareHoldings_TimeSplit"
DST_TS="data/processed/MutualFundShareHoldings_SemiAnnual_TimeSplit"

if [ -d "$SRC_TS" ]; then
    echo "Copying Q2/Q4 files to ${DST_TS}..."
    mkdir -p "${DST_TS}/pretrained" "${DST_TS}/finetune"
    for subdir in pretrained finetune; do
        if [ -d "${SRC_TS}/${subdir}" ]; then
            find "${SRC_TS}/${subdir}" -type f \( -name "*Q2.*" -o -name "*Q4.*" \) \
                -exec cp {} "${DST_TS}/${subdir}/" \;
        fi
    done
    echo "Semi-annual x time-split data copy done."
else
    echo "[Warning] ${SRC_TS} not found, skipping semi-annual x time-split copy."
fi

print_summary
