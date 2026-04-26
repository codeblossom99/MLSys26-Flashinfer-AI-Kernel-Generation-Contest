#!/bin/bash
set -e

cd "$(dirname "$0")"

echo "=========================================="
echo "Running GDN Prefill Benchmark on Local Machine"
echo "=========================================="
echo ""

START_TIME=$(date +%s.%N)

python scripts/pack_solution.py
python scripts/run_local.py

END_TIME=$(date +%s.%N)
ELAPSED=$(echo "$END_TIME - $START_TIME" | bc)

echo ""
echo "=========================================="
echo "Total execution time: ${ELAPSED} seconds"
echo "=========================================="