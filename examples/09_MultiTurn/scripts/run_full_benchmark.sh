#!/usr/bin/env bash
set -euo pipefail

CONFIG="$1"
METRICS_URL="${2:-http://localhost:8001}"

REPORT_DIR=$(grep 'report_dir:' "$CONFIG" | awk '{print $2}')
if [[ -z "$REPORT_DIR" ]]; then
    echo "ERROR: Could not extract report_dir from $CONFIG"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=== Full Benchmark Pipeline ==="
echo "  Config:     $CONFIG"
echo "  Report dir: $REPORT_DIR"
echo "  Metrics:    $METRICS_URL"
echo ""

mkdir -p "$REPORT_DIR"

echo "--- [1/3] Scraping /metrics (before) ---"
bash "$SCRIPT_DIR/scrape_vllm_metrics.sh" "$METRICS_URL" "$REPORT_DIR" before 2>&1 | head -5
echo ""

echo "--- [2/3] Running benchmark ---"
uv run inference-endpoint benchmark from-config --config "$CONFIG" 2>&1
BENCH_EXIT=$?
echo ""
echo "--- Benchmark exited with code $BENCH_EXIT ---"

echo "--- [3/3] Scraping /metrics (after) ---"
bash "$SCRIPT_DIR/scrape_vllm_metrics.sh" "$METRICS_URL" "$REPORT_DIR" after 2>&1 | head -5
echo ""

echo "=== Pipeline complete. Report dir: $REPORT_DIR ==="
ls -lh "$REPORT_DIR"
