#!/usr/bin/env bash
# Scrape vLLM/SGLang Prometheus /metrics for prefix cache stats.
#
# Usage:
#   # Snapshot before & after a benchmark run:
#   ./scrape_vllm_metrics.sh http://localhost:8000 logs/kimi_agentic before
#   <run benchmark>
#   ./scrape_vllm_metrics.sh http://localhost:8000 logs/kimi_agentic after
#
#   # Periodic scraping during a run (every 30s):
#   ./scrape_vllm_metrics.sh http://localhost:8000 logs/kimi_agentic poll 30
#
set -euo pipefail

SERVER_URL="${1:?Usage: $0 <server_url> <output_dir> <mode: before|after|poll> [interval_s]}"
OUTPUT_DIR="${2:?}"
MODE="${3:?Mode must be: before, after, or poll}"
INTERVAL="${4:-30}"

mkdir -p "$OUTPUT_DIR"

fetch_metrics() {
    local tag="$1"
    local ts
    ts=$(date +%Y%m%d_%H%M%S)
    local outfile="$OUTPUT_DIR/metrics_${tag}_${ts}.txt"
    local summary="$OUTPUT_DIR/cache_summary_${tag}_${ts}.txt"

    curl -s "$SERVER_URL/metrics" > "$outfile" 2>/dev/null || {
        echo "[$(date)] WARN: could not reach $SERVER_URL/metrics"
        return 1
    }

    {
        echo "=== Cache & Throughput Metrics ($tag @ $ts) ==="
        echo "Server: $SERVER_URL"
        echo ""

        echo "--- Prefix Cache ---"
        grep -E 'prefix_cache|cache_hit|cached_token|num_cached' "$outfile" 2>/dev/null || echo "(no prefix cache metrics found)"
        echo ""

        echo "--- GPU Cache Utilization ---"
        grep -E 'gpu_cache_usage|cpu_cache_usage|cache_usage_perc' "$outfile" 2>/dev/null || echo "(no cache usage metrics found)"
        echo ""

        echo "--- KV Cache Blocks ---"
        grep -E 'num_gpu_blocks|num_cpu_blocks|kv_cache' "$outfile" 2>/dev/null || echo "(no KV cache block metrics found)"
        echo ""

        echo "--- Request Throughput ---"
        grep -E 'num_requests|request_success|generation_tokens|prompt_tokens_total' "$outfile" 2>/dev/null || echo "(no throughput metrics found)"
        echo ""

        echo "--- Token Throughput ---"
        grep -E 'token_throughput|tokens_per_second|avg_prompt_throughput|avg_generation_throughput' "$outfile" 2>/dev/null || echo "(no token throughput metrics found)"
        echo ""

        echo "--- Scheduler ---"
        grep -E 'num_preemption|waiting_requests|running_requests|swapped_requests' "$outfile" 2>/dev/null || echo "(no scheduler metrics found)"
    } > "$summary"

    echo "[$(date)] Saved: $outfile ($( wc -l < "$outfile" ) lines), summary: $summary"
    cat "$summary"
}

case "$MODE" in
    before|after)
        fetch_metrics "$MODE"
        ;;
    poll)
        echo "[$(date)] Polling $SERVER_URL/metrics every ${INTERVAL}s. Ctrl-C to stop."
        seq_num=0
        while true; do
            fetch_metrics "poll_$(printf '%04d' $seq_num)" || true
            seq_num=$((seq_num + 1))
            sleep "$INTERVAL"
        done
        ;;
    *)
        echo "ERROR: mode must be before, after, or poll"
        exit 1
        ;;
esac
