#!/usr/bin/env python3
"""Extract per-turn token metrics from a multi-turn benchmark run.

Reads events.jsonl + /metrics scrapes and produces a summary table with:
  - Model output (median # tokens)
  - Model reasoning (median # tokens)
  - Tool output (median # tokens)  -- tool results fed back as context
  - Cache Prefill (avg # tokens per request, from /metrics aggregate)
  - Cache Hit %
  - # Turns

Usage:
    python extract_metrics.py \
        --report-dir logs/kimi_workflow \
        --tokenizer /data/workloads-inference/models/Kimi-K2.6-MXFP4 \
        [--metrics-before logs/kimi_workflow/metrics_before_*.txt] \
        [--metrics-after  logs/kimi_workflow/metrics_after_*.txt]
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import re
import statistics
from pathlib import Path

logger = logging.getLogger("extract_metrics")


def _load_tokenizer(model_path: str):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)


def _count_tokens(tokenizer, text: str | None) -> int:
    if not text:
        return 0
    return len(tokenizer.encode(text, add_special_tokens=False))


def _parse_model_output(data: list) -> tuple[str, str, str]:
    """Parse a sample.complete data list into (reasoning, content, tool_calls_text).

    Handles two formats:
    1. In-text: all in data[1] with </think> and <|tool_calls_section_begin|> markers
    2. Structured: data[2]=reasoning_content, data[3]=tool_calls list
    """
    chunks = data[1]
    if isinstance(chunks, list):
        full_text = "".join(str(c) for c in chunks)
    elif isinstance(chunks, str):
        full_text = chunks
    else:
        full_text = str(chunks) if chunks else ""

    reasoning_field = data[2] if len(data) > 2 else None
    tool_calls_field = data[3] if len(data) > 3 else None

    reasoning = ""
    content = ""
    tool_calls_text = ""

    think_end = full_text.find("</think>")
    if think_end != -1:
        think_start = full_text.find("<think>")
        if think_start != -1:
            reasoning = full_text[think_start + 7:think_end].strip()
        else:
            reasoning = full_text[:think_end].strip()
        after_think = full_text[think_end + 8:]
    else:
        after_think = full_text

    tool_section_start = after_think.find("<|tool_calls_section_begin|>")
    if tool_section_start != -1:
        content = after_think[:tool_section_start].strip()
        tool_calls_text = after_think[tool_section_start:]
    else:
        content = after_think.strip()

    if reasoning_field and isinstance(reasoning_field, str) and not reasoning:
        reasoning = reasoning_field

    if tool_calls_field and isinstance(tool_calls_field, list) and tool_calls_field and not tool_calls_text:
        tool_calls_text = json.dumps(tool_calls_field)

    return reasoning, content, tool_calls_text


def _parse_tool_results_from_prompt(prompt_text: str) -> list[str]:
    """Extract tool result content blocks from a rendered prompt string.

    Looks for {"result": "..."} JSON blocks which are the tool outputs
    returned by tools in the dataset.
    """
    results = []
    pattern = re.compile(r'\{"result":\s*"')
    for m in pattern.finditer(prompt_text):
        depth = 0
        start = m.start()
        for i in range(start, min(start + 50000, len(prompt_text))):
            if prompt_text[i] == '{':
                depth += 1
            elif prompt_text[i] == '}':
                depth -= 1
            if depth == 0:
                results.append(prompt_text[start:i + 1])
                break
    return results


def _parse_prometheus_counter(text: str, metric_name: str) -> float:
    pattern = rf'^{re.escape(metric_name)}\{{[^}}]*\}}\s+([\d.eE+\-]+)'
    for line in text.splitlines():
        m = re.match(pattern, line)
        if m:
            return float(m.group(1))
    return 0.0


def _parse_cache_metrics(before_path: str | None, after_path: str | None) -> dict:
    result = {
        "cache_hit_pct": None,
        "avg_cached_tokens_per_request": None,
        "total_prompt_tokens": None,
        "total_generation_tokens": None,
        "total_cache_queries": None,
        "total_cache_hits": None,
    }

    if not after_path:
        return result

    after_text = Path(after_path).read_text()
    before_text = Path(before_path).read_text() if before_path else ""

    def delta(metric: str) -> float:
        a = _parse_prometheus_counter(after_text, metric)
        b = _parse_prometheus_counter(before_text, metric) if before_text else 0.0
        return a - b

    queries = delta("vllm:prefix_cache_queries_total")
    hits = delta("vllm:prefix_cache_hits_total")
    prompt = delta("vllm:prompt_tokens_total")
    gen = delta("vllm:generation_tokens_total")

    result["total_cache_queries"] = queries
    result["total_cache_hits"] = hits
    result["total_prompt_tokens"] = prompt
    result["total_generation_tokens"] = gen

    if queries > 0:
        result["cache_hit_pct"] = round(hits / queries * 100, 1)

    return result


def extract_turn_metrics(report_dir: Path, tokenizer) -> dict:
    events_path = report_dir / "events.jsonl"
    if not events_path.exists():
        raise SystemExit(f"Missing {events_path}")

    output_tokens = []
    reasoning_tokens = []
    tool_call_tokens = []
    content_tokens = []
    tool_result_tokens = []
    n_turns = 0
    n_with_reasoning = 0
    n_with_tool_calls = 0
    n_with_tool_results = 0

    issued_prompts: dict[str, str] = {}

    with events_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            ev = json.loads(line)

            if ev.get("event_type") == "sample.issued":
                data = ev.get("data")
                if isinstance(data, list) and len(data) > 1 and isinstance(data[1], str):
                    key = f"{ev.get('conversation_id', '')}_{ev.get('turn', '')}"
                    issued_prompts[key] = data[1]
                continue

            if ev.get("event_type") != "sample.complete":
                continue

            data = ev.get("data")
            if not isinstance(data, list) or len(data) < 2:
                continue
            if data[0] != "TextModelOutput":
                continue

            n_turns += 1

            reasoning, content, tool_calls_text = _parse_model_output(data)

            reasoning_toks = _count_tokens(tokenizer, reasoning)
            content_toks = _count_tokens(tokenizer, content)
            tool_call_toks = _count_tokens(tokenizer, tool_calls_text)

            total_output = content_toks + tool_call_toks
            output_tokens.append(total_output)
            content_tokens.append(content_toks)

            if reasoning_toks > 0:
                n_with_reasoning += 1
                reasoning_tokens.append(reasoning_toks)

            if tool_call_toks > 0:
                n_with_tool_calls += 1
                tool_call_tokens.append(tool_call_toks)

            key = f"{ev.get('conversation_id', '')}_{ev.get('turn', '')}"
            prompt_text = issued_prompts.get(key, "")
            if prompt_text:
                tool_results = _parse_tool_results_from_prompt(prompt_text)
                if tool_results:
                    n_with_tool_results += 1
                    combined_result_text = "\n".join(tool_results)
                    tool_result_tokens.append(_count_tokens(tokenizer, combined_result_text))

    def safe_median(lst):
        return round(statistics.median(lst)) if lst else 0

    def safe_mean(lst):
        return round(statistics.mean(lst)) if lst else 0

    return {
        "n_turns": n_turns,
        "n_with_reasoning": n_with_reasoning,
        "n_with_tool_calls": n_with_tool_calls,
        "n_with_tool_results": n_with_tool_results,
        "model_output_median_tokens": safe_median(output_tokens),
        "model_output_mean_tokens": safe_mean(output_tokens),
        "model_output_p25_tokens": round(sorted(output_tokens)[len(output_tokens) // 4]) if output_tokens else 0,
        "model_output_p75_tokens": round(sorted(output_tokens)[3 * len(output_tokens) // 4]) if output_tokens else 0,
        "model_reasoning_median_tokens": safe_median(reasoning_tokens),
        "model_reasoning_mean_tokens": safe_mean(reasoning_tokens),
        "model_content_median_tokens": safe_median(content_tokens),
        "tool_call_median_tokens": safe_median(tool_call_tokens),
        "tool_call_mean_tokens": safe_mean(tool_call_tokens),
        "tool_result_median_tokens": safe_median(tool_result_tokens),
        "tool_result_mean_tokens": safe_mean(tool_result_tokens),
    }


def find_metrics_file(report_dir: Path, tag: str) -> str | None:
    pattern = str(report_dir / f"metrics_{tag}_*.txt")
    files = sorted(glob.glob(pattern))
    return files[-1] if files else None


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser(description="Extract per-turn token metrics from benchmark run.")
    p.add_argument("--report-dir", required=True, type=Path)
    p.add_argument("--tokenizer", required=True, type=str)
    p.add_argument("--metrics-before", type=str, default=None)
    p.add_argument("--metrics-after", type=str, default=None)
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()

    logger.info("Loading tokenizer from %s...", args.tokenizer)
    tokenizer = _load_tokenizer(args.tokenizer)
    logger.info("Tokenizer loaded: vocab_size=%d", tokenizer.vocab_size)

    logger.info("Processing events from %s...", args.report_dir)
    turn_metrics = extract_turn_metrics(args.report_dir, tokenizer)

    before = args.metrics_before or find_metrics_file(args.report_dir, "before")
    after = args.metrics_after or find_metrics_file(args.report_dir, "after")
    cache_metrics = _parse_cache_metrics(before, after)

    n_turns = turn_metrics["n_turns"]
    if cache_metrics["total_cache_hits"] is not None and n_turns > 0:
        cache_metrics["avg_cached_tokens_per_request"] = round(
            cache_metrics["total_cache_hits"] / n_turns
        )

    summary = {**turn_metrics, **cache_metrics}

    out_path = args.out or (args.report_dir / "token_metrics.json")
    out_path.write_text(json.dumps(summary, indent=2))
    logger.info("Wrote %s", out_path)

    print("\n" + "=" * 60)
    print("MULTI-TURN BENCHMARK METRICS")
    print("=" * 60)
    print(f"  Report dir: {args.report_dir}")
    print()
    print(f"  # Turns:                          {summary['n_turns']}")
    print(f"    - with reasoning:               {summary['n_with_reasoning']}")
    print(f"    - with tool_calls:              {summary['n_with_tool_calls']}")
    print(f"    - with tool results in prompt:  {summary['n_with_tool_results']}")
    print()
    print(f"  Model Output    (median tokens):  {summary['model_output_median_tokens']}")
    print(f"  Model Output    (mean tokens):    {summary['model_output_mean_tokens']}")
    print(f"  Model Output    (P25/P75):        {summary['model_output_p25_tokens']} / {summary['model_output_p75_tokens']}")
    print()
    print(f"  Model Reasoning (median tokens):  {summary['model_reasoning_median_tokens']}")
    print(f"  Model Reasoning (mean tokens):    {summary['model_reasoning_mean_tokens']}")
    print()
    print(f"  Model Content   (median tokens):  {summary['model_content_median_tokens']}")
    print()
    print(f"  Tool Calls      (median tokens):  {summary['tool_call_median_tokens']}")
    print(f"  Tool Calls      (mean tokens):    {summary['tool_call_mean_tokens']}")
    print()
    print(f"  Tool Results    (median tokens):  {summary['tool_result_median_tokens']}")
    print(f"  Tool Results    (mean tokens):    {summary['tool_result_mean_tokens']}")
    print()
    if cache_metrics["cache_hit_pct"] is not None:
        print(f"  Cache Hit %:                      {cache_metrics['cache_hit_pct']}%")
        print(f"  Avg Cached Tokens/Request:        {cache_metrics['avg_cached_tokens_per_request']}")
        print(f"  Total Prefix Cache Queries:       {int(cache_metrics['total_cache_queries'])}")
        print(f"  Total Prefix Cache Hits:          {int(cache_metrics['total_cache_hits'])}")
        print(f"  Total Prompt Tokens (delta):      {int(cache_metrics['total_prompt_tokens'])}")
        print(f"  Total Generation Tokens (delta):  {int(cache_metrics['total_generation_tokens'])}")
    else:
        print("  Cache metrics: not available (no /metrics scrapes found)")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
