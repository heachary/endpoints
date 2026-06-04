# # workflow v3
# bash examples/09_MultiTurn/scripts/run_full_benchmark.sh \
#   examples/09_MultiTurn/kimi_workflow_v3.yaml \
#   http://localhost:8001

# uv run python examples/09_MultiTurn/accuracy/score_inline_accuracy.py \
#   --gt /workspace/endpoints/examples/09_MultiTurn/datasets/agentic_workflow_v3.jsonl \
#   --domain workflow \
#   --report-dir logs/kimi_workflow_v3 \
#   --out logs/kimi_workflow_v3/scores.json

# coding v3 short
bash examples/09_MultiTurn/scripts/run_full_benchmark.sh \
  examples/09_MultiTurn/kimi_coding_v3_short.yaml \
  http://localhost:8001

uv run python examples/09_MultiTurn/accuracy/score_inline_accuracy.py \
  --gt /workspace/endpoints/examples/09_MultiTurn/datasets/agentic_coding_v3_short.jsonl \
  --domain coding \
  --report-dir logs/kimi_coding_v3_short_260kctx \
  --out logs/kimi_coding_v3_short_260kctx/scores.json