# Kimi-K2.6-MXFP4 Multi-Turn Benchmark — Launch Reference

Quick-reference commands for standing up the vLLM server and running the
agentic multi-turn benchmarks on AMD MI300X GPUs.

## 1. Create the container

The container is started with `sleep infinity` so the inductor patch can be
applied before the server launches.

```bash
docker run -d \
  --name kimi-bench \
  --ipc=host --shm-size=16g --network=host --privileged \
  --cap-add=CAP_SYS_ADMIN --cap-add=SYS_PTRACE \
  --device=/dev/kfd --device=/dev/dri --device=/dev/mem \
  --security-opt seccomp=unconfined \
  -e HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  -e ROCR_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  -v /data:/data \
  -v /home/rbrugaro/code/vllm/inductor_fallback_allow_list.patch:/tmp/inductor_patch.patch \
  --entrypoint /bin/bash \
  vllm/vllm-openai-rocm:nightly -c "sleep infinity"
```

## 2. Apply the inductor fallback allow-list patch

Prevents PyTorch Inductor compilation hangs on certain ROCm kernels.

```bash
docker exec kimi-bench bash -c '
  cd /opt/venv/lib/python3.12/site-packages &&
  patch -p1 < /tmp/inductor_patch.patch
'
```

## 3. Launch the vLLM server

### Server flags

| Flag | Value | Reason |
|------|-------|--------|
| `--tensor-parallel-size` | 8 | 8× MI300X for MXFP4 checkpoint |
| `--max-model-len` | 65536 | Agentic conversations can reach ~50K input tokens |
| `--max-num-seqs` | 512 | Concurrent sequence budget |
| `--attention-backend` | ROCM_AITER_MLA | MLA-optimised attention for MI300X |
| `--block-size` | 1 | Required by MLA attention backend |
| `--kv-cache-dtype` | fp8 | Fits 65K context in VRAM |
| `--gpu-memory-utilization` | 0.95 | Maximise KV-cache capacity |
| `--enable-prefix-caching` | — | Explicit; on by default in vLLM V1 but stated for clarity |
| `--tool-call-parser` | kimi_k2 | Native Kimi tool-call format (`<\|tool_call_begin\|>` markers) |
| `--reasoning-parser` | kimi_k2 | Separates `<think>` reasoning from visible output |
| `--enable-auto-tool-choice` | — | Required for tool-call parsing |
| `--compilation-config` | (see below) | Graph-capture with inductor partitioning |

### Resolved engine defaults (from engine config log, not set explicitly)

| Config | Resolved value | Note |
|--------|----------------|------|
| `dtype` | torch.bfloat16 | Auto-detected from model |
| `quantization` | quark | Auto-detected from MXFP4 checkpoint |
| `enable_chunked_prefill` | True | Default in V1 engine |
| `enable_prefix_caching` | True | Default in V1 engine (set explicitly for clarity) |
| `load_format` | auto | |
| `tokenizer_mode` | auto | |

### Command

```bash
docker exec -d kimi-bench bash -c '
export VLLM_ROCM_USE_AITER=1
export AMDGCN_USE_BUFFER_OPS=1
export VLLM_ROCM_USE_AITER_MLA_PS=1
export VLLM_ROCM_QUICK_REDUCE_QUANTIZATION=INT4
export VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS=1
export VLLM_ROCM_USE_AITER_TUNED_UNQUANTISED_GEMM=1
export VLLM_ROCM_DISABLE_ATTENTION_LINEAR_LAYER_DYNAMIC_MXFP4_QUANT=1
export VLLM_LOGGING_LEVEL=INFO

python -m vllm.entrypoints.openai.api_server \
  --model /data/workloads-inference/models/Kimi-K2.6-MXFP4 \
  --served-model-name Kimi-K2.6-MXFP4 \
  --port 8001 \
  --tensor-parallel-size 8 \
  --trust-remote-code \
  --max-model-len 65536 \
  --attention-backend ROCM_AITER_MLA \
  --block-size 1 \
  --gpu-memory-utilization 0.95 \
  --kv-cache-dtype fp8 \
  --max-num-seqs 512 \
  --enable-prefix-caching \
  --enable-auto-tool-choice \
  --tool-call-parser kimi_k2 \
  --reasoning-parser kimi_k2 \
  --compilation-config "{\"pass_config\": {\"fuse_allreduce_rms\": true, \"eliminate_noops\": true}, \"custom_ops\": [\"none\", \"+rms_norm\"], \"compile_ranges_endpoints\": [64], \"cudagraph_mode\": \"full_and_piecewise\", \"use_inductor_graph_partition\": true}" \
  2>&1 | tee /tmp/vllm_server.log
'
```

Wait for startup (graph capture takes several minutes):

```bash
docker exec kimi-bench tail -f /tmp/vllm_server.log
# Look for: "Application startup complete" or "Uvicorn running on ..."
```

Verify:

```bash
curl -s http://localhost:8001/health
curl -s http://localhost:8001/v1/models | python3 -m json.tool
```

### Post-launch sanity checks

```bash
# Confirm engine config in the log:
docker exec kimi-bench grep "non-default args" /tmp/vllm_server.log
# Should show: tool_call_parser='kimi_k2', reasoning_parser='kimi_k2'
# Should show: tensor_parallel_size=8, max_model_len=65536

# Confirm prefix caching is active:
docker exec kimi-bench grep "enable_prefix_caching" /tmp/vllm_server.log
# Should show: enable_prefix_caching=True
```

## 4. Prepare the datasets

v3 datasets are extracted from ppalanga's zip archive into a persistent
location (not `/tmp`).

```bash
mkdir -p data/v3
unzip -jo /home/ppalanga/agentic-ai-mlperf/Multi-Turn-20260519T182837Z-3-001.zip \
  "Multi-Turn/agentic_coding_v3.jsonl" \
  "Multi-Turn/agentic_workflow_v3.jsonl" \
  -d data/v3/

# Workflow v3 has a metadata header row — strip it
tail -n +2 data/v3/agentic_workflow_v3.jsonl > data/v3/agentic_workflow_v3_noheader.jsonl

# Coding v3 has no header — use as-is

# Create symlinks in the datasets directory
cd examples/09_MultiTurn/datasets
ln -sf "$(pwd)/../../../data/v3/agentic_workflow_v3_noheader.jsonl" agentic_workflow_v3.jsonl
ln -sf "$(pwd)/../../../data/v3/agentic_coding_v3.jsonl" agentic_coding_v3.jsonl
```

Expected row counts:

| Dataset | Rows | Conversations | Assistant turns |
|---------|------|---------------|-----------------|
| workflow v3 | 8,632 | 500 | 4,316 |
| coding v3 | 52,031 | 490 | 26,012 |

## 5. Client YAML settings

The benchmark YAML configs should use these model params to match the
canonical `kimi_agentic_benchmark.yaml`:

```yaml
model_params:
  name: "Kimi-K2.6-MXFP4"
  temperature: 1.0
  top_p: 0.95
  max_new_tokens: 20000   # covers longest observed assistant turn (~18k tokens)
  chat_template_kwargs:
    thinking: true
    preserve_thinking: true
```

| Client param | Value | Reason |
|--------------|-------|--------|
| `max_new_tokens` | 20000 | Matches canonical config; longest observed turn is ~18k tokens |
| `temperature` | 1.0 | Non-greedy sampling for realistic agentic behaviour |
| `top_p` | 0.95 | Standard nucleus sampling |
| `target_concurrency` | 8 | Max active conversations in flight |
| `turn_timeout_s` | 600.0 | Per-turn deadline (10 min) |
| `enable_salt` | true | Cache-bust across conversations, preserve within |
| `inject_tool_delay` | true | Honor `delay_seconds` from dataset |

## 6. Run the benchmarks

### Workflow

```bash
cd /home/rbrugaro/code/endpoints
bash examples/09_MultiTurn/scripts/run_full_benchmark.sh \
  examples/09_MultiTurn/kimi_workflow_v3.yaml \
  http://localhost:8001
```

Expected duration: ~2 hours, 4,316 client turns.
Output: `logs/kimi_workflow_v3/`

### Coding

```bash
bash examples/09_MultiTurn/scripts/run_full_benchmark.sh \
  examples/09_MultiTurn/kimi_coding_v3.yaml \
  http://localhost:8001
```

Expected duration: ~9 hours, 26,012 client turns (v3 is larger than v2).
Output: `logs/kimi_coding_v3/`

## 7. Score accuracy

The scorer reads `events.jsonl` + `sample_idx_map.json` from the benchmark
output and compares model outputs against the ground-truth dataset.

```bash
# Workflow — intent code match (binary per turn)
uv run python examples/09_MultiTurn/accuracy/score_inline_accuracy.py \
  --gt data/v3/agentic_workflow_v3.jsonl \
  --domain workflow \
  --report-dir logs/kimi_workflow_v3 \
  --out logs/kimi_workflow_v3/scores.json

# Coding — bash exe multiset IoU per turn
uv run python examples/09_MultiTurn/accuracy/score_inline_accuracy.py \
  --gt data/v3/agentic_coding_v3.jsonl \
  --domain coding \
  --report-dir logs/kimi_coding_v3 \
  --out logs/kimi_coding_v3/scores.json
```

> **Note:** `--report-dir` mode can be slow on large events files (1 GB+).
> If it hangs, build `model_assistants.jsonl` externally and use `--model`
> mode instead (see prior run notes for the extraction script).

## 8. Extract token metrics

```bash
uv run python examples/09_MultiTurn/scripts/extract_metrics.py \
  --report-dir logs/kimi_workflow_v3 \
  --tokenizer /data/workloads-inference/models/Kimi-K2.6-MXFP4 \
  --metrics-before logs/kimi_workflow_v3/metrics_before_*.txt \
  --metrics-after  logs/kimi_workflow_v3/metrics_after_*.txt \
  --out logs/kimi_workflow_v3/token_metrics.json
```

## Environment variables reference

| Variable | Value | Purpose |
|----------|-------|---------|
| `VLLM_ROCM_USE_AITER` | 1 | Enable AITER attention/tensor ops |
| `AMDGCN_USE_BUFFER_OPS` | 1 | Buffer ops for AMDGCN kernels |
| `VLLM_ROCM_USE_AITER_MLA_PS` | 1 | AITER multi-latent-attention PagedSeq |
| `VLLM_ROCM_QUICK_REDUCE_QUANTIZATION` | INT4 | INT4 all-reduce quantisation |
| `VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS` | 1 | Fused shared expert kernels |
| `VLLM_ROCM_USE_AITER_TUNED_UNQUANTISED_GEMM` | 1 | Tuned GEMM for unquantised layers |
| `VLLM_ROCM_DISABLE_ATTENTION_LINEAR_LAYER_DYNAMIC_MXFP4_QUANT` | 1 | Disable dynamic MXFP4 quant on attention linears |

## Prior runs (do not overwrite)

| Directory | Dataset | Server config | Notes |
|-----------|---------|---------------|-------|
| `logs/kimi_workflow_full` | workflow v2 | TP=4, hermes parser, no reasoning-parser, max_new_tokens=16000 | 4,316 turns, 0 errors |
| `logs/kimi_coding_full` | coding v2 | TP=4, hermes parser, no reasoning-parser, max_new_tokens=16000 | 22,590 turns, 11 errors |
| `logs/kimi_workflow_v3` | workflow v3 | TP=4, hermes parser, no reasoning-parser, max_new_tokens=16000 | 4,316 turns, 0 errors |
| `logs/agentic_workflow_pre_pr314` | workflow pre-PR314 | TP=8 (prior setup) | Backup of run before PR #314 |

### What was wrong in the prior runs

1. **`--tool-call-parser hermes`** instead of `kimi_k2` — Hermes format
   differs from Kimi's native `<|tool_call_begin|>` markers
2. **`--reasoning-parser` was never set** — the engine resolved it to `''`,
   meaning reasoning content was not separated from visible output
3. **`max_new_tokens: 16000`** instead of `20000` — could truncate the
   longest assistant turns (~18k tokens)
4. **TP=4** instead of TP=8 — used half the available GPUs
