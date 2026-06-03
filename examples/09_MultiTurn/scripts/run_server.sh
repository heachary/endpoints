#!/bin/bash

DATE=$(date +%Y%m%d_%H%M%S)
mkdir -p logs

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
  --gpu-memory-utilization 0.85 \
  --kv-cache-dtype fp8 \
  --max-num-seqs 512 \
  --enable-prefix-caching \
  --enable-auto-tool-choice \
  --tool-call-parser kimi_k2 \
  --reasoning-parser kimi_k2 \
  --compilation-config "{\"pass_config\": {\"fuse_allreduce_rms\": true, \"eliminate_noops\": true}, \"custom_ops\": [\"none\", \"+rms_norm\"], \"compile_ranges_endpoints\": [64], \"cudagraph_mode\": \"full_and_piecewise\", \"use_inductor_graph_partition\": true}" \
  2>&1 | tee logs/server_${DATE}.log