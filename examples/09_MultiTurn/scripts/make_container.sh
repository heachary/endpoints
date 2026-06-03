docker run -d \
  --name heachary.kk26_agentic_ai \
  --ipc=host --shm-size=16g --network=host --privileged \
  --cap-add=CAP_SYS_ADMIN --cap-add=SYS_PTRACE \
  --device=/dev/kfd --device=/dev/dri --device=/dev/mem \
  --security-opt seccomp=unconfined \
  -e HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  -e ROCR_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  -v /data:/data \
  -v /home/heachary/Code:/workspace \
  --entrypoint /bin/bash \
  rocm/workloads-perf-private:custom_vllm_custom_2026-05-26-14-54-47 -c "sleep infinity"