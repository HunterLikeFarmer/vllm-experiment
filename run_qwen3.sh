vllm serve Qwen/Qwen3-8B \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.8 \
  --max-num-seqs 256 \
  --max-num-batched-tokens 2048 \
  --block-size 16 \
  --enable-chunked-prefill \
  --disable-custom-all-reduce \
  --host 0.0.0.0 \
  --port 8000

# vllm serve Qwen/Qwen3-8B \
#   --tensor-parallel-size 1 \
#   --gpu-memory-utilization 0.8 \
#   --max-num-seqs 256 \
#   --max-num-batched-tokens 2048 \
#   --block-size 16 \
#   --enable-prefix-caching \
#   --disable-custom-all-reduce \
#   --host 0.0.0.0 \
#   --port 8000
