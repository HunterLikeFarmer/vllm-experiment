# python run_vllm_sweep.py \
#   --model Qwen/Qwen3-8B \
#   --host 0.0.0.0 --port 8000 \
#   --gpu-mem 0.80 \
#   --tensor-parallel 1 \
#   --max-num-seqs 64 128 256 \
#   --max-num-batched-tokens 1024 2048 4096 \
#   --block-size 16 32 \
#   --modes chunked prefix \
#   --concurrency 16 --num-requests 200 --max-new-tokens 128 \
#   --prompts-file prompts.txt \
#   --client-stream \
#   --out results.csv


CUDA_VISIBLE_DEVICES=0,1,2,3 \
python run_vllm_sweep.py \
  --model Qwen/Qwen3-8B \
  --host 0.0.0.0 --port 8000 \
  --gpu-mem 0.80 0.90 \
  --tensor-parallel 1 2 3 4 \
  --max-num-seqs 64 128 256 512 1024 2048 4096 8192 \
  --max-num-batched-tokens 64 128 256 512 1024 2048 4096 8192 \
  --block-size 8 16 32 \
  --modes chunked prefix \
  --concurrency 32 --num-requests 200 --max-new-tokens 128 \
  --prompts-file prompts.txt \
  --client-stream \
  --out results.csv
