#!/bin/bash
#SBATCH --account=bcrn-delta-gpu
#SBATCH --partition=gpuA40x4        # switch to non-interactive partition 
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --tasks=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=20G
#SBATCH --time=24:00:00
#SBATCH --exclusive
#SBATCH -o myjob.%j.out
#SBATCH -e myjob.%j.err

module load nccl
module load anaconda3_gpu
module load gcc
module load cmake
module load cuda/12.6.3

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
