#!/bin/bash
#SBATCH --account=bcrn-delta-gpu
#SBATCH --partition=gpuA40x4
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

# --- Modules / env ---
module load nccl
module load anaconda3_gpu
module load gcc
module load cmake
module load cuda/12.6.3

source activate vllm

# Use all 4 local GPUs; the warm-start sweep sets PP so TP*PP=4 (DP=1 single process).
export CUDA_VISIBLE_DEVICES=0,1,2,3
unset VLLM_OVERRIDE_PARALLELISM

# Single-node, sockets-only NCCL (stable on HPC nodes without IB)
export OMP_NUM_THREADS=1
export NCCL_DEBUG=WARN
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_IB_DISABLE=1
export NCCL_NET=Socket
export NCCL_NET_PLUGIN=none
export NCCL_SOCKET_IFNAME=lo
export NCCL_P2P_DISABLE=0
export NCCL_SHM_DISABLE=0
export CUDA_DEVICE_MAX_CONNECTIONS=1

# Optional caches for faster reuse (model + compiler)
export HF_HOME="$HOME/.cache/huggingface"
export TORCHINDUCTOR_CACHE_DIR="$HOME/.cache/torchinductor"
mkdir -p "$HF_HOME" "$TORCHINDUCTOR_CACHE_DIR"

# Avoid FD exhaustion
ulimit -n 65535 || true

echo "[SLURM] Node: $(hostname)"
echo "[SLURM] CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
nvidia-smi -L || true

# Unique results per job
OUT="results_warm_start.csv"

# --- Warm-start sweep over the FULL grid you requested ---
# The warm-start script will:
#  - keep a server up for each server-config combo (TP/PP plan, gpu-mem, mns, mnbt, block-size, mode),
#  - run the client once per combo (since concurrency/max-new-tokens are single-valued here),
#  - then tear down and move to the next group.

srun -N 1 -n 1 \
python warm_start_vllm_sweep.py \
  --model Qwen/Qwen3-8B \
  --host 0.0.0.0 --port 8000 \
  --gpu-mem 0.80 0.90 \
  --tensor-parallel 1 2 4 \
  --max-num-seqs 64 128 256 512 1024 2048 4096 8192 \
  --max-num-batched-tokens 64 128 256 512 1024 2048 4096 8192 \
  --block-size 16 32 \
  --modes chunked prefix \
  --concurrency 32 --num-requests 200 --max-new-tokens 128 \
  --prompts-file prompts.txt \
  --timeout-ready 180 \
  --client-stream \
  --out "$OUT"
