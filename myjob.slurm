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

# Always use exactly these 4 local GPUs; the sweep script will set PP so TP*PP=4.
export CUDA_VISIBLE_DEVICES=0,1,2,3

# Keep one server process (DP=1) and let the script adjust PP; ensure no stale overrides.
unset VLLM_OVERRIDE_PARALLELISM

# Reliability knobs for single-node comms (no IB/RoCE), loopback sockets only
export OMP_NUM_THREADS=1
export NCCL_DEBUG=WARN
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_IB_DISABLE=1           # force no InfiniBand
export NCCL_NET=Socket             # use socket backend
export NCCL_NET_PLUGIN=none        # disable any external net plugin
export NCCL_SOCKET_IFNAME=lo       # loopback only; stays on-node
export NCCL_P2P_DISABLE=0          # allow p2p over PCIe
export NCCL_SHM_DISABLE=0
export CUDA_DEVICE_MAX_CONNECTIONS=1

# (Optional) raise file descriptor limit; vLLM opens many sockets/pipes
ulimit -n 65535 || true

# Helpful logging
echo "[SLURM] Node: $(hostname)"
echo "[SLURM] CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
nvidia-smi -L || true

# --- Run sweep ---
# srun binds to the allocation cleanly; DP remains 1 because we launch a single process.
srun -N 1 -n 1 \
python run_vllm_sweep.py \
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
  --out results.csv
