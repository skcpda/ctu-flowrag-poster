#!/bin/bash
#SBATCH --job-name=rcrgat_ddp2
#SBATCH --partition=gpu-short
#SBATCH --gres=gpu:a100-80gb:2
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --output=logs/train_ddp2_%j.out
#SBATCH --error=logs/train_ddp2_%j.err

source /nfs_home/software/miniconda/etc/profile.d/conda.sh
conda activate ctu_flowrag

export WORLD_SIZE=2
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29500
export TORCH_ENABLE_CUDA_KERNEL_LAUNCH_BLOCKING=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8     # determinism
export PYTHONUNBUFFERED=1

mkdir -p logs ckpts

echo "Starting DDP RCR-GAT training at $(date)"
echo "GPUs: $(nvidia-smi --query-gpu=name --format=csv,noheader,nounits | wc -l)"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"

torchrun --nproc_per_node=2 -m ctu_flowrag.train.train_rcr_gat \
  --config ctu_flowrag/configs/rcr_gat.yaml \
  --tensor_dir ./tensors_dev \
  --doc_index ./data/train_index.json \
  --val_index ./data/dev_index.json \
  --epochs 60 \
  --amp \
  --grad_accum 2 \
  --ddp \
  --save_dir ./ckpts \
  --save_minutes 10 \
  --early_stop 5 \
  --seed 17

echo "DDP training completed at $(date)"
