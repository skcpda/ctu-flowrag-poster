#!/bin/bash
#SBATCH --job-name=rcrgat_1gpu_long
#SBATCH --partition=gpu-long
#SBATCH --gres=gpu:a100-80gb:1
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --output=logs/train_long_%j.out
#SBATCH --error=logs/train_long_%j.err

source /nfs_home/software/miniconda/etc/profile.d/conda.sh
conda activate ctu_flowrag

export TORCH_ENABLE_CUDA_KERNEL_LAUNCH_BLOCKING=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8     # determinism
export PYTHONUNBUFFERED=1

mkdir -p logs ckpts

echo "Starting long RCR-GAT training at $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader,nounits)"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"

python -m ctu_flowrag.train.train_rcr_gat \
  --config ctu_flowrag/configs/rcr_gat.yaml \
  --tensor_dir ./tensors_dev \
  --doc_index ./data/train_index.json \
  --val_index ./data/dev_index.json \
  --epochs 80 \
  --amp \
  --grad_accum 4 \
  --save_dir ./ckpts \
  --save_minutes 15 \
  --early_stop 5 \
  --seed 17

echo "Long training completed at $(date)"
