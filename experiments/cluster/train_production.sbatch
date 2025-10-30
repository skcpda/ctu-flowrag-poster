#!/bin/bash
#SBATCH --job-name=prod_train
#SBATCH --partition=gpu-short
#SBATCH --gres=gpu:a100-80gb:1
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --output=logs/train_prod_%j.out
#SBATCH --error=logs/train_prod_%j.err

source /nfs_home/software/miniconda/etc/profile.d/conda.sh
conda activate ctu_flowrag
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

echo "Starting production training at Wed Oct 15 13:38:06 IST 2025"
echo "GPU: "
echo "CUDA available: False"

python production_training.py

echo "Production training completed at Wed Oct 15 13:38:08 IST 2025"
