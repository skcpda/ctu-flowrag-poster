#!/bin/bash
#SBATCH --job-name=eval_paths
#SBATCH --partition=short
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=logs/eval_%j.out
#SBATCH --error=logs/eval_%j.err

source /nfs_home/software/miniconda/etc/profile.d/conda.sh
conda activate ctu_flowrag

export PYTHONUNBUFFERED=1

mkdir -p logs

echo "Starting path evaluation at $(date)"

python -m ctu_flowrag.eval.eval_paths \
  --config ctu_flowrag/configs/rcr_gat.yaml \
  --tensor_dir ./tensors_dev \
  --weights ./ckpts/rcr_gat_best.pt \
  --templates "[[0,1,3],[0,2,3]]" \
  --topk 10 \
  --out_csv ./logs/dev_paths.csv

echo "Path evaluation completed at $(date)"
