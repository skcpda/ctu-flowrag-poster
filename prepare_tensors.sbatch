#!/bin/bash
#SBATCH --job-name=tensors
#SBATCH --partition=cpu-short
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --output=logs/tensors_%j.out
#SBATCH --error=logs/tensors_%j.err

source /nfs_home/software/miniconda/etc/profile.d/conda.sh
conda activate ctu_flowrag

set -euo pipefail
mkdir -p logs tensors_dev

echo "Starting tensor preparation at $(date)"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"

python -m ctu_flowrag.scripts.prepare_tensors \
  --json_dir ./streamlined_pipeline/output_data/ctu_relations_production_ready \
  --tensor_dir ./tensors_dev \
  --embed_model e5-small \
  --text_dim 384 \
  

echo "Tensor preparation completed at $(date)"
