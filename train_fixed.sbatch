#!/bin/bash
#SBATCH --job-name=train_fixed
#SBATCH --partition=gpu-short
#SBATCH --gres=gpu:a100-80gb:1
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --output=logs/train_fixed_%j.out
#SBATCH --error=logs/train_fixed_%j.err

source /nfs_home/software/miniconda/etc/profile.d/conda.sh
conda activate ctu_flowrag

export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1

echo "Starting fixed RCR-GAT training at Wed Oct 15 12:36:29 IST 2025"
echo "GPU: "
echo "CUDA available: False"

# Create synthetic path pairs for training
python -c "
import sys
sys.path.insert(0, '.')
from ctu_flowrag.data_io.tensor_packs import load_tensor_pack
import json

# Load tensor pack
tensor_pack = load_tensor_pack('tensors_dev/documents')
num_nodes = tensor_pack.get_num_nodes()

# Create synthetic path pairs
path_pairs = []
for i in range(min(20, num_nodes // 3)):
    start_idx = i * 3
    middle_idx = i * 3 + 1
    end_idx = i * 3 + 2
    
    if end_idx < num_nodes:
        positive_path = [start_idx, middle_idx, end_idx]
        negative_path = [start_idx, end_idx]
        path_pairs.append({
            'positive': positive_path,
            'negative': negative_path
        })

# Save path pairs
with open('data/path_pairs.json', 'w') as f:
    json.dump(path_pairs, f)

print(f'Created {len(path_pairs)} path pairs for training')
"

# Run training with fixed path pairs
python -m ctu_flowrag.train.train_rcr_gat \
    --config ctu_flowrag/configs/rcr_gat.yaml \
    --tensor_dir ./tensors_dev \
    --doc_index ./data/dev_index.json \
    --epochs 10 \
    --amp \
    --grad_accum 2 \
    --save_dir ./ckpts \
    --save_minutes 15 \
    --early_stop 3 \
    --seed 17

echo "Fixed training completed at Wed Oct 15 12:36:32 IST 2025"
