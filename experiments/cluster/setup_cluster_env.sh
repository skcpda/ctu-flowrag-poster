#!/bin/bash
# Setup script for CTU-FlowRAG on GPU cluster
# Run this once per user session

set -euo pipefail

echo "Setting up CTU-FlowRAG environment on cluster..."

# Use system Miniconda
source /nfs_home/software/miniconda/etc/profile.d/conda.sh

# Create conda environment
echo "Creating conda environment..."
conda create -y -n ctu_flowrag python=3.10

# Activate environment
conda activate ctu_flowrag

# Install PyTorch with CUDA (bundled, no system CUDA module needed)
echo "Installing PyTorch with CUDA..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install core dependencies
echo "Installing core dependencies..."
pip install numpy pyyaml tqdm rich tensorboard networkx

# Install project in editable mode
echo "Installing CTU-FlowRAG project..."
cd ~/ctu-flowrag-poster  # or your repo clone path
pip install -e .

# Quick sanity check
echo "Running sanity check..."
python - <<'PY'
import torch
print("CUDA available:", torch.cuda.is_available())
print("GPU count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("GPU name:", torch.cuda.get_device_name(0))
PY

echo "Environment setup completed!"
echo "To activate: conda activate ctu_flowrag"
