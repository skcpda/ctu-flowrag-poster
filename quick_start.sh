#!/bin/bash
# Quick start script for CTU-FlowRAG training pipeline
# Run these commands in sequence

set -euo pipefail

echo "=== CTU-FlowRAG Quick Start ==="
echo ""

# Step 1: Prepare tensors
echo "Step 1: Preparing tensors..."
echo "Submitting tensor preparation job..."
TENSOR_JOB=$(sbatch prepare_tensors.sbatch | awk '{print $4}')
echo "Tensor job ID: $TENSOR_JOB"
echo "Monitor with: tail -f logs/tensors_${TENSOR_JOB}.out"
echo ""

# Step 2: Train on GPU (short)
echo "Step 2: Training RCR-GAT (short run)..."
echo "Submitting training job (depends on tensor job)..."
TRAIN_JOB=$(sbatch --dependency=afterok:$TENSOR_JOB train_rcrgat_gpu_short.sbatch | awk '{print $4}')
echo "Training job ID: $TRAIN_JOB"
echo "Monitor with: tail -f logs/train_${TRAIN_JOB}.out"
echo ""

# Step 3: Evaluate paths
echo "Step 3: Evaluating paths..."
echo "Submitting evaluation job (depends on training job)..."
EVAL_JOB=$(sbatch --dependency=afterok:$TRAIN_JOB eval_paths.sbatch | awk '{print $4}')
echo "Evaluation job ID: $EVAL_JOB"
echo "Monitor with: tail -f logs/eval_${EVAL_JOB}.out"
echo ""

echo "=== Pipeline Submitted ==="
echo "Job chain: $TENSOR_JOB -> $TRAIN_JOB -> $EVAL_JOB"
echo ""
echo "Check status with:"
echo "  squeue -u \$USER"
echo ""
echo "View results:"
echo "  ls -la logs/"
echo "  ls -la ckpts/"
echo ""
echo "For longer training, use:"
echo "  sbatch train_rcrgat_gpu_long.sbatch"
echo ""
echo "For 2-GPU training, use:"
echo "  sbatch train_rcrgat_ddp_2gpu.sbatch"
