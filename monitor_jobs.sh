#!/bin/bash
# Monitoring script for CTU-FlowRAG jobs

echo "=== CTU-FlowRAG Job Monitor ==="
echo ""

# Check job queue
echo "Current jobs:"
squeue -u $USER
echo ""

# Check recent logs
echo "Recent log files:"
ls -la logs/ | tail -10
echo ""

# Check GPU usage
echo "GPU usage:"
nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits
echo ""

# Check disk usage
echo "Disk usage:"
df -h . | tail -1
echo ""

# Show latest training progress
if [ -f "logs/train_*.out" ]; then
    echo "Latest training output:"
    tail -20 logs/train_*.out 2>/dev/null || echo "No training logs found"
    echo ""
fi

# Show latest evaluation results
if [ -f "logs/eval_*.out" ]; then
    echo "Latest evaluation output:"
    tail -20 logs/eval_*.out 2>/dev/null || echo "No evaluation logs found"
    echo ""
fi

# Check checkpoint directory
if [ -d "ckpts" ]; then
    echo "Checkpoints:"
    ls -la ckpts/ | tail -5
    echo ""
fi

echo "=== End Monitor ==="
