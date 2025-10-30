# CTU-FlowRAG Cluster Training Guide

This guide provides everything you need to run CTU-FlowRAG training on your GPU cluster.

## 🚀 Quick Start

### 1. Environment Setup (One-time)
```bash
# Login to cluster
ssh <username>@172.24.16.132

# Run setup script
bash setup_cluster_env.sh
```

### 2. Run Complete Pipeline
```bash
# Activate environment
conda activate ctu_flowrag

# Run complete pipeline
bash quick_start.sh
```

## 📋 Available Scripts

### Training Scripts
- `prepare_tensors.sbatch` - Prepare tensor packs (CPU, 4h)
- `train_rcrgat_gpu_short.sbatch` - Single GPU training (4h)
- `train_rcrgat_gpu_long.sbatch` - Single GPU training (12h)
- `train_rcrgat_ddp_2gpu.sbatch` - 2-GPU DDP training (4h)
- `eval_paths.sbatch` - Path evaluation (2h)

### Utility Scripts
- `setup_cluster_env.sh` - One-time environment setup
- `quick_start.sh` - Run complete pipeline
- `monitor_jobs.sh` - Monitor job status and logs
- `sweep_hyperparams.sbatch` - Hyperparameter sweep

## 🔧 Manual Job Submission

### Prepare Tensors
```bash
sbatch prepare_tensors.sbatch
```

### Train RCR-GAT (Short)
```bash
sbatch train_rcrgat_gpu_short.sbatch
```

### Train RCR-GAT (Long)
```bash
sbatch train_rcrgat_gpu_long.sbatch
```

### Train with 2 GPUs
```bash
sbatch train_rcrgat_ddp_2gpu.sbatch
```

### Evaluate Paths
```bash
sbatch eval_paths.sbatch
```

## 📊 Monitoring

### Check Job Status
```bash
squeue -u $USER
```

### Monitor Logs
```bash
# Real-time monitoring
tail -f logs/train_<jobid>.out
tail -f logs/train_<jobid>.err

# Or use the monitor script
bash monitor_jobs.sh
```

### Check Results
```bash
# Training checkpoints
ls -la ckpts/

# Evaluation results
ls -la logs/dev_paths.csv
```

## ⚙️ Hyperparameter Tuning

### Run Hyperparameter Sweep
```bash
sbatch sweep_hyperparams.sbatch
```

### Key Parameters to Tune
- `hidden_dim`: 384 → 512
- `num_layers`: 2 → 3
- `beta_conf`: 0.5 → 1.5 (confidence weight)
- `gamma_compat`: 0.3 → 1.0 (compatibility weight)
- `lambda_dist`: 0.08 → 0.16 (distance penalty)
- `alpha_scale`: 1.0 → 2.0 (type bias scaling)

## 📁 Directory Structure

```
ctu-flowrag/
├── logs/                    # Job logs and outputs
├── ckpts/                   # Model checkpoints
├── tensors_dev/            # Preprocessed tensor packs
├── data/                    # Data index files
│   ├── train_index.json
│   └── dev_index.json
├── *.sbatch                 # SLURM job scripts
├── *.sh                     # Utility scripts
└── ctu_flowrag/            # Source code
```

## 🔍 Troubleshooting

### Common Issues

1. **Out of Memory**: Reduce `hidden_dim` or use gradient accumulation
2. **Job Timeout**: Use `gpu-long` partition for longer training
3. **CUDA Issues**: Ensure PyTorch CUDA version matches cluster
4. **Permission Denied**: Check file permissions with `ls -la`

### Debug Commands
```bash
# Check GPU availability
python -c "import torch; print(torch.cuda.is_available())"

# Check CUDA version
python -c "import torch; print(torch.version.cuda)"

# Check job details
scontrol show job <jobid>

# Check partition limits
sinfo -p gpu-short
sinfo -p gpu-long
```

## 📈 Performance Tips

1. **Use AMP**: Always enable `--amp` for faster training
2. **Gradient Accumulation**: Use `--grad_accum 4` to simulate larger batches
3. **Checkpointing**: Use `--save_minutes 15` to prevent data loss
4. **Early Stopping**: Use `--early_stop 5` to prevent overfitting
5. **Resume Training**: Use `--resume <checkpoint>` to continue training

## 🎯 Expected Results

- **Training Time**: 2-4 hours for short runs, 8-12 hours for long runs
- **Memory Usage**: ~40-60GB for single GPU, ~80GB for 2-GPU
- **Checkpoints**: Saved every 15 minutes and on validation improvement
- **Metrics**: Recall@10, nDCG@10, MRR@10, MAP@10

## 📞 Support

For issues:
1. Check job logs: `tail -f logs/<jobid>.err`
2. Monitor GPU usage: `nvidia-smi`
3. Check disk space: `df -h`
4. Verify environment: `conda list | grep torch`
