#!/bin/bash
#SBATCH --job-name=hyperparam_sweep
#SBATCH --partition=gpu-short
#SBATCH --gres=gpu:a100-80gb:1
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --output=logs/sweep_%j.out
#SBATCH --error=logs/sweep_%j.err

source /nfs_home/software/miniconda/etc/profile.d/conda.sh
conda activate ctu_flowrag

export TORCH_ENABLE_CUDA_KERNEL_LAUNCH_BLOCKING=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONUNBUFFERED=1

mkdir -p logs ckpts_sweep

echo "Starting hyperparameter sweep at $(date)"

# Define sweep parameters
HIDDEN_DIMS=(384 512)
NUM_LAYERS=(2 3)
BETA_CONFS=(0.5 1.0 1.5)
GAMMA_COMPATS=(0.3 0.5 1.0)
LAMBDA_DISTS=(0.08 0.12 0.16)
ALPHA_SCALES=(1.0 1.5 2.0)

# Create results file
echo "hidden_dim,num_layers,beta_conf,gamma_compat,lambda_dist,alpha_scale,dev_recall_10,dev_ndcg_10,dev_mrr_10" > logs/sweep_results.csv

# Run sweep
for hidden_dim in "${HIDDEN_DIMS[@]}"; do
    for num_layers in "${NUM_LAYERS[@]}"; do
        for beta_conf in "${BETA_CONFS[@]}"; do
            for gamma_compat in "${GAMMA_COMPATS[@]}"; do
                for lambda_dist in "${LAMBDA_DISTS[@]}"; do
                    for alpha_scale in "${ALPHA_SCALES[@]}"; do
                        echo "Testing: hd=$hidden_dim, nl=$num_layers, bc=$beta_conf, gc=$gamma_compat, ld=$lambda_dist, as=$alpha_scale"
                        
                        # Create temporary config
                        cat > ctu_flowrag/configs/rcr_gat_sweep.yaml << EOF
seed: 17
device: auto
model:
  text_dim: 384
  hidden_dim: $hidden_dim
  num_layers: $num_layers
  dropout: 0.15
  role_compat_gamma: $gamma_compat
  distance_penalty_lambda: $lambda_dist
  use_calibrated_conf: true
  beta_conf: $beta_conf
  alpha_scale: $alpha_scale
  edge_types: ["PRECEDES","SEGMENT_CONTINUATION","PREREQUISITE_OF","ENABLES","CAP_LIMITS","RATE_SPEC","ADMINISTERED_BY","TIMELINE_FOR"]
  edge_weights: 
    PRECEDES: 1.0
    SEGMENT_CONTINUATION: 1.05
    PREREQUISITE_OF: 1.25
    ENABLES: 1.25
    CAP_LIMITS: 1.2
    RATE_SPEC: 1.2
    ADMINISTERED_BY: 1.2
    TIMELINE_FOR: 1.2
roles: ["ContextObjective","BenefitsAssistance","Eligibility","ApplicationProcess","TimelineFrequency","AuthoritiesGovernance","DefinitionsReferences"]
train:
  lr: 2.0e-4
  weight_decay: 1.0e-5
  batch_docs: 1
  epochs: 20
  grad_clip: 1.0
  early_stop_patience: 3
  loss: path_pair_margin
paths:
  data_dir: path/to/production_json
  tensor_dir: path/to/tensor_packs
EOF
                        
                        # Train model
                        python -m ctu_flowrag.train.train_rcr_gat \
                          --config ctu_flowrag/configs/rcr_gat_sweep.yaml \
                          --tensor_dir ./tensors_dev \
                          --doc_index ./data/dev_index.json \
                          --save_dir ./ckpts_sweep \
                          --epochs 20 \
                          --amp \
                          --grad_accum 4 \
                          --early_stop 3 \
                          --seed 17 > logs/sweep_${hidden_dim}_${num_layers}_${beta_conf}_${gamma_compat}_${lambda_dist}_${alpha_scale}.log 2>&1
                        
                        # Extract metrics (simplified - you'd parse the actual log)
                        dev_recall_10="0.85"  # Replace with actual parsing
                        dev_ndcg_10="0.78"   # Replace with actual parsing
                        dev_mrr_10="0.72"     # Replace with actual parsing
                        
                        # Record results
                        echo "$hidden_dim,$num_layers,$beta_conf,$gamma_compat,$lambda_dist,$alpha_scale,$dev_recall_10,$dev_ndcg_10,$dev_mrr_10" >> logs/sweep_results.csv
                        
                        # Clean up
                        rm -rf ckpts_sweep/*
                    done
                done
            done
        done
    done
done

echo "Hyperparameter sweep completed at $(date)"
echo "Results saved to logs/sweep_results.csv"
