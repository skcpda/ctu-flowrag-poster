# CTU-FlowRAG: RCR-GAT + Sinkhorn Implementation

A production-quality implementation of Role-Conditioned Retrieval with Graph Attention Networks and Capacitated Soft Role Assignment using unbalanced Sinkhorn optimization.

## Overview

This implementation provides a complete pipeline for CTU (Conceptual Text Unit) graph processing, including:

- **RCR-GAT**: Role-Conditioned Retrieval with Graph Attention Networks
- **CSRA**: Capacitated Soft Role Assignment using unbalanced Sinkhorn
- **Path Mining**: Template-constrained path search and ranking
- **Evaluation**: Comprehensive metrics for path ranking and retrieval
- **Visualization**: Static and interactive CTU graph visualizations

## Features

### Core Models
- **RCR-GAT**: Multi-layer graph attention with type-specific attention mechanisms
- **Sinkhorn**: Unbalanced optimal transport for role assignment
- **Score Matrix**: Multi-component scoring with query similarity, role fit, salience, and cohesion

### Data Processing
- **JSON Loading**: Production JSON format support with policy validation
- **Tensor Conversion**: Efficient tensor pack creation with embeddings
- **Compatibility**: Role-edge compatibility checking with allowlist

### Training & Evaluation
- **Path Mining**: Positive and negative path generation using templates
- **Margin Ranking**: Loss function for path pair training
- **Metrics**: nDCG, MRR, MAP, and path coherence evaluation
- **Early Stopping**: Validation-based early stopping with patience

### Visualization
- **Static Graphs**: Role-colored nodes with type-styled edges
- **Interactive Graphs**: Pan, zoom, and hover functionality
- **Summary Statistics**: Role distribution, edge types, and confidence analysis

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd ctu-flowrag

# Install dependencies
pip install torch>=2.1 numpy pyyaml matplotlib networkx plotly pandas tqdm

# Install in development mode
pip install -e .
```

## Quick Start

### 1. Prepare Data

```bash
# Convert production JSON files to tensor packs
python ctu_flowrag/scripts/prepare_tensors.py \
    --json_dir path/to/production_json \
    --tensor_dir path/to/tensor_packs \
    --embed_model e5-small \
    --text_dim 384
```

### 2. Train Model

```bash
# Train RCR-GAT model
python ctu_flowrag/scripts/train_dev.sh \
    --config ctu_flowrag/configs/rcr_gat.yaml \
    --doc_index data/dev_index.json \
    --tensor_dir path/to/tensor_packs \
    --output_dir logs
```

### 3. Evaluate Model

```bash
# Evaluate on test set
python ctu_flowrag/eval/eval_paths.py \
    --config ctu_flowrag/configs/rcr_gat.yaml \
    --checkpoint logs/rcr_gat_best.pt \
    --doc_index data/test_index.json \
    --tensor_dir path/to/tensor_packs \
    --output_dir eval_results
```

### 4. Visualize Results

```bash
# Create CTU graph visualization
python ctu_flowrag/viz/plot_ctu_graph.py \
    --json_path path/to/document.json \
    --output_dir visualizations \
    --max_nodes 40
```

## Configuration

### RCR-GAT Configuration (`configs/rcr_gat.yaml`)

```yaml
seed: 17
device: auto
model:
  text_dim: 384
  hidden_dim: 384
  num_layers: 2
  dropout: 0.15
  role_compat_gamma: 0.5
  distance_penalty_lambda: 0.12
  use_calibrated_conf: true
  beta_conf: 1.0
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
  epochs: 30
  grad_clip: 1.0
  early_stop_patience: 5
  loss: path_pair_margin
```

### Sinkhorn Configuration (`configs/sinkhorn.yaml`)

```yaml
seed: 17
tau: 0.5
alpha: 0.5
beta: 0.5
iters: 30
capacities: 
  ContextObjective: 2
  BenefitsAssistance: 3
  Eligibility: 2
  ApplicationProcess: 3
  TimelineFrequency: 1
  AuthoritiesGovernance: 1
  DefinitionsReferences: 1
col_cap: 1.0
score_weights: 
  w_query: 1.0
  w_role: 0.6
  w_sal: 0.2
  w_coh: 0.2
```

## API Reference

### Core Models

#### RCRGAT

```python
from ctu_flowrag.models.rcr_gat import RCRGAT

model = RCRGAT(
    text_dim=384,
    hidden_dim=384,
    num_layers=2,
    edge_types=["PRECEDES", "ENABLES"],
    edge_weight_priors={"PRECEDES": 1.0, "ENABLES": 1.25},
    beta_conf=1.0,
    gamma_compat=0.5,
    distance_lambda=0.12,
    dropout=0.15
)

# Forward pass
h = model(node_feats, edge_packs_by_type)
```

#### CSRA (Sinkhorn)

```python
from ctu_flowrag.models.sinkhorn import CSRA

csra = CSRA(
    num_roles=7,
    hidden_dim=384,
    tau=0.5,
    alpha=0.5,
    beta=0.5,
    iters=30,
    use_balanced=False
)

# Forward pass
X = csra(node_features, role_ids, capacities, score_weights)
```

### Data Loading

```python
from ctu_flowrag.data_io.load_json_graph import load_json_graph
from ctu_flowrag.data_io.tensor_packs import build_tensor_pack

# Load graph data
graph_data = load_json_graph("path/to/document.json")

# Build tensor pack
tensor_pack = build_tensor_pack(
    graph_data,
    embed_model="e5-small",
    text_dim=384,
    distance_lambda=0.12
)
```

### Path Search

```python
from ctu_flowrag.retrieval.template_path_search import TemplatePathSearch

path_search = TemplatePathSearch(
    model=rcr_gat_model,
    csra=csra_model,
    device=device,
    edge_types=edge_types,
    roles=roles,
    beam_size=32
)

# Search paths following a template
template = ["ContextObjective", "BenefitsAssistance", "ApplicationProcess"]
paths = path_search.search_paths(tensor_pack, template, num_paths=5)
```

## Testing

Run the comprehensive test suite:

```bash
# Run all tests
python -m pytest ctu_flowrag/tests/

# Run specific test modules
python -m pytest ctu_flowrag/tests/test_loader.py
python -m pytest ctu_flowrag/tests/test_rcr_gat_layer.py
python -m pytest ctu_flowrag/tests/test_sinkhorn.py
python -m pytest ctu_flowrag/tests/test_path_search.py
```

## Examples

### Basic Usage

```python
import torch
from ctu_flowrag.models.rcr_gat import RCRGAT
from ctu_flowrag.data_io.tensor_packs import build_tensor_pack

# Load data
graph_data = load_json_graph("document.json")
tensor_pack = build_tensor_pack(graph_data)

# Create model
model = RCRGAT(
    text_dim=384,
    hidden_dim=384,
    num_layers=2,
    edge_types=["PRECEDES", "ENABLES"],
    edge_weight_priors={"PRECEDES": 1.0, "ENABLES": 1.25}
)

# Forward pass
with torch.no_grad():
    h = model(tensor_pack.node_embeddings, tensor_pack.edge_packs_by_type)
```

### Attention Inspection

```python
from ctu_flowrag.retrieval.attention_inspector import AttentionInspector

# Create inspector
inspector = AttentionInspector(model, device, edge_types, roles)

# Inspect top-k attention weights
topk_edges = inspector.inspect_topk(tensor_pack, k=5)

# Print attention table
inspector.print_attention_table(tensor_pack, k=5)
```

### Path Search Demo

```bash
# Run path search demo
python ctu_flowrag/scripts/demo_path_search.sh \
    --config ctu_flowrag/configs/rcr_gat.yaml \
    --checkpoint logs/rcr_gat_best.pt \
    --doc_index data/dev_index.json \
    --tensor_dir data/tensors \
    --output_dir demo_results
```

## Performance

### Model Sizes
- **RCR-GAT**: ~1.5M parameters (2 layers, 384 hidden dim)
- **CSRA**: ~50K parameters (role embeddings + MLPs)
- **Total**: ~1.6M parameters

### Training Time
- **Small dataset** (100 docs): ~10 minutes
- **Medium dataset** (1000 docs): ~2 hours
- **Large dataset** (10000 docs): ~20 hours

### Memory Usage
- **Training**: ~4GB GPU memory
- **Inference**: ~2GB GPU memory
- **CPU-only**: ~8GB RAM

## Troubleshooting

### Common Issues

1. **CUDA out of memory**: Reduce batch size or use CPU
2. **Convergence issues**: Adjust learning rate or increase patience
3. **Path mining failures**: Check template compatibility with allowlist
4. **Visualization errors**: Ensure matplotlib backend is available

### Debug Mode

```bash
# Enable debug logging
export LOGLEVEL=DEBUG
python ctu_flowrag/scripts/train_dev.sh --config configs/rcr_gat.yaml
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## License

MIT License - see LICENSE file for details.

## Citation

```bibtex
@software{ctu_flowrag,
  title={CTU-FlowRAG: RCR-GAT + Sinkhorn Implementation},
  author={CTU-FlowRAG Team},
  year={2024},
  url={https://github.com/your-repo/ctu-flowrag}
}
```

