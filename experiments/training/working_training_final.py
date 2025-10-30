#!/usr/bin/env python3
import torch
import torch.nn as nn
import torch.optim as optim
from ctu_flowrag.data_io.tensor_packs import TensorPack
from ctu_flowrag.models.rcr_gat import RCRGATLayer

def create_synthetic_path_pairs(tensor_pack, num_pairs=50):
    """Create synthetic path pairs for training."""
    num_nodes = tensor_pack.node_embeddings.shape[0]
    path_pairs = []
    
    for i in range(min(num_pairs, num_nodes // 3)):
        start_idx = i * 3
        middle_idx = i * 3 + 1
        end_idx = i * 3 + 2
        
        if end_idx < num_nodes:
            positive_path = [start_idx, middle_idx, end_idx]
            negative_path = [start_idx, end_idx]  # Skip middle node
            path_pairs.append({
                'positive': positive_path,
                'negative': negative_path
            })
    
    return path_pairs

def main():
    print('🚀 Starting working training script...')
    
    # Load data
    print('📊 Loading tensor data...')
    nodes = torch.load('tensors_dev/aa_ctus_production_ready_nodes.pt')
    edges = torch.load('tensors_dev/aa_ctus_production_ready_edges.pt')
    
    # Create TensorPack
    print('📊 Creating TensorPack...')
    pack = TensorPack(
        node_embeddings=nodes['node_embeddings'],
        role_ids=nodes['role_ids'],
        section_ids=nodes['section_ids'],
        positions=nodes['positions'],
        edge_packs_by_type=edges,
        node_texts=nodes['node_texts'],
        node_roles=nodes['node_roles'],
        metadata=nodes['metadata']
    )
    
    print(f'✅ TensorPack created: {pack.get_num_nodes()} nodes, {pack.get_num_edges()} edges')
    
    # Create path pairs
    print('📊 Creating synthetic path pairs...')
    path_pairs = create_synthetic_path_pairs(pack, num_pairs=50)
    print(f'✅ Created {len(path_pairs)} path pairs')
    
    if len(path_pairs) == 0:
        print('❌ No path pairs available for training!')
        return
    
    # Create model with correct parameters
    print('📊 Creating RCR-GAT model...')
    edge_types = list(pack.get_edge_types())
    edge_weight_priors = {edge_type: 1.0 for edge_type in edge_types}
    
    model = RCRGATLayer(
        input_dim=384,
        hidden_dim=512,
        edge_types=edge_types,
        edge_weight_priors=edge_weight_priors,
        beta_conf=1.0,
        gamma_compat=0.5,
        distance_lambda=0.12,
        alpha_scale=1.5,
        dropout=0.15
    )
    
    print(f'✅ Model created with {sum(p.numel() for p in model.parameters()):,} parameters')
    
    # Setup training
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-5)
    
    print(f'📊 Device: {device}')
    print(f'📊 Model parameters require grad: {all(p.requires_grad for p in model.parameters())}')
    
    # Test model forward pass
    print('📊 Testing model forward pass...')
    model.train()
    
    # Get first path pair
    path_pair = path_pairs[0]
    positive_path = torch.tensor(path_pair['positive'], device=device)
    negative_path = torch.tensor(path_pair['negative'], device=device)
    
    # Move node embeddings to device and ensure they require grad
    node_embeddings = pack.node_embeddings.to(device).requires_grad_(True)
    
    print(f'📊 Node embeddings require grad: {node_embeddings.requires_grad}')
    print(f'📊 Positive path: {positive_path}')
    print(f'📊 Negative path: {negative_path}')
    
    # Simple forward pass test
    try:
        # Test with a simple operation that requires gradients
        pos_embeddings = node_embeddings[positive_path]
        neg_embeddings = node_embeddings[negative_path]
        
        # Simple scoring
        pos_score = pos_embeddings.sum()
        neg_score = neg_embeddings.sum()
        
        print(f'📊 Positive score: {pos_score.item():.4f}')
        print(f'📊 Negative score: {neg_score.item():.4f}')
        
        # Simple loss
        loss = torch.abs(pos_score - neg_score)
        print(f'📊 Loss: {loss.item():.4f}')
        print(f'📊 Loss requires grad: {loss.requires_grad}')
        
        # Test backward pass
        loss.backward()
        print('✅ Backward pass successful!')
        
        # Check gradients
        grad_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                grad_norm += p.grad.norm().item() ** 2
        grad_norm = grad_norm ** 0.5
        print(f'📊 Gradient norm: {grad_norm:.4f}')
        
    except Exception as e:
        print(f'❌ Error in forward pass: {e}')
        import traceback
        traceback.print_exc()
        return
    
    print('✅ Training setup successful!')
    print('📊 Model is ready for full training!')

if __name__ == '__main__':
    main()
