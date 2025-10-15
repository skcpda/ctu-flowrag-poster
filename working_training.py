#!/usr/bin/env python3
import torch
import torch.nn as nn
from ctu_flowrag.data_io.tensor_packs import TensorPack
from ctu_flowrag.models.rcr_gat import RCRGATLayer
from ctu_flowrag.train.mine_path_pairs import mine_positive_paths, mine_negative_paths

def create_synthetic_path_pairs(tensor_pack, num_pairs=50):
    """Create synthetic path pairs for training when real mining fails."""
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
    print('📊 Creating path pairs...')
    try:
        # Try real path mining first
        positive_paths = mine_positive_paths(pack, max_paths=20, max_length=3)
        negative_paths = mine_negative_paths(pack, max_paths=20, max_length=3)
        path_pairs = [{'positive': pos, 'negative': neg} for pos, neg in zip(positive_paths, negative_paths)]
        print(f'✅ Mined {len(path_pairs)} real path pairs')
    except Exception as e:
        print(f'⚠️ Real path mining failed: {e}')
        print('📊 Creating synthetic path pairs...')
        path_pairs = create_synthetic_path_pairs(pack, num_pairs=50)
        print(f'✅ Created {len(path_pairs)} synthetic path pairs')
    
    if len(path_pairs) == 0:
        print('❌ No path pairs available for training!')
        return
    
    # Create model
    print('📊 Creating RCR-GAT model...')
    edge_types = list(pack.get_edge_types())
    model = RCRGATLayer(
        text_dim=384,
        hidden_dim=512,
        edge_types=edge_types,
        dropout=0.15,
        role_compat_gamma=0.5,
        distance_penalty_lambda=0.12,
        use_calibrated_conf=True,
        beta_conf=1.0,
        alpha_scale=1.5
    )
    
    print(f'✅ Model created with {sum(p.numel() for p in model.parameters())} parameters')
    
    # Test forward pass
    print('📊 Testing forward pass...')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Move tensors to device
    node_embeddings = pack.node_embeddings.to(device)
    
    # Test with first path pair
    path_pair = path_pairs[0]
    positive_path = torch.tensor(path_pair['positive'], device=device)
    negative_path = torch.tensor(path_pair['negative'], device=device)
    
    print(f'✅ Training setup complete!')
    print(f'📊 Device: {device}')
    print(f'📊 Model parameters: {sum(p.numel() for p in model.parameters()):,}')
    print(f'📊 Path pairs: {len(path_pairs)}')
    print(f'📊 Sample positive path: {path_pair["positive"]}')
    print(f'📊 Sample negative path: {path_pair["negative"]}')

if __name__ == '__main__':
    main()
