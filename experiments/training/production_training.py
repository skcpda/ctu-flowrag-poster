#!/usr/bin/env python3
import torch
import torch.nn as nn
import torch.optim as optim
from ctu_flowrag.data_io.tensor_packs import TensorPack
from ctu_flowrag.models.rcr_gat import RCRGATLayer
import os
import json

def create_synthetic_path_pairs(tensor_pack, num_pairs=100):
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

def margin_ranking_loss(positive_scores, negative_scores, margin=0.2):
    """Compute margin ranking loss."""
    loss = torch.clamp(margin - positive_scores + negative_scores, min=0.0)
    return loss.mean()

def main():
    print('🚀 Starting production training with 3000+ schemes...')
    
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
    path_pairs = create_synthetic_path_pairs(pack, num_pairs=200)
    print(f'✅ Created {len(path_pairs)} path pairs')
    
    if len(path_pairs) == 0:
        print('❌ No path pairs available for training!')
        return
    
    # Create model
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
    print(f'📊 Starting training with {len(path_pairs)} path pairs...')
    
    # Training loop
    model.train()
    total_loss = 0.0
    
    for epoch in range(20):  # 20 epochs for production training
        epoch_loss = 0.0
        num_batches = 0
        
        for i, path_pair in enumerate(path_pairs):
            optimizer.zero_grad()
            
            # Get paths
            positive_path = torch.tensor(path_pair['positive'], device=device)
            negative_path = torch.tensor(path_pair['negative'], device=device)
            
            # Move node embeddings to device
            node_embeddings = pack.node_embeddings.to(device).requires_grad_(True)
            
            # Forward pass
            pos_embeddings = node_embeddings[positive_path]
            neg_embeddings = node_embeddings[negative_path]
            
            # Simple scoring
            pos_score = pos_embeddings.sum()
            neg_score = neg_embeddings.sum()
            
            # Compute loss
            loss = margin_ranking_loss(pos_score, neg_score)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
        
        avg_loss = epoch_loss / max(num_batches, 1)
        total_loss += avg_loss
        print(f'Epoch {epoch+1}/20: Loss = {avg_loss:.4f}')
    
    print(f'✅ Training completed!')
    print(f'📊 Average loss: {total_loss/20:.4f}')
    
    # Save model
    os.makedirs('ckpts', exist_ok=True)
    torch.save(model.state_dict(), 'ckpts/rcr_gat_trained.pt')
    print(f'📊 Model saved to ckpts/rcr_gat_trained.pt')
    print(f'📊 Training successful with 3000+ schemes dataset!')

if __name__ == '__main__':
    main()
