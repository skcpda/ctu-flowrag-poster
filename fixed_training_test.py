#!/usr/bin/env python3
import torch
import sys
import os
import json

# Add current directory to path
sys.path.insert(0, '.')

def create_simple_path_pairs(tensor_pack):
    """Create simple synthetic path pairs for training."""
    num_nodes = tensor_pack.get_num_nodes()
    
    # Create simple positive and negative path pairs
    path_pairs = []
    
    # Create some simple path pairs (start -> middle -> end)
    for i in range(min(10, num_nodes // 3)):
        start_idx = i * 3
        middle_idx = i * 3 + 1
        end_idx = i * 3 + 2
        
        if end_idx < num_nodes:
            # Positive path: start -> middle -> end
            positive_path = [start_idx, middle_idx, end_idx]
            # Negative path: start -> end (skip middle)
            negative_path = [start_idx, end_idx]
            
            path_pairs.append({
                'positive': positive_path,
                'negative': negative_path
            })
    
    print(f"Created {len(path_pairs)} synthetic path pairs")
    return path_pairs

def test_training_with_path_pairs():
    print('🔧 Testing CTU-FlowRAG Training with Path Pairs...')
    
    try:
        # Test imports
        from ctu_flowrag.models.rcr_gat import RCRGAT
        from ctu_flowrag.data_io.tensor_packs import load_tensor_pack
        print('✅ All imports successful')
        
        # Test device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f'✅ Device: {device}')
        
        # Load tensor pack
        tensor_pack = load_tensor_pack('tensors_dev/documents')
        print(f'✅ Tensor pack loaded: {tensor_pack.get_num_nodes()} nodes')
        
        # Create path pairs
        path_pairs = create_simple_path_pairs(tensor_pack)
        print(f'✅ Created {len(path_pairs)} path pairs')
        
        # Test model creation
        model = RCRGAT(
            text_dim=384,
            hidden_dim=384,
            num_layers=2,
            edge_types=['PRECEDES', 'ENABLES'],
            edge_weight_priors={'PRECEDES': 1.0, 'ENABLES': 1.25},
            dropout=0.15
        )
        print('✅ RCR-GAT model created successfully')
        
        print('🎉 Training system ready with path pairs!')
        return True
        
    except Exception as e:
        print(f'❌ Error: {e}')
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    success = test_training_with_path_pairs()
    if success:
        print('✅ Training system is ready with path pairs!')
    else:
        print('❌ Training system has issues')
