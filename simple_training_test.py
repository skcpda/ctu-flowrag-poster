#!/usr/bin/env python3
import torch
import sys
import os

# Add current directory to path
sys.path.insert(0, '.')

def test_training_components():
    print('🔧 Testing CTU-FlowRAG Training Components...')
    
    try:
        # Test imports
        from ctu_flowrag.models.rcr_gat import RCRGAT
        from ctu_flowrag.models.sinkhorn import UnbalancedSinkhorn
        from ctu_flowrag.data_io.tensor_packs import load_tensor_pack
        print('✅ All imports successful')
        
        # Test device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f'✅ Device: {device}')
        
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
        
        # Test tensor loading
        tensor_pack = load_tensor_pack('tensors_dev/documents')
        print(f'✅ Tensor pack loaded successfully')
        
        # Test model forward pass
        model.eval()
        with torch.no_grad():
            # Create dummy input
            batch_size = 1
            num_nodes = tensor_pack.node_features.shape[0]
            dummy_input = torch.randn(batch_size, num_nodes, 384).to(device)
            
            # Forward pass
            output = model(dummy_input, tensor_pack)
            print(f'✅ Model forward pass successful: {output.shape}')
        
        print('🎉 All training components working!')
        return True
        
    except Exception as e:
        print(f'❌ Error: {e}')
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    success = test_training_components()
    if success:
        print('✅ Training system is ready!')
    else:
        print('❌ Training system has issues')
