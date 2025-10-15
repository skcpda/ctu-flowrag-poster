import torch
import sys
import os

# Add current directory to path
sys.path.insert(0, '.')

try:
    from ctu_flowrag.models.rcr_gat import RCRGAT
    from ctu_flowrag.models.sinkhorn import UnbalancedSinkhorn
    print('✅ All imports successful')
    
    # Test CUDA
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
    print('✅ Model created successfully')
    
    # Test Sinkhorn
    sinkhorn = UnbalancedSinkhorn(tau=0.5, alpha=0.5, beta=0.5, iters=10)
    print('✅ Sinkhorn created successfully')
    
    print('🎉 All tests passed! System is working.')
    
except Exception as e:
    print(f'❌ Error: {e}')
    import traceback
    traceback.print_exc()
