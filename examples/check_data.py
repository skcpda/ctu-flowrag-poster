import torch
from ctu_flowrag.data_io.tensor_packs import TensorPack

try:
    pack = TensorPack.load('tensors_dev/aa_ctus_production_ready')
    print('✅ Tensor pack loaded successfully')
    print(f'📊 Nodes: {pack.node_embeddings.shape}')
    print(f'📊 Edges: {len(pack.edge_lists)} edge types')
    for edge_type, edges in pack.edge_lists.items():
        print(f'  {edge_type}: {edges.shape[0]} edges')
    
    # Check if we have the required attributes
    has_compat = hasattr(pack, 'compat_matrix')
    has_distances = hasattr(pack, 'distances')
    print(f'📊 Compat matrix: {pack.compat_matrix.shape if has_compat else "No compat matrix"}')
    print(f'📊 Distances: {len(pack.distances) if has_distances else "No distances"}')
    
    # Check edge structure
    if pack.edge_lists:
        first_edge_type = list(pack.edge_lists.keys())[0]
        first_edges = pack.edge_lists[first_edge_type]
        print(f'📊 First edge type {first_edge_type}: {first_edges.shape}')
        print(f'📊 Edge columns: {first_edges.shape[1]}')
        
except Exception as e:
    print(f'❌ Error loading tensor pack: {e}')
    import traceback
    traceback.print_exc()
