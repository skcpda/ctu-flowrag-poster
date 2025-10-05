"""
Convert loaded graphs to tensor packs for training.

Handles the conversion from graph data to PyTorch tensors with proper
embeddings, role IDs, and edge information.
"""

import torch
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import logging
from pathlib import Path

from .load_json_graph import load_json_graph
from .compat_allowlist import is_compatible, get_role_id

logger = logging.getLogger(__name__)

class TensorPack:
    """Container for tensorized graph data."""
    
    def __init__(self, 
                 node_embeddings: torch.Tensor,
                 role_ids: torch.Tensor,
                 section_ids: torch.Tensor,
                 positions: torch.Tensor,
                 edge_packs_by_type: Dict[str, Dict[str, torch.Tensor]],
                 node_texts: List[str],
                 node_roles: List[str],
                 metadata: Dict[str, Any]):
        self.node_embeddings = node_embeddings
        self.role_ids = role_ids
        self.section_ids = section_ids
        self.positions = positions
        self.edge_packs_by_type = edge_packs_by_type
        self.node_texts = node_texts
        self.node_roles = node_roles
        self.metadata = metadata
    
    def to(self, device: torch.device):
        """Move tensors to device."""
        self.node_embeddings = self.node_embeddings.to(device)
        self.role_ids = self.role_ids.to(device)
        self.section_ids = self.section_ids.to(device)
        self.positions = self.positions.to(device)
        
        for edge_type, edge_pack in self.edge_packs_by_type.items():
            for key, tensor in edge_pack.items():
                edge_pack[key] = tensor.to(device)
        
        return self
    
    def get_num_nodes(self) -> int:
        """Get number of nodes."""
        return self.node_embeddings.size(0)
    
    def get_num_edges(self) -> int:
        """Get total number of edges."""
        return sum(edge_pack['edge_index'].size(1) for edge_pack in self.edge_packs_by_type.values())
    
    def get_edge_types(self) -> List[str]:
        """Get list of edge types."""
        return list(self.edge_packs_by_type.keys())

def create_dummy_embeddings(texts: List[str], dim: int = 384, seed: int = 42) -> torch.Tensor:
    """
    Create dummy embeddings for texts (for testing/demo purposes).
    
    Args:
        texts: List of text strings
        dim: Embedding dimension
        seed: Random seed for reproducibility
        
    Returns:
        Tensor of shape (len(texts), dim) with dummy embeddings
    """
    torch.manual_seed(seed)
    embeddings = torch.randn(len(texts), dim)
    # Normalize to unit length
    embeddings = embeddings / torch.norm(embeddings, dim=1, keepdim=True)
    return embeddings

def encode_texts(texts: List[str], 
                 embed_model: str = "e5-small", 
                 text_dim: int = 384,
                 device: str = "cpu") -> torch.Tensor:
    """
    Encode texts using the specified embedding model.
    
    Args:
        texts: List of text strings
        embed_model: Name of embedding model to use
        text_dim: Expected embedding dimension
        device: Device to run on
        
    Returns:
        Tensor of shape (len(texts), text_dim) with embeddings
    """
    # For now, use dummy embeddings
    # In production, this would load the actual embedding model
    logger.warning(f"Using dummy embeddings for {len(texts)} texts (model: {embed_model})")
    return create_dummy_embeddings(texts, text_dim)

def build_tensor_pack(graph_data: Dict[str, Any],
                      embed_model: str = "e5-small",
                      text_dim: int = 384,
                      distance_lambda: float = 0.12,
                      device: str = "cpu") -> TensorPack:
    """
    Build a tensor pack from graph data.
    
    Args:
        graph_data: Graph data from load_json_graph
        embed_model: Embedding model to use
        text_dim: Embedding dimension
        distance_lambda: Distance penalty weight
        device: Device to run on
        
    Returns:
        TensorPack with all tensors
    """
    nodes = graph_data['nodes']
    edges_by_type = graph_data['edges_by_type']
    
    # Extract node information
    node_texts = [node['text'] for node in nodes]
    node_roles = [node['role'] for node in nodes]
    node_sections = [node['section_id'] for node in nodes]
    node_positions = [node['pos'] for node in nodes]
    
    # Create embeddings
    node_embeddings = encode_texts(node_texts, embed_model, text_dim, device)
    
    # Convert roles to IDs
    role_ids = torch.tensor([get_role_id(role) for role in node_roles], dtype=torch.long)
    
    # Convert sections to IDs (simple hash)
    section_to_id = {section: i for i, section in enumerate(set(node_sections))}
    section_ids = torch.tensor([section_to_id[section] for section in node_sections], dtype=torch.long)
    
    # Convert positions to tensor
    positions = torch.tensor(node_positions, dtype=torch.float)
    
    # Build edge packs by type
    edge_packs_by_type = {}
    
    for edge_type, edges in edges_by_type.items():
        if not edges:
            continue
            
        # Extract edge information
        src_indices = []
        dst_indices = []
        confidences = []
        compatibilities = []
        distances = []
        
        for edge in edges:
            # Find node indices
            src_idx = next(i for i, node in enumerate(nodes) if node['id'] == edge['src'])
            dst_idx = next(i for i, node in enumerate(nodes) if node['id'] == edge['dst'])
            
            src_indices.append(src_idx)
            dst_indices.append(dst_idx)
            confidences.append(edge['conf_cal'])
            
            # Compute compatibility
            src_role = node_roles[src_idx]
            dst_role = node_roles[dst_idx]
            compat = 1.0 if is_compatible(src_role, edge_type, dst_role) else 0.0
            compatibilities.append(compat)
            
            # Store raw distance (lambda will be applied in model)
            distance = edge['delta_sent']
            distances.append(distance)
        
        # Create edge index tensor
        edge_index = torch.tensor([src_indices, dst_indices], dtype=torch.long)
        
        # Create other tensors
        conf = torch.tensor(confidences, dtype=torch.float)
        compat = torch.tensor(compatibilities, dtype=torch.float)
        dist = torch.tensor(distances, dtype=torch.float)
        
        edge_packs_by_type[edge_type] = {
            'edge_index': edge_index,
            'conf': conf,
            'compat': compat,
            'dist': dist
        }
    
    # Create metadata
    metadata = {
        'num_nodes': len(nodes),
        'num_edges': sum(len(edges) for edges in edges_by_type.values()),
        'edge_types': list(edges_by_type.keys()),
        'embed_model': embed_model,
        'text_dim': text_dim,
        'distance_lambda': distance_lambda
    }
    
    return TensorPack(
        node_embeddings=node_embeddings,
        role_ids=role_ids,
        section_ids=section_ids,
        positions=positions,
        edge_packs_by_type=edge_packs_by_type,
        node_texts=node_texts,
        node_roles=node_roles,
        metadata=metadata
    )

def save_tensor_pack(tensor_pack: TensorPack, output_path: str):
    """
    Save a tensor pack to disk.
    
    Args:
        tensor_pack: TensorPack to save
        output_path: Path to save to (without extension)
    """
    output_path = Path(output_path)
    
    # Save node tensors
    torch.save({
        'node_embeddings': tensor_pack.node_embeddings,
        'role_ids': tensor_pack.role_ids,
        'section_ids': tensor_pack.section_ids,
        'positions': tensor_pack.positions,
        'node_texts': tensor_pack.node_texts,
        'node_roles': tensor_pack.node_roles,
        'metadata': tensor_pack.metadata
    }, f"{output_path}_nodes.pt")
    
    # Save edge tensors
    torch.save(tensor_pack.edge_packs_by_type, f"{output_path}_edges.pt")
    
    logger.info(f"Saved tensor pack to {output_path}_nodes.pt and {output_path}_edges.pt")

def load_tensor_pack(input_path: str) -> TensorPack:
    """
    Load a tensor pack from disk.
    
    Args:
        input_path: Path to load from (without extension)
        
    Returns:
        Loaded TensorPack
    """
    input_path = Path(input_path)
    
    # Load node tensors
    node_data = torch.load(f"{input_path}_nodes.pt")
    
    # Load edge tensors
    edge_packs_by_type = torch.load(f"{input_path}_edges.pt")
    
    return TensorPack(
        node_embeddings=node_data['node_embeddings'],
        role_ids=node_data['role_ids'],
        section_ids=node_data['section_ids'],
        positions=node_data['positions'],
        edge_packs_by_type=edge_packs_by_type,
        node_texts=node_data['node_texts'],
        node_roles=node_data['node_roles'],
        metadata=node_data['metadata']
    )

def validate_tensor_pack(tensor_pack: TensorPack) -> bool:
    """
    Validate a tensor pack for consistency.
    
    Args:
        tensor_pack: TensorPack to validate
        
    Returns:
        True if valid, False otherwise
    """
    num_nodes = tensor_pack.get_num_nodes()
    
    # Check tensor shapes
    if tensor_pack.node_embeddings.size(0) != num_nodes:
        logger.error("Node embeddings size mismatch")
        return False
    
    if tensor_pack.role_ids.size(0) != num_nodes:
        logger.error("Role IDs size mismatch")
        return False
    
    if tensor_pack.section_ids.size(0) != num_nodes:
        logger.error("Section IDs size mismatch")
        return False
    
    if tensor_pack.positions.size(0) != num_nodes:
        logger.error("Positions size mismatch")
        return False
    
    # Check edge tensors
    for edge_type, edge_pack in tensor_pack.edge_packs_by_type.items():
        edge_index = edge_pack['edge_index']
        if edge_index.size(0) != 2:
            logger.error(f"Edge index for {edge_type} has wrong shape")
            return False
        
        num_edges = edge_index.size(1)
        if edge_pack['conf'].size(0) != num_edges:
            logger.error(f"Confidence tensor for {edge_type} size mismatch")
            return False
        
        if edge_pack['compat'].size(0) != num_edges:
            logger.error(f"Compatibility tensor for {edge_type} size mismatch")
            return False
        
        if edge_pack['dist'].size(0) != num_edges:
            logger.error(f"Distance tensor for {edge_type} size mismatch")
            return False
        
        # Check edge indices are valid
        if edge_index.max() >= num_nodes:
            logger.error(f"Edge index for {edge_type} has invalid node indices")
            return False
    
    return True

