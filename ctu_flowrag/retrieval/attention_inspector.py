"""
Attention inspector for RCR-GAT model.

Provides functionality to inspect attention weights and analyze
which edges the model is focusing on.
"""

import torch
import torch.nn as nn
import yaml
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import json
import pandas as pd

from ..models.rcr_gat import RCRGAT
from ..data_io.tensor_packs import load_tensor_pack

logger = logging.getLogger(__name__)

class AttentionInspector:
    """
    Inspector for RCR-GAT attention weights.
    
    Provides methods to analyze attention patterns and identify
    important edges in the graph.
    """
    
    def __init__(self, 
                 model: RCRGAT,
                 device: torch.device,
                 edge_types: List[str],
                 roles: List[str]):
        self.model = model
        self.device = device
        self.edge_types = edge_types
        self.roles = roles
        
        # Role to ID mapping
        self.role_to_id = {role: i for i, role in enumerate(roles)}
        self.id_to_role = {i: role for i, role in enumerate(roles)}
    
    def inspect_topk(self, 
                    tensor_pack: TensorPack,
                    k: int = 5) -> Dict[str, List[Dict[str, Any]]]:
        """
        Inspect top-k attention weights for each node.
        
        Args:
            tensor_pack: TensorPack with graph data
            k: Number of top edges to return per node
            
        Returns:
            Dictionary mapping node indices to top-k edges
        """
        # Get attention weights
        attention_weights = self.model.get_attention_weights(
            tensor_pack.node_embeddings,
            tensor_pack.edge_packs_by_type
        )
        
        # Build edge information
        edge_info = self._build_edge_info(tensor_pack)
        
        # Find top-k edges for each node
        topk_edges = {}
        
        for node_idx in range(tensor_pack.get_num_nodes()):
            node_edges = []
            
            for edge_type, weights in attention_weights.items():
                edge_pack = tensor_pack.edge_packs_by_type[edge_type]
                edge_index = edge_pack['edge_index']
                conf = edge_pack['conf']
                compat = edge_pack['compat']
                dist = edge_pack['dist']
                
                # Find edges from this node
                for i in range(edge_index.size(1)):
                    src = edge_index[0, i].item()
                    dst = edge_index[1, i].item()
                    
                    if src == node_idx:
                        edge_data = {
                            'dst': dst,
                            'type': edge_type,
                            'weight': weights[i].item(),
                            'conf': conf[i].item(),
                            'compat': compat[i].item(),
                            'dist': dist[i].item(),
                            'dst_role': tensor_pack.node_roles[dst],
                            'dst_text': tensor_pack.node_texts[dst][:50] + "..." if len(tensor_pack.node_texts[dst]) > 50 else tensor_pack.node_texts[dst]
                        }
                        node_edges.append(edge_data)
            
            # Sort by attention weight and take top-k
            node_edges.sort(key=lambda x: x['weight'], reverse=True)
            topk_edges[node_idx] = node_edges[:k]
        
        return topk_edges
    
    def _build_edge_info(self, tensor_pack: TensorPack) -> Dict[str, List[Dict[str, Any]]]:
        """
        Build edge information dictionary.
        
        Args:
            tensor_pack: TensorPack with graph data
            
        Returns:
            Dictionary with edge information
        """
        edge_info = {}
        
        for edge_type, edge_pack in tensor_pack.edge_packs_by_type.items():
            edge_index = edge_pack['edge_index']
            conf = edge_pack['conf']
            compat = edge_pack['compat']
            dist = edge_pack['dist']
            
            edges = []
            for i in range(edge_index.size(1)):
                src = edge_index[0, i].item()
                dst = edge_index[1, i].item()
                
                edge_data = {
                    'src': src,
                    'dst': dst,
                    'type': edge_type,
                    'conf': conf[i].item(),
                    'compat': compat[i].item(),
                    'dist': dist[i].item(),
                    'src_role': tensor_pack.node_roles[src],
                    'dst_role': tensor_pack.node_roles[dst],
                    'src_text': tensor_pack.node_texts[src][:50] + "..." if len(tensor_pack.node_texts[src]) > 50 else tensor_pack.node_texts[src],
                    'dst_text': tensor_pack.node_texts[dst][:50] + "..." if len(tensor_pack.node_texts[dst]) > 50 else tensor_pack.node_texts[dst]
                }
                edges.append(edge_data)
            
            edge_info[edge_type] = edges
        
        return edge_info
    
    def analyze_attention_patterns(self, 
                                  tensor_pack: TensorPack) -> Dict[str, Any]:
        """
        Analyze attention patterns across the graph.
        
        Args:
            tensor_pack: TensorPack with graph data
            
        Returns:
            Dictionary with attention analysis
        """
        # Get attention weights
        attention_weights = self.model.get_attention_weights(
            tensor_pack.node_embeddings,
            tensor_pack.edge_packs_by_type
        )
        
        # Analyze attention by edge type
        edge_type_analysis = {}
        
        for edge_type, weights in attention_weights.items():
            if len(weights) == 0:
                continue
            
            edge_type_analysis[edge_type] = {
                'mean_weight': weights.mean().item(),
                'std_weight': weights.std().item(),
                'max_weight': weights.max().item(),
                'min_weight': weights.min().item(),
                'num_edges': len(weights)
            }
        
        # Analyze attention by role
        role_analysis = {}
        
        for node_idx in range(tensor_pack.get_num_nodes()):
            role = tensor_pack.node_roles[node_idx]
            
            if role not in role_analysis:
                role_analysis[role] = {
                    'outgoing_weights': [],
                    'incoming_weights': [],
                    'num_nodes': 0
                }
            
            role_analysis[role]['num_nodes'] += 1
            
            # Find outgoing edges
            for edge_type, weights in attention_weights.items():
                edge_pack = tensor_pack.edge_packs_by_type[edge_type]
                edge_index = edge_pack['edge_index']
                
                for i in range(edge_index.size(1)):
                    src = edge_index[0, i].item()
                    dst = edge_index[1, i].item()
                    
                    if src == node_idx:
                        role_analysis[role]['outgoing_weights'].append(weights[i].item())
                    if dst == node_idx:
                        role_analysis[role]['incoming_weights'].append(weights[i].item())
        
        # Compute statistics for each role
        for role, data in role_analysis.items():
            if data['outgoing_weights']:
                data['mean_outgoing'] = sum(data['outgoing_weights']) / len(data['outgoing_weights'])
                data['max_outgoing'] = max(data['outgoing_weights'])
            else:
                data['mean_outgoing'] = 0.0
                data['max_outgoing'] = 0.0
            
            if data['incoming_weights']:
                data['mean_incoming'] = sum(data['incoming_weights']) / len(data['incoming_weights'])
                data['max_incoming'] = max(data['incoming_weights'])
            else:
                data['mean_incoming'] = 0.0
                data['max_incoming'] = 0.0
        
        return {
            'edge_type_analysis': edge_type_analysis,
            'role_analysis': role_analysis
        }
    
    def print_attention_table(self, 
                            tensor_pack: TensorPack,
                            k: int = 5,
                            output_path: Optional[str] = None):
        """
        Print attention table for top-k edges.
        
        Args:
            tensor_pack: TensorPack with graph data
            k: Number of top edges to show per node
            output_path: Optional path to save table
        """
        topk_edges = self.inspect_topk(tensor_pack, k)
        
        # Create table data
        table_data = []
        
        for node_idx, edges in topk_edges.items():
            node_role = tensor_pack.node_roles[node_idx]
            node_text = tensor_pack.node_texts[node_idx][:50] + "..." if len(tensor_pack.node_texts[node_idx]) > 50 else tensor_pack.node_texts[node_idx]
            
            for edge in edges:
                table_data.append({
                    'src_node': node_idx,
                    'src_role': node_role,
                    'src_text': node_text,
                    'dst_node': edge['dst'],
                    'dst_role': edge['dst_role'],
                    'dst_text': edge['dst_text'],
                    'edge_type': edge['type'],
                    'attention_weight': edge['weight'],
                    'confidence': edge['conf'],
                    'compatibility': edge['compat'],
                    'distance': edge['dist']
                })
        
        # Create DataFrame
        df = pd.DataFrame(table_data)
        
        # Print table
        print("\n" + "="*120)
        print("ATTENTION INSPECTION RESULTS")
        print("="*120)
        
        for node_idx in sorted(topk_edges.keys()):
            node_edges = topk_edges[node_idx]
            if not node_edges:
                continue
            
            node_role = tensor_pack.node_roles[node_idx]
            node_text = tensor_pack.node_texts[node_idx][:100] + "..." if len(tensor_pack.node_texts[node_idx]) > 100 else tensor_pack.node_texts[node_idx]
            
            print(f"\nNode {node_idx} ({node_role}):")
            print(f"Text: {node_text}")
            print("-" * 80)
            
            for i, edge in enumerate(node_edges):
                print(f"  {i+1}. -> Node {edge['dst']} ({edge['dst_role']})")
                print(f"     Type: {edge['type']}, Weight: {edge['weight']:.4f}")
                print(f"     Conf: {edge['conf']:.3f}, Compat: {edge['compat']:.3f}, Dist: {edge['dist']:.3f}")
                print(f"     Text: {edge['dst_text']}")
                print()
        
        # Save to file if requested
        if output_path:
            df.to_csv(output_path, index=False)
            logger.info(f"Saved attention table to {output_path}")
    
    def save_attention_json(self, 
                          tensor_pack: TensorPack,
                          k: int = 5,
                          output_path: str):
        """
        Save attention results to JSON file.
        
        Args:
            tensor_pack: TensorPack with graph data
            k: Number of top edges to save per node
            output_path: Path to save JSON file
        """
        topk_edges = self.inspect_topk(tensor_pack, k)
        attention_analysis = self.analyze_attention_patterns(tensor_pack)
        
        results = {
            'doc_id': tensor_pack.metadata.get('doc_id', 'unknown'),
            'num_nodes': tensor_pack.get_num_nodes(),
            'num_edges': tensor_pack.get_num_edges(),
            'topk_edges': topk_edges,
            'attention_analysis': attention_analysis
        }
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Saved attention results to {output_path}")

def load_model(config_path: str, checkpoint_path: str, device: torch.device) -> RCRGAT:
    """
    Load trained RCR-GAT model.
    
    Args:
        config_path: Path to config file
        checkpoint_path: Path to model checkpoint
        device: Device to load on
        
    Returns:
        Loaded RCRGAT model
    """
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Load model
    model = RCRGAT(
        text_dim=config['model']['text_dim'],
        hidden_dim=config['model']['hidden_dim'],
        num_layers=config['model']['num_layers'],
        edge_types=config['model']['edge_types'],
        edge_weight_priors=config['model']['edge_weights'],
        beta_conf=config['model']['beta_conf'],
        gamma_compat=config['model']['role_compat_gamma'],
        distance_lambda=config['model']['distance_penalty_lambda'],
        dropout=config['model']['dropout']
    ).to(device)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    return model

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Inspect RCR-GAT attention weights')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--tensor_pack', type=str, required=True, help='Path to tensor pack')
    parser.add_argument('--output_dir', type=str, default='attention_results', help='Output directory')
    parser.add_argument('--k', type=int, default=5, help='Number of top edges to show per node')
    parser.add_argument('--device', type=str, default='auto', help='Device to use')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Setup device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    logger.info(f"Using device: {device}")
    
    # Load model
    model = load_model(args.config, args.checkpoint, device)
    
    # Load tensor pack
    tensor_pack = load_tensor_pack(args.tensor_pack)
    tensor_pack = tensor_pack.to(device)
    
    # Load config for roles and edge types
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Create attention inspector
    inspector = AttentionInspector(
        model=model,
        device=device,
        edge_types=config['model']['edge_types'],
        roles=config['roles']
    )
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Print attention table
    inspector.print_attention_table(
        tensor_pack, 
        k=args.k,
        output_path=str(output_dir / "attention_table.csv")
    )
    
    # Save attention JSON
    inspector.save_attention_json(
        tensor_pack,
        k=args.k,
        output_path=str(output_dir / "attention_results.json")
    )
    
    logger.info("Attention inspection completed!")

if __name__ == "__main__":
    main()

