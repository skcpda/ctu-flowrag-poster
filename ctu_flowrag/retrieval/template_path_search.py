"""
Template-constrained path search implementation.

Implements beam search for finding paths that follow role templates
using RCR-GAT contextualized features and CSRA assignments.
"""

import torch
import torch.nn as nn
import yaml
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import json
import numpy as np
from tqdm import tqdm

from ..models.rcr_gat import RCRGAT
from ..models.sinkhorn import CSRA
from ..data_io.tensor_packs import load_tensor_pack
from ..data_io.compat_allowlist import get_role_id, get_role_name

logger = logging.getLogger(__name__)

class PathSearchBeam:
    """Beam for path search."""
    
    def __init__(self, 
                 path: List[int],
                 score: float,
                 role_pos: int,
                 template: List[int]):
        self.path = path
        self.score = score
        self.role_pos = role_pos
        self.template = template
    
    def is_complete(self) -> bool:
        """Check if path is complete."""
        return self.role_pos >= len(self.template)
    
    def get_current_role(self) -> int:
        """Get current target role."""
        if self.role_pos < len(self.template):
            return self.template[self.role_pos]
        return -1

class TemplatePathSearch:
    """
    Template-constrained path search using beam search.
    
    Finds paths that follow role templates using RCR-GAT features
    and CSRA assignments.
    """
    
    def __init__(self, 
                 model: RCRGAT,
                 csra: CSRA,
                 device: torch.device,
                 edge_types: List[str],
                 roles: List[str],
                 beam_size: int = 32):
        self.model = model
        self.csra = csra
        self.device = device
        self.edge_types = edge_types
        self.roles = roles
        self.beam_size = beam_size
        
        # Role to ID mapping
        self.role_to_id = {role: i for i, role in enumerate(roles)}
        self.id_to_role = {i: role for i, role in enumerate(roles)}
    
    def search_paths(self, 
                    tensor_pack: TensorPack,
                    template: List[str],
                    num_paths: int = 5) -> List[Dict[str, Any]]:
        """
        Search for paths following a template.
        
        Args:
            tensor_pack: TensorPack with graph data
            template: List of role names in template
            num_paths: Number of paths to return
            
        Returns:
            List of best paths with scores and details
        """
        # Convert template to role IDs
        template_ids = [self.role_to_id[role] for role in template]
        
        # Get contextualized features
        with torch.no_grad():
            h = self.model(tensor_pack.node_embeddings, tensor_pack.edge_packs_by_type)
        
        # Get CSRA assignments
        X = self.csra(
            h,
            tensor_pack.role_ids,
            {},  # Capacities - use default
            {}   # Score weights - use default
        )
        
        # Initialize beam search
        beams = [PathSearchBeam([], 0.0, 0, template_ids)]
        
        # Build adjacency lists
        adj_lists = self._build_adjacency_lists(tensor_pack)
        
        # Beam search iterations
        for step in range(len(template_ids) * 2):  # Allow some flexibility
            new_beams = []
            
            for beam in beams:
                if beam.is_complete():
                    new_beams.append(beam)
                    continue
                
                # Get current target role
                target_role = beam.get_current_role()
                
                # Find candidate nodes
                candidates = self._find_candidates(
                    beam, 
                    target_role, 
                    tensor_pack, 
                    adj_lists, 
                    h, 
                    X
                )
                
                # Add candidates to new beams
                for node_idx, score in candidates:
                    new_beam = PathSearchBeam(
                        path=beam.path + [node_idx],
                        score=beam.score + score,
                        role_pos=beam.role_pos + 1,
                        template=template_ids
                    )
                    new_beams.append(new_beam)
            
            # Keep top beams
            beams = sorted(new_beams, key=lambda b: b.score, reverse=True)[:self.beam_size]
            
            # Check if all beams are complete
            if all(beam.is_complete() for beam in beams):
                break
        
        # Filter complete paths and return best ones
        complete_paths = [beam for beam in beams if beam.is_complete()]
        complete_paths = sorted(complete_paths, key=lambda b: b.score, reverse=True)
        
        # Convert to result format
        results = []
        for i, beam in enumerate(complete_paths[:num_paths]):
            result = {
                'path': beam.path,
                'score': beam.score,
                'roles': [tensor_pack.node_roles[node_idx] for node_idx in beam.path],
                'texts': [tensor_pack.node_texts[node_idx] for node_idx in beam.path],
                'rank': i + 1
            }
            results.append(result)
        
        return results
    
    def _build_adjacency_lists(self, tensor_pack: TensorPack) -> Dict[str, Dict[int, List[int]]]:
        """Build adjacency lists for each edge type."""
        adj_lists = {}
        
        for edge_type, edge_pack in tensor_pack.edge_packs_by_type.items():
            edge_index = edge_pack['edge_index']
            adj_list = {}
            
            for i in range(edge_index.size(1)):
                src = edge_index[0, i].item()
                dst = edge_index[1, i].item()
                
                if src not in adj_list:
                    adj_list[src] = []
                adj_list[src].append(dst)
            
            adj_lists[edge_type] = adj_list
        
        return adj_lists
    
    def _find_candidates(self, 
                        beam: PathSearchBeam,
                        target_role: int,
                        tensor_pack: TensorPack,
                        adj_lists: Dict[str, Dict[int, List[int]]],
                        h: torch.Tensor,
                        X: torch.Tensor) -> List[Tuple[int, float]]:
        """
        Find candidate nodes for the next step.
        
        Args:
            beam: Current beam
            target_role: Target role ID
            tensor_pack: TensorPack with graph data
            adj_lists: Adjacency lists by edge type
            h: Contextualized features
            X: CSRA assignment matrix
            
        Returns:
            List of (node_idx, score) tuples
        """
        candidates = []
        
        if not beam.path:
            # First step: find nodes with target role
            for node_idx in range(tensor_pack.get_num_nodes()):
                if tensor_pack.role_ids[node_idx].item() == target_role:
                    # Score based on CSRA assignment
                    score = X[target_role, node_idx].item()
                    candidates.append((node_idx, score))
        else:
            # Subsequent steps: find nodes reachable from current path
            current_node = beam.path[-1]
            current_role = tensor_pack.role_ids[current_node].item()
            
            # Look for edges that can connect to target role
            for edge_type, adj_list in adj_lists.items():
                if current_node not in adj_list:
                    continue
                
                for next_node in adj_list[current_node]:
                    next_role = tensor_pack.role_ids[next_node].item()
                    
                    # Check if next node has target role
                    if next_role == target_role:
                        # Compute score
                        score = self._compute_edge_score(
                            current_node, 
                            next_node, 
                            edge_type, 
                            tensor_pack, 
                            h, 
                            X
                        )
                        candidates.append((next_node, score))
        
        # Sort by score and return top candidates
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[:self.beam_size]
    
    def _compute_edge_score(self, 
                           src_node: int,
                           dst_node: int,
                           edge_type: str,
                           tensor_pack: TensorPack,
                           h: torch.Tensor,
                           X: torch.Tensor) -> float:
        """
        Compute score for an edge.
        
        Args:
            src_node: Source node index
            dst_node: Target node index
            edge_type: Edge type
            tensor_pack: TensorPack with graph data
            h: Contextualized features
            X: CSRA assignment matrix
            
        Returns:
            Edge score
        """
        # Base score from CSRA assignment
        dst_role = tensor_pack.role_ids[dst_node].item()
        base_score = X[dst_role, dst_node].item()
        
        # Add feature similarity
        src_feat = h[src_node]
        dst_feat = h[dst_node]
        similarity = torch.cosine_similarity(src_feat, dst_feat, dim=0).item()
        
        # Add edge type weight
        edge_weight = 1.0  # Default weight
        if edge_type == "PRECEDES":
            edge_weight = 1.0
        elif edge_type == "PREREQUISITE_OF":
            edge_weight = 1.25
        elif edge_type == "ENABLES":
            edge_weight = 1.25
        elif edge_type == "SEGMENT_CONTINUATION":
            edge_weight = 1.05
        
        # Combine scores
        total_score = base_score + 0.1 * similarity + 0.1 * edge_weight
        
        return total_score

def load_model_and_csra(config_path: str, 
                       checkpoint_path: str, 
                       device: torch.device) -> Tuple[RCRGAT, CSRA]:
    """
    Load trained model and CSRA.
    
    Args:
        config_path: Path to config file
        checkpoint_path: Path to model checkpoint
        device: Device to load on
        
    Returns:
        Tuple of (model, csra)
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
    
    # Create CSRA
    csra = CSRA(
        num_roles=len(config['roles']),
        hidden_dim=config['model']['hidden_dim'],
        tau=0.5,
        alpha=0.5,
        beta=0.5,
        iters=30,
        use_balanced=False
    ).to(device)
    
    return model, csra

def search_document_paths(tensor_pack: TensorPack,
                         model: RCRGAT,
                         csra: CSRA,
                         device: torch.device,
                         edge_types: List[str],
                         roles: List[str],
                         doc_id: str) -> Dict[str, Any]:
    """
    Search paths for a single document.
    
    Args:
        tensor_pack: TensorPack with document data
        model: RCRGAT model
        csra: CSRA model
        device: Device to run on
        edge_types: List of edge types
        roles: List of role names
        doc_id: Document identifier
        
    Returns:
        Dictionary with search results
    """
    # Initialize path search
    path_search = TemplatePathSearch(
        model=model,
        csra=csra,
        device=device,
        edge_types=edge_types,
        roles=roles,
        beam_size=32
    )
    
    # Define templates to search
    templates = [
        ["ContextObjective", "BenefitsAssistance", "ApplicationProcess"],
        ["ContextObjective", "Eligibility", "ApplicationProcess"],
        ["BenefitsAssistance", "Eligibility", "ApplicationProcess"],
        ["ContextObjective", "BenefitsAssistance", "Eligibility", "ApplicationProcess"],
        ["ApplicationProcess", "TimelineFrequency"],
        ["ContextObjective", "AuthoritiesGovernance", "ApplicationProcess"]
    ]
    
    # Search paths for each template
    all_results = {}
    
    for template in templates:
        try:
            paths = path_search.search_paths(tensor_pack, template, num_paths=3)
            all_results["_".join(template)] = paths
        except Exception as e:
            logger.warning(f"Error searching template {template} in {doc_id}: {e}")
            all_results["_".join(template)] = []
    
    return {
        'doc_id': doc_id,
        'num_nodes': tensor_pack.get_num_nodes(),
        'num_edges': tensor_pack.get_num_edges(),
        'templates': all_results
    }

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Template-constrained path search demo')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--doc_index', type=str, required=True, help='Path to document index file')
    parser.add_argument('--tensor_dir', type=str, required=True, help='Path to tensor directory')
    parser.add_argument('--output_dir', type=str, default='demo_results', help='Output directory')
    parser.add_argument('--device', type=str, default='auto', help='Device to use')
    parser.add_argument('--num_docs', type=int, default=5, help='Number of documents to process')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Setup device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    logger.info(f"Using device: {device}")
    
    # Load model and CSRA
    model, csra = load_model_and_csra(args.config, args.checkpoint, device)
    
    # Load document list
    with open(args.doc_index, 'r') as f:
        doc_list = json.load(f)
    
    # Limit number of documents
    doc_list = doc_list[:args.num_docs]
    
    logger.info(f"Processing {len(doc_list)} documents")
    
    # Process documents
    results = []
    
    for doc_id in tqdm(doc_list, desc="Processing documents"):
        try:
            # Load tensor pack
            tensor_pack_path = Path(args.tensor_dir) / doc_id
            tensor_pack = load_tensor_pack(str(tensor_pack_path))
            tensor_pack = tensor_pack.to(device)
            
            # Search paths
            doc_results = search_document_paths(
                tensor_pack,
                model,
                csra,
                device,
                config['model']['edge_types'],
                config['roles'],
                doc_id
            )
            
            results.append(doc_results)
            
        except Exception as e:
            logger.error(f"Error processing {doc_id}: {e}")
            continue
    
    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    with open(output_dir / "path_search_results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    logger.info("Path search completed!")
    logger.info(f"Processed {len(results)} documents")
    
    # Print sample results
    for result in results[:2]:  # Show first 2 documents
        logger.info(f"\nDocument: {result['doc_id']}")
        logger.info(f"Nodes: {result['num_nodes']}, Edges: {result['num_edges']}")
        
        for template_name, paths in result['templates'].items():
            if paths:
                logger.info(f"Template {template_name}: {len(paths)} paths found")
                for i, path in enumerate(paths[:1]):  # Show first path
                    logger.info(f"  Path {i+1}: {path['roles']} (score: {path['score']:.3f})")

if __name__ == "__main__":
    main()

