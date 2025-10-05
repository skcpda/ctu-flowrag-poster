"""
Path pair mining for training RCR-GAT.

Implements positive and negative path mining using templates and policy constraints.
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Set, Optional
import random
import logging
from pathlib import Path

from ..data_io.compat_allowlist import is_compatible, get_role_id, get_role_name
from ..data_io.tensor_packs import TensorPack

logger = logging.getLogger(__name__)

class PathMiner:
    """
    Mines positive and negative path pairs for training.
    
    Positive paths follow templates (e.g., Context→Benefits→Application)
    using only allowlisted edges. Negative paths violate constraints.
    """
    
    def __init__(self, 
                 edge_types: List[str],
                 roles: List[str],
                 max_path_length: int = 5,
                 min_path_length: int = 2,
                 num_negatives_per_positive: int = 3):
        """
        Initialize path miner.
        
        Args:
            edge_types: List of edge types
            roles: List of role names
            max_path_length: Maximum path length
            min_path_length: Minimum path length
            num_negatives_per_positive: Number of negatives per positive
        """
        self.edge_types = edge_types
        self.roles = roles
        self.max_path_length = max_path_length
        self.min_path_length = min_path_length
        self.num_negatives_per_positive = num_negatives_per_positive
        
        # Define path templates
        self.templates = self._define_templates()
        
        # Role to ID mapping
        self.role_to_id = {role: i for i, role in enumerate(roles)}
        self.id_to_role = {i: role for i, role in enumerate(roles)}
    
    def _define_templates(self) -> List[List[str]]:
        """
        Define path templates for positive mining.
        
        Returns:
            List of role sequences that form valid templates
        """
        templates = [
            # Basic flow templates
            ["ContextObjective", "BenefitsAssistance", "ApplicationProcess"],
            ["ContextObjective", "Eligibility", "ApplicationProcess"],
            ["BenefitsAssistance", "Eligibility", "ApplicationProcess"],
            ["ContextObjective", "BenefitsAssistance", "Eligibility", "ApplicationProcess"],
            
            # Authority flow
            ["ContextObjective", "AuthoritiesGovernance", "ApplicationProcess"],
            ["BenefitsAssistance", "AuthoritiesGovernance", "ApplicationProcess"],
            
            # Timeline flow
            ["ApplicationProcess", "TimelineFrequency"],
            ["ContextObjective", "ApplicationProcess", "TimelineFrequency"],
            
            # Definition flow
            ["DefinitionsReferences", "ContextObjective"],
            ["DefinitionsReferences", "ContextObjective", "BenefitsAssistance"],
            
            # Extended flows
            ["ContextObjective", "BenefitsAssistance", "Eligibility", "ApplicationProcess", "TimelineFrequency"],
            ["ContextObjective", "BenefitsAssistance", "AuthoritiesGovernance", "ApplicationProcess"],
        ]
        
        return templates
    
    def mine_paths(self, tensor_pack: TensorPack) -> List[Dict[str, List[int]]]:
        """
        Mine positive and negative paths from a tensor pack.
        
        Args:
            tensor_pack: TensorPack with graph data
            
        Returns:
            List of path pairs with 'pos' and 'neg' keys
        """
        # Build adjacency lists by edge type
        adj_lists = self._build_adjacency_lists(tensor_pack)
        
        # Find positive paths using templates
        positive_paths = self._find_positive_paths(tensor_pack, adj_lists)
        
        # Generate negative paths
        path_pairs = []
        for pos_path in positive_paths:
            neg_paths = self._generate_negative_paths(pos_path, tensor_pack, adj_lists)
            path_pairs.append({
                'pos': pos_path,
                'neg': neg_paths
            })
        
        logger.info(f"Mined {len(positive_paths)} positive paths with {len(path_pairs)} pairs")
        return path_pairs
    
    def _build_adjacency_lists(self, tensor_pack: TensorPack) -> Dict[str, Dict[int, List[int]]]:
        """
        Build adjacency lists for each edge type.
        
        Args:
            tensor_pack: TensorPack with graph data
            
        Returns:
            Dict mapping edge types to adjacency lists
        """
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
    
    def _find_positive_paths(self, 
                           tensor_pack: TensorPack,
                           adj_lists: Dict[str, Dict[int, List[int]]]) -> List[List[int]]:
        """
        Find positive paths using templates.
        
        Args:
            tensor_pack: TensorPack with graph data
            adj_lists: Adjacency lists by edge type
            
        Returns:
            List of positive paths (node ID sequences)
        """
        positive_paths = []
        
        for template in self.templates:
            # Convert template to role IDs
            template_role_ids = [self.role_to_id[role] for role in template]
            
            # Find paths that follow this template
            template_paths = self._find_template_paths(
                template_role_ids, 
                tensor_pack, 
                adj_lists
            )
            
            positive_paths.extend(template_paths)
        
        return positive_paths
    
    def _find_template_paths(self, 
                            template_role_ids: List[int],
                            tensor_pack: TensorPack,
                            adj_lists: Dict[str, Dict[int, List[int]]]) -> List[List[int]]:
        """
        Find paths that follow a specific template.
        
        Args:
            template_role_ids: Template as list of role IDs
            tensor_pack: TensorPack with graph data
            adj_lists: Adjacency lists by edge type
            
        Returns:
            List of paths following the template
        """
        paths = []
        
        # Find all nodes with the first role
        first_role = template_role_ids[0]
        start_nodes = [
            i for i, role_id in enumerate(tensor_pack.role_ids)
            if role_id.item() == first_role
        ]
        
        # For each start node, try to find a path following the template
        for start_node in start_nodes:
            path = self._dfs_template_path(
                start_node, 
                template_role_ids, 
                0, 
                tensor_pack, 
                adj_lists
            )
            if path:
                paths.append(path)
        
        return paths
    
    def _dfs_template_path(self, 
                          current_node: int,
                          template_role_ids: List[int],
                          template_pos: int,
                          tensor_pack: TensorPack,
                          adj_lists: Dict[str, Dict[int, List[int]]],
                          visited: Optional[Set[int]] = None) -> Optional[List[int]]:
        """
        DFS to find a path following a template.
        
        Args:
            current_node: Current node index
            template_role_ids: Template as list of role IDs
            template_pos: Current position in template
            tensor_pack: TensorPack with graph data
            adj_lists: Adjacency lists by edge type
            visited: Set of visited nodes
            
        Returns:
            Path if found, None otherwise
        """
        if visited is None:
            visited = set()
        
        if template_pos >= len(template_role_ids):
            return [current_node]
        
        current_role = tensor_pack.role_ids[current_node].item()
        target_role = template_role_ids[template_pos]
        
        # Check if current node matches template
        if current_role != target_role:
            return None
        
        # If we're at the last role, return the path
        if template_pos == len(template_role_ids) - 1:
            return [current_node]
        
        # Try to find next node with the next role
        next_role = template_role_ids[template_pos + 1]
        
        # Look for edges that can connect to the next role
        for edge_type, adj_list in adj_lists.items():
            if current_node not in adj_list:
                continue
            
            for next_node in adj_list[current_node]:
                if next_node in visited:
                    continue
                
                next_node_role = tensor_pack.role_ids[next_node].item()
                
                # Check if next node has the right role
                if next_node_role == next_role:
                    # Check if edge is compatible
                    current_role_name = self.id_to_role[current_role]
                    next_role_name = self.id_to_role[next_role]
                    
                    if is_compatible(current_role_name, edge_type, next_role_name):
                        visited.add(next_node)
                        subpath = self._dfs_template_path(
                            next_node, 
                            template_role_ids, 
                            template_pos + 1, 
                            tensor_pack, 
                            adj_lists, 
                            visited
                        )
                        if subpath:
                            return [current_node] + subpath
                        visited.remove(next_node)
        
        return None
    
    def _generate_negative_paths(self, 
                                pos_path: List[int],
                                tensor_pack: TensorPack,
                                adj_lists: Dict[str, Dict[int, List[int]]]) -> List[List[int]]:
        """
        Generate negative paths for a positive path.
        
        Args:
            pos_path: Positive path (node indices)
            tensor_pack: TensorPack with graph data
            adj_lists: Adjacency lists by edge type
            
        Returns:
            List of negative paths
        """
        negatives = []
        
        # Generate different types of negatives
        for _ in range(self.num_negatives_per_positive):
            neg_type = random.choice(['role_permutation', 'cross_section', 'invalid_edge'])
            
            if neg_type == 'role_permutation':
                neg_path = self._generate_role_permutation_negative(pos_path, tensor_pack)
            elif neg_type == 'cross_section':
                neg_path = self._generate_cross_section_negative(pos_path, tensor_pack)
            elif neg_type == 'invalid_edge':
                neg_path = self._generate_invalid_edge_negative(pos_path, tensor_pack, adj_lists)
            else:
                neg_path = self._generate_random_negative(pos_path, tensor_pack)
            
            if neg_path and neg_path != pos_path:
                negatives.append(neg_path)
        
        return negatives
    
    def _generate_role_permutation_negative(self, 
                                           pos_path: List[int],
                                           tensor_pack: TensorPack) -> List[int]:
        """
        Generate negative by permuting roles.
        
        Args:
            pos_path: Positive path
            tensor_pack: TensorPack with graph data
            
        Returns:
            Negative path with permuted roles
        """
        # Get roles of positive path
        pos_roles = [tensor_pack.role_ids[node].item() for node in pos_path]
        
        # Find nodes with different roles
        neg_path = []
        for i, target_role in enumerate(pos_roles):
            # Find a node with a different role
            candidates = [
                j for j, role_id in enumerate(tensor_pack.role_ids)
                if role_id.item() != target_role
            ]
            
            if candidates:
                neg_path.append(random.choice(candidates))
            else:
                # Fallback to original node
                neg_path.append(pos_path[i])
        
        return neg_path
    
    def _generate_cross_section_negative(self, 
                                        pos_path: List[int],
                                        tensor_pack: TensorPack) -> List[int]:
        """
        Generate negative by using cross-section edges.
        
        Args:
            pos_path: Positive path
            tensor_pack: TensorPack with graph data
            
        Returns:
            Negative path with cross-section violations
        """
        # This is a simplified version - in practice, we'd need to track sections
        # For now, just return a random permutation
        return self._generate_role_permutation_negative(pos_path, tensor_pack)
    
    def _generate_invalid_edge_negative(self, 
                                    pos_path: List[int],
                                    tensor_pack: TensorPack,
                                    adj_lists: Dict[str, Dict[int, List[int]]]) -> List[int]:
        """
        Generate negative by using invalid edges.
        
        Args:
            pos_path: Positive path
            tensor_pack: TensorPack with graph data
            adj_lists: Adjacency lists by edge type
            
        Returns:
            Negative path with invalid edges
        """
        # This is a simplified version - in practice, we'd need to check edge validity
        # For now, just return a random permutation
        return self._generate_role_permutation_negative(pos_path, tensor_pack)
    
    def _generate_random_negative(self, 
                                 pos_path: List[int],
                                 tensor_pack: TensorPack) -> List[int]:
        """
        Generate random negative path.
        
        Args:
            pos_path: Positive path
            tensor_pack: TensorPack with graph data
            
        Returns:
            Random negative path
        """
        # Generate random path of same length
        num_nodes = tensor_pack.get_num_nodes()
        neg_path = random.choices(range(num_nodes), k=len(pos_path))
        
        return neg_path
    
    def save_path_pairs(self, 
                       path_pairs: List[Dict[str, List[int]]], 
                       output_path: str):
        """
        Save path pairs to JSON file.
        
        Args:
            path_pairs: List of path pairs
            output_path: Path to save JSON file
        """
        import json
        
        # Convert to serializable format
        serializable_pairs = []
        for pair in path_pairs:
            serializable_pairs.append({
                'pos': pair['pos'],
                'neg': pair['neg']
            })
        
        with open(output_path, 'w') as f:
            json.dump(serializable_pairs, f, indent=2)
        
        logger.info(f"Saved {len(path_pairs)} path pairs to {output_path}")
    
    def load_path_pairs(self, input_path: str) -> List[Dict[str, List[int]]]:
        """
        Load path pairs from JSON file.
        
        Args:
            input_path: Path to load JSON file from
            
        Returns:
            List of path pairs
        """
        import json
        
        with open(input_path, 'r') as f:
            serializable_pairs = json.load(f)
        
        path_pairs = []
        for pair in serializable_pairs:
            path_pairs.append({
                'pos': pair['pos'],
                'neg': pair['neg']
            })
        
        logger.info(f"Loaded {len(path_pairs)} path pairs from {input_path}")
        return path_pairs

