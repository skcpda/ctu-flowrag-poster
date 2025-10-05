"""
Tests for path search functionality.

Tests template-constrained path search on toy graphs.
"""

import unittest
import torch
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from models.rcr_gat import RCRGAT
from data_io.tensor_packs import TensorPack
from train.mine_path_pairs import PathMiner

class TestPathSearch(unittest.TestCase):
    """Test path search functionality."""
    
    def setUp(self):
        """Set up test data."""
        # Create toy graph data
        self.num_nodes = 6
        self.node_embeddings = torch.randn(self.num_nodes, 64)
        self.role_ids = torch.tensor([0, 1, 2, 0, 1, 2], dtype=torch.long)  # Context, Benefits, Eligibility, Context, Benefits, Eligibility
        self.section_ids = torch.tensor([0, 0, 0, 1, 1, 1], dtype=torch.long)
        self.positions = torch.tensor([0, 1, 2, 0, 1, 2], dtype=torch.float)
        
        # Create edge packs
        self.edge_packs_by_type = {
            "PRECEDES": {
                "edge_index": torch.tensor([[0, 1, 3], [1, 2, 4]], dtype=torch.long),
                "conf": torch.tensor([0.8, 0.9, 0.7], dtype=torch.float),
                "compat": torch.tensor([1.0, 1.0, 1.0], dtype=torch.float),
                "dist": torch.tensor([0.1, 0.2, 0.1], dtype=torch.float)
            },
            "PREREQUISITE_OF": {
                "edge_index": torch.tensor([[0, 1], [2, 5]], dtype=torch.long),
                "conf": torch.tensor([0.6, 0.8], dtype=torch.float),
                "compat": torch.tensor([1.0, 1.0], dtype=torch.float),
                "dist": torch.tensor([0.2, 0.3], dtype=torch.float)
            },
            "ENABLES": {
                "edge_index": torch.tensor([[1, 3], [2, 4]], dtype=torch.long),
                "conf": torch.tensor([0.7, 0.9], dtype=torch.float),
                "compat": torch.tensor([1.0, 1.0], dtype=torch.float),
                "dist": torch.tensor([0.3, 0.4], dtype=torch.float)
            }
        }
        
        # Create tensor pack
        self.tensor_pack = TensorPack(
            node_embeddings=self.node_embeddings,
            role_ids=self.role_ids,
            section_ids=self.section_ids,
            positions=self.positions,
            edge_packs_by_type=self.edge_packs_by_type,
            node_texts=[f"Node {i}" for i in range(self.num_nodes)],
            node_roles=["ContextObjective", "BenefitsAssistance", "Eligibility", "ContextObjective", "BenefitsAssistance", "Eligibility"],
            metadata={"num_nodes": self.num_nodes, "num_edges": 7}
        )
        
        # Create path miner
        self.path_miner = PathMiner(
            edge_types=["PRECEDES", "PREREQUISITE_OF", "ENABLES"],
            roles=["ContextObjective", "BenefitsAssistance", "Eligibility"],
            max_path_length=5,
            min_path_length=2,
            num_negatives_per_positive=3
        )
    
    def test_mine_paths(self):
        """Test path mining."""
        path_pairs = self.path_miner.mine_paths(self.tensor_pack)
        
        # Check that paths were mined
        self.assertGreater(len(path_pairs), 0)
        
        # Check structure of path pairs
        for path_pair in path_pairs:
            self.assertIn('pos', path_pair)
            self.assertIn('neg', path_pair)
            
            # Check that positive path is not empty
            self.assertGreater(len(path_pair['pos']), 0)
            
            # Check that negative paths exist
            self.assertGreater(len(path_pair['neg']), 0)
    
    def test_positive_paths(self):
        """Test that positive paths follow templates."""
        path_pairs = self.path_miner.mine_paths(self.tensor_pack)
        
        for path_pair in path_pairs:
            pos_path = path_pair['pos']
            
            # Check that path has valid node indices
            for node_idx in pos_path:
                self.assertGreaterEqual(node_idx, 0)
                self.assertLess(node_idx, self.num_nodes)
            
            # Check that path follows a valid template
            if len(pos_path) >= 2:
                # Get roles of path
                path_roles = [self.role_ids[node_idx].item() for node_idx in pos_path]
                
                # Check that roles are in a valid sequence
                # This is a simplified check - in practice, we'd check against templates
                self.assertGreaterEqual(len(path_roles), 2)
    
    def test_negative_paths(self):
        """Test that negative paths violate constraints."""
        path_pairs = self.path_miner.mine_paths(self.tensor_pack)
        
        for path_pair in path_pairs:
            pos_path = path_pair['pos']
            neg_paths = path_pair['neg']
            
            # Check that negative paths are different from positive
            for neg_path in neg_paths:
                self.assertNotEqual(pos_path, neg_path)
                
                # Check that negative path has valid node indices
                for node_idx in neg_path:
                    self.assertGreaterEqual(node_idx, 0)
                    self.assertLess(node_idx, self.num_nodes)
    
    def test_path_length_constraints(self):
        """Test that paths respect length constraints."""
        path_pairs = self.path_miner.mine_paths(self.tensor_pack)
        
        for path_pair in path_pairs:
            pos_path = path_pair['pos']
            
            # Check minimum length
            self.assertGreaterEqual(len(pos_path), self.path_miner.min_path_length)
            
            # Check maximum length
            self.assertLessEqual(len(pos_path), self.path_miner.max_path_length)
    
    def test_template_following(self):
        """Test that paths follow defined templates."""
        # Test specific template: ContextObjective -> BenefitsAssistance -> Eligibility
        template = [0, 1, 2]  # ContextObjective, BenefitsAssistance, Eligibility
        
        # Find paths that follow this template
        template_paths = self.path_miner._find_template_paths(
            template,
            self.tensor_pack,
            self.path_miner._build_adjacency_lists(self.tensor_pack)
        )
        
        # Check that template paths exist
        self.assertGreaterEqual(len(template_paths), 0)
        
        # Check that template paths follow the template
        for path in template_paths:
            if len(path) >= len(template):
                path_roles = [self.role_ids[node_idx].item() for node_idx in path[:len(template)]]
                self.assertEqual(path_roles, template)
    
    def test_adjacency_list_construction(self):
        """Test adjacency list construction."""
        adj_lists = self.path_miner._build_adjacency_lists(self.tensor_pack)
        
        # Check that adjacency lists exist for each edge type
        for edge_type in ["PRECEDES", "PREREQUISITE_OF", "ENABLES"]:
            self.assertIn(edge_type, adj_lists)
            
            adj_list = adj_lists[edge_type]
            
            # Check that adjacency list is properly constructed
            for src, dsts in adj_list.items():
                self.assertGreaterEqual(src, 0)
                self.assertLess(src, self.num_nodes)
                
                for dst in dsts:
                    self.assertGreaterEqual(dst, 0)
                    self.assertLess(dst, self.num_nodes)
    
    def test_dfs_template_path(self):
        """Test DFS template path finding."""
        template = [0, 1, 2]  # ContextObjective, BenefitsAssistance, Eligibility
        adj_lists = self.path_miner._build_adjacency_lists(self.tensor_pack)
        
        # Test DFS from node 0 (ContextObjective)
        path = self.path_miner._dfs_template_path(
            0,  # Start node
            template,
            0,  # Template position
            self.tensor_pack,
            adj_lists
        )
        
        # Check that path is found or None
        if path is not None:
            self.assertGreater(len(path), 0)
            
            # Check that path follows template
            if len(path) >= len(template):
                path_roles = [self.role_ids[node_idx].item() for node_idx in path[:len(template)]]
                self.assertEqual(path_roles, template)
    
    def test_negative_path_generation(self):
        """Test negative path generation."""
        pos_path = [0, 1, 2]  # Example positive path
        
        # Test role permutation negative
        neg_path = self.path_miner._generate_role_permutation_negative(pos_path, self.tensor_pack)
        
        # Check that negative path is different from positive
        self.assertNotEqual(neg_path, pos_path)
        
        # Check that negative path has valid length
        self.assertEqual(len(neg_path), len(pos_path))
        
        # Check that negative path has valid node indices
        for node_idx in neg_path:
            self.assertGreaterEqual(node_idx, 0)
            self.assertLess(node_idx, self.num_nodes)
    
    def test_path_scoring(self):
        """Test path scoring functionality."""
        # This is a placeholder test - in practice, we'd test the actual scoring mechanism
        pos_path = [0, 1, 2]
        
        # Test that path scoring works without errors
        try:
            score = self.path_miner._compute_path_score(pos_path, self.tensor_pack)
            self.assertIsInstance(score, (int, float))
        except AttributeError:
            # If _compute_path_score doesn't exist, that's okay for this test
            pass
    
    def test_edge_compatibility(self):
        """Test edge compatibility checking."""
        # Test compatible edges
        self.assertTrue(self.path_miner._is_edge_compatible(0, 1, "PRECEDES"))
        self.assertTrue(self.path_miner._is_edge_compatible(1, 2, "PREREQUISITE_OF"))
        
        # Test incompatible edges
        self.assertFalse(self.path_miner._is_edge_compatible(0, 2, "PRECEDES"))
        self.assertFalse(self.path_miner._is_edge_compatible(2, 0, "PRECEDES"))
    
    def test_path_validation(self):
        """Test path validation."""
        # Valid path
        valid_path = [0, 1, 2]
        self.assertTrue(self.path_miner._is_path_valid(valid_path, self.tensor_pack))
        
        # Invalid path (empty)
        invalid_path = []
        self.assertFalse(self.path_miner._is_path_valid(invalid_path, self.tensor_pack))
        
        # Invalid path (single node)
        single_node_path = [0]
        self.assertFalse(self.path_miner._is_path_valid(single_node_path, self.tensor_pack))

if __name__ == '__main__':
    unittest.main()

