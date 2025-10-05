"""
Tests for data loading and tensor conversion.

Ensures no PRECEDES crosses sections, out-degree ≤ 2, and compat flags match allowlist.
"""

import unittest
import torch
import tempfile
import json
from pathlib import Path
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from data_io.load_json_graph import load_json_graph, validate_graph, get_graph_stats
from data_io.tensor_packs import build_tensor_pack, validate_tensor_pack
from data_io.compat_allowlist import is_compatible, validate_edge

class TestDataLoader(unittest.TestCase):
    """Test data loading functionality."""
    
    def setUp(self):
        """Set up test data."""
        self.test_data = {
            "scheme_name": "Test Scheme",
            "total_sentences": 5,
            "ctus": [
                {
                    "sid": 1,
                    "line_idx": 0,
                    "text": "This is a test CTU",
                    "role": "ContextObjective",
                    "confidence": 0.8,
                    "section_name": "Introduction",
                    "method": "rule_based"
                },
                {
                    "sid": 1,
                    "line_idx": 1,
                    "text": "This is another test CTU",
                    "role": "BenefitsAssistance",
                    "confidence": 0.7,
                    "section_name": "Introduction",
                    "method": "production_pipeline"
                },
                {
                    "sid": 1,
                    "line_idx": 2,
                    "text": "This is a third test CTU",
                    "role": "Eligibility",
                    "confidence": 0.9,
                    "section_name": "Introduction",
                    "method": "rule_based"
                },
                {
                    "sid": 2,
                    "line_idx": 0,
                    "text": "This is a different section CTU",
                    "role": "ApplicationProcess",
                    "confidence": 0.6,
                    "section_name": "Application",
                    "method": "production_pipeline"
                },
                {
                    "sid": 2,
                    "line_idx": 1,
                    "text": "This is another different section CTU",
                    "role": "AuthoritiesGovernance",
                    "confidence": 0.8,
                    "section_name": "Application",
                    "method": "rule_based"
                }
            ],
            "relations": [
                {
                    "ctu1": {"sid": 1, "line_idx": 0},
                    "ctu2": {"sid": 1, "line_idx": 1},
                    "type": "PRECEDES",
                    "confidence": 0.8,
                    "confidence_cal": 0.8,
                    "relation": "PRECEDES",
                    "method": "rule_based"
                },
                {
                    "ctu1": {"sid": 1, "line_idx": 1},
                    "ctu2": {"sid": 1, "line_idx": 2},
                    "type": "PREREQUISITE_OF",
                    "confidence": 0.7,
                    "confidence_cal": 0.7,
                    "relation": "PREREQUISITE_OF",
                    "method": "production_pipeline"
                },
                {
                    "ctu1": {"sid": 1, "line_idx": 2},
                    "ctu2": {"sid": 2, "line_idx": 0},
                    "type": "ENABLES",
                    "confidence": 0.6,
                    "confidence_cal": 0.6,
                    "relation": "ENABLES",
                    "method": "production_pipeline"
                },
                {
                    "ctu1": {"sid": 2, "line_idx": 0},
                    "ctu2": {"sid": 2, "line_idx": 1},
                    "type": "PRECEDES",
                    "confidence": 0.9,
                    "confidence_cal": 0.9,
                    "relation": "PRECEDES",
                    "method": "rule_based"
                }
            ]
        }
    
    def test_load_json_graph(self):
        """Test loading JSON graph."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(self.test_data, f)
            temp_path = f.name
        
        try:
            graph_data = load_json_graph(temp_path)
            
            # Check basic structure
            self.assertIn('nodes', graph_data)
            self.assertIn('edges_by_type', graph_data)
            self.assertIn('metadata', graph_data)
            
            # Check node count
            self.assertEqual(len(graph_data['nodes']), 5)
            
            # Check edge types
            self.assertIn('PRECEDES', graph_data['edges_by_type'])
            self.assertIn('PREREQUISITE_OF', graph_data['edges_by_type'])
            self.assertIn('ENABLES', graph_data['edges_by_type'])
            
        finally:
            os.unlink(temp_path)
    
    def test_validate_graph(self):
        """Test graph validation."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(self.test_data, f)
            temp_path = f.name
        
        try:
            graph_data = load_json_graph(temp_path)
            is_valid = validate_graph(graph_data)
            self.assertTrue(is_valid)
            
        finally:
            os.unlink(temp_path)
    
    def test_get_graph_stats(self):
        """Test graph statistics."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(self.test_data, f)
            temp_path = f.name
        
        try:
            graph_data = load_json_graph(temp_path)
            stats = get_graph_stats(graph_data)
            
            # Check basic stats
            self.assertEqual(stats['num_nodes'], 5)
            self.assertGreater(stats['num_edges'], 0)
            self.assertIn('role_distribution', stats)
            self.assertIn('edge_type_distribution', stats)
            
        finally:
            os.unlink(temp_path)
    
    def test_cross_section_precedes(self):
        """Test that cross-section PRECEDES edges are dropped."""
        # Create test data with cross-section PRECEDES
        cross_section_data = self.test_data.copy()
        cross_section_data['relations'].append({
            "ctu1": {"sid": 1, "line_idx": 0},
            "ctu2": {"sid": 2, "line_idx": 0},
            "type": "PRECEDES",
            "confidence": 0.5,
            "confidence_cal": 0.5,
            "relation": "PRECEDES",
            "method": "rule_based"
        })
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(cross_section_data, f)
            temp_path = f.name
        
        try:
            graph_data = load_json_graph(temp_path)
            
            # Check that cross-section PRECEDES was dropped
            precedes_edges = graph_data['edges_by_type'].get('PRECEDES', [])
            for edge in precedes_edges:
                src_node = next(n for n in graph_data['nodes'] if n['id'] == edge['src'])
                dst_node = next(n for n in graph_data['nodes'] if n['id'] == edge['dst'])
                self.assertEqual(src_node['section_id'], dst_node['section_id'])
            
        finally:
            os.unlink(temp_path)
    
    def test_out_degree_constraint(self):
        """Test that out-degree constraint is enforced."""
        # Create test data with high out-degree
        high_degree_data = self.test_data.copy()
        high_degree_data['relations'].extend([
            {
                "ctu1": {"sid": 1, "line_idx": 0},
                "ctu2": {"sid": 1, "line_idx": 1},
                "type": "SEGMENT_CONTINUATION",
                "confidence": 0.5,
                "confidence_cal": 0.5,
                "relation": "SEGMENT_CONTINUATION",
                "method": "rule_based"
            },
            {
                "ctu1": {"sid": 1, "line_idx": 0},
                "ctu2": {"sid": 1, "line_idx": 2},
                "type": "ENABLES",
                "confidence": 0.5,
                "confidence_cal": 0.5,
                "relation": "ENABLES",
                "method": "rule_based"
            }
        ])
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(high_degree_data, f)
            temp_path = f.name
        
        try:
            graph_data = load_json_graph(temp_path)
            
            # Check out-degree
            out_degree = {}
            for edge_type, edges in graph_data['edges_by_type'].items():
                for edge in edges:
                    src = edge['src']
                    out_degree[src] = out_degree.get(src, 0) + 1
            
            # All nodes should have out-degree <= 2
            for src, degree in out_degree.items():
                self.assertLessEqual(degree, 2)
            
        finally:
            os.unlink(temp_path)

class TestTensorPacks(unittest.TestCase):
    """Test tensor pack functionality."""
    
    def setUp(self):
        """Set up test data."""
        self.test_data = {
            "scheme_name": "Test Scheme",
            "total_sentences": 3,
            "ctus": [
                {
                    "sid": 1,
                    "line_idx": 0,
                    "text": "This is a test CTU",
                    "role": "ContextObjective",
                    "confidence": 0.8,
                    "section_name": "Introduction",
                    "method": "rule_based"
                },
                {
                    "sid": 1,
                    "line_idx": 1,
                    "text": "This is another test CTU",
                    "role": "BenefitsAssistance",
                    "confidence": 0.7,
                    "section_name": "Introduction",
                    "method": "production_pipeline"
                },
                {
                    "sid": 1,
                    "line_idx": 2,
                    "text": "This is a third test CTU",
                    "role": "Eligibility",
                    "confidence": 0.9,
                    "section_name": "Introduction",
                    "method": "rule_based"
                }
            ],
            "relations": [
                {
                    "ctu1": {"sid": 1, "line_idx": 0},
                    "ctu2": {"sid": 1, "line_idx": 1},
                    "type": "PRECEDES",
                    "confidence": 0.8,
                    "confidence_cal": 0.8,
                    "relation": "PRECEDES",
                    "method": "rule_based"
                },
                {
                    "ctu1": {"sid": 1, "line_idx": 1},
                    "ctu2": {"sid": 1, "line_idx": 2},
                    "type": "PREREQUISITE_OF",
                    "confidence": 0.7,
                    "confidence_cal": 0.7,
                    "relation": "PREREQUISITE_OF",
                    "method": "production_pipeline"
                }
            ]
        }
    
    def test_build_tensor_pack(self):
        """Test building tensor pack."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(self.test_data, f)
            temp_path = f.name
        
        try:
            graph_data = load_json_graph(temp_path)
            tensor_pack = build_tensor_pack(graph_data, text_dim=384)
            
            # Check basic structure
            self.assertIsInstance(tensor_pack.node_embeddings, torch.Tensor)
            self.assertIsInstance(tensor_pack.role_ids, torch.Tensor)
            self.assertIsInstance(tensor_pack.section_ids, torch.Tensor)
            self.assertIsInstance(tensor_pack.positions, torch.Tensor)
            
            # Check shapes
            self.assertEqual(tensor_pack.node_embeddings.size(0), 3)
            self.assertEqual(tensor_pack.node_embeddings.size(1), 384)
            self.assertEqual(tensor_pack.role_ids.size(0), 3)
            self.assertEqual(tensor_pack.section_ids.size(0), 3)
            self.assertEqual(tensor_pack.positions.size(0), 3)
            
            # Check edge packs
            self.assertIn('PRECEDES', tensor_pack.edge_packs_by_type)
            self.assertIn('PREREQUISITE_OF', tensor_pack.edge_packs_by_type)
            
        finally:
            os.unlink(temp_path)
    
    def test_validate_tensor_pack(self):
        """Test tensor pack validation."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(self.test_data, f)
            temp_path = f.name
        
        try:
            graph_data = load_json_graph(temp_path)
            tensor_pack = build_tensor_pack(graph_data, text_dim=384)
            
            is_valid = validate_tensor_pack(tensor_pack)
            self.assertTrue(is_valid)
            
        finally:
            os.unlink(temp_path)
    
    def test_compatibility_flags(self):
        """Test that compatibility flags match allowlist."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(self.test_data, f)
            temp_path = f.name
        
        try:
            graph_data = load_json_graph(temp_path)
            tensor_pack = build_tensor_pack(graph_data, text_dim=384)
            
            # Check compatibility flags
            for edge_type, edge_pack in tensor_pack.edge_packs_by_type.items():
                edge_index = edge_pack['edge_index']
                compat = edge_pack['compat']
                
                for i in range(edge_index.size(1)):
                    src_idx = edge_index[0, i].item()
                    dst_idx = edge_index[1, i].item()
                    
                    src_role = tensor_pack.node_roles[src_idx]
                    dst_role = tensor_pack.node_roles[dst_idx]
                    
                    expected_compat = 1.0 if is_compatible(src_role, edge_type, dst_role) else 0.0
                    actual_compat = compat[i].item()
                    
                    self.assertEqual(actual_compat, expected_compat)
            
        finally:
            os.unlink(temp_path)

class TestCompatibilityAllowlist(unittest.TestCase):
    """Test compatibility allowlist functionality."""
    
    def test_is_compatible(self):
        """Test compatibility checking."""
        # Test valid combinations
        self.assertTrue(is_compatible("ContextObjective", "PRECEDES", "ContextObjective"))
        self.assertTrue(is_compatible("ContextObjective", "PREREQUISITE_OF", "BenefitsAssistance"))
        self.assertTrue(is_compatible("BenefitsAssistance", "ENABLES", "ApplicationProcess"))
        
        # Test invalid combinations
        self.assertFalse(is_compatible("ContextObjective", "PRECEDES", "BenefitsAssistance"))
        self.assertFalse(is_compatible("BenefitsAssistance", "PRECEDES", "ContextObjective"))
        self.assertFalse(is_compatible("Eligibility", "PRECEDES", "AuthoritiesGovernance"))
    
    def test_validate_edge(self):
        """Test edge validation."""
        # Test valid edges
        self.assertTrue(validate_edge("ContextObjective", "PRECEDES", "ContextObjective"))
        self.assertTrue(validate_edge("ContextObjective", "PREREQUISITE_OF", "BenefitsAssistance"))
        
        # Test invalid edges
        self.assertFalse(validate_edge("ContextObjective", "PRECEDES", "BenefitsAssistance"))
        self.assertFalse(validate_edge("BenefitsAssistance", "PRECEDES", "ContextObjective"))

if __name__ == '__main__':
    unittest.main()

