"""
Tests for RCR-GAT layer functionality.

Constructs a toy 6-node graph and verifies attention mechanisms.
"""

import unittest
import torch
import torch.nn as nn
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from models.rcr_gat import RCRGATLayer, RCRGAT

class TestRCRGATLayer(unittest.TestCase):
    """Test RCR-GAT layer functionality."""
    
    def setUp(self):
        """Set up test data."""
        self.input_dim = 64
        self.hidden_dim = 32
        self.edge_types = ["PRECEDES", "PREREQUISITE_OF", "ENABLES"]
        self.edge_weight_priors = {
            "PRECEDES": 1.0,
            "PREREQUISITE_OF": 1.25,
            "ENABLES": 1.25
        }
        self.beta_conf = 1.0
        self.gamma_compat = 0.5
        self.distance_lambda = 0.12
        self.dropout = 0.15
        
        # Create layer
        self.layer = RCRGATLayer(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            edge_types=self.edge_types,
            edge_weight_priors=self.edge_weight_priors,
            beta_conf=self.beta_conf,
            gamma_compat=self.gamma_compat,
            distance_lambda=self.distance_lambda,
            dropout=self.dropout
        )
        
        # Create toy graph data
        self.num_nodes = 6
        self.node_feats = torch.randn(self.num_nodes, self.input_dim)
        
        # Create edge packs
        self.edge_packs_by_type = {
            "PRECEDES": {
                "edge_index": torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long),
                "conf": torch.tensor([0.8, 0.9, 0.7], dtype=torch.float),
                "compat": torch.tensor([1.0, 1.0, 1.0], dtype=torch.float),
                "dist": torch.tensor([0.1, 0.2, 0.3], dtype=torch.float)
            },
            "PREREQUISITE_OF": {
                "edge_index": torch.tensor([[0, 2], [2, 4]], dtype=torch.long),
                "conf": torch.tensor([0.6, 0.8], dtype=torch.float),
                "compat": torch.tensor([1.0, 1.0], dtype=torch.float),
                "dist": torch.tensor([0.2, 0.4], dtype=torch.float)
            },
            "ENABLES": {
                "edge_index": torch.tensor([[1, 3], [3, 5]], dtype=torch.long),
                "conf": torch.tensor([0.7, 0.9], dtype=torch.float),
                "compat": torch.tensor([1.0, 1.0], dtype=torch.float),
                "dist": torch.tensor([0.3, 0.5], dtype=torch.float)
            }
        }
    
    def test_layer_forward(self):
        """Test layer forward pass."""
        output = self.layer(self.node_feats, self.edge_packs_by_type)
        
        # Check output shape
        self.assertEqual(output.shape, (self.num_nodes, self.hidden_dim))
        
        # Check output is not all zeros
        self.assertFalse(torch.allclose(output, torch.zeros_like(output)))
    
    def test_attention_weights(self):
        """Test that attention weights are computed correctly."""
        # Get attention weights
        attention_weights = self.layer.get_attention_weights(
            self.node_feats, 
            self.edge_packs_by_type
        )
        
        # Check that attention weights exist for each edge type
        for edge_type in self.edge_types:
            self.assertIn(edge_type, attention_weights)
            
            weights = attention_weights[edge_type]
            edge_index = self.edge_packs_by_type[edge_type]["edge_index"]
            
            # Check shape
            self.assertEqual(weights.size(0), edge_index.size(1))
            
            # Check weights are non-negative
            self.assertTrue(torch.all(weights >= 0))
    
    def test_per_source_softmax(self):
        """Test that per-source softmax sums to approximately 1."""
        # Get attention weights
        attention_weights = self.layer.get_attention_weights(
            self.node_feats, 
            self.edge_packs_by_type
        )
        
        for edge_type in self.edge_types:
            weights = attention_weights[edge_type]
            edge_index = self.edge_packs_by_type[edge_type]["edge_index"]
            
            # Group weights by source node
            src_indices = edge_index[0]
            for src in src_indices.unique():
                src_mask = (src_indices == src)
                src_weights = weights[src_mask]
                
                # Sum should be approximately 1
                weight_sum = src_weights.sum().item()
                self.assertAlmostEqual(weight_sum, 1.0, places=5)
    
    def test_confidence_effect(self):
        """Test that higher confidence edges get higher attention."""
        # Create edge pack with varying confidence
        edge_pack = {
            "edge_index": torch.tensor([[0, 0], [1, 2]], dtype=torch.long),
            "conf": torch.tensor([0.9, 0.1], dtype=torch.float),  # High and low confidence
            "compat": torch.tensor([1.0, 1.0], dtype=torch.float),
            "dist": torch.tensor([0.1, 0.1], dtype=torch.float)
        }
        
        # Get attention weights
        attention_weights = self.layer.get_attention_weights(
            self.node_feats, 
            {"PRECEDES": edge_pack}
        )
        
        weights = attention_weights["PRECEDES"]
        
        # Higher confidence edge should have higher attention
        self.assertGreater(weights[0].item(), weights[1].item())
    
    def test_compatibility_effect(self):
        """Test that compatible edges get higher attention."""
        # Create edge pack with varying compatibility
        edge_pack = {
            "edge_index": torch.tensor([[0, 0], [1, 2]], dtype=torch.long),
            "conf": torch.tensor([0.5, 0.5], dtype=torch.float),
            "compat": torch.tensor([1.0, 0.0], dtype=torch.float),  # Compatible and incompatible
            "dist": torch.tensor([0.1, 0.1], dtype=torch.float)
        }
        
        # Get attention weights
        attention_weights = self.layer.get_attention_weights(
            self.node_feats, 
            {"PRECEDES": edge_pack}
        )
        
        weights = attention_weights["PRECEDES"]
        
        # Compatible edge should have higher attention
        self.assertGreater(weights[0].item(), weights[1].item())
    
    def test_distance_penalty(self):
        """Test that distance penalty reduces attention for far edges."""
        # Create edge pack with varying distances
        edge_pack = {
            "edge_index": torch.tensor([[0, 0], [1, 2]], dtype=torch.long),
            "conf": torch.tensor([0.5, 0.5], dtype=torch.float),
            "compat": torch.tensor([1.0, 1.0], dtype=torch.float),
            "dist": torch.tensor([0.1, 0.5], dtype=torch.float)  # Close and far
        }
        
        # Get attention weights
        attention_weights = self.layer.get_attention_weights(
            self.node_feats, 
            {"PRECEDES": edge_pack}
        )
        
        weights = attention_weights["PRECEDES"]
        
        # Closer edge should have higher attention
        self.assertGreater(weights[0].item(), weights[1].item())
    
    def test_type_biases(self):
        """Test that type biases affect attention."""
        # Create two edge packs with same features but different types
        edge_pack1 = {
            "edge_index": torch.tensor([[0], [1]], dtype=torch.long),
            "conf": torch.tensor([0.5], dtype=torch.float),
            "compat": torch.tensor([1.0], dtype=torch.float),
            "dist": torch.tensor([0.1], dtype=torch.float)
        }
        
        edge_pack2 = {
            "edge_index": torch.tensor([[0], [1]], dtype=torch.long),
            "conf": torch.tensor([0.5], dtype=torch.float),
            "compat": torch.tensor([1.0], dtype=torch.float),
            "dist": torch.tensor([0.1], dtype=torch.float)
        }
        
        # Get attention weights
        attention_weights1 = self.layer.get_attention_weights(
            self.node_feats, 
            {"PRECEDES": edge_pack1}
        )
        
        attention_weights2 = self.layer.get_attention_weights(
            self.node_feats, 
            {"PREREQUISITE_OF": edge_pack2}
        )
        
        # Attention weights should be different due to type biases
        # PREREQUISITE_OF has higher prior (1.25) than PRECEDES (1.0)
        self.assertNotEqual(
            attention_weights1["PRECEDES"][0].item(),
            attention_weights2["PREREQUISITE_OF"][0].item()
        )
        
        # PREREQUISITE_OF should have higher attention due to higher prior
        self.assertGreater(
            attention_weights2["PREREQUISITE_OF"][0].item(),
            attention_weights1["PRECEDES"][0].item()
        )

class TestRCRGAT(unittest.TestCase):
    """Test complete RCR-GAT model."""
    
    def setUp(self):
        """Set up test data."""
        self.text_dim = 64
        self.hidden_dim = 32
        self.num_layers = 2
        self.edge_types = ["PRECEDES", "PREREQUISITE_OF", "ENABLES"]
        self.edge_weight_priors = {
            "PRECEDES": 1.0,
            "PREREQUISITE_OF": 1.25,
            "ENABLES": 1.25
        }
        
        # Create model
        self.model = RCRGAT(
            text_dim=self.text_dim,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            edge_types=self.edge_types,
            edge_weight_priors=self.edge_weight_priors,
            beta_conf=1.0,
            gamma_compat=0.5,
            distance_lambda=0.12,
            dropout=0.15
        )
        
        # Create test data
        self.num_nodes = 6
        self.node_feats = torch.randn(self.num_nodes, self.text_dim)
        
        self.edge_packs_by_type = {
            "PRECEDES": {
                "edge_index": torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long),
                "conf": torch.tensor([0.8, 0.9, 0.7], dtype=torch.float),
                "compat": torch.tensor([1.0, 1.0, 1.0], dtype=torch.float),
                "dist": torch.tensor([0.1, 0.2, 0.3], dtype=torch.float)
            },
            "PREREQUISITE_OF": {
                "edge_index": torch.tensor([[0, 2], [2, 4]], dtype=torch.long),
                "conf": torch.tensor([0.6, 0.8], dtype=torch.float),
                "compat": torch.tensor([1.0, 1.0], dtype=torch.float),
                "dist": torch.tensor([0.2, 0.4], dtype=torch.float)
            },
            "ENABLES": {
                "edge_index": torch.tensor([[1, 3], [3, 5]], dtype=torch.long),
                "conf": torch.tensor([0.7, 0.9], dtype=torch.float),
                "compat": torch.tensor([1.0, 1.0], dtype=torch.float),
                "dist": torch.tensor([0.3, 0.5], dtype=torch.float)
            }
        }
    
    def test_model_forward(self):
        """Test model forward pass."""
        output = self.model(self.node_feats, self.edge_packs_by_type)
        
        # Check output shape
        self.assertEqual(output.shape, (self.num_nodes, self.hidden_dim))
        
        # Check output is not all zeros
        self.assertFalse(torch.allclose(output, torch.zeros_like(output)))
    
    def test_model_attention_weights(self):
        """Test model attention weights."""
        attention_weights = self.model.get_attention_weights(
            self.node_feats, 
            self.edge_packs_by_type
        )
        
        # Check that attention weights exist for each edge type
        for edge_type in self.edge_types:
            self.assertIn(edge_type, attention_weights)
            
            weights = attention_weights[edge_type]
            edge_index = self.edge_packs_by_type[edge_type]["edge_index"]
            
            # Check shape
            self.assertEqual(weights.size(0), edge_index.size(1))
            
            # Check weights are non-negative
            self.assertTrue(torch.all(weights >= 0))
    
    def test_model_parameters(self):
        """Test that model has learnable parameters."""
        num_params = sum(p.numel() for p in self.model.parameters())
        self.assertGreater(num_params, 0)
        
        # Check that parameters are learnable
        for name, param in self.model.named_parameters():
            self.assertTrue(param.requires_grad)
    
    def test_model_gradient_flow(self):
        """Test that gradients flow through the model."""
        output = self.model(self.node_feats, self.edge_packs_by_type)
        
        # Compute loss
        loss = output.sum()
        
        # Backward pass
        loss.backward()
        
        # Check that gradients exist
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                self.assertFalse(torch.allclose(param.grad, torch.zeros_like(param.grad)))

if __name__ == '__main__':
    unittest.main()

