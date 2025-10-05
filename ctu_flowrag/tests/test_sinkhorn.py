"""
Tests for Sinkhorn algorithm functionality.

Tests balanced and unbalanced Sinkhorn with synthetic score matrices.
"""

import unittest
import torch
import torch.nn as nn
import sys
import os
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from models.sinkhorn import UnbalancedSinkhorn, BalancedSinkhorn, CSRA

class TestUnbalancedSinkhorn(unittest.TestCase):
    """Test unbalanced Sinkhorn algorithm."""
    
    def setUp(self):
        """Set up test data."""
        self.sinkhorn = UnbalancedSinkhorn(
            tau=0.5,
            alpha=0.5,
            beta=0.5,
            iters=30,
            eps=1e-8
        )
        
        # Create synthetic score matrix
        self.S = torch.randn(3, 5)  # 3 roles, 5 CTUs
        self.kappa = torch.tensor([2.0, 3.0, 1.0])  # Row capacities
        self.rho = torch.ones(5)  # Column capacities
    
    def test_forward(self):
        """Test forward pass."""
        X = self.sinkhorn(self.S, self.kappa, self.rho)
        
        # Check output shape
        self.assertEqual(X.shape, (3, 5))
        
        # Check non-negativity
        self.assertTrue(torch.all(X >= 0))
        
        # Check no NaNs
        self.assertFalse(torch.any(torch.isnan(X)))
    
    def test_row_constraints(self):
        """Test that row sums are approximately bounded by kappa."""
        X = self.sinkhorn(self.S, self.kappa, self.rho)
        
        row_sums = torch.sum(X, dim=1)
        
        # Row sums should be <= kappa (within tolerance)
        for i in range(3):
            self.assertLessEqual(row_sums[i].item(), self.kappa[i].item() + 1e-6)
    
    def test_column_constraints(self):
        """Test that column sums are approximately bounded by rho."""
        X = self.sinkhorn(self.S, self.kappa, self.rho)
        
        col_sums = torch.sum(X, dim=0)
        
        # Column sums should be <= rho (within tolerance)
        for j in range(5):
            self.assertLessEqual(col_sums[j].item(), self.rho[j].item() + 1e-6)
    
    def test_log_space_forward(self):
        """Test log-space forward pass."""
        X = self.sinkhorn.forward_log_space(self.S, self.kappa, self.rho)
        
        # Check output shape
        self.assertEqual(X.shape, (3, 5))
        
        # Check non-negativity
        self.assertTrue(torch.all(X >= 0))
        
        # Check no NaNs
        self.assertFalse(torch.any(torch.isnan(X)))
    
    def test_convergence(self):
        """Test that algorithm converges."""
        # Use a simple case that should converge quickly
        S_simple = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        kappa_simple = torch.tensor([1.0, 1.0])
        rho_simple = torch.tensor([1.0, 1.0])
        
        X = self.sinkhorn(S_simple, kappa_simple, rho_simple)
        
        # Check that result is reasonable
        self.assertTrue(torch.all(X >= 0))
        self.assertFalse(torch.any(torch.isnan(X)))

class TestBalancedSinkhorn(unittest.TestCase):
    """Test balanced Sinkhorn algorithm."""
    
    def setUp(self):
        """Set up test data."""
        self.sinkhorn = BalancedSinkhorn(
            tau=0.5,
            iters=30,
            eps=1e-8
        )
        
        # Create synthetic score matrix
        self.S = torch.randn(3, 5)  # 3 roles, 5 CTUs
        self.kappa = torch.tensor([2.0, 3.0, 1.0])  # Row capacities
        self.rho = torch.ones(5)  # Column capacities
    
    def test_forward(self):
        """Test forward pass."""
        X = self.sinkhorn(self.S, self.kappa, self.rho)
        
        # Check output shape
        self.assertEqual(X.shape, (3, 5))
        
        # Check non-negativity
        self.assertTrue(torch.all(X >= 0))
        
        # Check no NaNs
        self.assertFalse(torch.any(torch.isnan(X)))
    
    def test_row_constraints(self):
        """Test that row sums are approximately equal to kappa."""
        X = self.sinkhorn(self.S, self.kappa, self.rho)
        
        row_sums = torch.sum(X, dim=1)
        
        # Row sums should be approximately equal to kappa
        for i in range(3):
            self.assertAlmostEqual(row_sums[i].item(), self.kappa[i].item(), places=5)
    
    def test_column_constraints(self):
        """Test that column sums are approximately equal to rho."""
        X = self.sinkhorn(self.S, self.kappa, self.rho)
        
        col_sums = torch.sum(X, dim=0)
        
        # Column sums should be approximately equal to rho
        for j in range(5):
            self.assertAlmostEqual(col_sums[j].item(), self.rho[j].item(), places=5)

class TestCSRA(unittest.TestCase):
    """Test CSRA (Capacitated Soft Role Assignment)."""
    
    def setUp(self):
        """Set up test data."""
        self.num_roles = 3
        self.hidden_dim = 32
        self.csra = CSRA(
            num_roles=self.num_roles,
            hidden_dim=self.hidden_dim,
            tau=0.5,
            alpha=0.5,
            beta=0.5,
            iters=30,
            use_balanced=False
        )
        
        # Create test data
        self.num_nodes = 5
        self.node_features = torch.randn(self.num_nodes, self.hidden_dim)
        self.role_ids = torch.tensor([0, 1, 2, 0, 1], dtype=torch.long)
        
        self.capacities = {
            "role_0": 2,
            "role_1": 3,
            "role_2": 1
        }
        
        self.score_weights = {
            "w_query": 1.0,
            "w_role": 0.6,
            "w_sal": 0.2,
            "w_coh": 0.2
        }
    
    def test_forward(self):
        """Test CSRA forward pass."""
        X = self.csra(
            self.node_features,
            self.role_ids,
            self.capacities,
            self.score_weights
        )
        
        # Check output shape
        self.assertEqual(X.shape, (self.num_roles, self.num_nodes))
        
        # Check non-negativity
        self.assertTrue(torch.all(X >= 0))
        
        # Check no NaNs
        self.assertFalse(torch.any(torch.isnan(X)))
    
    def test_hard_selection(self):
        """Test hard selection."""
        X = torch.tensor([
            [0.8, 0.2, 0.1, 0.9, 0.3],
            [0.1, 0.7, 0.6, 0.2, 0.8],
            [0.3, 0.4, 0.2, 0.1, 0.5]
        ])
        
        threshold = 0.5
        hard_X = self.csra.hard_selection(X, threshold)
        
        # Check that hard selection is binary
        self.assertTrue(torch.all((hard_X == 0) | (hard_X == 1)))
        
        # Check that values above threshold are selected
        expected = (X > threshold).float()
        self.assertTrue(torch.allclose(hard_X, expected))
    
    def test_topk_selection(self):
        """Test top-k selection."""
        X = torch.tensor([
            [0.8, 0.2, 0.1, 0.9, 0.3],
            [0.1, 0.7, 0.6, 0.2, 0.8],
            [0.3, 0.4, 0.2, 0.1, 0.5]
        ])
        
        k = 2
        topk_X = self.csra.topk_selection(X, k)
        
        # Check that top-k selection is binary
        self.assertTrue(torch.all((topk_X == 0) | (topk_X == 1)))
        
        # Check that exactly k items are selected per role
        for i in range(self.num_roles):
            selected_count = torch.sum(topk_X[i]).item()
            self.assertEqual(selected_count, k)
    
    def test_score_matrix_computation(self):
        """Test score matrix computation."""
        # Get score matrix directly
        S = self.csra._compute_score_matrix(
            self.node_features,
            self.role_ids,
            self.score_weights
        )
        
        # Check output shape
        self.assertEqual(S.shape, (self.num_roles, self.num_nodes))
        
        # Check no NaNs
        self.assertFalse(torch.any(torch.isnan(S)))
    
    def test_balanced_mode(self):
        """Test balanced Sinkhorn mode."""
        csra_balanced = CSRA(
            num_roles=self.num_roles,
            hidden_dim=self.hidden_dim,
            tau=0.5,
            alpha=0.5,
            beta=0.5,
            iters=30,
            use_balanced=True
        )
        
        X = csra_balanced(
            self.node_features,
            self.role_ids,
            self.capacities,
            self.score_weights
        )
        
        # Check output shape
        self.assertEqual(X.shape, (self.num_roles, self.num_nodes))
        
        # Check non-negativity
        self.assertTrue(torch.all(X >= 0))
        
        # Check no NaNs
        self.assertFalse(torch.any(torch.isnan(X)))

class TestSinkhornNumericalStability(unittest.TestCase):
    """Test numerical stability of Sinkhorn algorithms."""
    
    def test_large_scores(self):
        """Test with large score values."""
        sinkhorn = UnbalancedSinkhorn(tau=0.1, iters=50)
        
        # Large scores
        S = torch.tensor([[10.0, 5.0], [3.0, 8.0]])
        kappa = torch.tensor([1.0, 1.0])
        rho = torch.tensor([1.0, 1.0])
        
        X = sinkhorn(S, kappa, rho)
        
        # Check no NaNs or infs
        self.assertFalse(torch.any(torch.isnan(X)))
        self.assertFalse(torch.any(torch.isinf(X)))
    
    def test_small_scores(self):
        """Test with small score values."""
        sinkhorn = UnbalancedSinkhorn(tau=0.1, iters=50)
        
        # Small scores
        S = torch.tensor([[0.01, 0.005], [0.003, 0.008]])
        kappa = torch.tensor([1.0, 1.0])
        rho = torch.tensor([1.0, 1.0])
        
        X = sinkhorn(S, kappa, rho)
        
        # Check no NaNs or infs
        self.assertFalse(torch.any(torch.isnan(X)))
        self.assertFalse(torch.any(torch.isinf(X)))
    
    def test_extreme_capacities(self):
        """Test with extreme capacity values."""
        sinkhorn = UnbalancedSinkhorn(tau=0.5, iters=50)
        
        S = torch.randn(2, 3)
        kappa = torch.tensor([0.1, 10.0])  # Very different capacities
        rho = torch.tensor([1.0, 1.0, 1.0])
        
        X = sinkhorn(S, kappa, rho)
        
        # Check no NaNs or infs
        self.assertFalse(torch.any(torch.isnan(X)))
        self.assertFalse(torch.any(torch.isinf(X)))

if __name__ == '__main__':
    unittest.main()

