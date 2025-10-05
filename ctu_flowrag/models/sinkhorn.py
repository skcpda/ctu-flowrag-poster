"""
Unbalanced Sinkhorn implementation for CSRA (Capacitated Soft Role Assignment).

Implements the unbalanced Sinkhorn algorithm for optimal transport with
capacity constraints and relaxation parameters.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import math
import logging

logger = logging.getLogger(__name__)

class UnbalancedSinkhorn(nn.Module):
    """
    Unbalanced Sinkhorn algorithm for optimal transport.
    
    Implements the algorithm:
    K = exp(S/tau)
    u ← (kappa / (K v))^alpha
    v ← (rho / (K^T u))^beta
    X = diag(u) K diag(v)
    """
    
    def __init__(self, 
                 tau: float = 0.5,
                 alpha: float = 0.5,
                 beta: float = 0.5,
                 iters: int = 30,
                 eps: float = 1e-8):
        """
        Initialize unbalanced Sinkhorn.
        
        Args:
            tau: Temperature parameter
            alpha: Row relaxation parameter
            beta: Column relaxation parameter
            iters: Number of iterations
            eps: Numerical stability epsilon
        """
        super().__init__()
        
        self.tau = tau
        self.alpha = alpha
        self.beta = beta
        self.iters = iters
        self.eps = eps
    
    def forward(self, 
                S: torch.Tensor,
                kappa: torch.Tensor,
                rho: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of unbalanced Sinkhorn.
        
        Args:
            S: Score matrix [R, N] where R is number of roles, N is number of CTUs
            kappa: Row capacities [R] (max items per role)
            rho: Column capacities [N] (max roles per CTU)
            
        Returns:
            Assignment matrix X [R, N]
        """
        device = S.device
        R, N = S.size()
        
        # Initialize K = exp(S/tau)
        K = torch.exp(S / self.tau)
        
        # Initialize u and v
        u = torch.ones(R, device=device)
        v = torch.ones(N, device=device)
        
        # Iterative updates
        for i in range(self.iters):
            # u ← (kappa / (K v))^alpha
            Kv = torch.mm(K, v.unsqueeze(1)).squeeze(1)
            u_new = torch.pow(kappa / (Kv + self.eps), self.alpha)
            
            # v ← (rho / (K^T u))^beta
            KTu = torch.mm(K.t(), u.unsqueeze(1)).squeeze(1)
            v_new = torch.pow(rho / (KTu + self.eps), self.beta)
            
            # Update
            u = u_new
            v = v_new
            
            # Check convergence (optional)
            if i > 0 and torch.max(torch.abs(u - u_prev)) < self.eps:
                logger.debug(f"Sinkhorn converged after {i+1} iterations")
                break
            
            u_prev = u.clone()
        
        # Compute final assignment matrix
        X = torch.diag(u) @ K @ torch.diag(v)
        
        return X
    
    def forward_log_space(self, 
                         S: torch.Tensor,
                         kappa: torch.Tensor,
                         rho: torch.Tensor) -> torch.Tensor:
        """
        Forward pass in log-space for numerical stability.
        
        Args:
            S: Score matrix [R, N]
            kappa: Row capacities [R]
            rho: Column capacities [N]
            
        Returns:
            Assignment matrix X [R, N]
        """
        device = S.device
        R, N = S.size()
        
        # Initialize in log-space with clamping
        log_K = (S / self.tau).clamp(min=-60, max=60)
        log_u = torch.zeros(R, device=device)
        log_v = torch.zeros(N, device=device)
        
        # Iterative updates in log-space
        for i in range(self.iters):
            # log_u ← alpha * log(kappa / (K v))
            log_Kv = torch.logsumexp(log_K + log_v.unsqueeze(0), dim=1)
            log_u_new = self.alpha * (torch.log(kappa + self.eps) - log_Kv)
            
            # log_v ← beta * log(rho / (K^T u))
            log_KTu = torch.logsumexp(log_K.t() + log_u.unsqueeze(0), dim=1)
            log_v_new = self.beta * (torch.log(rho + self.eps) - log_KTu)
            
            # Check convergence
            if i > 0:
                delta_u = (log_u_new - log_u).abs().mean()
                delta_v = (log_v_new - log_v).abs().mean()
                delta = delta_u + delta_v
                
                if torch.isnan(delta):
                    raise RuntimeError("Sinkhorn diverged (NaN deltas)")
                
                if delta < self.eps:
                    logger.debug(f"Sinkhorn converged after {i+1} iterations")
                    break
                    
                if i % 5 == 0:
                    logger.debug(f"Sinkhorn iteration {i+1}: delta={delta:.6f}")
            
            # Update
            log_u = log_u_new
            log_v = log_v_new
            
            log_u_prev = log_u.clone()
        
        # Compute final assignment matrix
        log_X = log_u.unsqueeze(1) + log_K + log_v.unsqueeze(0)
        X = torch.exp(log_X)
        
        return X

class BalancedSinkhorn(nn.Module):
    """
    Balanced Sinkhorn algorithm (alpha=beta=1).
    
    Special case of unbalanced Sinkhorn where row and column sums
    are exactly equal to the target capacities.
    """
    
    def __init__(self, 
                 tau: float = 0.5,
                 iters: int = 30,
                 eps: float = 1e-8):
        """
        Initialize balanced Sinkhorn.
        
        Args:
            tau: Temperature parameter
            iters: Number of iterations
            eps: Numerical stability epsilon
        """
        super().__init__()
        
        self.tau = tau
        self.iters = iters
        self.eps = eps
    
    def forward(self, 
                S: torch.Tensor,
                kappa: torch.Tensor,
                rho: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of balanced Sinkhorn.
        
        Args:
            S: Score matrix [R, N]
            kappa: Row capacities [R]
            rho: Column capacities [N]
            
        Returns:
            Assignment matrix X [R, N]
        """
        device = S.device
        R, N = S.size()
        
        # Initialize K = exp(S/tau)
        K = torch.exp(S / self.tau)
        
        # Initialize u and v
        u = torch.ones(R, device=device)
        v = torch.ones(N, device=device)
        
        # Iterative updates
        for i in range(self.iters):
            # u ← kappa / (K v)
            Kv = torch.mm(K, v.unsqueeze(1)).squeeze(1)
            u = kappa / (Kv + self.eps)
            
            # v ← rho / (K^T u)
            KTu = torch.mm(K.t(), u.unsqueeze(1)).squeeze(1)
            v = rho / (KTu + self.eps)
        
        # Compute final assignment matrix
        X = torch.diag(u) @ K @ torch.diag(v)
        
        return X

class CSRA(nn.Module):
    """
    Capacitated Soft Role Assignment using Sinkhorn optimization.
    
    Implements the complete CSRA pipeline with score matrix computation
    and Sinkhorn optimization.
    """
    
    def __init__(self,
                 num_roles: int,
                 hidden_dim: int,
                 tau: float = 0.5,
                 alpha: float = 0.5,
                 beta: float = 0.5,
                 iters: int = 30,
                 use_balanced: bool = False):
        """
        Initialize CSRA.
        
        Args:
            num_roles: Number of roles
            hidden_dim: Hidden dimension
            tau: Temperature parameter
            alpha: Row relaxation parameter
            beta: Column relaxation parameter
            iters: Number of iterations
            use_balanced: Whether to use balanced Sinkhorn
        """
        super().__init__()
        
        self.num_roles = num_roles
        self.hidden_dim = hidden_dim
        self.tau = tau
        self.alpha = alpha
        self.beta = beta
        self.iters = iters
        self.use_balanced = use_balanced
        
        # Role embeddings
        self.role_embeddings = nn.Parameter(torch.randn(num_roles, hidden_dim))
        
        # Initialize role embeddings
        nn.init.xavier_uniform_(self.role_embeddings)
        
        # Sinkhorn algorithm
        if use_balanced:
            self.sinkhorn = BalancedSinkhorn(tau=tau, iters=iters)
        else:
            self.sinkhorn = UnbalancedSinkhorn(tau=tau, alpha=alpha, beta=beta, iters=iters)
    
    def forward(self, 
                node_features: torch.Tensor,
                role_ids: torch.Tensor,
                capacities: Dict[str, int],
                score_weights: Dict[str, float]) -> torch.Tensor:
        """
        Forward pass of CSRA.
        
        Args:
            node_features: Node features [N, hidden_dim]
            role_ids: Role IDs for each node [N]
            capacities: Role capacities
            score_weights: Weights for score components
            
        Returns:
            Assignment matrix X [R, N]
        """
        N = node_features.size(0)
        device = node_features.device
        
        # Build score matrix S [R, N]
        S = self._compute_score_matrix(node_features, role_ids, score_weights)
        
        # Build capacity vectors
        kappa = torch.tensor([capacities.get(f"role_{i}", 1) for i in range(self.num_roles)], device=device)
        rho = torch.ones(N, device=device)  # Each CTU can be assigned to at most 1 role
        
        # Apply Sinkhorn
        X = self.sinkhorn(S, kappa, rho)
        
        return X
    
    def _compute_score_matrix(self, 
                             node_features: torch.Tensor,
                             role_ids: torch.Tensor,
                             score_weights: Dict[str, float]) -> torch.Tensor:
        """
        Compute score matrix S [R, N].
        
        Args:
            node_features: Node features [N, hidden_dim]
            role_ids: Role IDs for each node [N]
            score_weights: Weights for score components
            
        Returns:
            Score matrix S [R, N]
        """
        R = self.num_roles
        N = node_features.size(0)
        device = node_features.device
        
        # Initialize score matrix
        S = torch.zeros(R, N, device=device)
        
        # Compute role fit scores
        role_fit = torch.mm(self.role_embeddings, node_features.t())  # [R, N]
        
        # Compute query similarity (using node features as queries)
        query_sim = torch.mm(node_features, node_features.t())  # [N, N]
        query_sim = torch.diag(query_sim)  # [N] - self-similarity
        
        # Compute salience (using L2 norm of features)
        salience = torch.norm(node_features, dim=1)  # [N]
        
        # Compute cohesion (using feature similarity to role embeddings)
        cohesion = torch.mm(node_features, self.role_embeddings.t())  # [N, R]
        cohesion = torch.max(cohesion, dim=1)[0]  # [N] - max similarity to any role
        
        # Combine score components
        w_query = score_weights.get('w_query', 1.0)
        w_role = score_weights.get('w_role', 0.6)
        w_sal = score_weights.get('w_sal', 0.2)
        w_coh = score_weights.get('w_coh', 0.2)
        
        for r in range(R):
            for n in range(N):
                # Base role fit
                role_score = w_role * role_fit[r, n]
                
                # Query similarity
                query_score = w_query * query_sim[n]
                
                # Salience
                sal_score = w_sal * salience[n]
                
                # Cohesion
                coh_score = w_coh * cohesion[n]
                
                # Total score
                S[r, n] = role_score + query_score + sal_score + coh_score
        
        return S
    
    def hard_selection(self, 
                      X: torch.Tensor, 
                      threshold: float = 0.5) -> torch.Tensor:
        """
        Convert soft assignment to hard selection.
        
        Args:
            X: Soft assignment matrix [R, N]
            threshold: Threshold for selection
            
        Returns:
            Hard assignment matrix [R, N]
        """
        return (X > threshold).float()
    
    def topk_selection(self, 
                      X: torch.Tensor, 
                      k: int = 1) -> torch.Tensor:
        """
        Select top-k assignments per role.
        
        Args:
            X: Soft assignment matrix [R, N]
            k: Number of top assignments per role
            
        Returns:
            Hard assignment matrix [R, N]
        """
        _, topk_indices = torch.topk(X, k, dim=1)
        hard_X = torch.zeros_like(X)
        hard_X.scatter_(1, topk_indices, 1.0)
        return hard_X

