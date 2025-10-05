"""
Score matrix computation for CSRA.

Implements the score matrix S_{r,i} computation with multiple components:
query similarity, role fit, salience, and cohesion.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import math
import logging

logger = logging.getLogger(__name__)

class ScoreMatrix(nn.Module):
    """
    Score matrix computation for CSRA.
    
    Computes S_{r,i} = w_query*sim(q,h_i) + w_role*fit(r,h_i) + w_sal*salience_i + w_coh*cohesion_i
    """
    
    def __init__(self, 
                 num_roles: int,
                 hidden_dim: int,
                 score_weights: Dict[str, float],
                 use_learnable_role_emb: bool = True):
        """
        Initialize score matrix computation.
        
        Args:
            num_roles: Number of roles
            hidden_dim: Hidden dimension
            score_weights: Weights for score components
            use_learnable_role_emb: Whether to use learnable role embeddings
        """
        super().__init__()
        
        self.num_roles = num_roles
        self.hidden_dim = hidden_dim
        self.score_weights = score_weights
        self.use_learnable_role_emb = use_learnable_role_emb
        
        # Role embeddings
        if use_learnable_role_emb:
            self.role_embeddings = nn.Parameter(torch.randn(num_roles, hidden_dim))
            nn.init.xavier_uniform_(self.role_embeddings)
        else:
            self.register_parameter('role_embeddings', None)
        
        # Role fit MLP (optional)
        self.role_fit_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        # Query encoder (for learned queries)
        self.query_encoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Salience MLP
        self.salience_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # Cohesion MLP
        self.cohesion_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward(self, 
                node_features: torch.Tensor,
                role_ids: torch.Tensor,
                query_features: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute score matrix S [R, N].
        
        Args:
            node_features: Node features [N, hidden_dim]
            role_ids: Role IDs for each node [N]
            query_features: Optional query features [N, hidden_dim]
            
        Returns:
            Score matrix S [R, N]
        """
        N = node_features.size(0)
        device = node_features.device
        
        # Initialize score matrix
        S = torch.zeros(self.num_roles, N, device=device)
        
        # Compute query similarity
        if query_features is not None:
            query_sim = self._compute_query_similarity(node_features, query_features)
        else:
            # Use self-similarity as query similarity
            query_sim = torch.norm(node_features, dim=1)  # [N]
        
        # Compute role fit
        role_fit = self._compute_role_fit(node_features, role_ids)
        
        # Compute salience
        salience = self._compute_salience(node_features)
        
        # Compute cohesion
        cohesion = self._compute_cohesion(node_features)
        
        # Combine score components
        w_query = self.score_weights.get('w_query', 1.0)
        w_role = self.score_weights.get('w_role', 0.6)
        w_sal = self.score_weights.get('w_sal', 0.2)
        w_coh = self.score_weights.get('w_coh', 0.2)
        
        for r in range(self.num_roles):
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
    
    def _compute_query_similarity(self, 
                                 node_features: torch.Tensor,
                                 query_features: torch.Tensor) -> torch.Tensor:
        """
        Compute query similarity.
        
        Args:
            node_features: Node features [N, hidden_dim]
            query_features: Query features [N, hidden_dim]
            
        Returns:
            Query similarity scores [N]
        """
        # Cosine similarity
        node_norm = F.normalize(node_features, p=2, dim=1)
        query_norm = F.normalize(query_features, p=2, dim=1)
        
        similarity = torch.sum(node_norm * query_norm, dim=1)  # [N]
        
        return similarity
    
    def _compute_role_fit(self, 
                         node_features: torch.Tensor,
                         role_ids: torch.Tensor) -> torch.Tensor:
        """
        Compute role fit scores.
        
        Args:
            node_features: Node features [N, hidden_dim]
            role_ids: Role IDs for each node [N]
            
        Returns:
            Role fit scores [R, N]
        """
        if self.role_embeddings is not None:
            # Use learnable role embeddings
            role_fit = torch.mm(self.role_embeddings, node_features.t())  # [R, N]
        else:
            # Use MLP on concatenated features
            role_fit = torch.zeros(self.num_roles, node_features.size(0), device=node_features.device)
            
            for r in range(self.num_roles):
                # Create role embedding (one-hot)
                role_emb = torch.zeros(self.num_roles, device=node_features.device)
                role_emb[r] = 1.0
                
                # Concatenate with node features
                concat_features = torch.cat([
                    node_features, 
                    role_emb.unsqueeze(0).expand(node_features.size(0), -1)
                ], dim=1)  # [N, hidden_dim + num_roles]
                
                # Apply MLP
                role_fit[r] = self.role_fit_mlp(concat_features).squeeze(1)
        
        return role_fit
    
    def _compute_salience(self, node_features: torch.Tensor) -> torch.Tensor:
        """
        Compute salience scores.
        
        Args:
            node_features: Node features [N, hidden_dim]
            
        Returns:
            Salience scores [N]
        """
        # Use MLP to compute salience
        salience = self.salience_mlp(node_features).squeeze(1)  # [N]
        
        # Apply sigmoid to normalize to [0, 1]
        salience = torch.sigmoid(salience)
        
        return salience
    
    def _compute_cohesion(self, node_features: torch.Tensor) -> torch.Tensor:
        """
        Compute cohesion scores.
        
        Args:
            node_features: Node features [N, hidden_dim]
            
        Returns:
            Cohesion scores [N]
        """
        # Use MLP to compute cohesion
        cohesion = self.cohesion_mlp(node_features).squeeze(1)  # [N]
        
        # Apply sigmoid to normalize to [0, 1]
        cohesion = torch.sigmoid(cohesion)
        
        return cohesion
    
    def compute_template_query(self, 
                              template: List[int],
                              node_features: torch.Tensor,
                              role_ids: torch.Tensor) -> torch.Tensor:
        """
        Compute query features from a template.
        
        Args:
            template: List of role IDs in the template
            node_features: Node features [N, hidden_dim]
            role_ids: Role IDs for each node [N]
            
        Returns:
            Query features [N, hidden_dim]
        """
        # Find nodes that match the template roles
        template_mask = torch.zeros(node_features.size(0), dtype=torch.bool, device=node_features.device)
        
        for role_id in template:
            template_mask |= (role_ids == role_id)
        
        if template_mask.sum() == 0:
            # No nodes match template, return zero query
            return torch.zeros_like(node_features)
        
        # Compute query as average of matching nodes
        matching_features = node_features[template_mask]
        query_features = matching_features.mean(dim=0, keepdim=True).expand(node_features.size(0), -1)
        
        # Apply query encoder
        query_features = self.query_encoder(query_features)
        
        return query_features

