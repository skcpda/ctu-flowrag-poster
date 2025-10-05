"""
RCR-GAT: Role-Conditioned Retrieval with Graph Attention Networks

Implements the core RCR-GAT model with type-specific attention mechanisms,
learnable type biases, confidence and role-compatibility priors, and distance penalties.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import math
import logging

logger = logging.getLogger(__name__)

class RCRGATLayer(nn.Module):
    """
    Single RCR-GAT layer with type-specific attention.
    
    Implements the attention mechanism described in the design:
    - Type-specific Q/K/V projections
    - Learnable type biases
    - Confidence and role-compatibility priors
    - Distance penalty
    - Per-source softmax (segment softmax)
    """
    
    def __init__(self, 
                 input_dim: int,
                 hidden_dim: int,
                 edge_types: List[str],
                 edge_weight_priors: Dict[str, float],
                 beta_conf: float = 1.0,
                 gamma_compat: float = 0.5,
                 distance_lambda: float = 0.12,
                 dropout: float = 0.15):
        """
        Initialize RCR-GAT layer.
        
        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden dimension
            edge_types: List of edge type names
            edge_weight_priors: Prior weights for each edge type
            beta_conf: Confidence weight
            gamma_compat: Compatibility weight
            distance_lambda: Distance penalty weight
            dropout: Dropout rate
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.edge_types = edge_types
        self.edge_weight_priors = edge_weight_priors
        self.beta_conf = beta_conf
        self.gamma_compat = gamma_compat
        self.distance_lambda = distance_lambda
        self.dropout = dropout
        
        # Type-specific projections
        self.type_projections = nn.ModuleDict()
        for edge_type in edge_types:
            self.type_projections[edge_type] = nn.ModuleDict({
                'W_q': nn.Linear(input_dim, hidden_dim, bias=False),
                'W_k': nn.Linear(input_dim, hidden_dim, bias=False),
                'W_v': nn.Linear(input_dim, hidden_dim, bias=False)
            })
        
        # Learnable type biases
        self.type_biases = nn.ParameterDict()
        for edge_type in edge_types:
            self.type_biases[edge_type] = nn.Parameter(torch.zeros(1))
        
        # Residual connection
        self.W0 = nn.Linear(input_dim, hidden_dim)
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
        # Dropout
        self.dropout_layer = nn.Dropout(dropout)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights using Xavier initialization."""
        for edge_type in self.edge_types:
            for proj in ['W_q', 'W_k', 'W_v']:
                nn.init.xavier_uniform_(self.type_projections[edge_type][proj].weight)
        
        nn.init.xavier_uniform_(self.W0.weight)
        nn.init.zeros_(self.W0.bias)
    
    def forward(self, 
                node_feats: torch.Tensor,
                edge_packs_by_type: Dict[str, Dict[str, torch.Tensor]]) -> torch.Tensor:
        """
        Forward pass of RCR-GAT layer.
        
        Args:
            node_feats: Node features [N, input_dim]
            edge_packs_by_type: Dict mapping edge types to edge tensors
            
        Returns:
            Updated node features [N, hidden_dim]
        """
        N = node_feats.size(0)
        device = node_feats.device
        
        # Initialize output features
        out_feats = torch.zeros(N, self.hidden_dim, device=device)
        
        # Process each edge type
        for edge_type in self.edge_types:
            if edge_type not in edge_packs_by_type:
                continue
                
            edge_pack = edge_packs_by_type[edge_type]
            edge_index = edge_pack['edge_index']  # [2, E]
            conf = edge_pack['conf']  # [E]
            compat = edge_pack['compat']  # [E]
            dist = edge_pack['dist']  # [E]
            
            if edge_index.size(1) == 0:
                continue
            
            # Get type-specific projections
            W_q = self.type_projections[edge_type]['W_q']
            W_k = self.type_projections[edge_type]['W_k']
            W_v = self.type_projections[edge_type]['W_v']
            
            # Compute Q, K, V
            Q = W_q(node_feats)  # [N, hidden_dim]
            K = W_k(node_feats)  # [N, hidden_dim]
            V = W_v(node_feats)  # [N, hidden_dim]
            
            # Get edge indices
            src_indices = edge_index[0]  # [E]
            dst_indices = edge_index[1]  # [E]
            
            # Compute attention scores
            # score = dot(Q_i, K_j) / sqrt(d) + alpha_t + beta * conf + gamma * compat - dist
            q_src = Q[src_indices]  # [E, hidden_dim]
            k_dst = K[dst_indices]  # [E, hidden_dim]
            
            # Dot product attention
            dot_scores = torch.sum(q_src * k_dst, dim=1) / math.sqrt(self.hidden_dim)  # [E]
            
            # Add type bias
            alpha_t = self.type_biases[edge_type]
            type_scores = dot_scores + alpha_t
            
            # Add confidence and compatibility priors
            conf_scores = self.beta_conf * conf
            compat_scores = self.gamma_compat * compat
            
            # Add distance penalty
            dist_penalty = self.distance_lambda * dist
            
            # Final attention scores
            attention_scores = type_scores + conf_scores + compat_scores - dist_penalty
            
            # Per-source softmax (segment softmax)
            attention_weights = self._segment_softmax(attention_scores, src_indices, N)
            
            # Aggregate messages
            v_dst = V[dst_indices]  # [E, hidden_dim]
            weighted_messages = attention_weights.unsqueeze(1) * v_dst  # [E, hidden_dim]
            
            # Sum messages by source (using scatter_add)
            out_feats = out_feats.scatter_add(0, src_indices.unsqueeze(1).expand(-1, self.hidden_dim), weighted_messages)
        
        # Residual connection
        residual = self.W0(node_feats)
        out_feats = out_feats + residual
        
        # Layer normalization and activation
        out_feats = self.layer_norm(out_feats)
        out_feats = F.relu(out_feats)
        out_feats = self.dropout_layer(out_feats)
        
        return out_feats
    
    def _segment_softmax(self, 
                        scores: torch.Tensor, 
                        src_indices: torch.Tensor, 
                        num_nodes: int) -> torch.Tensor:
        """
        Compute per-source softmax (segment softmax).
        
        Args:
            scores: Attention scores [E]
            src_indices: Source node indices [E]
            num_nodes: Total number of nodes
            
        Returns:
            Softmax weights [E]
        """
        # Create a mask for each source node
        src_mask = torch.zeros(num_nodes, scores.size(0), device=scores.device)
        src_mask[src_indices, torch.arange(scores.size(0), device=scores.device)] = 1
        
        # Compute softmax for each source
        max_scores = torch.max(scores.unsqueeze(0) * src_mask, dim=1)[0]  # [N]
        exp_scores = torch.exp(scores - max_scores[src_indices])  # [E]
        
        # Sum exp scores for each source
        sum_exp = torch.sum(exp_scores.unsqueeze(0) * src_mask, dim=1)  # [N]
        
        # Normalize
        softmax_weights = exp_scores / (sum_exp[src_indices] + 1e-8)  # [E]
        
        return softmax_weights

class RCRGAT(nn.Module):
    """
    Multi-layer RCR-GAT model.
    
    Implements the complete RCR-GAT architecture with multiple layers
    and type-specific attention mechanisms.
    """
    
    def __init__(self,
                 text_dim: int,
                 hidden_dim: int,
                 num_layers: int,
                 edge_types: List[str],
                 edge_weight_priors: Dict[str, float],
                 beta_conf: float = 1.0,
                 gamma_compat: float = 0.5,
                 distance_lambda: float = 0.12,
                 dropout: float = 0.15):
        """
        Initialize RCR-GAT model.
        
        Args:
            text_dim: Input text embedding dimension
            hidden_dim: Hidden dimension
            num_layers: Number of GAT layers
            edge_types: List of edge type names
            edge_weight_priors: Prior weights for each edge type
            beta_conf: Confidence weight
            gamma_compat: Compatibility weight
            distance_lambda: Distance penalty weight
            dropout: Dropout rate
        """
        super().__init__()
        
        self.text_dim = text_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.edge_types = edge_types
        self.edge_weight_priors = edge_weight_priors
        self.beta_conf = beta_conf
        self.gamma_compat = gamma_compat
        self.distance_lambda = distance_lambda
        self.dropout = dropout
        
        # Input projection
        self.input_proj = nn.Linear(text_dim, hidden_dim)
        
        # GAT layers
        self.gat_layers = nn.ModuleList()
        for i in range(num_layers):
            self.gat_layers.append(RCRGATLayer(
                input_dim=hidden_dim,
                hidden_dim=hidden_dim,
                edge_types=edge_types,
                edge_weight_priors=edge_weight_priors,
                beta_conf=beta_conf,
                gamma_compat=gamma_compat,
                distance_lambda=distance_lambda,
                dropout=dropout
            ))
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights using Xavier initialization."""
        nn.init.xavier_uniform_(self.input_proj.weight)
        nn.init.zeros_(self.input_proj.bias)
        nn.init.xavier_uniform_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)
    
    def forward(self, 
                node_feats: torch.Tensor,
                edge_packs_by_type: Dict[str, Dict[str, torch.Tensor]]) -> torch.Tensor:
        """
        Forward pass of RCR-GAT model.
        
        Args:
            node_feats: Node features [N, text_dim]
            edge_packs_by_type: Dict mapping edge types to edge tensors
            
        Returns:
            Contextualized node features [N, hidden_dim]
        """
        # Input projection
        h = self.input_proj(node_feats)  # [N, hidden_dim]
        
        # Apply GAT layers
        for layer in self.gat_layers:
            h = layer(h, edge_packs_by_type)
        
        # Output projection
        h = self.output_proj(h)
        
        return h
    
    def get_attention_weights(self, 
                             node_feats: torch.Tensor,
                             edge_packs_by_type: Dict[str, Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """
        Get attention weights for inspection.
        
        Args:
            node_feats: Node features [N, text_dim]
            edge_packs_by_type: Dict mapping edge types to edge tensors
            
        Returns:
            Dict mapping edge types to attention weights
        """
        attention_weights = {}
        
        # Input projection
        h = self.input_proj(node_feats)
        
        # Get attention from first layer
        if len(self.gat_layers) > 0:
            layer = self.gat_layers[0]
            
            for edge_type in self.edge_types:
                if edge_type not in edge_packs_by_type:
                    continue
                    
                edge_pack = edge_packs_by_type[edge_type]
                edge_index = edge_pack['edge_index']
                conf = edge_pack['conf']
                compat = edge_pack['compat']
                dist = edge_pack['dist']
                
                if edge_index.size(1) == 0:
                    continue
                
                # Get type-specific projections
                W_q = layer.type_projections[edge_type]['W_q']
                W_k = layer.type_projections[edge_type]['W_k']
                
                # Compute Q, K
                Q = W_q(h)
                K = W_k(h)
                
                # Get edge indices
                src_indices = edge_index[0]
                dst_indices = edge_index[1]
                
                # Compute attention scores
                q_src = Q[src_indices]
                k_dst = K[dst_indices]
                
                dot_scores = torch.sum(q_src * k_dst, dim=1) / math.sqrt(self.hidden_dim)
                alpha_t = layer.type_biases[edge_type]
                type_scores = dot_scores + alpha_t
                conf_scores = layer.beta_conf * conf
                compat_scores = layer.gamma_compat * compat
                dist_penalty = layer.distance_lambda * dist
                
                attention_scores = type_scores + conf_scores + compat_scores - dist_penalty
                
                # Compute softmax weights
                attention_weights[edge_type] = layer._segment_softmax(attention_scores, src_indices, h.size(0))
        
        return attention_weights

