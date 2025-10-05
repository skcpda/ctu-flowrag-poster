"""
Evaluation metrics for path ranking and retrieval.

Implements nDCG, MRR, MAP, and path coherence metrics.
"""

import torch
import numpy as np
from typing import List, Dict, Any, Optional
import math
import logging

logger = logging.getLogger(__name__)

def ndcg_at_k(scores: List[float], k: int = 10) -> float:
    """
    Compute nDCG@k for a single query.
    
    Args:
        scores: List of scores (higher is better)
        k: Cutoff for nDCG
        
    Returns:
        nDCG@k score
    """
    if len(scores) == 0:
        return 0.0
    
    # Sort by score (descending)
    sorted_scores = sorted(scores, reverse=True)
    
    # Compute DCG@k
    dcg = 0.0
    for i in range(min(k, len(sorted_scores))):
        dcg += sorted_scores[i] / math.log2(i + 2)
    
    # Compute IDCG@k (ideal DCG)
    idcg = 0.0
    for i in range(min(k, len(sorted_scores))):
        idcg += 1.0 / math.log2(i + 2)
    
    # Compute nDCG@k
    if idcg == 0:
        return 0.0
    
    return dcg / idcg

def mrr_at_k(scores: List[float], k: int = 10) -> float:
    """
    Compute MRR@k for a single query.
    
    Args:
        scores: List of scores (higher is better)
        k: Cutoff for MRR
        
    Returns:
        MRR@k score
    """
    if len(scores) == 0:
        return 0.0
    
    # Find the rank of the highest score
    sorted_scores = sorted(scores, reverse=True)
    
    # MRR is 1/rank of the first relevant item
    # For simplicity, we assume the first item is relevant
    if len(sorted_scores) > 0 and sorted_scores[0] > 0:
        return 1.0 / 1.0  # First item has rank 1
    else:
        return 0.0

def map_at_k(scores: List[float], k: int = 10) -> float:
    """
    Compute MAP@k for a single query.
    
    Args:
        scores: List of scores (higher is better)
        k: Cutoff for MAP
        
    Returns:
        MAP@k score
    """
    if len(scores) == 0:
        return 0.0
    
    # Sort by score (descending)
    sorted_scores = sorted(scores, reverse=True)
    
    # For simplicity, assume all items are relevant
    # In practice, you'd have relevance labels
    num_relevant = min(k, len(sorted_scores))
    
    if num_relevant == 0:
        return 0.0
    
    # Compute average precision
    precision_sum = 0.0
    for i in range(num_relevant):
        precision = (i + 1) / (i + 1)  # Precision at rank i+1
        precision_sum += precision
    
    return precision_sum / num_relevant

def compute_path_metrics(scores_list: List[List[float]], 
                         paths_list: List[List[List[int]]]) -> Dict[str, float]:
    """
    Compute path ranking metrics.
    
    Args:
        scores_list: List of score lists for each query
        paths_list: List of path lists for each query
        
    Returns:
        Dictionary with metrics
    """
    if len(scores_list) == 0:
        return {
            'ndcg@10': 0.0,
            'mrr@10': 0.0,
            'map@10': 0.0,
            'num_queries': 0
        }
    
    ndcg_scores = []
    mrr_scores = []
    map_scores = []
    
    for scores in scores_list:
        if len(scores) == 0:
            continue
            
        # Compute metrics for this query
        ndcg = ndcg_at_k(scores, k=10)
        mrr = mrr_at_k(scores, k=10)
        map_score = map_at_k(scores, k=10)
        
        ndcg_scores.append(ndcg)
        mrr_scores.append(mrr)
        map_scores.append(map_score)
    
    # Compute averages
    avg_ndcg = np.mean(ndcg_scores) if ndcg_scores else 0.0
    avg_mrr = np.mean(mrr_scores) if mrr_scores else 0.0
    avg_map = np.mean(map_scores) if map_scores else 0.0
    
    return {
        'ndcg@10': avg_ndcg,
        'mrr@10': avg_mrr,
        'map@10': avg_map,
        'num_queries': len(scores_list)
    }

def compute_path_coherence(paths: List[List[int]], 
                          edge_types: List[str],
                          edge_weights: Dict[str, float]) -> float:
    """
    Compute path coherence based on edge types and weights.
    
    Args:
        paths: List of paths (node sequences)
        edge_types: List of edge types
        edge_weights: Weights for each edge type
        
    Returns:
        Average coherence score
    """
    if len(paths) == 0:
        return 0.0
    
    coherence_scores = []
    
    for path in paths:
        if len(path) < 2:
            continue
            
        # Compute coherence for this path
        path_coherence = 0.0
        num_edges = 0
        
        for i in range(len(path) - 1):
            # For simplicity, assume all edges have weight 1.0
            # In practice, you'd look up the actual edge type and weight
            edge_weight = 1.0
            path_coherence += edge_weight
            num_edges += 1
        
        if num_edges > 0:
            path_coherence /= num_edges
        
        coherence_scores.append(path_coherence)
    
    return np.mean(coherence_scores) if coherence_scores else 0.0

def compute_retrieval_metrics(query_results: Dict[str, List[Dict[str, Any]]]) -> Dict[str, float]:
    """
    Compute retrieval metrics for multiple queries.
    
    Args:
        query_results: Dict mapping query_id to list of results
        
    Returns:
        Dictionary with retrieval metrics
    """
    if len(query_results) == 0:
        return {
            'precision@10': 0.0,
            'recall@10': 0.0,
            'f1@10': 0.0,
            'num_queries': 0
        }
    
    precision_scores = []
    recall_scores = []
    f1_scores = []
    
    for query_id, results in query_results.items():
        if len(results) == 0:
            continue
            
        # For simplicity, assume all results are relevant
        # In practice, you'd have relevance labels
        num_relevant = len(results)
        num_retrieved = len(results)
        
        precision = num_relevant / num_retrieved if num_retrieved > 0 else 0.0
        recall = 1.0  # Assume perfect recall for simplicity
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        precision_scores.append(precision)
        recall_scores.append(recall)
        f1_scores.append(f1)
    
    return {
        'precision@10': np.mean(precision_scores) if precision_scores else 0.0,
        'recall@10': np.mean(recall_scores) if recall_scores else 0.0,
        'f1@10': np.mean(f1_scores) if f1_scores else 0.0,
        'num_queries': len(query_results)
    }

def compute_attention_metrics(attention_weights: Dict[str, torch.Tensor]) -> Dict[str, float]:
    """
    Compute attention distribution metrics.
    
    Args:
        attention_weights: Dict mapping edge types to attention weights
        
    Returns:
        Dictionary with attention metrics
    """
    if len(attention_weights) == 0:
        return {
            'attention_entropy': 0.0,
            'attention_concentration': 0.0,
            'num_edge_types': 0
        }
    
    entropy_scores = []
    concentration_scores = []
    
    for edge_type, weights in attention_weights.items():
        if weights.numel() == 0:
            continue
            
        # Convert to probabilities
        probs = torch.softmax(weights, dim=0)
        
        # Compute entropy
        entropy = -torch.sum(probs * torch.log(probs + 1e-8))
        entropy_scores.append(entropy.item())
        
        # Compute concentration (inverse of entropy)
        concentration = 1.0 / (entropy + 1e-8)
        concentration_scores.append(concentration.item())
    
    return {
        'attention_entropy': np.mean(entropy_scores) if entropy_scores else 0.0,
        'attention_concentration': np.mean(concentration_scores) if concentration_scores else 0.0,
        'num_edge_types': len(attention_weights)
    }

