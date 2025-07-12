# src/ctu/graph_builder.py
import numpy as np
import json
from pathlib import Path
from typing import List, Dict, Tuple
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import scipy.sparse as sp

# Load sentence transformer for embeddings
EMB = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

def compute_jaccard_similarity(text1: str, text2: str) -> float:
    """Compute Jaccard similarity between two texts."""
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())
    
    intersection = len(words1.intersection(words2))
    union = len(words1.union(words2))
    
    return intersection / union if union > 0 else 0.0

def compute_cosine_similarity(emb1: np.ndarray, emb2: np.ndarray) -> float:
    """Compute cosine similarity between embeddings."""
    return np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))

def compute_temporal_adjacency(ctu1_idx: int, ctu2_idx: int, num_ctus: int) -> float:
    """Compute temporal adjacency score."""
    distance = abs(ctu1_idx - ctu2_idx)
    # Exponential decay with distance
    return np.exp(-distance / 2.0)

def build_discourse_graph(ctus: List[Dict], alpha: float = 0.3, beta: float = 0.3, 
                         gamma: float = 0.4, min_weight: float = 0.15) -> Dict:
    """
    Build discourse graph G=(U,E,w) with edge weights.
    
    Args:
        ctus: List of CTU dictionaries with 'text' and 'ctu_id' keys
        alpha: Weight for temporal adjacency
        beta: Weight for Jaccard similarity  
        gamma: Weight for cosine similarity
        min_weight: Minimum edge weight to keep
        
    Returns:
        Dictionary with graph data (adjacency matrix, edge weights, etc.)
    """
    num_ctus = len(ctus)
    
    # Extract texts and compute embeddings
    texts = [ctu['text'] for ctu in ctus]
    embeddings = EMB.encode(texts, batch_size=32, show_progress_bar=False)
    
    # Initialize adjacency matrix
    adjacency = np.zeros((num_ctus, num_ctus))
    edge_weights = {}
    
    print(f"Building discourse graph with {num_ctus} CTUs...")
    
    # Compute edge weights for all pairs
    for i in range(num_ctus):
        for j in range(i + 1, num_ctus):
            # Temporal adjacency
            temp_adj = compute_temporal_adjacency(i, j, num_ctus)
            
            # Jaccard similarity
            jaccard = compute_jaccard_similarity(texts[i], texts[j])
            
            # Cosine similarity
            cosine_sim = compute_cosine_similarity(embeddings[i], embeddings[j])
            
            # Combined weight
            weight = alpha * temp_adj + beta * jaccard + gamma * cosine_sim
            
            # Only keep edges above threshold
            if weight >= min_weight:
                adjacency[i, j] = weight
                adjacency[j, i] = weight  # Undirected graph
                edge_weights[(i, j)] = weight
                edge_weights[(j, i)] = weight
    
    # Convert to sparse matrix
    sparse_adj = sp.csr_matrix(adjacency)
    
    # Compute graph statistics
    num_edges = len(edge_weights) // 2  # Undirected, so divide by 2
    avg_degree = 2 * num_edges / num_ctus if num_ctus > 0 else 0
    
    graph_data = {
        'num_nodes': num_ctus,
        'num_edges': num_edges,
        'avg_degree': avg_degree,
        'adjacency_matrix': sparse_adj,
        'edge_weights': edge_weights,
        'embeddings': embeddings,
        'texts': texts,
        'parameters': {
            'alpha': alpha,
            'beta': beta, 
            'gamma': gamma,
            'min_weight': min_weight
        }
    }
    
    print(f"Graph built: {num_ctus} nodes, {num_edges} edges, avg degree: {avg_degree:.2f}")
    
    return graph_data

def save_graph(graph_data: Dict, output_path: Path) -> None:
    """Save graph data to file."""
    # Convert numpy arrays to lists for JSON serialization
    save_data = {
        'num_nodes': graph_data['num_nodes'],
        'num_edges': graph_data['num_edges'],
        'avg_degree': graph_data['avg_degree'],
        'edge_weights': graph_data['edge_weights'],
        'texts': graph_data['texts'],
        'parameters': graph_data['parameters']
    }
    
    # Save sparse adjacency matrix separately
    sp.save_npz(output_path.with_suffix('.npz'), graph_data['adjacency_matrix'])
    
    # Save other data as JSON
    with open(output_path, 'w') as f:
        json.dump(save_data, f, indent=2)
    
    print(f"Graph saved to {output_path} and {output_path.with_suffix('.npz')}")

def load_graph(graph_path: Path) -> Dict:
    """Load graph data from file."""
    # Load JSON data
    with open(graph_path, 'r') as f:
        data = json.load(f)
    
    # Load sparse adjacency matrix
    adj_matrix = sp.load_npz(graph_path.with_suffix('.npz'))
    data['adjacency_matrix'] = adj_matrix
    
    return data

def analyze_graph(graph_data: Dict) -> Dict:
    """Analyze graph properties."""
    adj_matrix = graph_data['adjacency_matrix']
    
    # Degree distribution
    degrees = np.array(adj_matrix.sum(axis=1)).flatten()
    
    # Connected components
    from scipy.sparse.csgraph import connected_components
    n_components, labels = connected_components(adj_matrix)
    
    # Clustering coefficient (simplified)
    clustering_coeff = 0.0  # Would need to implement triangle counting
    
    analysis = {
        'num_nodes': graph_data['num_nodes'],
        'num_edges': graph_data['num_edges'],
        'avg_degree': graph_data['avg_degree'],
        'max_degree': int(degrees.max()),
        'min_degree': int(degrees.min()),
        'num_components': n_components,
        'largest_component_size': int(np.bincount(labels).max()),
        'clustering_coefficient': clustering_coeff
    }
    
    return analysis

def main():
    """CLI interface for graph building."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Build discourse graph from CTUs")
    parser.add_argument("--input", required=True, help="Input CTU JSON file")
    parser.add_argument("--output", required=True, help="Output graph file")
    parser.add_argument("--alpha", type=float, default=0.3, help="Temporal adjacency weight")
    parser.add_argument("--beta", type=float, default=0.3, help="Jaccard similarity weight")
    parser.add_argument("--gamma", type=float, default=0.4, help="Cosine similarity weight")
    parser.add_argument("--min-weight", type=float, default=0.15, help="Minimum edge weight")
    parser.add_argument("--analyze", action="store_true", help="Analyze graph properties")
    
    args = parser.parse_args()
    
    # Load CTU data
    with open(args.input, 'r') as f:
        ctus = json.load(f)
    
    # Build graph
    graph_data = build_discourse_graph(
        ctus, 
        alpha=args.alpha,
        beta=args.beta,
        gamma=args.gamma,
        min_weight=args.min_weight
    )
    
    # Save graph
    save_graph(graph_data, Path(args.output))
    
    # Analyze if requested
    if args.analyze:
        analysis = analyze_graph(graph_data)
        print("\nGraph Analysis:")
        for key, value in analysis.items():
            print(f"{key}: {value}")

if __name__ == "__main__":
    main() 