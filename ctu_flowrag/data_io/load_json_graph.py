"""
Load production JSON graphs and convert to internal format.

Handles the conversion from production JSON format to internal graph representation,
enforcing policy constraints like no cross-section PRECEDES edges.
"""

import json
from typing import Dict, List, Any, Tuple
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

def load_json_graph(json_path: str) -> Dict[str, Any]:
    """
    Load a production JSON graph and convert to internal format.
    
    Args:
        json_path: Path to the production JSON file
        
    Returns:
        Dictionary with 'nodes' and 'edges_by_type' keys
        
    Raises:
        FileNotFoundError: If JSON file doesn't exist
        ValueError: If JSON format is invalid
    """
    json_path = Path(json_path)
    if not json_path.exists():
        raise FileNotFoundError(f"JSON file not found: {json_path}")
    
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Validate required fields
    required_fields = ['scheme_name', 'ctus', 'relations']
    for field in required_fields:
        if field not in data:
            raise ValueError(f"Missing required field: {field}")
    
    # Convert CTUs to nodes
    nodes = []
    node_id_map = {}  # (sid, line_idx) -> node_id
    
    for i, ctu in enumerate(data['ctus']):
        node_id = f"{ctu['sid']}_{ctu['line_idx']}"
        node_id_map[(ctu['sid'], ctu['line_idx'])] = node_id
        
        node = {
            'id': node_id,
            'text': ctu['text'],
            'role': ctu['role'],
            'section_id': ctu.get('section_name', 'unknown'),
            'pos': ctu['line_idx'],
            'confidence': ctu.get('confidence', 0.0),
            'sid': ctu['sid'],
            'line_idx': ctu['line_idx'],
            'flags': {
                'is_structural': ctu.get('method', '') == 'rule_based',
                'is_semantic': ctu.get('method', '') == 'production_pipeline'
            }
        }
        nodes.append(node)
    
    # Convert relations to edges by type
    edges_by_type = {}
    
    for relation in data['relations']:
        # Get source and target node IDs
        src_key = (relation['ctu1']['sid'], relation['ctu1']['line_idx'])
        dst_key = (relation['ctu2']['sid'], relation['ctu2']['line_idx'])
        
        if src_key not in node_id_map or dst_key not in node_id_map:
            logger.warning(f"Skipping relation with missing nodes: {src_key} -> {dst_key}")
            continue
            
        src_id = node_id_map[src_key]
        dst_id = node_id_map[dst_key]
        
        # Get source and target nodes for section check
        src_node = next(n for n in nodes if n['id'] == src_id)
        dst_node = next(n for n in nodes if n['id'] == dst_id)
        
        edge_type = relation['type']
        
        # Enforce policy: drop PRECEDES if cross-section
        if edge_type == 'PRECEDES' and src_node['section_id'] != dst_node['section_id']:
            logger.debug(f"Dropping cross-section PRECEDES: {src_id} -> {dst_id}")
            continue
        
        # Compute delta_sent (sentence distance)
        delta_sent = abs(dst_node['pos'] - src_node['pos'])
        
        edge = {
            'src': src_id,
            'dst': dst_id,
            'conf_cal': float(max(0.0, min(1.0, relation.get('confidence_cal', relation.get('confidence', 0.0))))),
            'conf_raw': relation.get('confidence', 0.0),
            'delta_sent': delta_sent,
            'same_section': src_node['section_id'] == dst_node['section_id'],
            'relation': relation.get('relation', edge_type),
            'method': relation.get('method', 'unknown')
        }
        
        if edge_type not in edges_by_type:
            edges_by_type[edge_type] = []
        edges_by_type[edge_type].append(edge)
    
    # Enforce out-degree constraints (keep highest confidence edges)
    out_degree = {}
    edges_to_remove = []
    
    for edge_type, edges in edges_by_type.items():
        # Group edges by source
        src_edges = {}
        for i, edge in enumerate(edges):
            src = edge['src']
            if src not in src_edges:
                src_edges[src] = []
            src_edges[src].append((i, edge))
        
        # For each source with >2 edges, keep only top 2 by confidence
        for src, edge_list in src_edges.items():
            if len(edge_list) > 2:
                # Sort by confidence (descending) and keep top 2
                edge_list.sort(key=lambda x: x[1]['conf_cal'], reverse=True)
                for i, (edge_idx, edge) in enumerate(edge_list[2:], start=2):
                    edges_to_remove.append((edge_type, edge_idx))
                    logger.warning(f"Removing edge {src}->{edge['dst']} (out-degree > 2, conf: {edge['conf_cal']:.3f})")
    
    # Remove excess edges
    for edge_type, edge_idx in sorted(edges_to_remove, key=lambda x: x[1], reverse=True):
        edges_by_type[edge_type].pop(edge_idx)
    
    # Handle edge deduplication (keep highest confidence)
    edge_groups = {}
    for edge_type, edges in edges_by_type.items():
        for edge in edges:
            edge_key = (edge['src'], edge['dst'], edge_type)
            if edge_key not in edge_groups:
                edge_groups[edge_key] = []
            edge_groups[edge_key].append(edge)
    
    # Rebuild edges_by_type with deduplicated edges
    edges_by_type = {}
    for edge_key, edge_list in edge_groups.items():
        if len(edge_list) > 1:
            # Keep highest confidence edge
            best_edge = max(edge_list, key=lambda x: x['conf_cal'])
            logger.warning(f"Duplicate edge {edge_key}, keeping highest confidence: {best_edge['conf_cal']:.3f}")
        else:
            best_edge = edge_list[0]
        
        edge_type = edge_key[2]
        if edge_type not in edges_by_type:
            edges_by_type[edge_type] = []
        edges_by_type[edge_type].append(best_edge)
    
    result = {
        'metadata': {
            'scheme_name': data['scheme_name'],
            'total_sentences': data.get('total_sentences', len(nodes)),
            'num_nodes': len(nodes),
            'num_edges': sum(len(edges) for edges in edges_by_type.values()),
            'edge_types': list(edges_by_type.keys())
        },
        'nodes': nodes,
        'edges_by_type': edges_by_type
    }
    
    logger.info(f"Loaded graph: {len(nodes)} nodes, {sum(len(edges) for edges in edges_by_type.values())} edges")
    return result

def validate_graph(graph_data: Dict[str, Any]) -> bool:
    """
    Validate a loaded graph against policy constraints.
    
    Args:
        graph_data: Graph data from load_json_graph
        
    Returns:
        True if graph is valid, False otherwise
    """
    nodes = graph_data['nodes']
    edges_by_type = graph_data['edges_by_type']
    
    # Check node IDs are unique
    node_ids = [node['id'] for node in nodes]
    if len(node_ids) != len(set(node_ids)):
        logger.error("Duplicate node IDs found")
        return False
    
    # Check out-degree constraints
    out_degree = {}
    for edge_type, edges in edges_by_type.items():
        for edge in edges:
            src = edge['src']
            out_degree[src] = out_degree.get(src, 0) + 1
            if out_degree[src] > 2:
                logger.error(f"Node {src} violates out-degree constraint: {out_degree[src]} > 2")
                return False
    
    # Check no cross-section PRECEDES
    for edge in edges_by_type.get('PRECEDES', []):
        src_node = next(n for n in nodes if n['id'] == edge['src'])
        dst_node = next(n for n in nodes if n['id'] == edge['dst'])
        if src_node['section_id'] != dst_node['section_id']:
            logger.error(f"Cross-section PRECEDES edge: {edge['src']} -> {edge['dst']}")
            return False
    
    # Check edge deduplication
    edge_set = set()
    for edge_type, edges in edges_by_type.items():
        for edge in edges:
            edge_key = (edge['src'], edge['dst'], edge_type)
            if edge_key in edge_set:
                logger.error(f"Duplicate edge: {edge_key}")
                return False
            edge_set.add(edge_key)
    
    return True

def get_graph_stats(graph_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Get statistics about a loaded graph.
    
    Args:
        graph_data: Graph data from load_json_graph
        
    Returns:
        Dictionary with graph statistics
    """
    nodes = graph_data['nodes']
    edges_by_type = graph_data['edges_by_type']
    
    # Role distribution
    role_counts = {}
    for node in nodes:
        role = node['role']
        role_counts[role] = role_counts.get(role, 0) + 1
    
    # Edge type distribution
    edge_type_counts = {edge_type: len(edges) for edge_type, edges in edges_by_type.items()}
    
    # Confidence statistics
    all_confidences = []
    for edges in edges_by_type.values():
        for edge in edges:
            all_confidences.append(edge['conf_cal'])
    
    conf_stats = {
        'mean': sum(all_confidences) / len(all_confidences) if all_confidences else 0,
        'min': min(all_confidences) if all_confidences else 0,
        'max': max(all_confidences) if all_confidences else 0
    }
    
    return {
        'num_nodes': len(nodes),
        'num_edges': sum(len(edges) for edges in edges_by_type.values()),
        'role_distribution': role_counts,
        'edge_type_distribution': edge_type_counts,
        'confidence_stats': conf_stats,
        'edge_types': list(edges_by_type.keys())
    }

