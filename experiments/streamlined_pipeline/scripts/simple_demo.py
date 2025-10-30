#!/usr/bin/env python3
"""
Simple CTU Graph Demo

This creates a very simple, easy-to-read visualization showing just the key concepts
and their relationships in a clean, hierarchical format.
"""

import os
import sys
import json
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path

def load_data():
    """Load the CTU relation data."""
    data_file = "../output_data/ctu_relations_production_ready/aa_ctus_production_ready.json"
    with open(data_file, 'r', encoding='utf-8') as f:
        return json.load(f)

def create_simple_visualization():
    """Create a simple, readable CTU graph visualization."""
    data = load_data()
    
    # Build graph
    G = nx.DiGraph()
    
    # Add nodes
    for ctu in data['ctus']:
        node_id = f"{ctu['sid']}_{ctu['line_idx']}"
        G.add_node(node_id, **ctu)
    
    # Add edges
    for relation in data['relations']:
        ctu1_id = f"{relation['ctu1']['sid']}_{relation['ctu1']['line_idx']}"
        ctu2_id = f"{relation['ctu2']['sid']}_{relation['ctu2']['line_idx']}"
        G.add_edge(ctu1_id, ctu2_id, **relation)
    
    # Select key CTUs by role and importance
    key_ctus = []
    role_priority = {
        'ContextObjective': 1,
        'BenefitsAssistance': 2, 
        'Eligibility': 3,
        'ApplicationProcess': 4,
        'AuthoritiesGovernance': 5
    }
    
    # Get one representative CTU from each role
    role_representatives = {}
    for node, data in G.nodes(data=True):
        role = data['role']
        if role in role_priority and role not in role_representatives:
            role_representatives[role] = node
    
    # Add a few more high-confidence CTUs
    high_confidence_nodes = [(node, data['confidence']) for node, data in G.nodes(data=True)]
    high_confidence_nodes.sort(key=lambda x: x[1], reverse=True)
    
    # Select 8-10 key nodes
    key_ctus = list(role_representatives.values())[:5]  # One from each role
    for node, conf in high_confidence_nodes[:3]:  # Top 3 by confidence
        if node not in key_ctus:
            key_ctus.append(node)
    
    # Create subgraph
    subgraph = G.subgraph(key_ctus)
    
    # Color mapping
    role_colors = {
        'ContextObjective': '#2E86AB',      # Blue
        'BenefitsAssistance': '#A23B72',   # Purple  
        'Eligibility': '#F18F01',          # Orange
        'ApplicationProcess': '#C73E1D',    # Red
        'AuthoritiesGovernance': '#7209B7' # Dark Purple
    }
    
    # Create figure
    plt.figure(figsize=(20, 12))
    
    # Simple hierarchical layout
    pos = {}
    y_positions = [0, -2, -4, -6, -8]
    x_positions = [-3, -1, 1, 3]
    
    for i, node in enumerate(subgraph.nodes()):
        role = subgraph.nodes[node]['role']
        if role == 'ContextObjective':
            pos[node] = (0, 0)  # Center top
        elif role == 'BenefitsAssistance':
            pos[node] = (-2, -2)
        elif role == 'Eligibility':
            pos[node] = (2, -2)
        elif role == 'ApplicationProcess':
            pos[node] = (-2, -4)
        elif role == 'AuthoritiesGovernance':
            pos[node] = (2, -4)
        else:
            # Distribute others
            pos[node] = (x_positions[i % len(x_positions)], y_positions[i // len(x_positions)])
    
    # Draw nodes
    node_colors = []
    node_sizes = []
    node_labels = {}
    
    for node in subgraph.nodes():
        role = subgraph.nodes[node]['role']
        confidence = subgraph.nodes[node]['confidence']
        
        node_colors.append(role_colors.get(role, '#808080'))
        node_sizes.append(1200 + confidence * 600)
        
        # Create very short, clean labels
        text = subgraph.nodes[node]['sentence']
        if len(text) > 40:
            text = text[:37] + "..."
        node_labels[node] = text
    
    # Draw the graph
    nx.draw_networkx_nodes(subgraph, pos, 
                          node_color=node_colors,
                          node_size=node_sizes,
                          alpha=0.9,
                          edgecolors='black',
                          linewidths=3)
    
    # Draw edges
    nx.draw_networkx_edges(subgraph, pos,
                          edge_color='gray',
                          width=2,
                          alpha=0.6,
                          arrows=True,
                          arrowsize=30,
                          arrowstyle='->')
    
    # Add labels
    nx.draw_networkx_labels(subgraph, pos, node_labels, 
                           font_size=11, font_weight='bold',
                           bbox=dict(boxstyle="round,pad=0.4", facecolor='white', alpha=0.9))
    
    # Create legend
    legend_elements = []
    for role, color in role_colors.items():
        if role in [node[1]['role'] for node in subgraph.nodes(data=True)]:
            legend_elements.append(mpatches.Patch(color=color, label=role))
    
    plt.legend(handles=legend_elements, loc='upper right', fontsize=12)
    
    # Add title
    title = f"Simple CTU Graph: {data['scheme_name']}\n"
    title += f"Key Concepts and Relationships"
    
    plt.title(title, fontsize=18, fontweight='bold', pad=20)
    plt.axis('off')
    plt.tight_layout()
    
    # Save
    output_file = "../demo_visualizations/aa_simple_demo.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Simple visualization saved to: {output_file}")
    
    # Print summary
    print("\n" + "="*80)
    print("SIMPLE CTU GRAPH - KEY CONCEPTS")
    print("="*80)
    print(f"Scheme: {data['scheme_name']}")
    print(f"Showing: {len(subgraph.nodes())} key CTUs")
    print("\nKey concepts by role:")
    print("-" * 50)
    
    for node in subgraph.nodes():
        ctu_data = subgraph.nodes[node]
        role = ctu_data['role']
        text = ctu_data['sentence'][:60] + "..." if len(ctu_data['sentence']) > 60 else ctu_data['sentence']
        print(f"[{role:20}] {text}")
    
    return output_file

if __name__ == "__main__":
    print("Creating simple CTU graph visualization...")
    output_file = create_simple_visualization()
    print(f"\n✅ Simple visualization created: {output_file}")
    print("\nThis shows only the key concepts in a clean, easy-to-read format!")
