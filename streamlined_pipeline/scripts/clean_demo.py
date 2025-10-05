#!/usr/bin/env python3
"""
Clean CTU Graph Demo

This creates a much cleaner, more readable visualization by:
1. Showing only the most important CTUs (by confidence and centrality)
2. Using better layout and spacing
3. Truncating text labels for readability
4. Using larger, clearer fonts
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

def create_clean_visualization():
    """Create a clean, readable CTU graph visualization."""
    data = load_data()
    
    # Build graph
    G = nx.DiGraph()
    
    # Add nodes with importance scoring
    for ctu in data['ctus']:
        node_id = f"{ctu['sid']}_{ctu['line_idx']}"
        # Calculate importance: confidence + centrality + role importance
        role_importance = {
            'ContextObjective': 1.0,
            'BenefitsAssistance': 0.9,
            'Eligibility': 0.8,
            'ApplicationProcess': 0.7,
            'AuthoritiesGovernance': 0.6,
            'TimelineFrequency': 0.5,
            'DefinitionsReferences': 0.4
        }.get(ctu['role'], 0.5)
        
        importance = ctu['confidence'] * role_importance
        G.add_node(node_id, **ctu, importance=importance)
    
    # Add edges
    for relation in data['relations']:
        ctu1_id = f"{relation['ctu1']['sid']}_{relation['ctu1']['line_idx']}"
        ctu2_id = f"{relation['ctu2']['sid']}_{relation['ctu2']['line_idx']}"
        G.add_edge(ctu1_id, ctu2_id, **relation)
    
    # Select top 15 most important nodes
    node_importance = [(node, data['importance']) for node, data in G.nodes(data=True)]
    node_importance.sort(key=lambda x: x[1], reverse=True)
    top_nodes = [node for node, _ in node_importance[:15]]
    
    # Create subgraph with only top nodes
    subgraph = G.subgraph(top_nodes)
    
    # Color mapping
    role_colors = {
        'ContextObjective': '#2E86AB',      # Blue
        'BenefitsAssistance': '#A23B72',   # Purple  
        'Eligibility': '#F18F01',          # Orange
        'ApplicationProcess': '#C73E1D',    # Red
        'AuthoritiesGovernance': '#7209B7', # Dark Purple
        'TimelineFrequency': '#06A77D',    # Teal
        'DefinitionsReferences': '#F4A261'  # Light Orange
    }
    
    # Create figure with better sizing
    plt.figure(figsize=(24, 16))
    
    # Use hierarchical layout for better readability
    pos = {}
    sections = {}
    
    # Group nodes by section and role
    for node in subgraph.nodes():
        section = subgraph.nodes[node]['section_name']
        role = subgraph.nodes[node]['role']
        if section not in sections:
            sections[section] = {}
        if role not in sections[section]:
            sections[section][role] = []
        sections[section][role].append(node)
    
    # Position nodes in a clean hierarchical layout
    y_pos = 0
    for section_name, roles in sections.items():
        x_pos = 0
        for role, nodes in roles.items():
            for i, node in enumerate(nodes):
                pos[node] = (x_pos + i * 2, y_pos)
            x_pos += len(nodes) * 2 + 1
        y_pos += 3
    
    # Draw nodes with better styling
    node_colors = []
    node_sizes = []
    node_labels = {}
    
    for node in subgraph.nodes():
        role = subgraph.nodes[node]['role']
        confidence = subgraph.nodes[node]['confidence']
        
        node_colors.append(role_colors.get(role, '#808080'))
        node_sizes.append(800 + confidence * 400)  # Larger nodes
        
        # Create clean, truncated labels
        text = subgraph.nodes[node]['sentence']
        if len(text) > 60:
            text = text[:57] + "..."
        node_labels[node] = text
    
    # Draw the graph
    nx.draw_networkx_nodes(subgraph, pos, 
                          node_color=node_colors,
                          node_size=node_sizes,
                          alpha=0.9,
                          edgecolors='black',
                          linewidths=2)
    
    # Draw edges with better styling
    edge_colors = []
    edge_widths = []
    
    for edge in subgraph.edges():
        confidence = subgraph.edges[edge]['confidence']
        edge_colors.append(plt.cm.RdYlBu(confidence))
        edge_widths.append(2 + confidence * 3)
    
    nx.draw_networkx_edges(subgraph, pos,
                          edge_color=edge_colors,
                          width=edge_widths,
                          alpha=0.7,
                          arrows=True,
                          arrowsize=25,
                          arrowstyle='->')
    
    # Add labels with better formatting
    nx.draw_networkx_labels(subgraph, pos, node_labels, 
                           font_size=10, font_weight='bold',
                           bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
    
    # Create legend
    legend_elements = []
    for role, color in role_colors.items():
        if role in [node[1]['role'] for node in subgraph.nodes(data=True)]:
            legend_elements.append(mpatches.Patch(color=color, label=role))
    
    plt.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(0, 1), fontsize=12)
    
    # Add title and stats
    stats = {
        'total_ctus': len(data['ctus']),
        'shown_ctus': len(subgraph.nodes()),
        'avg_confidence': np.mean([r['confidence'] for r in data['relations']])
    }
    
    title = f"Clean CTU Graph: {data['scheme_name']}\n"
    title += f"Showing top {stats['shown_ctus']} most important CTUs (out of {stats['total_ctus']} total)\n"
    title += f"Average confidence: {stats['avg_confidence']:.2f}"
    
    plt.title(title, fontsize=16, fontweight='bold', pad=20)
    plt.axis('off')
    plt.tight_layout()
    
    # Save the clean visualization
    output_file = "../demo_visualizations/aa_clean_demo.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Clean visualization saved to: {output_file}")
    
    # Also create a summary
    print("\n" + "="*80)
    print("CLEAN CTU GRAPH SUMMARY")
    print("="*80)
    print(f"Scheme: {data['scheme_name']}")
    print(f"Total CTUs: {stats['total_ctus']}")
    print(f"Showing: {stats['shown_ctus']} most important CTUs")
    print(f"Average confidence: {stats['avg_confidence']:.2f}")
    print("\nTop CTUs by importance:")
    print("-" * 50)
    
    for i, (node, importance) in enumerate(node_importance[:10], 1):
        ctu_data = G.nodes[node]
        role = ctu_data['role']
        text = ctu_data['sentence'][:50] + "..." if len(ctu_data['sentence']) > 50 else ctu_data['sentence']
        print(f"{i:2d}. [{role:20}] {text}")
    
    return output_file

if __name__ == "__main__":
    print("Creating clean CTU graph visualization...")
    output_file = create_clean_visualization()
    print(f"\n✅ Clean visualization created: {output_file}")
    print("\nThis shows only the most important CTUs with better layout and readability!")
