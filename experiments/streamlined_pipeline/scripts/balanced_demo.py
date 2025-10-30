#!/usr/bin/env python3
"""
Balanced CTU Demo

Creates a visualization with a good mix of different role types
and meaningful relationships between them.
"""

import os
import sys
import json
import networkx as nx
import plotly.graph_objects as go
import numpy as np
from pathlib import Path

def load_data():
    """Load the CTU relation data."""
    data_file = "../output_data/ctu_relations_production_ready/aa_ctus_production_ready.json"
    with open(data_file, 'r', encoding='utf-8') as f:
        return json.load(f)

def create_balanced_visualization():
    """Create a balanced visualization with diverse role types."""
    data = load_data()
    
    # Build full graph
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
    
    # Select nodes to ensure good role diversity
    selected_nodes = set()
    
    # Get role distribution from the data
    role_counts = {}
    for node, data in G.nodes(data=True):
        role = data['role']
        role_counts[role] = role_counts.get(role, 0) + 1
    
    print("Available roles and counts:")
    for role, count in sorted(role_counts.items()):
        print(f"  {role}: {count}")
    
    # Select one representative from each major role type
    role_representatives = {}
    for node, data in G.nodes(data=True):
        role = data['role']
        if role not in role_representatives:
            role_representatives[role] = node
    
    # Add representatives from key roles
    key_roles = ['ContextObjective', 'BenefitsAssistance', 'Eligibility', 'ApplicationProcess', 'AuthoritiesGovernance']
    for role in key_roles:
        if role in role_representatives:
            selected_nodes.add(role_representatives[role])
    
    # Add a few more high-confidence nodes from different roles
    role_nodes = {}
    for node, data in G.nodes(data=True):
        role = data['role']
        if role not in role_nodes:
            role_nodes[role] = []
        role_nodes[role].append((node, data['confidence']))
    
    # Add 1-2 more nodes from roles that have good representation
    for role, nodes in role_nodes.items():
        if len(nodes) > 1 and role in ['ContextObjective', 'BenefitsAssistance', 'Eligibility']:
            # Sort by confidence and add top 1-2
            nodes.sort(key=lambda x: x[1], reverse=True)
            for node, conf in nodes[:2]:
                if node not in selected_nodes and len(selected_nodes) < 12:
                    selected_nodes.add(node)
    
    # Create subgraph
    subgraph = G.subgraph(selected_nodes)
    
    print(f"\nSelected {len(subgraph.nodes())} CTUs with {len(subgraph.edges())} relationships")
    
    # Show role distribution of selected nodes
    selected_role_counts = {}
    for node in subgraph.nodes():
        role = subgraph.nodes[node]['role']
        selected_role_counts[role] = selected_role_counts.get(role, 0) + 1
    
    print("\nSelected role distribution:")
    for role, count in sorted(selected_role_counts.items()):
        print(f"  {role}: {count}")
    
    # Use spring layout for better positioning
    pos = nx.spring_layout(subgraph, k=2, iterations=100, seed=42)
    
    # Color mapping
    role_colors = {
        'ContextObjective': '#2E86AB',      # Blue
        'BenefitsAssistance': '#A23B72',   # Purple  
        'Eligibility': '#F18F01',          # Orange
        'ApplicationProcess': '#C73E1D',   # Red
        'AuthoritiesGovernance': '#7209B7', # Dark Purple
        'TimelineFrequency': '#06A77D',    # Teal
        'DefinitionsReferences': '#F4A261'  # Light Orange
    }
    
    # Prepare node data
    node_x = []
    node_y = []
    node_text = []
    node_colors = []
    node_sizes = []
    node_labels = []
    
    for node in subgraph.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        
        # Get node data
        ctu_data = subgraph.nodes[node]
        role = ctu_data['role']
        confidence = ctu_data['confidence']
        sentence = ctu_data['sentence']
        
        # Create hover text
        hover_text = f"<b>{role}</b><br>"
        hover_text += f"Confidence: {confidence:.2f}<br>"
        hover_text += f"Section: {ctu_data['section_name']}<br><br>"
        hover_text += f"<i>{sentence}</i>"
        
        node_text.append(hover_text)
        node_colors.append(role_colors.get(role, '#808080'))
        node_sizes.append(25 + confidence * 15)
        node_labels.append(f"{role}<br>{sentence[:25]}...")
    
    # Prepare edge data
    edge_x = []
    edge_y = []
    
    for edge in subgraph.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
    
    # Create edge trace
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=3, color='rgba(125,125,125,0.7)'),
        hoverinfo='none',
        mode='lines',
        showlegend=False
    )
    
    # Create node trace
    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        hoverinfo='text',
        hovertext=node_text,
        text=node_labels,
        textposition="middle center",
        textfont=dict(size=10, color="white", family="Arial Black"),
        marker=dict(
            size=node_sizes,
            color=node_colors,
            line=dict(width=3, color='black'),
            opacity=0.9
        ),
        showlegend=False
    )
    
    # Create the figure
    fig = go.Figure(data=[edge_trace, node_trace],
                   layout=go.Layout(
                       title=dict(
                           text=f"Balanced CTU Graph: {data['scheme_name']}<br><sub>Diverse role types with meaningful relationships</sub>",
                           font=dict(size=18)
                       ),
                       showlegend=False,
                       hovermode='closest',
                       margin=dict(b=20,l=5,r=5,t=80),
                       xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                       yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                       plot_bgcolor='white',
                       paper_bgcolor='white',
                       width=1000,
                       height=700
                   ))
    
    # Add role legend
    legend_annotations = []
    y_pos = 0.95
    for role, color in role_colors.items():
        if role in [subgraph.nodes[node]['role'] for node in subgraph.nodes()]:
            legend_annotations.append(dict(
                x=0.02, y=y_pos,
                xref="paper", yref="paper",
                text=f"<span style='color:{color}'>●</span> {role}",
                showarrow=False,
                font=dict(size=12, family="Arial Black"),
                bgcolor="rgba(255,255,255,0.9)",
                bordercolor="black",
                borderwidth=1
            ))
            y_pos -= 0.05
    
    fig.update_layout(annotations=list(fig.layout.annotations) + legend_annotations)
    
    # Save the visualization
    output_file = "../demo_visualizations/balanced_ctu_demo.html"
    fig.write_html(output_file, config={'displayModeBar': True, 'displaylogo': False})
    print(f"Balanced CTU visualization saved to: {output_file}")
    
    # Print summary
    print("\n" + "="*80)
    print("BALANCED CTU GRAPH - DIVERSE ROLE TYPES")
    print("="*80)
    print(f"Scheme: {data['scheme_name']}")
    print(f"Showing: {len(subgraph.nodes())} CTUs with {len(subgraph.edges())} relationships")
    
    # Show selected CTUs by role
    print("\nSelected CTUs by role:")
    print("-" * 50)
    for role in sorted(selected_role_counts.keys()):
        print(f"\n{role}:")
        for node in subgraph.nodes():
            if subgraph.nodes[node]['role'] == role:
                ctu_data = subgraph.nodes[node]
                text = ctu_data['sentence'][:60] + "..." if len(ctu_data['sentence']) > 60 else ctu_data['sentence']
                print(f"  • {text}")
    
    return output_file

if __name__ == "__main__":
    print("Creating balanced CTU graph visualization...")
    output_file = create_balanced_visualization()
    print(f"\n✅ Balanced CTU visualization created: {output_file}")
    print("\nThis should have a good mix of different role types!")
