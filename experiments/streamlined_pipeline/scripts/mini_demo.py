#!/usr/bin/env python3
"""
Mini CTU Demo

Creates a very small, clean visualization with only the most important CTUs (8-12 nodes max).
Perfect for clear demonstration and exploration.
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

def create_mini_visualization():
    """Create a mini visualization with only the most important CTUs."""
    data = load_data()
    
    # Build full graph first
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
    
    # Select only the most important CTUs
    # Strategy: One representative from each role + highest confidence CTUs
    role_representatives = {}
    high_confidence_nodes = []
    
    # Get one representative from each role
    for node, data in G.nodes(data=True):
        role = data['role']
        if role not in role_representatives:
            role_representatives[role] = node
    
    # Get highest confidence nodes
    node_confidence = [(node, data['confidence']) for node, data in G.nodes(data=True)]
    node_confidence.sort(key=lambda x: x[1], reverse=True)
    
    # Select top nodes (max 10)
    selected_nodes = set()
    
    # Add role representatives (max 6)
    for role, node in list(role_representatives.items())[:6]:
        selected_nodes.add(node)
    
    # Add highest confidence nodes (max 4 more)
    for node, conf in node_confidence[:4]:
        if node not in selected_nodes:
            selected_nodes.add(node)
    
    # Create subgraph with only selected nodes
    subgraph = G.subgraph(selected_nodes)
    
    print(f"Selected {len(subgraph.nodes())} most important CTUs out of {len(G.nodes())} total")
    
    # Use a simple grid layout for better positioning
    pos = {}
    nodes = list(subgraph.nodes())
    
    # Arrange in a grid
    cols = 3
    rows = (len(nodes) + cols - 1) // cols
    
    for i, node in enumerate(nodes):
        row = i // cols
        col = i % cols
        pos[node] = (col * 3, -row * 2)
    
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
        node_sizes.append(40 + confidence * 20)  # Larger nodes
        node_labels.append(f"{role}<br>{sentence[:30]}...")
    
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
        line=dict(width=4, color='rgba(125,125,125,0.7)'),
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
        textfont=dict(size=12, color="white", family="Arial Black"),
        marker=dict(
            size=node_sizes,
            color=node_colors,
            line=dict(width=4, color='black'),
            opacity=0.9
        ),
        showlegend=False
    )
    
    # Create the figure
    fig = go.Figure(data=[edge_trace, node_trace],
                   layout=go.Layout(
                       title=dict(
                           text=f"Mini CTU Graph: {data['scheme_name']}<br><sub>Key Concepts Only - Hover for details</sub>",
                           font=dict(size=20)
                       ),
                       showlegend=False,
                       hovermode='closest',
                       margin=dict(b=20,l=5,r=5,t=80),
                       xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                       yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                       plot_bgcolor='white',
                       paper_bgcolor='white',
                       width=1000,
                       height=600
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
                font=dict(size=14, family="Arial Black"),
                bgcolor="rgba(255,255,255,0.9)",
                bordercolor="black",
                borderwidth=2
            ))
            y_pos -= 0.06
    
    fig.update_layout(annotations=list(fig.layout.annotations) + legend_annotations)
    
    # Save the visualization
    output_file = "../demo_visualizations/mini_ctu_demo.html"
    fig.write_html(output_file, config={'displayModeBar': True, 'displaylogo': False})
    print(f"Mini CTU visualization saved to: {output_file}")
    
    # Print summary
    print("\n" + "="*80)
    print("MINI CTU GRAPH - KEY CONCEPTS ONLY")
    print("="*80)
    print(f"Scheme: {data['scheme_name']}")
    print(f"Showing: {len(subgraph.nodes())} key CTUs (out of {len(G.nodes())} total)")
    print(f"Relations: {len(subgraph.edges())}")
    
    # Show selected CTUs
    print("\nSelected CTUs:")
    print("-" * 50)
    for node in subgraph.nodes():
        ctu_data = subgraph.nodes[node]
        role = ctu_data['role']
        confidence = ctu_data['confidence']
        text = ctu_data['sentence'][:60] + "..." if len(ctu_data['sentence']) > 60 else ctu_data['sentence']
        print(f"[{role:20}] (conf: {confidence:.2f}) {text}")
    
    return output_file

if __name__ == "__main__":
    print("Creating mini CTU graph visualization...")
    output_file = create_mini_visualization()
    print(f"\n✅ Mini CTU visualization created: {output_file}")
    print("\nThis shows only the most important concepts - much cleaner!")
