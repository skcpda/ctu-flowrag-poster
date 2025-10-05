#!/usr/bin/env python3
"""
Better Interactive CTU Graph Demo

Creates a proper interactive network visualization with:
- Force-directed layout for natural positioning
- Pan and zoom capabilities
- Hover tooltips with full CTU text
- Click to highlight connections
- Better node positioning and edge visibility
"""

import os
import sys
import json
import networkx as nx
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
from pathlib import Path

def load_data():
    """Load the CTU relation data."""
    data_file = "../output_data/ctu_relations_production_ready/aa_ctus_production_ready.json"
    with open(data_file, 'r', encoding='utf-8') as f:
        return json.load(f)

def create_better_interactive_visualization():
    """Create a much better interactive CTU graph visualization."""
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
    
    # Use NetworkX spring layout for better positioning
    pos = nx.spring_layout(G, k=3, iterations=50, seed=42)
    
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
    node_info = []
    node_colors = []
    node_sizes = []
    node_labels = []
    
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        
        # Get node data
        ctu_data = G.nodes[node]
        role = ctu_data['role']
        confidence = ctu_data['confidence']
        sentence = ctu_data['sentence']
        
        # Create hover text
        hover_text = f"<b>{role}</b><br>"
        hover_text += f"Confidence: {confidence:.2f}<br>"
        hover_text += f"Section: {ctu_data['section_name']}<br><br>"
        hover_text += f"<i>{sentence}</i>"
        
        node_text.append(hover_text)
        node_info.append(f"{role}: {sentence[:50]}...")
        node_colors.append(role_colors.get(role, '#808080'))
        node_sizes.append(15 + confidence * 10)  # Size based on confidence
        node_labels.append(f"{role}<br>{sentence[:30]}...")
    
    # Prepare edge data
    edge_x = []
    edge_y = []
    edge_info = []
    
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        
        # Get edge data
        edge_data = G.edges[edge]
        relation = edge_data['relation']
        confidence = edge_data['confidence']
        
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
        edge_info.append(f"{relation} (conf: {confidence:.2f})")
    
    # Create edge trace
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=2, color='rgba(125,125,125,0.5)'),
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
        textfont=dict(size=8, color="white"),
        marker=dict(
            size=node_sizes,
            color=node_colors,
            line=dict(width=2, color='black'),
            opacity=0.8
        ),
        showlegend=False
    )
    
    # Create the figure
    fig = go.Figure(data=[edge_trace, node_trace],
                   layout=go.Layout(
                       title=dict(
                           text=f"Interactive CTU Graph: {data['scheme_name']}<br><sub>Hover over nodes for details • Click and drag to pan • Scroll to zoom</sub>",
                           font=dict(size=20)
                       ),
                       showlegend=False,
                       hovermode='closest',
                       margin=dict(b=20,l=5,r=5,t=80),
                       annotations=[dict(
                           text="Interactive CTU Graph - Pan, Zoom, and Explore!",
                           showarrow=False,
                           xref="paper", yref="paper",
                           x=0.005, y=-0.002,
                           xanchor='left', yanchor='bottom',
                           font=dict(color="gray", size=12)
                       )],
                       xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                       yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                       plot_bgcolor='white',
                       paper_bgcolor='white',
                       width=1200,
                       height=800
                   ))
    
    # Add role legend
    legend_annotations = []
    y_pos = 0.95
    for role, color in role_colors.items():
        if role in [G.nodes[node]['role'] for node in G.nodes()]:
            legend_annotations.append(dict(
                x=0.02, y=y_pos,
                xref="paper", yref="paper",
                text=f"<span style='color:{color}'>●</span> {role}",
                showarrow=False,
                font=dict(size=12),
                bgcolor="rgba(255,255,255,0.8)",
                bordercolor="gray",
                borderwidth=1
            ))
            y_pos -= 0.05
    
    fig.update_layout(annotations=list(fig.layout.annotations) + legend_annotations)
    
    # Save the interactive visualization
    output_file = "../demo_visualizations/aa_better_interactive.html"
    fig.write_html(output_file, config={'displayModeBar': True, 'displaylogo': False})
    print(f"Better interactive visualization saved to: {output_file}")
    
    # Print summary
    print("\n" + "="*80)
    print("BETTER INTERACTIVE CTU GRAPH")
    print("="*80)
    print(f"Scheme: {data['scheme_name']}")
    print(f"Total CTUs: {len(G.nodes())}")
    print(f"Total Relations: {len(G.edges())}")
    print(f"Average Confidence: {np.mean([r['confidence'] for r in data['relations']]):.2f}")
    print("\nFeatures:")
    print("- Pan and zoom with mouse")
    print("- Hover over nodes for full CTU text")
    print("- Color-coded by role type")
    print("- Node size based on confidence")
    print("- Force-directed layout for natural positioning")
    
    return output_file

if __name__ == "__main__":
    print("Creating better interactive CTU graph visualization...")
    output_file = create_better_interactive_visualization()
    print(f"\n✅ Better interactive visualization created: {output_file}")
    print("\nOpen in browser to explore with pan, zoom, and hover!")
