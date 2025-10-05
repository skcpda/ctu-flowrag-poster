#!/usr/bin/env python3
"""
Small Document CTU Demo

Finds a document with fewer CTUs and creates a clean visualization for it.
"""

import os
import sys
import json
import networkx as nx
import plotly.graph_objects as go
import numpy as np
from pathlib import Path

def find_small_document():
    """Find a document with fewer CTUs."""
    relations_dir = Path("../output_data/ctu_relations_production_ready")
    json_files = list(relations_dir.glob("*_production_ready.json"))
    
    print("Searching for documents with fewer CTUs...")
    
    # Check files to find one with fewer CTUs
    for file_path in json_files[:30]:  # Check first 30 files
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                ctus = len(data.get('ctus', []))
                if ctus <= 30:  # Look for docs with 30 or fewer CTUs
                    name = data.get('scheme_name', 'Unknown')
                    print(f"Found: {file_path.name} - {ctus} CTUs - {name}")
                    return str(file_path), data
        except Exception as e:
            print(f"Error with {file_path.name}: {e}")
            continue
    
    # If no small docs found, use the first one
    print("No small documents found, using first available...")
    with open(json_files[0], 'r', encoding='utf-8') as f:
        data = json.load(f)
    return str(json_files[0]), data

def create_small_doc_visualization():
    """Create visualization for a smaller document."""
    file_path, data = find_small_document()
    
    print(f"\nCreating visualization for: {data['scheme_name']}")
    print(f"CTUs: {len(data['ctus'])}, Relations: {len(data['relations'])}")
    
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
    pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
    
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
        node_colors.append(role_colors.get(role, '#808080'))
        node_sizes.append(20 + confidence * 15)  # Larger nodes for smaller graph
        node_labels.append(f"{role}<br>{sentence[:25]}...")
    
    # Prepare edge data
    edge_x = []
    edge_y = []
    
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
    
    # Create edge trace
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=3, color='rgba(125,125,125,0.6)'),
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
        textfont=dict(size=10, color="white", family="Arial"),
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
                           text=f"Small CTU Graph: {data['scheme_name']}<br><sub>Hover for details • Pan and zoom to explore</sub>",
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
    
    # Save the visualization
    output_file = "../demo_visualizations/small_doc_interactive.html"
    fig.write_html(output_file, config={'displayModeBar': True, 'displaylogo': False})
    print(f"Small document visualization saved to: {output_file}")
    
    # Print summary
    print("\n" + "="*80)
    print("SMALL DOCUMENT CTU GRAPH")
    print("="*80)
    print(f"Scheme: {data['scheme_name']}")
    print(f"Total CTUs: {len(G.nodes())}")
    print(f"Total Relations: {len(G.edges())}")
    print(f"Average Confidence: {np.mean([r['confidence'] for r in data['relations']]):.2f}")
    
    # Show role distribution
    role_counts = {}
    for node in G.nodes():
        role = G.nodes[node]['role']
        role_counts[role] = role_counts.get(role, 0) + 1
    
    print("\nRole Distribution:")
    for role, count in sorted(role_counts.items()):
        print(f"  {role}: {count}")
    
    return output_file

if __name__ == "__main__":
    print("Creating small document CTU graph visualization...")
    output_file = create_small_doc_visualization()
    print(f"\n✅ Small document visualization created: {output_file}")
    print("\nThis should be much cleaner and easier to explore!")
