#!/usr/bin/env python3
"""
Rich CTU Demo

Finds a document with good connectivity and creates a mini visualization
with 8-12 nodes that have meaningful relationships between them.
"""

import os
import sys
import json
import networkx as nx
import plotly.graph_objects as go
import numpy as np
from pathlib import Path

def find_rich_document():
    """Find a document with good connectivity."""
    relations_dir = Path("../output_data/ctu_relations_production_ready")
    json_files = list(relations_dir.glob("*_production_ready.json"))
    
    print("Searching for documents with good connectivity...")
    
    best_doc = None
    best_score = 0
    
    # Check several files to find one with good connectivity
    for file_path in json_files[:20]:  # Check first 20 files
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                ctus = len(data.get('ctus', []))
                relations = len(data.get('relations', []))
                
                # Calculate connectivity score
                if ctus > 0:
                    connectivity_ratio = relations / ctus
                    if 10 <= ctus <= 40 and connectivity_ratio > 0.8:  # Good connectivity
                        score = ctus * connectivity_ratio
                        if score > best_score:
                            best_score = score
                            best_doc = (str(file_path), data)
                            print(f"Found candidate: {file_path.name} - {ctus} CTUs, {relations} relations (ratio: {connectivity_ratio:.2f})")
        except Exception as e:
            continue
    
    if best_doc:
        return best_doc
    else:
        # Fallback to first available
        print("No rich document found, using first available...")
        with open(json_files[0], 'r', encoding='utf-8') as f:
            data = json.load(f)
        return str(json_files[0]), data

def create_rich_visualization():
    """Create a rich mini visualization with good connectivity."""
    file_path, data = find_rich_document()
    
    print(f"\nCreating visualization for: {data['scheme_name']}")
    print(f"CTUs: {len(data['ctus'])}, Relations: {len(data['relations'])}")
    
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
    
    # Find a connected component with good size (8-15 nodes)
    connected_components = list(nx.weakly_connected_components(G))
    
    # Find the best component
    best_component = None
    for component in connected_components:
        if 8 <= len(component) <= 15:
            best_component = component
            break
    
    # If no perfect component, take the largest one and trim it
    if not best_component:
        largest_component = max(connected_components, key=len)
        # Select top nodes by centrality
        centrality = nx.degree_centrality(G.subgraph(largest_component))
        top_nodes = sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:12]
        best_component = [node for node, _ in top_nodes]
    
    # Create subgraph
    subgraph = G.subgraph(best_component)
    
    print(f"Selected {len(subgraph.nodes())} CTUs with {len(subgraph.edges())} relationships")
    
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
    
    # Prepare edge data with relationship info
    edge_x = []
    edge_y = []
    edge_info = []
    
    for edge in subgraph.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        
        edge_data = subgraph.edges[edge]
        relation = edge_data['relation']
        confidence = edge_data['confidence']
        
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
        edge_info.append(f"{relation} (conf: {confidence:.2f})")
    
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
                           text=f"Rich CTU Graph: {data['scheme_name']}<br><sub>Well-connected concepts - Hover for details</sub>",
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
    output_file = "../demo_visualizations/rich_ctu_demo.html"
    fig.write_html(output_file, config={'displayModeBar': True, 'displaylogo': False})
    print(f"Rich CTU visualization saved to: {output_file}")
    
    # Print summary
    print("\n" + "="*80)
    print("RICH CTU GRAPH - WELL-CONNECTED CONCEPTS")
    print("="*80)
    print(f"Scheme: {data['scheme_name']}")
    print(f"Showing: {len(subgraph.nodes())} CTUs with {len(subgraph.edges())} relationships")
    print(f"Connectivity: {len(subgraph.edges())/len(subgraph.nodes()):.2f} edges per node")
    
    # Show role distribution
    role_counts = {}
    for node in subgraph.nodes():
        role = subgraph.nodes[node]['role']
        role_counts[role] = role_counts.get(role, 0) + 1
    
    print("\nRole Distribution:")
    for role, count in sorted(role_counts.items()):
        print(f"  {role}: {count}")
    
    # Show some key relationships
    print("\nKey Relationships:")
    print("-" * 50)
    for edge in list(subgraph.edges())[:5]:  # Show first 5 relationships
        edge_data = subgraph.edges[edge]
        relation = edge_data['relation']
        confidence = edge_data['confidence']
        print(f"  {relation} (conf: {confidence:.2f})")
    
    return output_file

if __name__ == "__main__":
    print("Creating rich CTU graph visualization...")
    output_file = create_rich_visualization()
    print(f"\n✅ Rich CTU visualization created: {output_file}")
    print("\nThis should have much better connectivity and relationships!")
