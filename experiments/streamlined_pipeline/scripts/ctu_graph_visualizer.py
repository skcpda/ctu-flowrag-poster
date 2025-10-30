#!/usr/bin/env python3
"""
CTU Graph Visualizer

This script creates interactive visualizations of CTU (Conceptual Text Units) graphs
from the relation data files. It supports multiple visualization formats and interactive
features for demonstration purposes.

Author: AI Assistant
Date: 2025-01-27
"""

import json
import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import numpy as np
from collections import defaultdict, Counter
import warnings
warnings.filterwarnings('ignore')

# Try to import plotly for interactive visualizations
try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    print("Warning: Plotly not available. Interactive features will be limited.")

class CTUGraphVisualizer:
    """Main class for visualizing CTU graphs from relation data."""
    
    def __init__(self, data_file: str):
        """Initialize the visualizer with a CTU relation data file."""
        self.data_file = data_file
        self.data = self._load_data()
        self.graph = self._build_graph()
        self.role_colors = self._get_role_colors()
        
    def _load_data(self) -> Dict:
        """Load the CTU relation data from JSON file."""
        try:
            with open(self.data_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"Error: File {self.data_file} not found.")
            sys.exit(1)
        except json.JSONDecodeError:
            print(f"Error: Invalid JSON in {self.data_file}")
            sys.exit(1)
    
    def _build_graph(self) -> nx.DiGraph:
        """Build a NetworkX directed graph from the CTU relations."""
        G = nx.DiGraph()
        
        # Add nodes (CTUs)
        for ctu in self.data['ctus']:
            node_id = f"{ctu['sid']}_{ctu['line_idx']}"
            G.add_node(node_id, **ctu)
        
        # Add edges (relations)
        for relation in self.data['relations']:
            ctu1_id = f"{relation['ctu1']['sid']}_{relation['ctu1']['line_idx']}"
            ctu2_id = f"{relation['ctu2']['sid']}_{relation['ctu2']['line_idx']}"
            
            G.add_edge(
                ctu1_id, 
                ctu2_id,
                relation=relation['relation'],
                confidence=relation['confidence'],
                method=relation['method']
            )
        
        return G
    
    def _get_role_colors(self) -> Dict[str, str]:
        """Define color mapping for different CTU roles."""
        return {
            'ContextObjective': '#2E86AB',      # Blue
            'BenefitsAssistance': '#A23B72',   # Purple
            'Eligibility': '#F18F01',          # Orange
            'ApplicationProcess': '#C73E1D',   # Red
            'AuthoritiesGovernance': '#7209B7', # Dark Purple
            'TimelineFrequency': '#06A77D',    # Teal
            'DefinitionsReferences': '#F4A261'  # Light Orange
        }
    
    def get_graph_stats(self) -> Dict:
        """Get comprehensive statistics about the CTU graph."""
        stats = {
            'scheme_name': self.data['scheme_name'],
            'total_ctus': len(self.data['ctus']),
            'total_relations': len(self.data['relations']),
            'nodes': self.graph.number_of_nodes(),
            'edges': self.graph.number_of_edges(),
            'role_distribution': dict(self.data['role_tagging']['role_distribution']),
            'relation_types': Counter([r['relation'] for r in self.data['relations']]),
            'avg_confidence': np.mean([r['confidence'] for r in self.data['relations']]),
            'sections': len(set(ctu['section_name'] for ctu in self.data['ctus']))
        }
        return stats
    
    def create_static_visualization(self, output_file: str = None, 
                                  layout: str = 'hierarchical',
                                  max_nodes: int = 50) -> str:
        """Create a static matplotlib visualization of the CTU graph."""
        
        # Limit nodes for readability
        if self.graph.number_of_nodes() > max_nodes:
            print(f"Warning: Graph has {self.graph.number_of_nodes()} nodes. "
                  f"Showing first {max_nodes} nodes for readability.")
            nodes_to_show = list(self.graph.nodes())[:max_nodes]
            subgraph = self.graph.subgraph(nodes_to_show)
        else:
            subgraph = self.graph
        
        # Create figure
        plt.figure(figsize=(20, 16))
        
        # Choose layout
        if layout == 'hierarchical':
            pos = self._hierarchical_layout(subgraph)
        elif layout == 'spring':
            pos = nx.spring_layout(subgraph, k=3, iterations=50)
        elif layout == 'circular':
            pos = nx.circular_layout(subgraph)
        else:
            pos = nx.spring_layout(subgraph, k=2, iterations=30)
        
        # Draw nodes
        node_colors = []
        node_sizes = []
        for node in subgraph.nodes():
            role = subgraph.nodes[node]['role']
            node_colors.append(self.role_colors.get(role, '#808080'))
            # Size based on confidence
            confidence = subgraph.nodes[node].get('confidence', 0.5)
            node_sizes.append(300 + confidence * 200)
        
        # Draw edges
        edge_colors = []
        edge_widths = []
        for edge in subgraph.edges():
            confidence = subgraph.edges[edge]['confidence']
            edge_colors.append(plt.cm.RdYlBu(confidence))
            edge_widths.append(1 + confidence * 2)
        
        # Draw the graph
        nx.draw_networkx_nodes(subgraph, pos, 
                              node_color=node_colors,
                              node_size=node_sizes,
                              alpha=0.8)
        
        nx.draw_networkx_edges(subgraph, pos,
                              edge_color=edge_colors,
                              width=edge_widths,
                              alpha=0.6,
                              arrows=True,
                              arrowsize=20,
                              arrowstyle='->')
        
        # Add labels (truncated for readability)
        labels = {}
        for node in subgraph.nodes():
            text = subgraph.nodes[node]['sentence'][:50] + "..." if len(subgraph.nodes[node]['sentence']) > 50 else subgraph.nodes[node]['sentence']
            labels[node] = text
        
        nx.draw_networkx_labels(subgraph, pos, labels, font_size=8, font_weight='bold')
        
        # Create legend
        legend_elements = []
        for role, color in self.role_colors.items():
            if role in [node[1]['role'] for node in subgraph.nodes(data=True)]:
                legend_elements.append(mpatches.Patch(color=color, label=role))
        
        plt.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(0, 1))
        
        # Add title and stats
        stats = self.get_graph_stats()
        title = f"CTU Graph: {stats['scheme_name']}\n"
        title += f"Nodes: {stats['nodes']}, Edges: {stats['edges']}, "
        title += f"Avg Confidence: {stats['avg_confidence']:.2f}"
        
        plt.title(title, fontsize=16, fontweight='bold', pad=20)
        plt.axis('off')
        plt.tight_layout()
        
        # Save or show
        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"Static visualization saved to: {output_file}")
            return output_file
        else:
            plt.show()
            return "displayed"
    
    def _hierarchical_layout(self, graph: nx.DiGraph) -> Dict:
        """Create a hierarchical layout for the graph."""
        pos = {}
        
        # Group nodes by section
        sections = defaultdict(list)
        for node in graph.nodes():
            section = graph.nodes[node]['section_name']
            sections[section].append(node)
        
        # Position nodes by section
        y_offset = 0
        for section, nodes in sections.items():
            x_positions = np.linspace(0, 10, len(nodes))
            for i, node in enumerate(nodes):
                pos[node] = (x_positions[i], y_offset)
            y_offset += 2
        
        return pos
    
    def create_interactive_visualization(self, output_file: str = None) -> str:
        """Create an interactive Plotly visualization."""
        if not PLOTLY_AVAILABLE:
            print("Error: Plotly not available. Install with: pip install plotly")
            return None
        
        # Prepare data for plotly
        edge_x = []
        edge_y = []
        edge_info = []
        
        for edge in self.graph.edges(data=True):
            x0, y0 = 0, 0  # Simplified positioning
            x1, y1 = 1, 1
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
            edge_info.append(f"Relation: {edge[2]['relation']}<br>Confidence: {edge[2]['confidence']:.2f}")
        
        # Create edge trace
        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=2, color='#888'),
            hoverinfo='none',
            mode='lines'
        )
        
        # Create node trace
        node_x = []
        node_y = []
        node_text = []
        node_colors = []
        
        for node in self.graph.nodes(data=True):
            node_x.append(0)  # Simplified positioning
            node_y.append(0)
            node_text.append(f"Role: {node[1]['role']}<br>Text: {node[1]['sentence'][:100]}...")
            node_colors.append(self.role_colors.get(node[1]['role'], '#808080'))
        
        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            hoverinfo='text',
            text=node_text,
            marker=dict(
                size=20,
                color=node_colors,
                line=dict(width=2, color='white')
            )
        )
        
        # Create figure
        fig = go.Figure(data=[edge_trace, node_trace],
                      layout=go.Layout(
                          title=dict(text=f"Interactive CTU Graph: {self.data['scheme_name']}", font=dict(size=16)),
                          showlegend=False,
                          hovermode='closest',
                          margin=dict(b=20,l=5,r=5,t=40),
                          annotations=[ dict(
                              text="Interactive CTU Graph Visualization",
                              showarrow=False,
                              xref="paper", yref="paper",
                              x=0.005, y=-0.002,
                              xanchor='left', yanchor='bottom',
                              font=dict(color="black", size=12)
                          )],
                          xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                          yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
                      ))
        
        if output_file:
            fig.write_html(output_file)
            print(f"Interactive visualization saved to: {output_file}")
            return output_file
        else:
            fig.show()
            return "displayed"
    
    def create_summary_dashboard(self, output_file: str = None) -> str:
        """Create a comprehensive dashboard with multiple visualizations."""
        if not PLOTLY_AVAILABLE:
            print("Error: Plotly not available for dashboard creation.")
            return None
        
        stats = self.get_graph_stats()
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Role Distribution', 'Relation Types', 
                          'Confidence Distribution', 'Graph Overview'),
            specs=[[{"type": "pie"}, {"type": "bar"}],
                   [{"type": "histogram"}, {"type": "scatter"}]]
        )
        
        # Role distribution pie chart
        roles = list(stats['role_distribution'].keys())
        role_counts = list(stats['role_distribution'].values())
        role_colors_list = [self.role_colors.get(role, '#808080') for role in roles]
        
        fig.add_trace(
            go.Pie(labels=roles, values=role_counts, name="Roles"),
            row=1, col=1
        )
        
        # Relation types bar chart
        rel_types = list(stats['relation_types'].keys())
        rel_counts = list(stats['relation_types'].values())
        
        fig.add_trace(
            go.Bar(x=rel_types, y=rel_counts, name="Relations"),
            row=1, col=2
        )
        
        # Confidence distribution
        confidences = [r['confidence'] for r in self.data['relations']]
        fig.add_trace(
            go.Histogram(x=confidences, name="Confidence"),
            row=2, col=1
        )
        
        # Graph overview (simplified)
        node_roles = [self.graph.nodes[node]['role'] for node in self.graph.nodes()]
        role_positions = {role: i for i, role in enumerate(set(node_roles))}
        x_pos = [role_positions[role] for role in node_roles]
        y_pos = [np.random.random() for _ in range(len(node_roles))]
        
        fig.add_trace(
            go.Scatter(x=x_pos, y=y_pos, mode='markers', 
                      marker=dict(size=10, color=[self.role_colors.get(role, '#808080') for role in node_roles]),
                      name="Nodes"),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            title_text=f"CTU Graph Dashboard: {stats['scheme_name']}",
            showlegend=False,
            height=800
        )
        
        if output_file:
            fig.write_html(output_file)
            print(f"Dashboard saved to: {output_file}")
            return output_file
        else:
            fig.show()
            return "displayed"
    
    def export_graph_data(self, output_file: str) -> str:
        """Export the graph data in various formats for further analysis."""
        # Export as GraphML for network analysis tools
        nx.write_graphml(self.graph, output_file.replace('.json', '.graphml'))
        
        # Export as JSON for web visualization
        graph_data = {
            'nodes': [
                {
                    'id': node,
                    'role': data['role'],
                    'sentence': data['sentence'],
                    'confidence': data['confidence'],
                    'section': data['section_name']
                }
                for node, data in self.graph.nodes(data=True)
            ],
            'edges': [
                {
                    'source': edge[0],
                    'target': edge[1],
                    'relation': data['relation'],
                    'confidence': data['confidence']
                }
                for edge, data in self.graph.edges(data=True)
            ],
            'metadata': self.get_graph_stats()
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(graph_data, f, indent=2, ensure_ascii=False)
        
        print(f"Graph data exported to: {output_file}")
        return output_file
    
    def print_summary(self):
        """Print a text summary of the CTU graph."""
        stats = self.get_graph_stats()
        
        print("=" * 80)
        print(f"CTU GRAPH SUMMARY: {stats['scheme_name']}")
        print("=" * 80)
        print(f"Total CTUs: {stats['total_ctus']}")
        print(f"Total Relations: {stats['total_relations']}")
        print(f"Graph Nodes: {stats['nodes']}")
        print(f"Graph Edges: {stats['edges']}")
        print(f"Average Confidence: {stats['avg_confidence']:.3f}")
        print(f"Sections: {stats['sections']}")
        print()
        
        print("ROLE DISTRIBUTION:")
        print("-" * 40)
        for role, count in stats['role_distribution'].items():
            percentage = (count / stats['total_ctus']) * 100
            print(f"{role:25} {count:3d} ({percentage:5.1f}%)")
        print()
        
        print("RELATION TYPES:")
        print("-" * 40)
        for rel_type, count in stats['relation_types'].items():
            percentage = (count / stats['total_relations']) * 100
            print(f"{rel_type:25} {count:3d} ({percentage:5.1f}%)")
        print()
        
        print("GRAPH CONNECTIVITY:")
        print("-" * 40)
        if nx.is_weakly_connected(self.graph):
            print("✓ Graph is weakly connected")
        else:
            print("✗ Graph has disconnected components")
        
        if nx.is_strongly_connected(self.graph):
            print("✓ Graph is strongly connected")
        else:
            print("✗ Graph is not strongly connected")
        
        print(f"Average clustering coefficient: {nx.average_clustering(self.graph.to_undirected()):.3f}")
        print("=" * 80)


def main():
    """Main function to run the CTU graph visualizer."""
    parser = argparse.ArgumentParser(description='Visualize CTU graphs from relation data')
    parser.add_argument('data_file', help='Path to the CTU relation JSON file')
    parser.add_argument('--output-dir', default='./visualizations', 
                       help='Output directory for visualizations (default: ./visualizations)')
    parser.add_argument('--format', choices=['static', 'interactive', 'dashboard', 'all'], 
                       default='all', help='Visualization format')
    parser.add_argument('--layout', choices=['hierarchical', 'spring', 'circular'], 
                       default='hierarchical', help='Layout for static visualization')
    parser.add_argument('--max-nodes', type=int, default=50, 
                       help='Maximum nodes to show in static visualization')
    parser.add_argument('--export-data', action='store_true', 
                       help='Export graph data in various formats')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize visualizer
    print(f"Loading CTU data from: {args.data_file}")
    visualizer = CTUGraphVisualizer(args.data_file)
    
    # Print summary
    visualizer.print_summary()
    
    # Generate base filename
    base_name = Path(args.data_file).stem
    output_dir = Path(args.output_dir)
    
    # Create visualizations based on format
    if args.format in ['static', 'all']:
        print("\nCreating static visualization...")
        static_file = output_dir / f"{base_name}_static.png"
        visualizer.create_static_visualization(
            str(static_file), 
            layout=args.layout, 
            max_nodes=args.max_nodes
        )
    
    if args.format in ['interactive', 'all'] and PLOTLY_AVAILABLE:
        print("\nCreating interactive visualization...")
        interactive_file = output_dir / f"{base_name}_interactive.html"
        visualizer.create_interactive_visualization(str(interactive_file))
    
    if args.format in ['dashboard', 'all'] and PLOTLY_AVAILABLE:
        print("\nCreating dashboard...")
        dashboard_file = output_dir / f"{base_name}_dashboard.html"
        visualizer.create_summary_dashboard(str(dashboard_file))
    
    if args.export_data:
        print("\nExporting graph data...")
        data_file = output_dir / f"{base_name}_graph_data.json"
        visualizer.export_graph_data(str(data_file))
    
    print(f"\nVisualization complete! Check the '{args.output_dir}' directory for outputs.")


if __name__ == "__main__":
    main()
