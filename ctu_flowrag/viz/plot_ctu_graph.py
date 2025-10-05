"""
CTU graph visualization.

Reads production JSON and creates static visualizations of CTU graphs
with role-colored nodes, type-styled edges, and legends.
"""

import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import argparse
import logging

from ..data_io.load_json_graph import load_json_graph
from ..data_io.compat_allowlist import is_compatible

logger = logging.getLogger(__name__)

class CTUGraphVisualizer:
    """
    Visualizer for CTU graphs.
    
    Creates static visualizations of CTU graphs with proper
    role coloring, edge styling, and legends.
    """
    
    def __init__(self, 
                 roles: List[str],
                 edge_types: List[str]):
        self.roles = roles
        self.edge_types = edge_types
        
        # Color mapping for roles
        self.role_colors = {
            'ContextObjective': '#2E86AB',      # Blue
            'BenefitsAssistance': '#A23B72',   # Purple  
            'Eligibility': '#F18F01',          # Orange
            'ApplicationProcess': '#C73E1D',   # Red
            'TimelineFrequency': '#06A77D',    # Teal
            'AuthoritiesGovernance': '#7209B7', # Dark Purple
            'DefinitionsReferences': '#F4A261'  # Light Orange
        }
        
        # Edge style mapping
        self.edge_styles = {
            'PRECEDES': 'solid',
            'SEGMENT_CONTINUATION': 'dashed',
            'PREREQUISITE_OF': 'dotted',
            'ENABLES': 'dashdot',
            'CAP_LIMITS': 'solid',
            'RATE_SPEC': 'dashed',
            'ADMINISTERED_BY': 'dotted',
            'TIMELINE_FOR': 'dashdot'
        }
        
        # Edge colors
        self.edge_colors = {
            'PRECEDES': '#666666',
            'SEGMENT_CONTINUATION': '#999999',
            'PREREQUISITE_OF': '#FF6B6B',
            'ENABLES': '#4ECDC4',
            'CAP_LIMITS': '#45B7D1',
            'RATE_SPEC': '#96CEB4',
            'ADMINISTERED_BY': '#FFEAA7',
            'TIMELINE_FOR': '#DDA0DD'
        }
    
    def visualize_graph(self, 
                       graph_data: Dict[str, Any],
                       output_path: str,
                       max_nodes: int = 40,
                       center_node: Optional[str] = None) -> None:
        """
        Create visualization of CTU graph.
        
        Args:
            graph_data: Graph data from load_json_graph
            output_path: Path to save visualization
            max_nodes: Maximum number of nodes to show
            center_node: Node to center the graph on
        """
        # Build NetworkX graph
        G = self._build_networkx_graph(graph_data)
        
        if len(G.nodes()) == 0:
            logger.warning("Empty graph, skipping visualization")
            return
        
        # Select subgraph if too many nodes
        if len(G.nodes()) > max_nodes:
            G = self._select_subgraph(G, max_nodes, center_node)
        
        # Create figure
        plt.figure(figsize=(20, 16))
        
        # Use spring layout for positioning
        pos = nx.spring_layout(G, k=3, iterations=50, seed=42)
        
        # Draw edges by type
        self._draw_edges_by_type(G, pos)
        
        # Draw nodes
        self._draw_nodes(G, pos)
        
        # Add labels
        self._draw_labels(G, pos)
        
        # Add legends
        self._add_legends()
        
        # Add title
        scheme_name = graph_data['metadata']['scheme_name']
        plt.title(f"CTU Graph: {scheme_name}\n{len(G.nodes())} nodes, {len(G.edges())} edges", 
                 fontsize=16, fontweight='bold', pad=20)
        
        # Remove axes
        plt.axis('off')
        
        # Save figure
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        logger.info(f"Saved visualization to {output_path}")
    
    def _build_networkx_graph(self, graph_data: Dict[str, Any]) -> nx.DiGraph:
        """Build NetworkX graph from graph data."""
        G = nx.DiGraph()
        
        # Add nodes
        for node in graph_data['nodes']:
            G.add_node(node['id'], **node)
        
        # Add edges
        for edge_type, edges in graph_data['edges_by_type'].items():
            for edge in edges:
                G.add_edge(edge['src'], edge['dst'], 
                          edge_type=edge_type,
                          conf=edge['conf_cal'],
                          compat=edge.get('same_section', True))
        
        return G
    
    def _select_subgraph(self, 
                        G: nx.DiGraph, 
                        max_nodes: int, 
                        center_node: Optional[str] = None) -> nx.DiGraph:
        """Select subgraph with maximum number of nodes."""
        if center_node and center_node in G.nodes():
            # Start from center node and expand
            subgraph_nodes = {center_node}
            queue = [center_node]
            
            while len(subgraph_nodes) < max_nodes and queue:
                current = queue.pop(0)
                
                # Add neighbors
                for neighbor in G.neighbors(current):
                    if neighbor not in subgraph_nodes and len(subgraph_nodes) < max_nodes:
                        subgraph_nodes.add(neighbor)
                        queue.append(neighbor)
                
                # Add predecessors
                for predecessor in G.predecessors(current):
                    if predecessor not in subgraph_nodes and len(subgraph_nodes) < max_nodes:
                        subgraph_nodes.add(predecessor)
                        queue.append(predecessor)
        else:
            # Select nodes with highest degree
            degrees = dict(G.degree())
            sorted_nodes = sorted(degrees.items(), key=lambda x: x[1], reverse=True)
            subgraph_nodes = {node for node, _ in sorted_nodes[:max_nodes]}
        
        return G.subgraph(subgraph_nodes)
    
    def _draw_edges_by_type(self, G: nx.DiGraph, pos: Dict[str, Tuple[float, float]]):
        """Draw edges grouped by type."""
        for edge_type in self.edge_types:
            edges = [(u, v) for u, v, d in G.edges(data=True) if d.get('edge_type') == edge_type]
            
            if not edges:
                continue
            
            # Get edge properties
            edge_weights = [G[u][v].get('conf', 0.5) for u, v in edges]
            edge_colors = [self.edge_colors.get(edge_type, '#666666') for _ in edges]
            
            # Draw edges
            nx.draw_networkx_edges(
                G, pos,
                edgelist=edges,
                edge_color=edge_colors,
                width=[w * 3 for w in edge_weights],
                style=self.edge_styles.get(edge_type, 'solid'),
                alpha=0.7,
                arrows=True,
                arrowsize=20,
                arrowstyle='->'
            )
    
    def _draw_nodes(self, G: nx.DiGraph, pos: Dict[str, Tuple[float, float]]):
        """Draw nodes with role-based coloring."""
        # Group nodes by role
        role_groups = {}
        for node in G.nodes():
            role = G.nodes[node]['role']
            if role not in role_groups:
                role_groups[role] = []
            role_groups[role].append(node)
        
        # Draw nodes for each role
        for role, nodes in role_groups.items():
            color = self.role_colors.get(role, '#808080')
            
            # Get node sizes based on degree
            sizes = [G.degree(node) * 100 + 200 for node in nodes]
            
            nx.draw_networkx_nodes(
                G, pos,
                nodelist=nodes,
                node_color=color,
                node_size=sizes,
                alpha=0.8,
                edgecolors='black',
                linewidths=2
            )
    
    def _draw_labels(self, G: nx.DiGraph, pos: Dict[str, Tuple[float, float]]):
        """Draw node labels."""
        # Create label dictionary
        labels = {}
        for node in G.nodes():
            role = G.nodes[node]['role']
            text = G.nodes[node]['text']
            
            # Truncate text for labels
            if len(text) > 30:
                text = text[:27] + "..."
            
            labels[node] = f"{role}\n{text}"
        
        # Draw labels
        nx.draw_networkx_labels(
            G, pos,
            labels=labels,
            font_size=8,
            font_weight='bold',
            bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8)
        )
    
    def _add_legends(self):
        """Add legends for roles and edge types."""
        # Role legend
        role_patches = []
        for role, color in self.role_colors.items():
            role_patches.append(mpatches.Patch(color=color, label=role))
        
        plt.legend(handles=role_patches, 
                  loc='upper left', 
                  bbox_to_anchor=(0.02, 0.98),
                  fontsize=10,
                  title="Roles",
                  title_fontsize=12)
        
        # Edge type legend
        edge_patches = []
        for edge_type, color in self.edge_colors.items():
            edge_patches.append(mpatches.Patch(color=color, label=edge_type))
        
        plt.legend(handles=edge_patches,
                  loc='upper right',
                  bbox_to_anchor=(0.98, 0.98),
                  fontsize=10,
                  title="Edge Types",
                  title_fontsize=12)
    
    def create_summary_visualization(self, 
                                   graph_data: Dict[str, Any],
                                   output_path: str) -> None:
        """
        Create summary visualization with statistics.
        
        Args:
            graph_data: Graph data from load_json_graph
            output_path: Path to save visualization
        """
        # Create subplot layout
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))
        
        # 1. Role distribution
        self._plot_role_distribution(graph_data, ax1)
        
        # 2. Edge type distribution
        self._plot_edge_type_distribution(graph_data, ax2)
        
        # 3. Confidence distribution
        self._plot_confidence_distribution(graph_data, ax3)
        
        # 4. Graph statistics
        self._plot_graph_statistics(graph_data, ax4)
        
        # Add main title
        scheme_name = graph_data['metadata']['scheme_name']
        fig.suptitle(f"CTU Graph Summary: {scheme_name}", fontsize=16, fontweight='bold')
        
        # Save figure
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        logger.info(f"Saved summary visualization to {output_path}")
    
    def _plot_role_distribution(self, graph_data: Dict[str, Any], ax):
        """Plot role distribution pie chart."""
        role_counts = {}
        for node in graph_data['nodes']:
            role = node['role']
            role_counts[role] = role_counts.get(role, 0) + 1
        
        if not role_counts:
            ax.text(0.5, 0.5, 'No roles found', ha='center', va='center')
            return
        
        labels = list(role_counts.keys())
        sizes = list(role_counts.values())
        colors = [self.role_colors.get(role, '#808080') for role in labels]
        
        ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        ax.set_title('Role Distribution')
    
    def _plot_edge_type_distribution(self, graph_data: Dict[str, Any], ax):
        """Plot edge type distribution bar chart."""
        edge_type_counts = {}
        for edge_type, edges in graph_data['edges_by_type'].items():
            edge_type_counts[edge_type] = len(edges)
        
        if not edge_type_counts:
            ax.text(0.5, 0.5, 'No edges found', ha='center', va='center')
            return
        
        edge_types = list(edge_type_counts.keys())
        counts = list(edge_type_counts.values())
        colors = [self.edge_colors.get(edge_type, '#666666') for edge_type in edge_types]
        
        bars = ax.bar(edge_types, counts, color=colors)
        ax.set_title('Edge Type Distribution')
        ax.set_ylabel('Number of Edges')
        ax.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, count in zip(bars, counts):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                   str(count), ha='center', va='bottom')
    
    def _plot_confidence_distribution(self, graph_data: Dict[str, Any], ax):
        """Plot confidence distribution histogram."""
        all_confidences = []
        for edges in graph_data['edges_by_type'].values():
            for edge in edges:
                all_confidences.append(edge['conf_cal'])
        
        if not all_confidences:
            ax.text(0.5, 0.5, 'No confidence data', ha='center', va='center')
            return
        
        ax.hist(all_confidences, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        ax.set_title('Confidence Distribution')
        ax.set_xlabel('Confidence')
        ax.set_ylabel('Frequency')
        ax.axvline(np.mean(all_confidences), color='red', linestyle='--', 
                  label=f'Mean: {np.mean(all_confidences):.3f}')
        ax.legend()
    
    def _plot_graph_statistics(self, graph_data: Dict[str, Any], ax):
        """Plot graph statistics text."""
        stats = {
            'Total Nodes': graph_data['metadata']['num_nodes'],
            'Total Edges': graph_data['metadata']['num_edges'],
            'Edge Types': len(graph_data['edges_by_type']),
            'Roles': len(set(node['role'] for node in graph_data['nodes']))
        }
        
        # Add confidence statistics
        all_confidences = []
        for edges in graph_data['edges_by_type'].values():
            for edge in edges:
                all_confidences.append(edge['conf_cal'])
        
        if all_confidences:
            stats['Mean Confidence'] = f"{np.mean(all_confidences):.3f}"
            stats['Min Confidence'] = f"{np.min(all_confidences):.3f}"
            stats['Max Confidence'] = f"{np.max(all_confidences):.3f}"
        
        # Display statistics
        ax.text(0.1, 0.9, 'Graph Statistics', fontsize=14, fontweight='bold', 
                transform=ax.transAxes)
        
        y_pos = 0.8
        for key, value in stats.items():
            ax.text(0.1, y_pos, f"{key}: {value}", fontsize=12, 
                   transform=ax.transAxes)
            y_pos -= 0.1
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Visualize CTU graphs')
    parser.add_argument('--json_path', type=str, required=True, help='Path to JSON file')
    parser.add_argument('--output_dir', type=str, default='visualizations', help='Output directory')
    parser.add_argument('--max_nodes', type=int, default=40, help='Maximum number of nodes to show')
    parser.add_argument('--center_node', type=str, help='Node to center the graph on')
    parser.add_argument('--create_summary', action='store_true', help='Create summary visualization')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Load graph data
    graph_data = load_json_graph(args.json_path)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Define roles and edge types
    roles = ["ContextObjective", "BenefitsAssistance", "Eligibility", "ApplicationProcess", 
             "TimelineFrequency", "AuthoritiesGovernance", "DefinitionsReferences"]
    edge_types = ["PRECEDES", "SEGMENT_CONTINUATION", "PREREQUISITE_OF", "ENABLES", 
                  "CAP_LIMITS", "RATE_SPEC", "ADMINISTERED_BY", "TIMELINE_FOR"]
    
    # Create visualizer
    visualizer = CTUGraphVisualizer(roles, edge_types)
    
    # Create main visualization
    output_path = output_dir / f"{Path(args.json_path).stem}_graph.png"
    visualizer.visualize_graph(
        graph_data, 
        str(output_path),
        max_nodes=args.max_nodes,
        center_node=args.center_node
    )
    
    # Create summary visualization if requested
    if args.create_summary:
        summary_path = output_dir / f"{Path(args.json_path).stem}_summary.png"
        visualizer.create_summary_visualization(graph_data, str(summary_path))
    
    logger.info("Visualization completed!")

if __name__ == "__main__":
    main()

