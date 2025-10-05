#!/usr/bin/env python3
"""
Demo script for CTU Graph Visualization

This script provides an easy way to demonstrate CTU graphs to supervisors.
It automatically finds available relation files and creates visualizations.

Usage:
    python demo_visualization.py [scheme_name]
    
If no scheme name is provided, it will list available schemes and let you choose.
"""

import os
import sys
import json
from pathlib import Path
import argparse
from ctu_graph_visualizer import CTUGraphVisualizer

def find_available_schemes(relations_dir: str) -> list:
    """Find all available CTU relation files."""
    relations_path = Path(relations_dir)
    if not relations_path.exists():
        print(f"Error: Relations directory {relations_dir} not found.")
        return []
    
    json_files = list(relations_path.glob("*_production_ready.json"))
    schemes = []
    
    for file_path in json_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                schemes.append({
                    'file': str(file_path),
                    'name': data.get('scheme_name', file_path.stem),
                    'ctus': len(data.get('ctus', [])),
                    'relations': len(data.get('relations', []))
                })
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Warning: Could not parse {file_path}: {e}")
            continue
    
    return sorted(schemes, key=lambda x: x['name'])

def display_scheme_menu(schemes: list):
    """Display a menu of available schemes."""
    print("\n" + "="*80)
    print("AVAILABLE CTU SCHEMES")
    print("="*80)
    print(f"{'#':<3} {'Scheme Name':<40} {'CTUs':<8} {'Relations':<10} {'File'}")
    print("-"*80)
    
    for i, scheme in enumerate(schemes, 1):
        name = scheme['name'][:37] + "..." if len(scheme['name']) > 40 else scheme['name']
        file_name = Path(scheme['file']).name
        print(f"{i:<3} {name:<40} {scheme['ctus']:<8} {scheme['relations']:<10} {file_name}")
    
    print("="*80)
    return len(schemes)

def get_user_choice(max_choice: int) -> int:
    """Get user's choice from the menu."""
    while True:
        try:
            choice = input(f"\nEnter your choice (1-{max_choice}, or 'q' to quit): ").strip()
            if choice.lower() == 'q':
                return None
            choice_num = int(choice)
            if 1 <= choice_num <= max_choice:
                return choice_num - 1  # Convert to 0-based index
            else:
                print(f"Please enter a number between 1 and {max_choice}")
        except ValueError:
            print("Please enter a valid number or 'q' to quit")

def create_demo_visualizations(visualizer: CTUGraphVisualizer, output_dir: str, scheme_name: str):
    """Create all types of visualizations for demonstration."""
    print(f"\nCreating visualizations for: {scheme_name}")
    print("-" * 50)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Static visualization (hierarchical layout)
    print("1. Creating static visualization (hierarchical layout)...")
    static_file = os.path.join(output_dir, f"{scheme_name}_hierarchical.png")
    visualizer.create_static_visualization(static_file, layout='hierarchical', max_nodes=30)
    
    # 2. Static visualization (spring layout)
    print("2. Creating static visualization (spring layout)...")
    spring_file = os.path.join(output_dir, f"{scheme_name}_spring.png")
    visualizer.create_static_visualization(spring_file, layout='spring', max_nodes=30)
    
    # 3. Interactive visualization (if plotly available)
    try:
        import plotly
        print("3. Creating interactive visualization...")
        interactive_file = os.path.join(output_dir, f"{scheme_name}_interactive.html")
        visualizer.create_interactive_visualization(interactive_file)
    except ImportError:
        print("3. Skipping interactive visualization (plotly not available)")
    
    # 4. Dashboard (if plotly available)
    try:
        import plotly
        print("4. Creating dashboard...")
        dashboard_file = os.path.join(output_dir, f"{scheme_name}_dashboard.html")
        visualizer.create_summary_dashboard(dashboard_file)
    except ImportError:
        print("4. Skipping dashboard (plotly not available)")
    
    # 5. Export graph data
    print("5. Exporting graph data...")
    data_file = os.path.join(output_dir, f"{scheme_name}_graph_data.json")
    visualizer.export_graph_data(data_file)
    
    print(f"\n✅ All visualizations created in: {output_dir}")
    print("\nFiles created:")
    for file in os.listdir(output_dir):
        if scheme_name in file:
            print(f"  - {file}")

def main():
    """Main demo function."""
    parser = argparse.ArgumentParser(description='Demo CTU Graph Visualization')
    parser.add_argument('scheme_name', nargs='?', help='Name of the scheme to visualize')
    parser.add_argument('--relations-dir', default='../output_data/ctu_relations_production_ready',
                       help='Directory containing relation files')
    parser.add_argument('--output-dir', default='./demo_visualizations',
                       help='Output directory for visualizations')
    parser.add_argument('--list-only', action='store_true',
                       help='Only list available schemes')
    
    args = parser.parse_args()
    
    # Find available schemes
    print("Scanning for available CTU relation files...")
    schemes = find_available_schemes(args.relations_dir)
    
    if not schemes:
        print("No CTU relation files found!")
        return
    
    if args.list_only:
        display_scheme_menu(schemes)
        return
    
    # Select scheme
    if args.scheme_name:
        # Find scheme by name
        selected_scheme = None
        for scheme in schemes:
            if args.scheme_name.lower() in scheme['name'].lower():
                selected_scheme = scheme
                break
        
        if not selected_scheme:
            print(f"Scheme '{args.scheme_name}' not found!")
            print("Available schemes:")
            display_scheme_menu(schemes)
            return
    else:
        # Interactive selection
        display_scheme_menu(schemes)
        choice = get_user_choice(len(schemes))
        if choice is None:
            print("Goodbye!")
            return
        selected_scheme = schemes[choice]
    
    # Create visualizations
    print(f"\nSelected scheme: {selected_scheme['name']}")
    print(f"CTUs: {selected_scheme['ctus']}, Relations: {selected_scheme['relations']}")
    
    try:
        visualizer = CTUGraphVisualizer(selected_scheme['file'])
        create_demo_visualizations(visualizer, args.output_dir, selected_scheme['name'])
        
        # Print summary
        print("\n" + "="*80)
        print("GRAPH SUMMARY")
        print("="*80)
        visualizer.print_summary()
        
    except Exception as e:
        print(f"Error creating visualizations: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
