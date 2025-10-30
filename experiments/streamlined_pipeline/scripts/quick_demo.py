#!/usr/bin/env python3
"""
Quick Demo Script for CTU Graph Visualization

This is the simplest way to demonstrate a CTU graph to your supervisor.
Just run this script and it will show you the Advance Authorisation scheme graph.

Usage:
    python quick_demo.py
"""

import os
import sys
from pathlib import Path

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from ctu_graph_visualizer import CTUGraphVisualizer
except ImportError as e:
    print(f"Error importing CTUGraphVisualizer: {e}")
    print("Make sure you're running this from the scripts directory.")
    sys.exit(1)

def main():
    """Quick demo of CTU graph visualization."""
    print("="*80)
    print("CTU GRAPH VISUALIZATION - QUICK DEMO")
    print("="*80)
    
    # Find the AA (Advance Authorisation) scheme file
    relations_dir = Path("../output_data/ctu_relations_production_ready")
    aa_file = relations_dir / "aa_ctus_production_ready.json"
    
    if not aa_file.exists():
        print(f"Error: Could not find {aa_file}")
        print("Make sure you're running this from the scripts directory.")
        return
    
    print(f"Loading CTU data from: {aa_file}")
    
    try:
        # Create visualizer
        visualizer = CTUGraphVisualizer(str(aa_file))
        
        # Print summary
        print("\n" + "="*80)
        print("GRAPH SUMMARY")
        print("="*80)
        visualizer.print_summary()
        
        # Create output directory
        output_dir = "../demo_visualizations"
        os.makedirs(output_dir, exist_ok=True)
        
        print("\nCreating visualizations...")
        print("="*80)
        
        # 1. Static visualization
        print("1. Creating static visualization...")
        static_file = os.path.join(output_dir, "aa_demo_static.png")
        visualizer.create_static_visualization(static_file, layout='hierarchical', max_nodes=25)
        print(f"   ✓ Saved to: {static_file}")
        
        # 2. Interactive visualization (if available)
        try:
            import plotly
            print("2. Creating interactive visualization...")
            interactive_file = os.path.join(output_dir, "aa_demo_interactive.html")
            visualizer.create_interactive_visualization(interactive_file)
            print(f"   ✓ Saved to: {interactive_file}")
        except ImportError:
            print("2. Skipping interactive visualization (plotly not installed)")
            print("   Install with: pip install plotly")
        
        # 3. Dashboard (if available)
        try:
            import plotly
            print("3. Creating dashboard...")
            dashboard_file = os.path.join(output_dir, "aa_demo_dashboard.html")
            visualizer.create_summary_dashboard(dashboard_file)
            print(f"   ✓ Saved to: {dashboard_file}")
        except ImportError:
            print("3. Skipping dashboard (plotly not installed)")
        
        print("\n" + "="*80)
        print("DEMO COMPLETE!")
        print("="*80)
        print(f"Visualizations saved to: {output_dir}")
        print("\nFiles created:")
        for file in os.listdir(output_dir):
            if "aa_demo" in file:
                print(f"  📊 {file}")
        
        print("\nTo view the visualizations:")
        print("  • Open .html files in your web browser for interactive views")
        print("  • Open .png files in any image viewer for static views")
        print("  • The dashboard provides a comprehensive overview")
        
    except Exception as e:
        print(f"Error creating visualizations: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
