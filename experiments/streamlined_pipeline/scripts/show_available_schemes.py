#!/usr/bin/env python3
"""
Show Available CTU Schemes

This script lists all available CTU relation files and their basic statistics.
Useful for choosing which scheme to visualize.
"""

import os
import sys
import json
from pathlib import Path

def main():
    """List all available CTU schemes."""
    relations_dir = Path("../output_data/ctu_relations_production_ready")
    
    if not relations_dir.exists():
        print(f"Error: Relations directory {relations_dir} not found.")
        print("Make sure you're running this from the scripts directory.")
        return
    
    print("="*100)
    print("AVAILABLE CTU SCHEMES")
    print("="*100)
    print(f"{'#':<3} {'Scheme Name':<50} {'CTUs':<8} {'Relations':<10} {'File'}")
    print("-"*100)
    
    json_files = sorted(relations_dir.glob("*_production_ready.json"))
    
    if not json_files:
        print("No CTU relation files found!")
        return
    
    for i, file_path in enumerate(json_files, 1):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                scheme_name = data.get('scheme_name', file_path.stem)
                ctus = len(data.get('ctus', []))
                relations = len(data.get('relations', []))
                
                # Truncate long names
                display_name = scheme_name[:47] + "..." if len(scheme_name) > 50 else scheme_name
                file_name = file_path.name
                
                print(f"{i:<3} {display_name:<50} {ctus:<8} {relations:<10} {file_name}")
                
        except (json.JSONDecodeError, KeyError) as e:
            print(f"{i:<3} {'ERROR':<50} {'N/A':<8} {'N/A':<10} {file_path.name}")
            print(f"     Error: {e}")
    
    print("="*100)
    print(f"Total schemes found: {len(json_files)}")
    print("\nTo visualize a specific scheme:")
    print("  python demo_visualization.py")
    print("  python quick_demo.py  # (for Advance Authorisation only)")
    print("\nTo visualize a specific scheme by name:")
    print("  python demo_visualization.py 'scheme_name'")

if __name__ == "__main__":
    main()
