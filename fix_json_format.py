#!/usr/bin/env python3
import json
import os
import sys
from pathlib import Path

def fix_json_file(file_path):
    """Fix JSON file by adding missing 'type' field to relations."""
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        if 'relations' in data:
            for relation in data['relations']:
                if 'type' not in relation:
                    # Use the 'relation' field as the 'type' field
                    relation['type'] = relation.get('relation', 'UNKNOWN')
        
        # Write back the fixed data
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        return True
    except Exception as e:
        print(f"Error fixing {file_path}: {e}")
        return False

def main():
    data_dir = Path("streamlined_pipeline/output_data/ctu_relations_production_ready")
    
    if not data_dir.exists():
        print(f"Data directory {data_dir} not found")
        return
    
    json_files = list(data_dir.glob("*.json"))
    print(f"Found {len(json_files)} JSON files to process")
    
    successful = 0
    failed = 0
    
    for json_file in json_files:
        print(f"Processing: {json_file.name}")
        if fix_json_file(json_file):
            successful += 1
        else:
            failed += 1
    
    print(f"\nResults: {successful} successful, {failed} failed")

if __name__ == "__main__":
    main()
