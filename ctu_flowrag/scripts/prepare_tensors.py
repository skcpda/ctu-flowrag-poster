#!/usr/bin/env python3
"""
Prepare tensors from production JSON files.

Converts production JSON files to tensor packs for training and evaluation.
"""

import argparse
import logging
import json
import yaml
from pathlib import Path
from typing import List, Dict, Any
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from data_io.load_json_graph import load_json_graph, validate_graph
from data_io.tensor_packs import build_tensor_pack, save_tensor_pack, validate_tensor_pack

logger = logging.getLogger(__name__)

def load_json_files(json_dir: str) -> List[Path]:
    """
    Load all JSON files from directory.
    
    Args:
        json_dir: Directory containing JSON files
        
    Returns:
        List of JSON file paths
    """
    json_dir = Path(json_dir)
    json_files = list(json_dir.glob("*.json"))
    
    logger.info(f"Found {len(json_files)} JSON files in {json_dir}")
    return json_files

def process_json_file(json_path: Path, 
                     tensor_dir: Path,
                     embed_model: str,
                     text_dim: int,
                     distance_lambda: float) -> bool:
    """
    Process a single JSON file and save tensor pack.
    
    Args:
        json_path: Path to JSON file
        tensor_dir: Directory to save tensor packs
        embed_model: Embedding model to use
        text_dim: Text embedding dimension
        distance_lambda: Distance penalty weight
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Load graph data
        graph_data = load_json_graph(str(json_path))
        
        # Validate graph
        if not validate_graph(graph_data):
            logger.error(f"Graph validation failed for {json_path}")
            return False
        
        # Build tensor pack
        tensor_pack = build_tensor_pack(
            graph_data,
            embed_model=embed_model,
            text_dim=text_dim,
            distance_lambda=distance_lambda
        )
        
        # Validate tensor pack
        if not validate_tensor_pack(tensor_pack):
            logger.error(f"Tensor pack validation failed for {json_path}")
            return False
        
        # Save tensor pack
        doc_id = json_path.stem
        output_path = tensor_dir / doc_id
        save_tensor_pack(tensor_pack, str(output_path))
        
        logger.info(f"Processed {json_path} -> {output_path}")
        return True
        
    except Exception as e:
        logger.error(f"Error processing {json_path}: {e}")
        return False

def create_doc_index(json_files: List[Path], output_path: str):
    """
    Create document index file.
    
    Args:
        json_files: List of JSON file paths
        output_path: Path to save document index
    """
    doc_ids = [f.stem for f in json_files]
    
    with open(output_path, 'w') as f:
        json.dump(doc_ids, f, indent=2)
    
    logger.info(f"Created document index with {len(doc_ids)} documents")

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Prepare tensors from production JSON files')
    parser.add_argument('--json_dir', type=str, required=True, help='Directory containing JSON files')
    parser.add_argument('--tensor_dir', type=str, required=True, help='Directory to save tensor packs')
    parser.add_argument('--embed_model', type=str, default='e5-small', help='Embedding model to use')
    parser.add_argument('--text_dim', type=int, default=384, help='Text embedding dimension')
    parser.add_argument('--distance_lambda', type=float, default=0.12, help='Distance penalty weight')
    parser.add_argument('--config', type=str, help='Path to config file')
    parser.add_argument('--doc_index', type=str, help='Path to save document index')
    parser.add_argument('--log_level', type=str, default='INFO', help='Logging level')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Load configuration if provided
    if args.config:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
        
        # Override arguments with config values
        if 'model' in config:
            args.text_dim = config['model'].get('text_dim', args.text_dim)
            args.distance_lambda = config['model'].get('distance_penalty_lambda', args.distance_lambda)
    
    # Create output directory
    tensor_dir = Path(args.tensor_dir)
    tensor_dir.mkdir(parents=True, exist_ok=True)
    
    # Load JSON files
    json_files = load_json_files(args.json_dir)
    
    if len(json_files) == 0:
        logger.error(f"No JSON files found in {args.json_dir}")
        return 1
    
    # Process JSON files
    successful = 0
    failed = 0
    
    for json_file in json_files:
        if process_json_file(json_file, tensor_dir, args.embed_model, args.text_dim, args.distance_lambda):
            successful += 1
        else:
            failed += 1
    
    logger.info(f"Processing complete: {successful} successful, {failed} failed")
    
    # Create document index
    if args.doc_index:
        create_doc_index(json_files, args.doc_index)
    
    return 0 if failed == 0 else 1

if __name__ == "__main__":
    sys.exit(main())

