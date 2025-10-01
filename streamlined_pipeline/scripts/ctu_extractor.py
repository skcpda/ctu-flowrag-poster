#!/usr/bin/env python3
"""
CTU Extractor - Extract Content Thematic Units from GPT-4o mini descriptions
"""

import os
import json
import re
from typing import List, Dict
from pathlib import Path

def extract_ctus_from_scheme(scheme_data: Dict) -> List[Dict]:
    """Extract CTUs from a scheme description"""
    sentences = scheme_data.get('sentences', [])
    scheme_name = scheme_data.get('scheme_name', 'Unknown Scheme')
    
    ctus = []
    for i, sentence in enumerate(sentences):
        # Clean sentence
        sentence = sentence.strip()
        if not sentence:
            continue
            
        # Create CTU
        ctu = {
            'sentence': sentence,
            'text': sentence,  # Alias for compatibility
            'role': 'Unknown',  # Will be labeled by role tagger
            'confidence': 1.0,
            'sid': 1,  # Single section for now
            'line_idx': i,
            'scheme_name': scheme_name
        }
        ctus.append(ctu)
    
    return ctus

def process_scheme_file(input_file: Path, output_file: Path) -> Dict:
    """Process a single scheme file"""
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            scheme_data = json.load(f)
        
        # Extract CTUs
        ctus = extract_ctus_from_scheme(scheme_data)
        
        # Create output structure
        output_data = {
            'scheme_name': scheme_data.get('scheme_name', 'Unknown Scheme'),
            'total_sentences': len(ctus),
            'ctus': ctus,
            'processing_timestamp': scheme_data.get('timestamp', ''),
            'model_used': scheme_data.get('model', 'gpt-4o-mini')
        }
        
        # Save output
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        return {
            'success': True,
            'ctus_count': len(ctus),
            'scheme_name': scheme_data.get('scheme_name', 'Unknown')
        }
        
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'file': str(input_file)
        }

def main():
    """Extract CTUs from all scheme descriptions"""
    input_dir = Path("input_data/scheme_descriptions")
    output_dir = Path("output_data/ctu_extracted")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get all scheme files
    scheme_files = list(input_dir.glob("*.json"))
    print(f"Found {len(scheme_files)} scheme files")
    
    results = {
        'total_files': len(scheme_files),
        'successful': 0,
        'failed': 0,
        'total_ctus': 0,
        'errors': []
    }
    
    for i, scheme_file in enumerate(scheme_files, 1):
        output_file = output_dir / f"{scheme_file.stem}_ctus.json"
        
        print(f"Processing {i}/{len(scheme_files)}: {scheme_file.name}")
        result = process_scheme_file(scheme_file, output_file)
        
        if result['success']:
            results['successful'] += 1
            results['total_ctus'] += result['ctus_count']
            print(f"  ✓ Extracted {result['ctus_count']} CTUs")
        else:
            results['failed'] += 1
            results['errors'].append(result['error'])
            print(f"  ❌ Error: {result['error']}")
    
    # Save summary
    with open(output_dir / "extraction_summary.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n=== CTU Extraction Complete ===")
    print(f"Total files: {results['total_files']}")
    print(f"Successful: {results['successful']}")
    print(f"Failed: {results['failed']}")
    print(f"Total CTUs: {results['total_ctus']}")
    print(f"Average CTUs per scheme: {results['total_ctus'] / results['successful'] if results['successful'] > 0 else 0:.1f}")

if __name__ == '__main__':
    main()
