#!/usr/bin/env python3
"""
CTU Extractor - Extract Content Thematic Units from GPT-4o mini descriptions
with intelligent section detection
"""

import os
import json
import re
from typing import List, Dict, Tuple
from pathlib import Path

def detect_sections(sentences: List[str]) -> List[Tuple[int, str, str]]:
    """
    Detect section boundaries based on content patterns.
    Returns list of (start_idx, section_name, section_type)
    """
    sections = []
    current_section = "Introduction"
    current_start = 0
    
    # Section detection patterns - more specific and less aggressive
    section_patterns = {
        'Introduction': [
            r'\b(overview|introduction|background|about|scheme|program|initiative)\b',
            r'\b(launched|started|initiated|established)\b',
            r'\b(government|ministry|department)\b'
        ],
        'Objectives': [
            r'\b(primary objective|main objective|goal|aim|purpose|target|mission)\b',
            r'\b(enhance|improve|provide|develop|create)\b.*\b(skill|training|education|development)\b'
        ],
        'Benefits': [
            r'\b(benefit|advantage|support|assistance|help)\b',
            r'\b(financial|monetary|stipend|allowance|grant)\b',
            r'\b(certificate|recognition|credential)\b',
            r'\b(job|employment|career|opportunity)\b'
        ],
        'Eligibility': [
            r'\b(eligible|eligibility|criteria|requirement|condition)\b',
            r'\b(age|aged|years old|minimum|maximum)\b',
            r'\b(education|qualification|degree|grade)\b',
            r'\b(income|salary|earning|financial)\b'
        ],
        'Application': [
            r'\b(application process|application|apply|registration|enroll|register)\b',
            r'\b(process|procedure|step|method)\b',
            r'\b(form|document|paperwork|submission)\b',
            r'\b(center|institution|organization)\b'
        ],
        'Implementation': [
            r'\b(implementation|conduct|training|session)\b',
            r'\b(ministry|authority|department|organization)\b',
            r'\b(guideline|instruction|procedure|protocol)\b',
            r'\b(contact|inquiry|support|help)\b'
        ],
        'Results': [
            r'\b(result|outcome|achievement|success)\b',
            r'\b(participant|beneficiary|individual)\b',
            r'\b(statistic|number|count|total)\b',
            r'\b(feedback|response|evaluation)\b'
        ]
    }
    
    # Minimum section size to avoid too many small sections
    min_section_size = max(3, len(sentences) // 8)  # At most 8 sections
    
    for i, sentence in enumerate(sentences):
        sentence_lower = sentence.lower()
        
        # Only consider section transitions if we have enough content in current section
        if i - current_start < min_section_size:
            continue
        
        # Check for section transitions
        for section_name, patterns in section_patterns.items():
            if section_name == current_section:
                continue
                
            # Check if this sentence strongly indicates a new section
            pattern_matches = sum(1 for pattern in patterns if re.search(pattern, sentence_lower))
            if pattern_matches >= 2:  # At least 2 pattern matches
                # End current section and start new one
                if i > current_start:  # Only create section if there are sentences
                    sections.append((current_start, current_section, 'content'))
                current_section = section_name
                current_start = i
                break
    
    # Add the final section
    if current_start < len(sentences):
        sections.append((current_start, current_section, 'content'))
    
    # If no sections detected, create a single "Introduction" section
    if not sections:
        sections.append((0, "Introduction", 'content'))
    
    # Merge very small sections with previous section
    merged_sections = []
    for i, (start_idx, section_name, section_type) in enumerate(sections):
        if i == 0:
            merged_sections.append((start_idx, section_name, section_type))
        else:
            prev_start = merged_sections[-1][0]
            current_size = start_idx - prev_start
            if current_size < min_section_size and len(merged_sections) > 1:
                # Merge with previous section
                merged_sections[-1] = (merged_sections[-1][0], merged_sections[-1][1], merged_sections[-1][2])
            else:
                merged_sections.append((start_idx, section_name, section_type))
    
    return merged_sections

def extract_ctus_from_scheme(scheme_data: Dict) -> List[Dict]:
    """Extract CTUs from a scheme description with section detection"""
    sentences = scheme_data.get('sentences', [])
    scheme_name = scheme_data.get('scheme_name', 'Unknown Scheme')
    
    # Detect sections
    sections = detect_sections(sentences)
    
    ctus = []
    for i, sentence in enumerate(sentences):
        # Clean sentence
        sentence = sentence.strip()
        if not sentence:
            continue
        
        # Find which section this sentence belongs to
        section_id = 1
        section_name = "Introduction"
        for start_idx, sec_name, sec_type in sections:
            if i >= start_idx:
                section_id = sections.index((start_idx, sec_name, sec_type)) + 1
                section_name = sec_name
        
        # Create CTU
        ctu = {
            'sentence': sentence,
            'text': sentence,  # Alias for compatibility
            'role': 'Unknown',  # Will be labeled by role tagger
            'confidence': 1.0,
            'sid': section_id,  # Section ID based on detection
            'section_name': section_name,  # Human-readable section name
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
