#!/usr/bin/env python3
"""
Identify the largest fields in JSON files that we're not extracting yet
"""

import json
import os
from pathlib import Path
from collections import Counter, defaultdict
import re

def count_words(text):
    """Count words in text"""
    if not text:
        return 0
    return len(str(text).split())

def analyze_field_sizes(json_data, current_path="", field_sizes=None, max_depth=10, depth=0):
    """Recursively analyze field sizes in JSON"""
    if field_sizes is None:
        field_sizes = defaultdict(list)
    
    if depth > max_depth:
        return field_sizes
    
    if isinstance(json_data, dict):
        for key, value in json_data.items():
            new_path = f"{current_path}.{key}" if current_path else key
            
            if isinstance(value, str):
                word_count = count_words(value)
                if word_count > 0:
                    field_sizes[new_path].append(word_count)
            elif isinstance(value, (dict, list)):
                # Recursively analyze nested structures
                analyze_field_sizes(value, new_path, field_sizes, max_depth, depth + 1)
    
    elif isinstance(json_data, list):
        for i, item in enumerate(json_data):
            new_path = f"{current_path}[{i}]"
            if isinstance(item, (dict, list)):
                analyze_field_sizes(item, new_path, field_sizes, max_depth, depth + 1)
            elif isinstance(item, str):
                word_count = count_words(item)
                if word_count > 0:
                    field_sizes[new_path].append(word_count)
    
    return field_sizes

def identify_large_fields():
    """Identify the largest fields across all JSON files"""
    input_dir = Path("/Users/priyankjairaj/Downloads/MoTA/mySchemeData")
    
    all_field_sizes = defaultdict(list)
    processed_files = 0
    
    print("Analyzing field sizes across JSON files...")
    
    # Find all JSON files
    json_files = list(input_dir.glob("*/longDescription.txt"))
    
    for json_file in json_files:
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                json_data = json.load(f)
            
            # Analyze field sizes
            field_sizes = analyze_field_sizes(json_data)
            
            # Merge with global field sizes
            for field_path, sizes in field_sizes.items():
                all_field_sizes[field_path].extend(sizes)
            
            processed_files += 1
            if processed_files % 100 == 0:
                print(f"Processed {processed_files} files...")
                
        except Exception as e:
            print(f"Error processing {json_file}: {e}")
            continue
    
    print(f"\nAnalyzed {processed_files} JSON files")
    
    # Calculate statistics for each field
    field_stats = {}
    for field_path, sizes in all_field_sizes.items():
        if sizes:
            field_stats[field_path] = {
                'total_words': sum(sizes),
                'avg_words': sum(sizes) / len(sizes),
                'max_words': max(sizes),
                'count': len(sizes),
                'total_occurrences': len(sizes)
            }
    
    # Sort by total words (most impactful fields)
    sorted_by_total = sorted(field_stats.items(), key=lambda x: x[1]['total_words'], reverse=True)
    
    print(f"\n=== TOP 50 LARGEST FIELDS BY TOTAL WORDS ===")
    for i, (field_path, stats) in enumerate(sorted_by_total[:50]):
        print(f"{i+1:2d}. {field_path}")
        print(f"    Total: {stats['total_words']:,} words | Avg: {stats['avg_words']:.1f} | Max: {stats['max_words']} | Count: {stats['count']}")
        print()
    
    # Sort by average words (fields with most content per occurrence)
    sorted_by_avg = sorted(field_stats.items(), key=lambda x: x[1]['avg_words'], reverse=True)
    
    print(f"\n=== TOP 50 LARGEST FIELDS BY AVERAGE WORDS ===")
    for i, (field_path, stats) in enumerate(sorted_by_avg[:50]):
        print(f"{i+1:2d}. {field_path}")
        print(f"    Avg: {stats['avg_words']:.1f} words | Total: {stats['total_words']:,} | Max: {stats['max_words']} | Count: {stats['count']}")
        print()
    
    # Identify fields we're likely not extracting
    current_extraction_fields = {
        'data.en.basicDetails.schemeName',
        'data.en.basicDetails.schemeShortTitle', 
        'data.en.basicDetails.briefDescription',
        'data.en.basicDetails.schemeOpenDate',
        'data.en.basicDetails.schemeCloseDate',
        'data.en.basicDetails.schemeFor',
        'data.en.basicDetails.dbtScheme',
        'data.en.basicDetails.tags',
        'data.en.basicDetails.targetBeneficiaries',
        'data.en.basicDetails.schemeCategory',
        'data.en.basicDetails.schemeSubCategory',
        'data.en.basicDetails.state',
        'data.en.basicDetails.level',
        'data.en.basicDetails.nodalDepartmentName',
        'data.en.basicDetails.implementingAgency',
        'data.en.schemeContent.detailedDescription_md',
        'data.en.schemeContent.detailedDescription',
        'data.en.schemeContent.benefits_md',
        'data.en.schemeContent.benefits',
        'data.en.schemeContent.exclusions_md',
        'data.en.schemeContent.exclusions',
        'data.en.schemeContent.eligibilityCriteria',
        'data.en.schemeContent.objectives_md',
        'data.en.schemeContent.objectives',
        'data.en.schemeContent.definitions_md',
        'data.en.schemeContent.schemeDefinitions',
        'data.en.schemeContent.documentsRequired_md',
        'data.en.schemeContent.documentsRequired',
        'data.en.schemeContent.timeline_md',
        'data.en.schemeContent.timeline',
        'data.en.schemeContent.applicationProcess',
        'data.en.schemeContent.references_md',
        'data.en.schemeContent.references',
        'data.en.schemeContent.benefitTypes',
        'data.en.schemeContent.schemeImageUrl'
    }
    
    # Find fields we're not extracting
    missing_fields = []
    for field_path, stats in field_stats.items():
        # Check if this field is in our current extraction
        is_extracted = False
        for extracted_field in current_extraction_fields:
            if field_path.startswith(extracted_field) or extracted_field in field_path:
                is_extracted = True
                break
        
        if not is_extracted and stats['total_words'] > 1000:  # Only fields with significant content
            missing_fields.append((field_path, stats))
    
    # Sort missing fields by total words
    missing_fields.sort(key=lambda x: x[1]['total_words'], reverse=True)
    
    print(f"\n=== TOP MISSING FIELDS (NOT CURRENTLY EXTRACTED) ===")
    for i, (field_path, stats) in enumerate(missing_fields[:30]):
        print(f"{i+1:2d}. {field_path}")
        print(f"    Total: {stats['total_words']:,} words | Avg: {stats['avg_words']:.1f} | Max: {stats['max_words']} | Count: {stats['count']}")
        print()
    
    return field_stats, missing_fields

if __name__ == "__main__":
    field_stats, missing_fields = identify_large_fields()
