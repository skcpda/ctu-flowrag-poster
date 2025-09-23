#!/usr/bin/env python3
"""
JSON Structure Analysis - Find missing fields and bulky content
"""

import json
import os
from pathlib import Path
from typing import Dict, Any, Set
import re

def analyze_json_structure(json_data: Dict, path: str = "", all_keys: Set[str] = None) -> Set[str]:
    """Recursively analyze JSON structure to find all keys"""
    if all_keys is None:
        all_keys = set()
    
    if isinstance(json_data, dict):
        for key, value in json_data.items():
            current_path = f"{path}.{key}" if path else key
            all_keys.add(current_path)
            
            if isinstance(value, (dict, list)):
                analyze_json_structure(value, current_path, all_keys)
    elif isinstance(json_data, list):
        for i, item in enumerate(json_data):
            if isinstance(item, (dict, list)):
                analyze_json_structure(item, f"{path}[{i}]", all_keys)
    
    return all_keys

def get_text_content_length(text: str) -> int:
    """Get the length of meaningful text content"""
    if not text:
        return 0
    # Remove HTML tags and normalize whitespace
    clean_text = re.sub(r'<[^>]+>', '', str(text))
    clean_text = re.sub(r'\s+', ' ', clean_text).strip()
    return len(clean_text.split())

def analyze_scheme_content(json_data: Dict) -> Dict[str, Any]:
    """Analyze the content of a single scheme"""
    analysis = {
        'total_keys': 0,
        'text_fields': {},
        'bulky_content': {},
        'missing_fields': [],
        'available_content_length': 0
    }
    
    # Get all keys in the JSON
    all_keys = analyze_json_structure(json_data)
    analysis['total_keys'] = len(all_keys)
    
    # Fields we're currently extracting
    extracted_fields = {
        'data.en.basicDetails.schemeName',
        'data.en.basicDetails.schemeShortTitle', 
        'data.en.basicDetails.briefDescription',
        'data.en.basicDetails.schemeOpenDate',
        'data.en.basicDetails.schemeCloseDate',
        'data.en.basicDetails.tags',
        'data.en.basicDetails.targetBeneficiaries',
        'data.en.basicDetails.schemeCategory',
        'data.en.basicDetails.schemeSubCategory',
        'data.en.basicDetails.state',
        'data.en.basicDetails.level',
        'data.en.basicDetails.nodalDepartmentName',
        'data.en.basicDetails.implementingAgency',
        'data.en.basicDetails.dbtScheme',
        'data.en.schemeContent.detailedDescription_md',
        'data.en.schemeContent.benefits_md',
        'data.en.schemeContent.exclusions_md',
        'data.en.schemeContent.eligibilityCriteria.eligibilityDescription_md',
        'data.en.schemeContent.applicationProcess',
        'data.en.schemeContent.references',
        'data.en.schemeContent.benefitTypes',
        'data.en.schemeContent.documentsRequired_md',
        'data.en.schemeContent.objectives_md',
        'data.en.schemeContent.definitions_md',
        'data.en.schemeContent.timeline_md',
        'data.en.schemeContent.references_md'
    }
    
    # Find missing fields
    missing_fields = all_keys - extracted_fields
    analysis['missing_fields'] = list(missing_fields)
    
    # Analyze text content in various fields
    def safe_get(data, *keys, default=""):
        try:
            for key in keys:
                if isinstance(data, dict) and key in data:
                    data = data[key]
                else:
                    return default
            return data if data is not None else default
        except:
            return default
    
    # Check for bulky content in various fields
    bulky_fields = [
        ('detailedDescription', 'data.en.schemeContent.detailedDescription'),
        ('benefits', 'data.en.schemeContent.benefits'),
        ('exclusions', 'data.en.schemeContent.exclusions'),
        ('applicationProcess.process', 'data.en.schemeContent.applicationProcess.0.process'),
        ('schemeDefinitions', 'data.en.schemeContent.schemeDefinitions'),
        ('documentsRequired', 'data.en.schemeContent.documentsRequired'),
        ('objectives', 'data.en.schemeContent.objectives'),
        ('timeline', 'data.en.schemeContent.timeline'),
        ('references', 'data.en.schemeContent.references')
    ]
    
    for field_name, field_path in bulky_fields:
        content = safe_get(json_data, *field_path.split('.'))
        if content:
            if isinstance(content, list):
                # Extract text from structured content
                text_content = ""
                for item in content:
                    if isinstance(item, dict):
                        if 'text' in item:
                            text_content += item['text'] + " "
                        elif 'children' in item:
                            for child in item['children']:
                                if isinstance(child, dict) and 'text' in child:
                                    text_content += child['text'] + " "
            else:
                text_content = str(content)
            
            content_length = get_text_content_length(text_content)
            if content_length > 0:
                analysis['text_fields'][field_name] = content_length
                analysis['available_content_length'] += content_length
                
                if content_length > 50:  # Consider bulky if more than 50 words
                    analysis['bulky_content'][field_name] = content_length
    
    return analysis

def analyze_multiple_schemes(input_dir: str, sample_size: int = 10):
    """Analyze multiple schemes to find patterns"""
    input_path = Path(input_dir)
    long_desc_files = list(input_path.glob("*/longDescription.txt"))
    
    print(f"Analyzing {min(sample_size, len(long_desc_files))} schemes...")
    
    all_analyses = []
    total_available_content = 0
    all_missing_fields = set()
    all_bulky_fields = set()
    
    for i, file_path in enumerate(long_desc_files[:sample_size]):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                json_data = json.load(f)
            
            analysis = analyze_scheme_content(json_data)
            all_analyses.append(analysis)
            total_available_content += analysis['available_content_length']
            all_missing_fields.update(analysis['missing_fields'])
            all_bulky_fields.update(analysis['bulky_content'].keys())
            
            print(f"\n[{i+1}] {file_path.parent.name}:")
            print(f"  Total keys: {analysis['total_keys']}")
            print(f"  Available content: {analysis['available_content_length']} words")
            print(f"  Bulky fields: {list(analysis['bulky_content'].keys())}")
            print(f"  Missing fields: {len(analysis['missing_fields'])}")
            
        except Exception as e:
            print(f"Error processing {file_path.parent.name}: {e}")
    
    # Summary
    print(f"\n=== ANALYSIS SUMMARY ===")
    print(f"Average available content per scheme: {total_available_content / len(all_analyses):.1f} words")
    print(f"Total unique missing fields: {len(all_missing_fields)}")
    print(f"Most common bulky fields: {sorted(all_bulky_fields, key=lambda x: sum(a['bulky_content'].get(x, 0) for a in all_analyses), reverse=True)[:10]}")
    
    # Show some missing fields
    print(f"\nSample missing fields:")
    for field in sorted(list(all_missing_fields))[:20]:
        print(f"  - {field}")
    
    return all_analyses

def main():
    """Main function"""
    input_directory = "/Users/priyankjairaj/Downloads/MoTA/mySchemeData"
    
    print("=== JSON Structure Analysis ===")
    print("Analyzing JSON structure to find missing fields and bulky content...")
    
    analyze_multiple_schemes(input_directory, sample_size=20)

if __name__ == "__main__":
    main()
