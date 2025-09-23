#!/usr/bin/env python3
"""
Enhanced Scheme Processor - Converts JSON scheme data into clean paragraph-style descriptions
"""

import json
import os
import time
from pathlib import Path
from typing import Dict, Any, Optional
import re

def safe_get(data: Dict, *keys, default: str = "") -> str:
    """Safely get nested dictionary values with fallback"""
    try:
        for key in keys:
            if isinstance(data, dict) and key in data:
                data = data[key]
            else:
                return default
        return str(data) if data is not None else default
    except:
        return default

def clean_text(text: str) -> str:
    """Clean and format text content"""
    if not text:
        return ""
    
    # Remove extra whitespace and normalize
    text = re.sub(r'\s+', ' ', str(text).strip())
    
    # Remove HTML-like tags if present
    text = re.sub(r'<[^>]+>', '', text)
    
    return text

def extract_scheme_data(json_data: Dict) -> Dict[str, Any]:
    """Extract and structure scheme data from JSON"""
    scheme = {}
    
    # Basic scheme information
    scheme['schemeName'] = safe_get(json_data, 'data', 'en', 'basicDetails', 'schemeName')
    scheme['shortDescription'] = safe_get(json_data, 'data', 'en', 'basicDetails', 'briefDescription')
    
    # Target population and scope
    target_beneficiaries = safe_get(json_data, 'data', 'en', 'basicDetails', 'targetBeneficiaries')
    if target_beneficiaries and isinstance(target_beneficiaries, list):
        scheme['targetPopulation'] = ', '.join([str(b.get('label', b)) for b in target_beneficiaries])
    
    scheme['category'] = safe_get(json_data, 'data', 'en', 'basicDetails', 'schemeCategory', 0, 'label')
    scheme['sector'] = safe_get(json_data, 'data', 'en', 'basicDetails', 'schemeSubCategory', 0, 'label')
    
    # Geography
    scheme['geography'] = safe_get(json_data, 'data', 'en', 'basicDetails', 'state', 'label')
    scheme['jurisdiction'] = safe_get(json_data, 'data', 'en', 'basicDetails', 'level', 'label')
    
    # Scheme content
    scheme_content = json_data.get('data', {}).get('en', {}).get('schemeContent', {})
    
    scheme['detailedDescription_md'] = clean_text(safe_get(scheme_content, 'detailedDescription_md'))
    scheme['objectives_md'] = clean_text(safe_get(scheme_content, 'objectives_md'))
    scheme['benefits_md'] = clean_text(safe_get(scheme_content, 'benefits_md'))
    scheme['eligibilityDescription_md'] = clean_text(safe_get(scheme_content, 'eligibilityDescription_md'))
    scheme['exclusions_md'] = clean_text(safe_get(scheme_content, 'exclusions_md'))
    scheme['definitions_md'] = clean_text(safe_get(scheme_content, 'definitions_md'))
    scheme['documents_md'] = clean_text(safe_get(scheme_content, 'documentsRequired_md'))
    scheme['timeline_md'] = clean_text(safe_get(scheme_content, 'timeline_md'))
    scheme['references_md'] = clean_text(safe_get(scheme_content, 'references_md'))
    
    # Application process
    app_process = scheme_content.get('applicationProcess', [])
    if app_process and len(app_process) > 0:
        scheme['applicationProcess'] = {
            'process_md': clean_text(safe_get(app_process[0], 'process_md')),
            'mode': safe_get(app_process[0], 'mode'),
            'portalUrl': safe_get(app_process[0], 'portalUrl'),
            'steps': safe_get(app_process[0], 'process', default=[])
        }
    
    # Implementing agency
    scheme['implementingAgency'] = safe_get(json_data, 'data', 'en', 'basicDetails', 'implementingAgency')
    
    # References
    references = scheme_content.get('references', [])
    if references:
        scheme['references'] = [{'label': r.get('title', ''), 'url': r.get('url', '')} for r in references]
    
    # Language counts (if available)
    if 'lang_counts' in json_data:
        scheme['lang_counts'] = json_data['lang_counts']
    
    return scheme

def generate_description_from_template(scheme: Dict[str, Any]) -> str:
    """Generate description using the template"""
    
    # Template with conditional sections
    template_parts = []
    
    # Header
    if scheme.get('schemeName'):
        template_parts.append(f"# {scheme['schemeName']}")
    
    # Summary
    if scheme.get('shortDescription'):
        template_parts.append(f"\n**Summary:** {scheme['shortDescription']}")
    
    # Detailed description
    if scheme.get('detailedDescription_md'):
        template_parts.append(f"\n{scheme['detailedDescription_md']}")
    
    # Target population
    target_info = []
    if scheme.get('targetPopulation'):
        target_info.append(scheme['targetPopulation'])
    if scheme.get('category'):
        target_info.append(f"Category: {scheme['category']}")
    if scheme.get('sector'):
        target_info.append(f"Sector: {scheme['sector']}")
    
    if target_info:
        template_parts.append(f"\n**Who is this for:** {'; '.join(target_info)}")
    
    # Geography
    geo_info = []
    if scheme.get('geography'):
        geo_info.append(scheme['geography'])
    if scheme.get('jurisdiction'):
        geo_info.append(scheme['jurisdiction'])
    
    if geo_info:
        template_parts.append(f"\n**Where it applies:** {'; '.join(geo_info)}")
    
    # Objectives
    if scheme.get('objectives_md'):
        template_parts.append(f"\n**Objectives:** {scheme['objectives_md']}")
    
    # Benefits
    if scheme.get('benefits_md'):
        template_parts.append(f"\n**Benefits / Assistance:** {scheme['benefits_md']}")
    
    # Eligibility
    if scheme.get('eligibilityDescription_md'):
        template_parts.append(f"\n**Eligibility:** {scheme['eligibilityDescription_md']}")
    
    # Exclusions
    if scheme.get('exclusions_md'):
        template_parts.append(f"\n**Exclusions / Not eligible:** {scheme['exclusions_md']}")
    
    # Definitions
    if scheme.get('definitions_md'):
        template_parts.append(f"\n**Definitions:** {scheme['definitions_md']}")
    
    # Required documents
    if scheme.get('documents_md'):
        template_parts.append(f"\n**Required documents:** {scheme['documents_md']}")
    
    # Application process
    app_process = scheme.get('applicationProcess', {})
    if app_process.get('process_md'):
        template_parts.append(f"\n**How to apply:** {app_process['process_md']}")
        if app_process.get('mode'):
            template_parts.append(f"Mode: {app_process['mode']}.")
        if app_process.get('portalUrl'):
            template_parts.append(f"Apply at: {app_process['portalUrl']}")
    
    # Timeline
    if scheme.get('timeline_md'):
        template_parts.append(f"\n**Timeline / Cycle:** {scheme['timeline_md']}")
    
    # Contacts
    if scheme.get('implementingAgency'):
        template_parts.append(f"\n**Contacts & Authorities:** Implementing agency: {scheme['implementingAgency']}.")
    
    # References
    if scheme.get('references_md'):
        template_parts.append(f"\n**References / Annexures:** {scheme['references_md']}")
    elif scheme.get('references'):
        ref_text = "\n".join([f"- {r['label']}: {r['url']}" for r in scheme['references'] if r['label'] and r['url']])
        if ref_text:
            template_parts.append(f"\n**References / Annexures:**\n{ref_text}")
    
    # Language metadata
    if scheme.get('lang_counts'):
        lang_info = ", ".join([f"{code}={count}" for code, count in scheme['lang_counts'].items()])
        template_parts.append(f"\n<sub>Language mix: {lang_info}</sub>")
    
    return "\n".join(template_parts)

def process_single_scheme(input_file: Path, output_dir: Path) -> Dict[str, Any]:
    """Process a single scheme file"""
    result = {
        'success': False,
        'scheme_name': 'Unknown',
        'error': None
    }
    
    try:
        # Read JSON data
        with open(input_file, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
        
        # Extract scheme data
        scheme = extract_scheme_data(json_data)
        result['scheme_name'] = scheme.get('schemeName', 'Unknown Scheme')
        
        # Generate description using template
        description = generate_description_from_template(scheme)
        
        # Create output directory
        scheme_dir = output_dir / input_file.parent.name
        scheme_dir.mkdir(exist_ok=True)
        
        # Save description
        description_file = scheme_dir / "description.txt"
        with open(description_file, 'w', encoding='utf-8') as f:
            f.write(description)
        
        # Save metadata
        metadata = {
            'scheme_name': result['scheme_name'],
            'processing_timestamp': time.time(),
            'source_file': str(input_file),
            'template_used': 'enhanced_template_v1'
        }
        
        metadata_file = scheme_dir / "processing_metadata.json"
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        result['success'] = True
        print(f"✓ Processed: {result['scheme_name']}")
        
    except Exception as e:
        result['error'] = str(e)
        print(f"✗ Failed: {input_file.name} - {e}")
    
    return result

def process_all_schemes(input_dir: str, output_dir: str):
    """Process all schemes individually"""
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Find all longDescription.txt files
    long_desc_files = list(input_path.glob("*/longDescription.txt"))
    print(f"Found {len(long_desc_files)} scheme files")
    
    # Process each scheme individually
    results = {
        'successful': 0,
        'failed': 0,
        'errors': []
    }
    
    for i, file_path in enumerate(long_desc_files, 1):
        print(f"\n[{i}/{len(long_desc_files)}] Processing: {file_path.parent.name}")
        
        result = process_single_scheme(file_path, output_path)
        
        if result['success']:
            results['successful'] += 1
        else:
            results['failed'] += 1
            results['errors'].append(f"{file_path.parent.name}: {result['error']}")
        
        # Small delay to avoid overwhelming the system
        time.sleep(0.1)
    
    # Print final summary
    print(f"\n=== PROCESSING COMPLETE ===")
    print(f"Total schemes processed: {len(long_desc_files)}")
    print(f"Successful: {results['successful']}")
    print(f"Failed: {results['failed']}")
    
    if results['errors']:
        print(f"\nErrors encountered:")
        for error in results['errors'][:10]:  # Show first 10 errors
            print(f"- {error}")
        if len(results['errors']) > 10:
            print(f"... and {len(results['errors']) - 10} more errors")

def main():
    """Main function"""
    input_directory = "/Users/priyankjairaj/Downloads/MoTA/mySchemeData"
    output_directory = "/Users/priyankjairaj/Downloads/MoTA/enhanced_schemes"
    
    print("=== Enhanced Scheme Processor ===")
    print(f"Input directory: {input_directory}")
    print(f"Output directory: {output_directory}")
    print("Processing schemes individually with template-based formatting...")
    
    process_all_schemes(input_directory, output_directory)

if __name__ == "__main__":
    main()
