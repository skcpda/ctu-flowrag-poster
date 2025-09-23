#!/usr/bin/env python3
"""
Targeted Missing Content Processor - Focus on the 460,103 missing words
Target: Capture the largest missing fields to reach 60%+ efficiency
"""

import json
import os
import time
from pathlib import Path
from typing import Dict, Any, Optional, List, Set
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

def extract_targeted_missing_content(json_data: Dict) -> Dict[str, Any]:
    """Extract ONLY the largest missing content fields to reach 60% efficiency"""
    scheme = {}
    
    # Get the main data structure
    data = json_data.get('data', {})
    en_data = data.get('en', {})
    basic_details = en_data.get('basicDetails', {})
    scheme_content = en_data.get('schemeContent', {})
    
    # BASIC SCHEME INFO (keep minimal)
    scheme['schemeName'] = basic_details.get('schemeName', '')
    scheme['schemeShortTitle'] = basic_details.get('schemeShortTitle', '')
    scheme['schemeOpenDate'] = basic_details.get('schemeOpenDate', '')
    
    # TARGET 1: BIGGEST MISSING FIELD - Brief Description (118,128 words potential)
    brief_desc = scheme_content.get('briefDescription', '')
    if brief_desc:
        scheme['schemeBriefDescription'] = clean_text(brief_desc)
    else:
        scheme['schemeBriefDescription'] = ''
    
    # TARGET 2: Scheme Definitions (40,000+ words potential)
    scheme_definitions = scheme_content.get('schemeDefinitions', [])
    if scheme_definitions:
        def_text = ""
        for i, def_item in enumerate(scheme_definitions):
            if isinstance(def_item, dict):
                # Extract definitions_md if available
                def_md = def_item.get('definitions_md', '')
                if def_md:
                    def_text += f"**Definition Set {i+1}:** {clean_text(def_md)}\n\n"
                
                # Extract individual definitions
                definitions = def_item.get('definition', [])
                if definitions:
                    for j, definition in enumerate(definitions):
                        if isinstance(definition, dict):
                            name = definition.get('name', f'Definition {j+1}')
                            definition_text = definition.get('definition', '')
                            if definition_text:
                                def_text += f"**{name}:** {clean_text(definition_text)}\n"
        scheme['schemeDefinitions'] = clean_text(def_text)
    else:
        scheme['schemeDefinitions'] = ''
    
    # TARGET 3: Application Process - process_md (massive content)
    app_process = en_data.get('applicationProcess', [])
    if app_process and len(app_process) > 0:
        # Try markdown version first (this is the big one!)
        process_md = app_process[0].get('process_md', '')
        if process_md:
            scheme['applicationProcess_md'] = clean_text(process_md)
        else:
            scheme['applicationProcess_md'] = ''
        
        # Also extract structured process
        process_structured = app_process[0].get('process', [])
        if process_structured:
            scheme['applicationProcess_structured'] = clean_text(str(process_structured))
        else:
            scheme['applicationProcess_structured'] = ''
        
        scheme['applicationMode'] = app_process[0].get('mode', '')
        scheme['portalUrl'] = app_process[0].get('portalUrl', '')
        
        # Extract additional application processes
        if len(app_process) > 1:
            additional_processes = []
            for i, process in enumerate(app_process[1:], 1):
                process_md = process.get('process_md', '')
                if process_md:
                    additional_processes.append(f"**Process {i+1}:** {clean_text(process_md)}")
                else:
                    process_structured = process.get('process', [])
                    if process_structured:
                        additional_processes.append(f"**Process {i+1}:** {clean_text(str(process_structured))}")
            if additional_processes:
                scheme['additionalApplicationProcesses'] = '\n\n'.join(additional_processes)
    else:
        scheme['applicationProcess_md'] = ''
        scheme['applicationProcess_structured'] = ''
        scheme['applicationMode'] = ''
        scheme['portalUrl'] = ''
        scheme['additionalApplicationProcesses'] = ''
    
    # TARGET 4: Eligibility Criteria - eligibilityDescription_md (massive content)
    eligibility_criteria = scheme_content.get('eligibilityCriteria', {})
    eligibility_md = eligibility_criteria.get('eligibilityDescription_md', '')
    if eligibility_md:
        scheme['eligibilityDescription_md'] = clean_text(eligibility_md)
    else:
        scheme['eligibilityDescription_md'] = ''
    
    # TARGET 5: All other markdown fields that might have massive content
    # Detailed Description
    detailed_desc_md = scheme_content.get('detailedDescription_md', '')
    if detailed_desc_md:
        scheme['detailedDescription_md'] = clean_text(detailed_desc_md)
    else:
        scheme['detailedDescription_md'] = ''
    
    # Benefits
    benefits_md = scheme_content.get('benefits_md', '')
    if benefits_md:
        scheme['benefits_md'] = clean_text(benefits_md)
    else:
        scheme['benefits_md'] = ''
    
    # Exclusions
    exclusions_md = scheme_content.get('exclusions_md', '')
    if exclusions_md:
        scheme['exclusions_md'] = clean_text(exclusions_md)
    else:
        scheme['exclusions_md'] = ''
    
    # Objectives
    objectives_md = scheme_content.get('objectives_md', '')
    if objectives_md:
        scheme['objectives_md'] = clean_text(objectives_md)
    else:
        scheme['objectives_md'] = ''
    
    # Documents Required
    documents_md = scheme_content.get('documentsRequired_md', '')
    if documents_md:
        scheme['documentsRequired_md'] = clean_text(documents_md)
    else:
        scheme['documentsRequired_md'] = ''
    
    # Timeline
    timeline_md = scheme_content.get('timeline_md', '')
    if timeline_md:
        scheme['timeline_md'] = clean_text(timeline_md)
    else:
        scheme['timeline_md'] = ''
    
    # References
    references_md = scheme_content.get('references_md', '')
    if references_md:
        scheme['references_md'] = clean_text(references_md)
    else:
        scheme['references_md'] = ''
    
    # TARGET 6: Additional high-content fields
    # Nodal Ministry (2,058 words potential)
    nodal_ministry = basic_details.get('nodalMinistryName', {})
    scheme['nodalMinistry'] = nodal_ministry.get('label', '')
    
    # Scheme Type (1,065 words potential)
    scheme_type = basic_details.get('schemeType', {})
    scheme['schemeType'] = scheme_type.get('label', '')
    
    # Target population
    target_beneficiaries = basic_details.get('targetBeneficiaries', [])
    if target_beneficiaries:
        scheme['targetPopulation'] = ', '.join([str(b.get('label', b)) for b in target_beneficiaries])
    
    # Category and sector
    scheme_category = basic_details.get('schemeCategory', [])
    if scheme_category:
        scheme['category'] = scheme_category[0].get('label', '')
    
    scheme_subcategory = basic_details.get('schemeSubCategory', [])
    if scheme_subcategory:
        scheme['sector'] = scheme_subcategory[0].get('label', '')
    
    # Geography
    state_info = basic_details.get('state', {})
    scheme['geography'] = state_info.get('label', '')
    
    level_info = basic_details.get('level', {})
    scheme['jurisdiction'] = level_info.get('label', '')
    
    # Implementing agency
    scheme['implementingAgency'] = basic_details.get('implementingAgency', '')
    
    # Nodal department
    nodal_dept = basic_details.get('nodalDepartmentName', {})
    scheme['nodalDepartment'] = nodal_dept.get('label', '')
    
    # DBT Scheme
    scheme['dbtScheme'] = basic_details.get('dbtScheme', False)
    
    # Tags
    tags = basic_details.get('tags', [])
    if tags:
        scheme['tags'] = ', '.join(tags)
    
    # Language counts
    if 'lang_counts' in json_data:
        scheme['lang_counts'] = json_data['lang_counts']
    
    return scheme

def generate_targeted_description(scheme: Dict[str, Any]) -> str:
    """Generate description focused on the largest missing content fields"""
    template_parts = []
    
    # Header
    if scheme.get('schemeName'):
        template_parts.append(f"# {scheme['schemeName']}")
    
    # Short title
    if scheme.get('schemeShortTitle') and scheme.get('schemeShortTitle') != scheme.get('schemeName'):
        template_parts.append(f"\n**Short Title:** {scheme['schemeShortTitle']}")
    
    # TARGET 1: Scheme Brief Description (BIGGEST MISSING FIELD!)
    if scheme.get('schemeBriefDescription'):
        template_parts.append(f"\n**Scheme Brief Description:** {scheme['schemeBriefDescription']}")
    
    # Launch date
    if scheme.get('schemeOpenDate'):
        template_parts.append(f"\n**Launch Date:** {scheme['schemeOpenDate']}")
    
    # TARGET 2: Scheme Definitions (40,000+ words potential)
    if scheme.get('schemeDefinitions'):
        template_parts.append(f"\n**Scheme Definitions:** {scheme['schemeDefinitions']}")
    
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
    
    # DBT scheme
    if scheme.get('dbtScheme'):
        template_parts.append(f"\n**DBT Scheme:** Yes (Direct Benefit Transfer)")
    
    # TARGET 3: Detailed Description
    if scheme.get('detailedDescription_md'):
        template_parts.append(f"\n**Detailed Description:** {scheme['detailedDescription_md']}")
    
    # TARGET 4: Objectives
    if scheme.get('objectives_md'):
        template_parts.append(f"\n**Objectives:** {scheme['objectives_md']}")
    
    # TARGET 5: Benefits
    if scheme.get('benefits_md'):
        template_parts.append(f"\n**Benefits / Assistance:** {scheme['benefits_md']}")
    
    # TARGET 6: Eligibility (MASSIVE CONTENT)
    if scheme.get('eligibilityDescription_md'):
        template_parts.append(f"\n**Eligibility Criteria:** {scheme['eligibilityDescription_md']}")
    
    # TARGET 7: Exclusions
    if scheme.get('exclusions_md'):
        template_parts.append(f"\n**Exclusions / Not eligible:** {scheme['exclusions_md']}")
    
    # TARGET 8: Application Process (MASSIVE CONTENT)
    if scheme.get('applicationProcess_md'):
        template_parts.append(f"\n**Application Process:** {scheme['applicationProcess_md']}")
        if scheme.get('applicationMode'):
            template_parts.append(f"Mode: {scheme['applicationMode']}")
        if scheme.get('portalUrl'):
            template_parts.append(f"Apply at: {scheme['portalUrl']}")
    elif scheme.get('applicationProcess_structured'):
        template_parts.append(f"\n**Application Process:** {scheme['applicationProcess_structured']}")
        if scheme.get('applicationMode'):
            template_parts.append(f"Mode: {scheme['applicationMode']}")
        if scheme.get('portalUrl'):
            template_parts.append(f"Apply at: {scheme['portalUrl']}")
    
    # Additional application processes
    if scheme.get('additionalApplicationProcesses'):
        template_parts.append(f"\n**Additional Application Processes:** {scheme['additionalApplicationProcesses']}")
    
    # TARGET 9: Required documents
    if scheme.get('documentsRequired_md'):
        template_parts.append(f"\n**Required documents:** {scheme['documentsRequired_md']}")
    
    # TARGET 10: Timeline
    if scheme.get('timeline_md'):
        template_parts.append(f"\n**Timeline / Cycle:** {scheme['timeline_md']}")
    
    # Contacts
    contact_info = []
    if scheme.get('implementingAgency'):
        contact_info.append(f"Implementing agency: {scheme['implementingAgency']}")
    if scheme.get('nodalDepartment'):
        contact_info.append(f"Nodal Department: {scheme['nodalDepartment']}")
    if scheme.get('nodalMinistry'):
        contact_info.append(f"Nodal Ministry: {scheme['nodalMinistry']}")
    
    if contact_info:
        template_parts.append(f"\n**Contacts & Authorities:** {'; '.join(contact_info)}")
    
    # TARGET 11: References
    if scheme.get('references_md'):
        template_parts.append(f"\n**References / Annexures:** {scheme['references_md']}")
    
    # Tags
    if scheme.get('tags'):
        template_parts.append(f"\n**Tags:** {scheme['tags']}")
    
    # Language metadata
    if scheme.get('lang_counts'):
        lang_info = ", ".join([f"{code}={count}" for code, count in scheme['lang_counts'].items()])
        template_parts.append(f"\n<sub>Language mix: {lang_info}</sub>")
    
    return "\n".join(template_parts)

def process_single_scheme(input_file: Path, output_dir: Path) -> Dict[str, Any]:
    """Process a single scheme file with targeted missing content extraction"""
    result = {
        'success': False,
        'scheme_name': 'Unknown',
        'error': None,
        'content_length': 0,
        'extracted_fields': 0,
        'targeted_content_length': 0
    }
    
    try:
        # Read JSON data
        with open(input_file, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
        
        # Extract targeted missing content from scheme data
        scheme = extract_targeted_missing_content(json_data)
        result['scheme_name'] = scheme.get('schemeName', 'Unknown Scheme')
        result['extracted_fields'] = len(scheme)
        
        # Generate description with targeted content
        description = generate_targeted_description(scheme)
        result['content_length'] = len(description.split())
        
        # Calculate targeted content (excluding template words)
        template_words = 74  # Our corrected estimate
        result['targeted_content_length'] = max(0, result['content_length'] - template_words)
        
        # Create output directory
        scheme_dir = output_dir / input_file.parent.name
        scheme_dir.mkdir(exist_ok=True)
        
        # Save description
        description_file = scheme_dir / "description.txt"
        with open(description_file, 'w', encoding='utf-8') as f:
            f.write(description)
        
        result['success'] = True
        print(f"✓ Processed: {result['scheme_name']} ({result['content_length']} words, {result['targeted_content_length']} content words, {result['extracted_fields']} fields)")
        
    except Exception as e:
        result['error'] = str(e)
        print(f"✗ Failed: {input_file.name} - {e}")
    
    return result

def process_all_schemes(input_dir: str, output_dir: str):
    """Process all schemes with targeted missing content extraction"""
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Find all longDescription.txt files
    long_desc_files = list(input_path.glob("*/longDescription.txt"))
    print(f"Found {len(long_desc_files)} scheme files")
    print("Processing schemes with TARGETED MISSING CONTENT extraction...")
    print("Focus: Capture the 460,103 missing words to reach 60%+ efficiency!")
    
    # Process each scheme individually
    results = {
        'successful': 0,
        'failed': 0,
        'total_content': 0,
        'total_targeted_content': 0,
        'total_fields': 0,
        'errors': []
    }
    
    for i, file_path in enumerate(long_desc_files, 1):
        print(f"\n[{i}/{len(long_desc_files)}] Processing: {file_path.parent.name}")
        
        result = process_single_scheme(file_path, output_path)
        
        if result['success']:
            results['successful'] += 1
            results['total_content'] += result['content_length']
            results['total_targeted_content'] += result['targeted_content_length']
            results['total_fields'] += result['extracted_fields']
        else:
            results['failed'] += 1
            results['errors'].append(f"{file_path.parent.name}: {result['error']}")
        
        # Small delay to avoid overwhelming the system
        time.sleep(0.1)
    
    # Print final summary
    print(f"\n=== TARGETED MISSING CONTENT EXTRACTION COMPLETE ===")
    print(f"Total schemes processed: {len(long_desc_files)}")
    print(f"Successful: {results['successful']}")
    print(f"Failed: {results['failed']}")
    if results['successful'] > 0:
        print(f"Average content per scheme: {results['total_content'] / results['successful']:.1f} words")
        print(f"Average targeted content per scheme: {results['total_targeted_content'] / results['successful']:.1f} words")
        print(f"Average fields per scheme: {results['total_fields'] / results['successful']:.1f} fields")
        print(f"Total targeted content extracted: {results['total_targeted_content']} words")
        
        # Calculate efficiency
        total_json_words = 1078.6 * results['successful']  # Average JSON words * successful schemes
        efficiency = (results['total_targeted_content'] / total_json_words) * 100
        print(f"Targeted extraction efficiency: {efficiency:.1f}%")
        
        if efficiency >= 60:
            print(f"🎯 SUCCESS! Reached {efficiency:.1f}% efficiency (target: 60%+)")
        else:
            missing_words = int(total_json_words * 0.6 - results['total_targeted_content'])
            print(f"📈 Need {missing_words:,} more words to reach 60% efficiency")
    
    if results['errors']:
        print(f"\nErrors encountered:")
        for error in results['errors'][:10]:  # Show first 10 errors
            print(f"- {error}")
        if len(results['errors']) > 10:
            print(f"... and {len(results['errors']) - 10} more errors")

def main():
    """Main function"""
    input_directory = "/Users/priyankjairaj/Downloads/MoTA/mySchemeData"
    output_directory = "/Users/priyankjairaj/Downloads/ctu-flowrag/targeted_schemes"
    
    print("=== Targeted Missing Content Processor ===")
    print(f"Input directory: {input_directory}")
    print(f"Output directory: {output_directory}")
    print("Processing schemes with TARGETED MISSING CONTENT extraction...")
    print("🎯 Goal: Capture 460,103 missing words to reach 60%+ efficiency!")
    print("Focus: briefDescription, schemeDefinitions, applicationProcess_md, eligibilityDescription_md")
    
    process_all_schemes(input_directory, output_directory)

if __name__ == "__main__":
    main()
