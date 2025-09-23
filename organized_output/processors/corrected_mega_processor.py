#!/usr/bin/env python3
"""
Corrected Mega Processor - Actually extract the MASSIVE missing fields with correct paths
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

def extract_corrected_mega_fields_from_json(json_data: Dict) -> Dict[str, Any]:
    """Extract MEGA fields with CORRECT paths - the massive missing content we just discovered"""
    scheme = {}
    
    # Get the main data structure
    data = json_data.get('data', {})
    en_data = data.get('en', {})
    basic_details = en_data.get('basicDetails', {})
    scheme_content = en_data.get('schemeContent', {})
    
    # BASIC DETAILS - Extract everything
    scheme['schemeName'] = basic_details.get('schemeName', '')
    scheme['schemeShortTitle'] = basic_details.get('schemeShortTitle', '')
    scheme['briefDescription'] = basic_details.get('briefDescription', '')
    scheme['schemeOpenDate'] = basic_details.get('schemeOpenDate', '')
    scheme['schemeCloseDate'] = basic_details.get('schemeCloseDate', '')
    scheme['schemeFor'] = basic_details.get('schemeFor', '')
    scheme['dbtScheme'] = basic_details.get('dbtScheme', False)
    scheme['schemeImageUrl'] = basic_details.get('schemeImageUrl', '')
    
    # Tags
    tags = basic_details.get('tags', [])
    if tags:
        scheme['tags'] = ', '.join(tags)
    
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
    
    # Nodal department
    nodal_dept = basic_details.get('nodalDepartmentName', {})
    scheme['nodalDepartment'] = nodal_dept.get('label', '')
    
    # Implementing agency
    scheme['implementingAgency'] = basic_details.get('implementingAgency', '')
    
    # SCHEME CONTENT - Extract EVERYTHING with full structure parsing
    
    # Detailed Description - Try ALL possible versions
    detailed_desc_md = scheme_content.get('detailedDescription_md', '')
    detailed_desc_structured = scheme_content.get('detailedDescription', [])
    
    if detailed_desc_md:
        scheme['detailedDescription'] = clean_text(detailed_desc_md)
    elif detailed_desc_structured:
        scheme['detailedDescription'] = clean_text(str(detailed_desc_structured))
    else:
        scheme['detailedDescription'] = ''
    
    # Benefits - Try ALL versions
    benefits_md = scheme_content.get('benefits_md', '')
    benefits_structured = scheme_content.get('benefits', [])
    
    if benefits_md:
        scheme['benefits'] = clean_text(benefits_md)
    elif benefits_structured:
        scheme['benefits'] = clean_text(str(benefits_structured))
    else:
        scheme['benefits'] = ''
    
    # Benefit types
    benefit_types = scheme_content.get('benefitTypes', {})
    if benefit_types:
        scheme['benefitTypes'] = benefit_types.get('label', '')
    
    # Eligibility - Try ALL versions (THE SECOND BIGGEST MISSING FIELD!)
    eligibility_md = scheme_content.get('eligibilityDescription_md', '')
    eligibility_criteria = scheme_content.get('eligibilityCriteria', {})
    eligibility_structured = eligibility_criteria.get('eligibilityDescription', [])
    
    if eligibility_md:
        scheme['eligibility'] = clean_text(eligibility_md)
    elif eligibility_structured:
        scheme['eligibility'] = clean_text(str(eligibility_structured))
    else:
        scheme['eligibility'] = ''
    
    # Exclusions - Try ALL versions
    exclusions_md = scheme_content.get('exclusions_md', '')
    exclusions_structured = scheme_content.get('exclusions', [])
    
    if exclusions_md:
        scheme['exclusions'] = clean_text(exclusions_md)
    elif exclusions_structured:
        scheme['exclusions'] = clean_text(str(exclusions_structured))
    else:
        scheme['exclusions'] = ''
    
    # Objectives - Try ALL versions
    objectives_md = scheme_content.get('objectives_md', '')
    objectives_structured = scheme_content.get('objectives', [])
    
    if objectives_md:
        scheme['objectives'] = clean_text(objectives_md)
    elif objectives_structured:
        scheme['objectives'] = clean_text(str(objectives_structured))
    else:
        scheme['objectives'] = ''
    
    # Definitions - Try ALL versions
    definitions_md = scheme_content.get('definitions_md', '')
    scheme_definitions = scheme_content.get('schemeDefinitions', [])
    
    if definitions_md:
        scheme['definitions'] = clean_text(definitions_md)
    elif scheme_definitions:
        def_text = ""
        for def_item in scheme_definitions:
            if isinstance(def_item, dict):
                name = def_item.get('name', '')
                definition = def_item.get('definition', '')
                if name and definition:
                    def_text += f"**{name}**: {definition}\n"
        scheme['definitions'] = clean_text(def_text)
    else:
        scheme['definitions'] = ''
    
    # Documents Required - Try ALL versions
    documents_md = scheme_content.get('documentsRequired_md', '')
    documents_structured = scheme_content.get('documentsRequired', [])
    
    if documents_md:
        scheme['documents'] = clean_text(documents_md)
    elif documents_structured:
        scheme['documents'] = clean_text(str(documents_structured))
    else:
        scheme['documents'] = ''
    
    # Timeline - Try ALL versions
    timeline_md = scheme_content.get('timeline_md', '')
    timeline_structured = scheme_content.get('timeline', [])
    
    if timeline_md:
        scheme['timeline'] = clean_text(timeline_md)
    elif timeline_structured:
        scheme['timeline'] = clean_text(str(timeline_structured))
    else:
        scheme['timeline'] = ''
    
    # APPLICATION PROCESS - Extract EVERYTHING (THE BIGGEST MISSING FIELD!)
    app_process = scheme_content.get('applicationProcess', [])
    if app_process and len(app_process) > 0:
        # Try markdown version first (THE MASSIVE FIELD!)
        process_md = app_process[0].get('process_md', '')
        if process_md:
            scheme['applicationProcess'] = clean_text(process_md)
        else:
            # Extract from structured process with full formatting
            process_structured = app_process[0].get('process', [])
            if process_structured:
                scheme['applicationProcess'] = clean_text(str(process_structured))
            else:
                scheme['applicationProcess'] = ''
        
        scheme['applicationMode'] = app_process[0].get('mode', '')
        scheme['portalUrl'] = app_process[0].get('portalUrl', '')
        
        # Extract additional application processes (THE SECOND BIGGEST MISSING FIELD!)
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
        scheme['applicationProcess'] = ''
        scheme['applicationMode'] = ''
        scheme['portalUrl'] = ''
        scheme['additionalApplicationProcesses'] = ''
    
    # REFERENCES - Extract EVERYTHING
    references_md = scheme_content.get('references_md', '')
    references_list = scheme_content.get('references', [])
    
    if references_md:
        scheme['references'] = clean_text(references_md)
    elif references_list:
        ref_text = ""
        for ref in references_list:
            if isinstance(ref, dict):
                title = ref.get('title', '')
                url = ref.get('url', '')
                if title and url:
                    ref_text += f"- **{title}**: {url}\n"
        scheme['references'] = clean_text(ref_text)
    else:
        scheme['references'] = ''
    
    # ADDITIONAL FIELDS - Extract everything else we can find
    scheme['schemeImageUrl'] = scheme_content.get('schemeImageUrl', '')
    
    # Language counts
    if 'lang_counts' in json_data:
        scheme['lang_counts'] = json_data['lang_counts']
    
    # Extract any other fields we might have missed
    for key, value in scheme_content.items():
        if key not in [
            'detailedDescription_md', 'detailedDescription', 'benefits_md', 'benefits',
            'exclusions_md', 'exclusions', 'eligibilityCriteria', 'objectives_md', 'objectives',
            'definitions_md', 'schemeDefinitions', 'documentsRequired_md', 'documentsRequired',
            'timeline_md', 'timeline', 'applicationProcess', 'references_md', 'references',
            'benefitTypes', 'schemeImageUrl'
        ]:
            if isinstance(value, (str, int, float, bool)) and value:
                scheme[f'extra_{key}'] = str(value)
            elif isinstance(value, (dict, list)) and value:
                scheme[f'extra_{key}'] = clean_text(str(value))
    
    return scheme

def generate_corrected_mega_description(scheme: Dict[str, Any]) -> str:
    """Generate description with ALL extracted content - CORRECTED MEGA TEMPLATE"""
    template_parts = []
    
    # Header
    if scheme.get('schemeName'):
        template_parts.append(f"# {scheme['schemeName']}")
    
    # Short title
    if scheme.get('schemeShortTitle') and scheme.get('schemeShortTitle') != scheme.get('schemeName'):
        template_parts.append(f"\n**Short Title:** {scheme['schemeShortTitle']}")
    
    # Brief description (THE BIGGEST MISSING FIELD!)
    if scheme.get('briefDescription'):
        template_parts.append(f"\n**Summary:** {scheme['briefDescription']}")
    
    # Launch date
    if scheme.get('schemeOpenDate'):
        template_parts.append(f"\n**Launch Date:** {scheme['schemeOpenDate']}")
    
    # Closing date
    if scheme.get('schemeCloseDate'):
        template_parts.append(f"\n**Closing Date:** {scheme['schemeCloseDate']}")
    
    # Scheme for
    if scheme.get('schemeFor'):
        template_parts.append(f"\n**Scheme For:** {scheme['schemeFor']}")
    
    # Tags
    if scheme.get('tags'):
        template_parts.append(f"\n**Tags:** {scheme['tags']}")
    
    # Detailed description
    if scheme.get('detailedDescription'):
        template_parts.append(f"\n{scheme['detailedDescription']}")
    
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
    
    # Objectives
    if scheme.get('objectives'):
        template_parts.append(f"\n**Objectives:** {scheme['objectives']}")
    
    # Benefits
    if scheme.get('benefits'):
        template_parts.append(f"\n**Benefits / Assistance:** {scheme['benefits']}")
    
    # Benefit types
    if scheme.get('benefitTypes'):
        template_parts.append(f"\n**Benefit Type:** {scheme['benefitTypes']}")
    
    # Eligibility (THE SECOND BIGGEST MISSING FIELD!)
    if scheme.get('eligibility'):
        template_parts.append(f"\n**Eligibility:** {scheme['eligibility']}")
    
    # Exclusions
    if scheme.get('exclusions'):
        template_parts.append(f"\n**Exclusions / Not eligible:** {scheme['exclusions']}")
    
    # Definitions
    if scheme.get('definitions'):
        template_parts.append(f"\n**Definitions:** {scheme['definitions']}")
    
    # Required documents
    if scheme.get('documents'):
        template_parts.append(f"\n**Required documents:** {scheme['documents']}")
    
    # Application process (THE BIGGEST MISSING FIELD!)
    if scheme.get('applicationProcess'):
        template_parts.append(f"\n**How to apply:** {scheme['applicationProcess']}")
        if scheme.get('applicationMode'):
            template_parts.append(f"Mode: {scheme['applicationMode']}.")
        if scheme.get('portalUrl'):
            template_parts.append(f"Apply at: {scheme['portalUrl']}")
    
    # Additional application processes (THE SECOND BIGGEST MISSING FIELD!)
    if scheme.get('additionalApplicationProcesses'):
        template_parts.append(f"\n**Additional Application Processes:** {scheme['additionalApplicationProcesses']}")
    
    # Timeline
    if scheme.get('timeline'):
        template_parts.append(f"\n**Timeline / Cycle:** {scheme['timeline']}")
    
    # Contacts
    contact_info = []
    if scheme.get('implementingAgency'):
        contact_info.append(f"Implementing agency: {scheme['implementingAgency']}")
    if scheme.get('nodalDepartment'):
        contact_info.append(f"Nodal Department: {scheme['nodalDepartment']}")
    
    if contact_info:
        template_parts.append(f"\n**Contacts & Authorities:** {'; '.join(contact_info)}.")
    
    # References
    if scheme.get('references'):
        template_parts.append(f"\n**References / Annexures:** {scheme['references']}")
    
    # Extra fields - Add any additional content we extracted
    extra_fields = {k: v for k, v in scheme.items() if k.startswith('extra_')}
    if extra_fields:
        template_parts.append(f"\n**Additional Information:**")
        for key, value in extra_fields.items():
            field_name = key.replace('extra_', '').replace('_', ' ').title()
            template_parts.append(f"{field_name}: {value}")
    
    # Language metadata
    if scheme.get('lang_counts'):
        lang_info = ", ".join([f"{code}={count}" for code, count in scheme['lang_counts'].items()])
        template_parts.append(f"\n<sub>Language mix: {lang_info}</sub>")
    
    return "\n".join(template_parts)

def process_single_scheme(input_file: Path, output_dir: Path) -> Dict[str, Any]:
    """Process a single scheme file with corrected mega extraction"""
    result = {
        'success': False,
        'scheme_name': 'Unknown',
        'error': None,
        'content_length': 0,
        'extracted_fields': 0
    }
    
    try:
        # Read JSON data
        with open(input_file, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
        
        # Extract CORRECTED MEGA fields from scheme data
        scheme = extract_corrected_mega_fields_from_json(json_data)
        result['scheme_name'] = scheme.get('schemeName', 'Unknown Scheme')
        result['extracted_fields'] = len(scheme)
        
        # Generate description with ALL content
        description = generate_corrected_mega_description(scheme)
        result['content_length'] = len(description.split())
        
        # Create output directory
        scheme_dir = output_dir / input_file.parent.name
        scheme_dir.mkdir(exist_ok=True)
        
        # Save description
        description_file = scheme_dir / "description.txt"
        with open(description_file, 'w', encoding='utf-8') as f:
            f.write(description)
        
        result['success'] = True
        print(f"✓ Processed: {result['scheme_name']} ({result['content_length']} words, {result['extracted_fields']} fields)")
        
    except Exception as e:
        result['error'] = str(e)
        print(f"✗ Failed: {input_file.name} - {e}")
    
    return result

def process_all_schemes(input_dir: str, output_dir: str):
    """Process all schemes with corrected mega extraction"""
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Find all longDescription.txt files
    long_desc_files = list(input_path.glob("*/longDescription.txt"))
    print(f"Found {len(long_desc_files)} scheme files")
    print("Processing schemes with CORRECTED MEGA content extraction...")
    
    # Process each scheme individually
    results = {
        'successful': 0,
        'failed': 0,
        'total_content': 0,
        'total_fields': 0,
        'errors': []
    }
    
    for i, file_path in enumerate(long_desc_files, 1):
        print(f"\n[{i}/{len(long_desc_files)}] Processing: {file_path.parent.name}")
        
        result = process_single_scheme(file_path, output_path)
        
        if result['success']:
            results['successful'] += 1
            results['total_content'] += result['content_length']
            results['total_fields'] += result['extracted_fields']
        else:
            results['failed'] += 1
            results['errors'].append(f"{file_path.parent.name}: {result['error']}")
        
        # Small delay to avoid overwhelming the system
        time.sleep(0.1)
    
    # Print final summary
    print(f"\n=== CORRECTED MEGA EXTRACTION PROCESSING COMPLETE ===")
    print(f"Total schemes processed: {len(long_desc_files)}")
    print(f"Successful: {results['successful']}")
    print(f"Failed: {results['failed']}")
    if results['successful'] > 0:
        print(f"Average content per scheme: {results['total_content'] / results['successful']:.1f} words")
        print(f"Average fields per scheme: {results['total_fields'] / results['successful']:.1f} fields")
        print(f"Total content extracted: {results['total_content']} words")
    
    if results['errors']:
        print(f"\nErrors encountered:")
        for error in results['errors'][:10]:  # Show first 10 errors
            print(f"- {error}")
        if len(results['errors']) > 10:
            print(f"... and {len(results['errors']) - 10} more errors")

def main():
    """Main function"""
    input_directory = "/Users/priyankjairaj/Downloads/MoTA/mySchemeData"
    output_directory = "/Users/priyankjairaj/Downloads/ctu-flowrag/corrected_mega_schemes"
    
    print("=== Corrected Mega Extraction Processor ===")
    print(f"Input directory: {input_directory}")
    print(f"Output directory: {output_directory}")
    print("Processing schemes with CORRECTED MEGA content extraction...")
    
    process_all_schemes(input_directory, output_directory)

if __name__ == "__main__":
    main()
