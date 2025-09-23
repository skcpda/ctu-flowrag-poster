#!/usr/bin/env python3
"""
Fixed Scheme Processor - Correctly extracts all data from JSON
"""

import json
import os
import time
from pathlib import Path
from typing import Dict, Any, Optional
import re

# Try to import Jinja2, fall back to simple string formatting if not available
try:
    from jinja2 import Template
    JINJA2_AVAILABLE = True
except ImportError:
    JINJA2_AVAILABLE = False
    print("Jinja2 not available, using simple string formatting")

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

def extract_structured_content(content_list: list) -> str:
    """Extract text from structured content arrays"""
    if not content_list:
        return ""
    
    text_parts = []
    for item in content_list:
        if isinstance(item, dict):
            if 'text' in item:
                text_parts.append(item['text'])
            elif 'children' in item:
                # Recursively extract from children
                child_text = extract_structured_content(item['children'])
                if child_text:
                    text_parts.append(child_text)
    
    return ' '.join(text_parts)

def extract_scheme_data(json_data: Dict) -> Dict[str, Any]:
    """Extract and structure scheme data from JSON"""
    scheme = {}
    
    # Get the main data structure
    data = json_data.get('data', {})
    en_data = data.get('en', {})
    basic_details = en_data.get('basicDetails', {})
    scheme_content = en_data.get('schemeContent', {})
    
    # Basic scheme information
    scheme['schemeName'] = basic_details.get('schemeName', '')
    scheme['shortDescription'] = basic_details.get('briefDescription', '')
    
    # Target population and scope
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
    
    # Detailed description - try both structured and markdown versions
    detailed_desc = scheme_content.get('detailedDescription_md', '')
    if not detailed_desc:
        # Try structured version
        structured_desc = scheme_content.get('detailedDescription', [])
        detailed_desc = extract_structured_content(structured_desc)
    
    scheme['detailedDescription_md'] = clean_text(detailed_desc)
    
    # Benefits - try both versions
    benefits_md = scheme_content.get('benefits_md', '')
    if not benefits_md:
        # Try structured version
        structured_benefits = scheme_content.get('benefits', [])
        benefits_md = extract_structured_content(structured_benefits)
    
    scheme['benefits_md'] = clean_text(benefits_md)
    
    # Eligibility
    eligibility_md = scheme_content.get('eligibilityDescription_md', '')
    if not eligibility_md:
        # Try structured version
        eligibility_criteria = scheme_content.get('eligibilityCriteria', {})
        eligibility_md = eligibility_criteria.get('eligibilityDescription_md', '')
        if not eligibility_md:
            structured_eligibility = eligibility_criteria.get('eligibilityDescription', [])
            eligibility_md = extract_structured_content(structured_eligibility)
    
    scheme['eligibilityDescription_md'] = clean_text(eligibility_md)
    
    # Exclusions
    exclusions_md = scheme_content.get('exclusions_md', '')
    if not exclusions_md:
        # Try structured version
        structured_exclusions = scheme_content.get('exclusions', [])
        exclusions_md = extract_structured_content(structured_exclusions)
    
    scheme['exclusions_md'] = clean_text(exclusions_md)
    
    # Documents required
    scheme['documents_md'] = clean_text(scheme_content.get('documentsRequired_md', ''))
    
    # Application process
    app_process = scheme_content.get('applicationProcess', [])
    if app_process and len(app_process) > 0:
        process_data = app_process[0]
        scheme['applicationProcess'] = {
            'process_md': clean_text(process_data.get('process_md', '')),
            'mode': process_data.get('mode', ''),
            'portalUrl': process_data.get('portalUrl', ''),
            'steps': process_data.get('process', [])
        }
    
    # Implementing agency
    scheme['implementingAgency'] = basic_details.get('implementingAgency', '')
    
    # References
    references = scheme_content.get('references', [])
    if references:
        scheme['references'] = [{'label': r.get('title', ''), 'url': r.get('url', '')} for r in references]
    
    # Additional fields that might be present
    scheme['objectives_md'] = clean_text(scheme_content.get('objectives_md', ''))
    scheme['definitions_md'] = clean_text(scheme_content.get('definitions_md', ''))
    scheme['timeline_md'] = clean_text(scheme_content.get('timeline_md', ''))
    scheme['references_md'] = clean_text(scheme_content.get('references_md', ''))
    
    # Language counts (if available)
    if 'lang_counts' in json_data:
        scheme['lang_counts'] = json_data['lang_counts']
    
    return scheme

def get_template() -> str:
    """Get the Jinja2 template"""
    return """{# ====== HEADER ====== #}
# {{ scheme.schemeName }}

{% if scheme.shortDescription %}
**Summary:** {{ scheme.shortDescription }}
{% endif %}

{% if scheme.detailedDescription_md %}
{{ scheme.detailedDescription_md }}
{% endif %}

{# ====== CONTEXT / TARGET POPULATION / SCOPE ====== #}
{% if scheme.targetPopulation or scheme.category or scheme.sector %}
**Who is this for:** 
{% if scheme.targetPopulation %}{{ scheme.targetPopulation }}{% if scheme.category or scheme.sector %}; {% endif %}{% endif %}
{% if scheme.category %}Category: {{ scheme.category }}{% if scheme.sector %}; {% endif %}{% endif %}
{% if scheme.sector %}Sector: {{ scheme.sector }}{% endif %}
{% endif %}

{% if scheme.geography or scheme.jurisdiction %}
**Where it applies:** 
{% if scheme.geography %}{{ scheme.geography }}{% if scheme.jurisdiction %}; {% endif %}{% endif %}
{% if scheme.jurisdiction %}{{ scheme.jurisdiction }}{% endif %}
{% endif %}

{# ====== OBJECTIVES (optional) ====== #}
{% if scheme.objectives_md %}
**Objectives:** {{ scheme.objectives_md }}
{% endif %}

{# ====== BENEFITS ====== #}
{% if scheme.benefits_md %}
**Benefits / Assistance:** {{ scheme.benefits_md }}
{% endif %}

{# ====== ELIGIBILITY ====== #}
{% if scheme.eligibilityDescription_md %}
**Eligibility:** {{ scheme.eligibilityDescription_md }}
{% endif %}

{# ====== EXCLUSIONS / NEGATIVE LIST ====== #}
{% if scheme.exclusions_md %}
**Exclusions / Not eligible:** {{ scheme.exclusions_md }}
{% endif %}

{# ====== DEFINITIONS ====== #}
{% if scheme.definitions_md %}
**Definitions:** {{ scheme.definitions_md }}
{% endif %}

{# ====== REQUIRED DOCUMENTS ====== #}
{% if scheme.documents_md %}
**Required documents:** {{ scheme.documents_md }}
{% endif %}

{# ====== APPLICATION PROCESS ====== #}
{% if scheme.applicationProcess %}
**How to apply:** 
{% if scheme.applicationProcess.process_md %}{{ scheme.applicationProcess.process_md }}{% endif %}
{% if scheme.applicationProcess.mode %}
Mode: {{ scheme.applicationProcess.mode }}.
{% endif %}
{% if scheme.applicationProcess.portalUrl %}
Apply at: {{ scheme.applicationProcess.portalUrl }}
{% endif %}
{% endif %}

{# ====== TIMELINES / CYCLE ====== #}
{% if scheme.timeline_md %}
**Timeline / Cycle:** {{ scheme.timeline_md }}
{% endif %}

{# ====== CONTACTS / AUTHORITIES ====== #}
{% if scheme.implementingAgency %}
**Contacts & Authorities:** Implementing agency: {{ scheme.implementingAgency }}.
{% endif %}

{# ====== REFERENCES ====== #}
{% if scheme.references_md %}
**References / Annexures:** {{ scheme.references_md }}
{% elif scheme.references %}
**References / Annexures:** 
{% for r in scheme.references %}
- {{ r.label }}: {{ r.url }}
{% endfor %}
{% endif %}

{# ====== METADATA (optional for debugging/provenance) ====== #}
{% if scheme.lang_counts %}
<sub>Language mix: {% for code, n in scheme.lang_counts.items() %}{{ code }}={{ n }}{% if not loop.last %}, {% endif %}{% endfor %}</sub>
{% endif %}"""

def generate_description_with_jinja2(scheme: Dict[str, Any]) -> str:
    """Generate description using Jinja2 template"""
    if not JINJA2_AVAILABLE:
        raise ImportError("Jinja2 is required for template processing")
    
    template = Template(get_template())
    return template.render(scheme=scheme)

def generate_description_simple(scheme: Dict[str, Any]) -> str:
    """Generate description using simple string formatting (fallback)"""
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
        try:
            if JINJA2_AVAILABLE:
                description = generate_description_with_jinja2(scheme)
            else:
                description = generate_description_simple(scheme)
        except Exception as e:
            print(f"Template processing failed, using simple formatting: {e}")
            description = generate_description_simple(scheme)
        
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
            'template_used': 'jinja2_template' if JINJA2_AVAILABLE else 'simple_template',
            'jinja2_available': JINJA2_AVAILABLE,
            'extracted_fields': list(scheme.keys())
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
    
    if JINJA2_AVAILABLE:
        print("Using Jinja2 template processing")
    else:
        print("Jinja2 not available, using simple string formatting")
        print("To install Jinja2: pip install jinja2")
    
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
    output_directory = "/Users/priyankjairaj/Downloads/ctu-flowrag/processed_schemes"
    
    print("=== Fixed Scheme Processor ===")
    print(f"Input directory: {input_directory}")
    print(f"Output directory: {output_directory}")
    print("Processing schemes individually with improved data extraction...")
    
    process_all_schemes(input_directory, output_directory)

if __name__ == "__main__":
    main()
