#!/usr/bin/env python3
"""
Analyze extraction efficiency by comparing source JSON content with generated descriptions
"""

import json
import os
from pathlib import Path
import re

def count_words(text):
    """Count words in text"""
    if not text:
        return 0
    return len(str(text).split())

def extract_all_text_from_json(json_data, current_path="", extracted_texts=None):
    """Recursively extract ALL text content from JSON"""
    if extracted_texts is None:
        extracted_texts = []
    
    if isinstance(json_data, dict):
        for key, value in json_data.items():
            if isinstance(value, str) and value.strip():
                # Clean the text
                clean_text = re.sub(r'\s+', ' ', value.strip())
                if clean_text and len(clean_text) > 3:  # Only meaningful text
                    extracted_texts.append(clean_text)
            elif isinstance(value, (dict, list)):
                extract_all_text_from_json(value, f"{current_path}.{key}", extracted_texts)
    
    elif isinstance(json_data, list):
        for i, item in enumerate(json_data):
            if isinstance(item, str) and item.strip():
                clean_text = re.sub(r'\s+', ' ', item.strip())
                if clean_text and len(clean_text) > 3:
                    extracted_texts.append(clean_text)
            elif isinstance(item, (dict, list)):
                extract_all_text_from_json(item, f"{current_path}[{i}]", extracted_texts)
    
    elif isinstance(json_data, str) and json_data.strip():
        clean_text = re.sub(r'\s+', ' ', json_data.strip())
        if clean_text and len(clean_text) > 3:
            extracted_texts.append(clean_text)
    
    return extracted_texts

def count_template_words(description_text):
    """Count template/formatting words in description"""
    template_words = [
        "scheme", "name", "title", "description", "launch", "date", "tags", "summary",
        "who", "this", "for", "where", "applies", "dbt", "benefits", "assistance",
        "benefit", "type", "eligibility", "exclusions", "not", "eligible", "definitions",
        "required", "documents", "how", "apply", "timeline", "cycle", "contacts",
        "authorities", "references", "annexures", "additional", "information",
        "language", "mix", "implementing", "agency", "nodal", "department"
    ]
    
    words = description_text.lower().split()
    template_word_count = sum(1 for word in words if word in template_words)
    return template_word_count

def analyze_single_scheme(json_file_path, description_file_path):
    """Analyze a single scheme's extraction efficiency"""
    try:
        # Read JSON file
        with open(json_file_path, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
        
        # Extract all text from JSON
        all_json_texts = extract_all_text_from_json(json_data)
        json_total_words = sum(count_words(text) for text in all_json_texts)
        
        # Read description file
        with open(description_file_path, 'r', encoding='utf-8') as f:
            description_text = f.read()
        
        description_total_words = count_words(description_text)
        template_words = count_template_words(description_text)
        content_words = description_total_words - template_words
        
        return {
            'json_words': json_total_words,
            'description_words': description_total_words,
            'template_words': template_words,
            'content_words': content_words,
            'extraction_ratio': content_words / json_total_words if json_total_words > 0 else 0
        }
    
    except Exception as e:
        return None

def analyze_extraction_efficiency():
    """Analyze extraction efficiency across all schemes"""
    json_dir = Path("/Users/priyankjairaj/Downloads/MoTA/mySchemeData")
    description_dir = Path("/Users/priyankjairaj/Downloads/ctu-flowrag/complete_schemes")
    
    results = []
    processed = 0
    
    print("Analyzing extraction efficiency...")
    
    # Find all JSON files
    json_files = list(json_dir.glob("*/longDescription.txt"))
    
    for json_file in json_files:
        scheme_name = json_file.parent.name
        description_file = description_dir / scheme_name / "description.txt"
        
        if description_file.exists():
            result = analyze_single_scheme(json_file, description_file)
            if result:
                results.append(result)
                processed += 1
                
                if processed % 100 == 0:
                    print(f"Processed {processed} schemes...")
    
    if not results:
        print("No results found!")
        return
    
    # Calculate statistics
    total_json_words = sum(r['json_words'] for r in results)
    total_description_words = sum(r['description_words'] for r in results)
    total_template_words = sum(r['template_words'] for r in results)
    total_content_words = sum(r['content_words'] for r in results)
    
    avg_json_words = total_json_words / len(results)
    avg_description_words = total_description_words / len(results)
    avg_template_words = total_template_words / len(results)
    avg_content_words = total_content_words / len(results)
    avg_extraction_ratio = sum(r['extraction_ratio'] for r in results) / len(results)
    
    print(f"\n=== EXTRACTION EFFICIENCY ANALYSIS ===")
    print(f"Schemes analyzed: {len(results)}")
    print(f"\n📊 AVERAGE WORDS PER SCHEME:")
    print(f"  • Source JSON content: {avg_json_words:.1f} words")
    print(f"  • Generated description: {avg_description_words:.1f} words")
    print(f"  • Template/formatting words: {avg_template_words:.1f} words")
    print(f"  • Actual content extracted: {avg_content_words:.1f} words")
    print(f"\n📈 EXTRACTION EFFICIENCY:")
    print(f"  • Content extraction ratio: {avg_extraction_ratio:.1%}")
    print(f"  • Template overhead: {(avg_template_words/avg_description_words)*100:.1f}%")
    print(f"  • Content vs JSON ratio: {(avg_content_words/avg_json_words)*100:.1f}%")
    
    # Show some examples
    print(f"\n📋 SAMPLE RESULTS:")
    sorted_results = sorted(results, key=lambda x: x['extraction_ratio'], reverse=True)
    
    print(f"\nTop 5 extraction efficiency:")
    for i, result in enumerate(sorted_results[:5]):
        print(f"  {i+1}. JSON: {result['json_words']} words → Content: {result['content_words']} words ({result['extraction_ratio']:.1%})")
    
    print(f"\nBottom 5 extraction efficiency:")
    for i, result in enumerate(sorted_results[-5:]):
        print(f"  {i+1}. JSON: {result['json_words']} words → Content: {result['content_words']} words ({result['extraction_ratio']:.1%})")
    
    # Analysis
    high_efficiency = len([r for r in results if r['extraction_ratio'] > 0.5])
    medium_efficiency = len([r for r in results if 0.2 <= r['extraction_ratio'] <= 0.5])
    low_efficiency = len([r for r in results if r['extraction_ratio'] < 0.2])
    
    print(f"\n📊 EFFICIENCY DISTRIBUTION:")
    print(f"  • High efficiency (>50%): {high_efficiency} schemes ({high_efficiency/len(results)*100:.1f}%)")
    print(f"  • Medium efficiency (20-50%): {medium_efficiency} schemes ({medium_efficiency/len(results)*100:.1f}%)")
    print(f"  • Low efficiency (<20%): {low_efficiency} schemes ({low_efficiency/len(results)*100:.1f}%)")
    
    return {
        'avg_json_words': avg_json_words,
        'avg_content_words': avg_content_words,
        'extraction_ratio': avg_extraction_ratio,
        'total_schemes': len(results)
    }

if __name__ == "__main__":
    analyze_extraction_efficiency()
