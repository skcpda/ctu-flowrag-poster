#!/usr/bin/env python3
"""
Document Splitter - Convert scheme descriptions into structured sentences for graph building
"""

import json
import os
import re
from pathlib import Path
from typing import List, Dict, Any, Optional
import argparse

class DocumentSplitter:
    """Split policy documents into structured sentences for graph building"""
    
    def __init__(self):
        self.sentence_id_counter = 0
    
    def reset_sentence_counter(self):
        """Reset sentence ID counter for new document"""
        self.sentence_id_counter = 0
    
    def get_next_sentence_id(self) -> str:
        """Get next sentence ID"""
        self.sentence_id_counter += 1
        return f"S{self.sentence_id_counter}"
    
    def is_heading(self, line: str) -> bool:
        """Check if line is a heading"""
        line = line.strip()
        # Headings start with # or are in **bold** format
        return (line.startswith('#') or 
                (line.startswith('**') and line.endswith('**')) or
                (line.startswith('*') and line.endswith('*')))
    
    def is_table_row(self, line: str) -> bool:
        """Check if line is a table row"""
        line = line.strip()
        return '|' in line and line.count('|') >= 2
    
    def is_list_item(self, line: str) -> bool:
        """Check if line is a list item"""
        line = line.strip()
        # Various list markers
        list_patterns = [
            r'^\s*[-*•]\s+',  # Bullet points
            r'^\s*\d+[.)]\s+',  # Numbered lists
            r'^\s*\([a-zA-Z0-9]+\)\s+',  # Lettered lists like (a), (1), etc.
            r'^\s*[a-zA-Z]\.\s+',  # Single letter lists like a., b., etc.
        ]
        return any(re.match(pattern, line) for pattern in list_patterns)
    
    def clean_heading(self, line: str) -> str:
        """Clean heading text"""
        line = line.strip()
        # Remove markdown formatting
        line = re.sub(r'^#+\s*', '', line)  # Remove # symbols
        line = re.sub(r'\*\*(.*?)\*\*', r'\1', line)  # Remove **bold**
        line = re.sub(r'\*(.*?)\*', r'\1', line)  # Remove *italic*
        return line.strip()
    
    def clean_sentence(self, line: str) -> str:
        """Clean sentence text while preserving list markers"""
        line = line.strip()
        # Don't remove list markers - they're part of the structure
        return line
    
    def split_compound_sentences(self, text: str) -> List[str]:
        """Split compound sentences that can stand alone"""
        # Split on "; and" and "; or" if they start new clauses
        sentences = []
        
        # Pattern for splitting on "; and" or "; or" followed by capital letter
        pattern = r';\s+(and|or)\s+([A-Z])'
        parts = re.split(pattern, text)
        
        if len(parts) == 1:
            # No splits found, return original
            return [text]
        
        # Reconstruct sentences
        current_sentence = parts[0]
        for i in range(1, len(parts), 3):
            if i + 1 < len(parts):
                connector = parts[i]
                next_part = parts[i + 1]
                # Check if this creates a meaningful standalone sentence
                potential_sentence = current_sentence + f"; {connector} {next_part}"
                if len(potential_sentence.split()) > 5:  # Minimum word count for standalone
                    sentences.append(current_sentence.strip())
                    current_sentence = next_part
                else:
                    current_sentence += f"; {connector} {next_part}"
        
        if current_sentence.strip():
            sentences.append(current_sentence.strip())
        
        return sentences if sentences else [text]
    
    def process_line(self, line: str, line_idx: int) -> List[Dict[str, Any]]:
        """Process a single line and return sentence objects"""
        sentences = []
        line = line.rstrip()  # Remove trailing whitespace
        
        if not line.strip():
            return sentences
        
        if self.is_heading(line):
            # Process as heading
            cleaned_text = self.clean_heading(line)
            if cleaned_text:
                sentences.append({
                    "sid": self.get_next_sentence_id(),
                    "text": cleaned_text,
                    "type": "heading",
                    "line_idx": line_idx
                })
        
        elif self.is_table_row(line):
            # Process as table row
            cleaned_text = self.clean_sentence(line)
            sentences.append({
                "sid": self.get_next_sentence_id(),
                "text": cleaned_text,
                "type": "table_row",
                "line_idx": line_idx
            })
        
        elif self.is_list_item(line):
            # Process as list item (preserve markers)
            cleaned_text = self.clean_sentence(line)
            sentences.append({
                "sid": self.get_next_sentence_id(),
                "text": cleaned_text,
                "type": "sentence",
                "line_idx": line_idx
            })
        
        else:
            # Regular sentence - check for compound sentences
            compound_sentences = self.split_compound_sentences(line)
            for sentence in compound_sentences:
                cleaned_text = self.clean_sentence(sentence)
                if cleaned_text:
                    sentences.append({
                        "sid": self.get_next_sentence_id(),
                        "text": cleaned_text,
                        "type": "sentence",
                        "line_idx": line_idx
                    })
        
        return sentences
    
    def split_document(self, doc_id: str, text: str, section: str = "FULL_DOC") -> Dict[str, Any]:
        """Split a document into structured sentences"""
        self.reset_sentence_counter()
        
        lines = text.split('\n')
        sentences = []
        
        for line_idx, line in enumerate(lines, 1):
            line_sentences = self.process_line(line, line_idx)
            sentences.extend(line_sentences)
        
        return {
            "doc_id": doc_id,
            "section": section,
            "sentences": sentences
        }
    
    def split_multiple_sections(self, doc_id: str, sections: Dict[str, str]) -> Dict[str, Any]:
        """Split multiple sections of a document"""
        section_results = []
        
        for section_name, section_text in sections.items():
            section_result = self.split_document(doc_id, section_text, section_name)
            section_results.append(section_result)
        
        return {
            "doc_id": doc_id,
            "sections": section_results
        }

def process_scheme_file(input_file: Path, output_file: Path, splitter: DocumentSplitter):
    """Process a single scheme description file"""
    try:
        # Read the description file
        with open(input_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Get scheme name from directory
        scheme_name = input_file.parent.name
        
        # Split the document
        result = splitter.split_document(scheme_name, content)
        
        # Save the result
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        sentence_count = len(result['sentences'])
        print(f"✓ Processed: {scheme_name} ({sentence_count} sentences)")
        return True, sentence_count
        
    except Exception as e:
        print(f"✗ Failed: {input_file.name} - {e}")
        return False, 0

def process_all_schemes(input_dir: str, output_dir: str):
    """Process all scheme description files"""
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Find all description.txt files
    description_files = list(input_path.glob("*/description.txt"))
    print(f"Found {len(description_files)} scheme description files")
    print("Splitting documents into structured sentences...")
    
    splitter = DocumentSplitter()
    results = {
        'successful': 0,
        'failed': 0,
        'total_sentences': 0,
        'errors': []
    }
    
    for i, file_path in enumerate(description_files, 1):
        print(f"\n[{i}/{len(description_files)}] Processing: {file_path.parent.name}")
        
        # Create output file path
        output_file = output_path / f"{file_path.parent.name}_sentences.json"
        
        success, sentence_count = process_scheme_file(file_path, output_file, splitter)
        
        if success:
            results['successful'] += 1
            results['total_sentences'] += sentence_count
        else:
            results['failed'] += 1
            results['errors'].append(f"{file_path.parent.name}: Processing failed")
    
    # Print final summary
    print(f"\n=== DOCUMENT SPLITTING COMPLETE ===")
    print(f"Total schemes processed: {len(description_files)}")
    print(f"Successful: {results['successful']}")
    print(f"Failed: {results['failed']}")
    if results['successful'] > 0:
        print(f"Average sentences per scheme: {results['total_sentences'] / results['successful']:.1f}")
        print(f"Total sentences created: {results['total_sentences']}")
    
    if results['errors']:
        print(f"\nErrors encountered:")
        for error in results['errors'][:10]:  # Show first 10 errors
            print(f"- {error}")
        if len(results['errors']) > 10:
            print(f"... and {len(results['errors']) - 10} more errors")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Split scheme documents into structured sentences')
    parser.add_argument('--input', '-i', default='/Users/priyankjairaj/Downloads/ctu-flowrag/targeted_schemes',
                       help='Input directory containing description.txt files')
    parser.add_argument('--output', '-o', default='/Users/priyankjairaj/Downloads/ctu-flowrag/split_sentences',
                       help='Output directory for sentence JSON files')
    
    args = parser.parse_args()
    
    print("=== Document Splitter ===")
    print(f"Input directory: {args.input}")
    print(f"Output directory: {args.output}")
    print("Splitting scheme documents into structured sentences for graph building...")
    
    process_all_schemes(args.input, args.output)

if __name__ == "__main__":
    main()
