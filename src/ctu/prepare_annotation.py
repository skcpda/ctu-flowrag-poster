# src/ctu/prepare_annotation.py
import json
from pathlib import Path
from typing import List, Dict
from src.io.load_scheme import load_scheme
from src.prep.sent_split_lid import sent_split_lid

def prepare_document_for_annotation(scheme_path: Path) -> Dict:
    """
    Prepare a single document for annotation.
    
    Args:
        scheme_path: Path to scheme directory
        
    Returns:
        Dictionary with document data for annotation
    """
    # Load scheme data
    scheme_data = load_scheme(scheme_path)
    
    # Get the main description text
    main_text = scheme_data.get('longDescription', '')
    if not main_text:
        # Fallback to other fields
        for field in ['shortDescription', 'name', 'objective']:
            if scheme_data.get(field):
                main_text = scheme_data[field]
                break
    
    # Split into sentences with language detection
    sent_records = sent_split_lid(main_text)
    
    # Extract just the sentences
    sentences = [rec['sent'] for rec in sent_records]
    
    # Create annotation format
    doc_data = {
        'doc_id': scheme_path.name,
        'sentences': sentences,
        'scheme_name': scheme_data.get('name', ''),
        'scheme_objective': scheme_data.get('objective', ''),
        'total_sentences': len(sentences)
    }
    
    return doc_data

def prepare_annotation_dataset(schemes_dir: Path, output_file: Path, max_docs: int = 50) -> None:
    """
    Prepare annotation dataset from scheme documents.
    
    Args:
        schemes_dir: Directory containing scheme subdirectories
        output_file: Output JSON file path
        max_docs: Maximum number of documents to include
    """
    scheme_dirs = list(schemes_dir.iterdir())
    scheme_dirs = [d for d in scheme_dirs if d.is_dir()]
    
    # Sort by name for consistent ordering
    scheme_dirs.sort(key=lambda x: x.name)
    
    # Limit to max_docs
    scheme_dirs = scheme_dirs[:max_docs]
    
    annotation_data = []
    
    for scheme_dir in scheme_dirs:
        try:
            doc_data = prepare_document_for_annotation(scheme_dir)
            annotation_data.append(doc_data)
            print(f"Prepared {scheme_dir.name} ({doc_data['total_sentences']} sentences)")
        except Exception as e:
            print(f"Error processing {scheme_dir.name}: {e}")
            continue
    
    # Save annotation dataset
    with open(output_file, 'w') as f:
        json.dump(annotation_data, f, indent=2)
    
    print(f"\nAnnotation dataset saved to {output_file}")
    print(f"Total documents: {len(annotation_data)}")
    print(f"Total sentences: {sum(d['total_sentences'] for d in annotation_data)}")

def create_doccano_format(annotation_file: Path, output_file: Path) -> None:
    """
    Convert annotation data to Doccano format.
    
    Args:
        annotation_file: Input annotation JSON file
        output_file: Output Doccano JSONL file
    """
    with open(annotation_file, 'r') as f:
        annotation_data = json.load(f)
    
    doccano_data = []
    
    for doc in annotation_data:
        # Create text for Doccano (sentences with line numbers)
        sentences = doc['sentences']
        text_lines = []
        
        for i, sent in enumerate(sentences):
            text_lines.append(f"{i+1:3d}. {sent}")
        
        text = "\n".join(text_lines)
        
        # Create Doccano entry
        doccano_entry = {
            'text': text,
            'meta': {
                'doc_id': doc['doc_id'],
                'scheme_name': doc['scheme_name'],
                'scheme_objective': doc['scheme_objective'],
                'total_sentences': doc['total_sentences']
            }
        }
        
        doccano_data.append(doccano_entry)
    
    # Save in JSONL format
    with open(output_file, 'w') as f:
        for entry in doccano_data:
            f.write(json.dumps(entry) + '\n')
    
    print(f"Doccano format saved to {output_file}")

def main():
    """CLI interface for annotation preparation."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Prepare documents for CTU annotation")
    parser.add_argument("--schemes-dir", required=True, help="Directory containing scheme data")
    parser.add_argument("--output", required=True, help="Output annotation JSON file")
    parser.add_argument("--doccano-output", help="Output Doccano JSONL file")
    parser.add_argument("--max-docs", type=int, default=50, help="Maximum documents to include")
    
    args = parser.parse_args()
    
    schemes_dir = Path(args.schemes_dir)
    output_file = Path(args.output)
    
    # Prepare annotation dataset
    prepare_annotation_dataset(schemes_dir, output_file, args.max_docs)
    
    # Create Doccano format if requested
    if args.doccano_output:
        create_doccano_format(output_file, Path(args.doccano_output))

if __name__ == "__main__":
    main() 