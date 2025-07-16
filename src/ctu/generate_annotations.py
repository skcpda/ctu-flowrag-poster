# src/ctu/generate_annotations.py
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

from src.prep.sent_split_lid import sent_split_lid
from src.ctu.segment import segment_scheme
from src.role.tag import hybrid_classifier

__all__ = ["CTUAnnotationGenerator"]


class CTUAnnotationGenerator:
    """Generate (synthetic) CTU annotations suitable for quick testing."""

    def __init__(self, output_dir: Optional[Path] = None):
        self.output_dir = Path(output_dir) if output_dir else Path("output")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    def generate_ctu_annotations(self, text: str, scheme_name: str) -> Dict:
        """Return a JSON-serialisable annotation object for *text*."""
        sent_records = sent_split_lid(text)
        ctus = segment_scheme(sent_records)

        annotated_ctus: List[Dict] = []
        for ctu in ctus:
            role_info = hybrid_classifier(ctu["text"])
            annotated_ctus.append(
                {
                    "ctu_id": ctu["ctu_id"],
                    "topic": role_info["role"],
                    "start_sentence": ctu["start"],
                    "end_sentence": ctu["end"],
                    "confidence": role_info["confidence"],
                    "text": ctu["text"],
                }
            )

        annotations = {
            "scheme_name": scheme_name,
            "metadata": {
                "num_ctus": len(annotated_ctus),
                "total_sentences": len(sent_records),
            },
            "ctus": annotated_ctus,
        }
        return annotations

    # ------------------------------------------------------------------
    def process_scheme_file(self, file_path: Path) -> Optional[Dict]:
        """Load a text file and return annotation dict; returns *None* if empty."""
        if not file_path.exists():
            return None
        text = file_path.read_text(encoding="utf-8")
        if not text.strip():
            return None
        scheme_name = file_path.stem
        return {
            "scheme_name": scheme_name,
            "annotations": self.generate_ctu_annotations(text, scheme_name),
        }

def create_evaluation_dataset(annotations_file: Path, output_dir: Path) -> None:
    """
    Create evaluation dataset from LLM-generated annotations.
    
    Args:
        annotations_file: Path to annotations JSON file
        output_dir: Directory to save evaluation dataset
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(annotations_file, 'r') as f:
        annotations = json.load(f)
    
    # Create evaluation format
    eval_data = []
    
    for scheme in annotations:
        scheme_name = scheme['scheme_name']
        original_text = scheme['original_text']
        ctus = scheme['annotations']['ctus']
        
        # Create evaluation entry
        eval_entry = {
            "scheme_name": scheme_name,
            "text": original_text,
            "gold_standard_ctus": ctus,  # LLM-generated as "silver standard"
            "metadata": {
                "num_ctus": len(ctus),
                "generation_method": "llm_gpt35",
                "confidence_scores": [ctu.get('confidence', 0.8) for ctu in ctus]
            }
        }
        
        eval_data.append(eval_entry)
    
    # Save evaluation dataset
    eval_file = output_dir / "llm_generated_eval_dataset.json"
    with open(eval_file, 'w') as f:
        json.dump(eval_data, f, indent=2)
    
    print(f"✅ Created evaluation dataset: {eval_file}")
    print(f"   Schemes: {len(eval_data)}")
    print(f"   Total CTUs: {sum(len(e['gold_standard_ctus']) for e in eval_data)}")

def main():
    """CLI interface for annotation generation."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate CTU annotations using LLM")
    parser.add_argument("--schemes-dir", required=True, help="Directory containing scheme files")
    parser.add_argument("--output-file", required=True, help="Output file for annotations")
    parser.add_argument("--create-eval-dataset", help="Create evaluation dataset in this directory")
    
    args = parser.parse_args()
    
    # Initialize generator
    generator = CTUAnnotationGenerator()
    
    # Process schemes
    schemes_dir = Path(args.schemes_dir)
    output_file = Path(args.output_file)
    
    generator.batch_process_schemes(schemes_dir, output_file)
    
    # Create evaluation dataset if requested
    if args.create_eval_dataset:
        create_evaluation_dataset(output_file, Path(args.create_eval_dataset))

if __name__ == "__main__":
    main() 