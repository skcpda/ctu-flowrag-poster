# src/ctu/generate_annotations.py
import json
import os
from pathlib import Path
from typing import List, Dict, Any
from openai import OpenAI
from tqdm import tqdm

class CTUAnnotationGenerator:
    """Generate CTU boundary annotations using LLM."""
    
    def __init__(self, api_key: str = None):
        """Initialize with OpenAI API key."""
        if api_key:
            self.client = OpenAI(api_key=api_key)
        elif os.getenv("OPENAI_API_KEY"):
            self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        else:
            raise ValueError("OpenAI API key required")
    
    def generate_ctu_annotations(self, text: str, scheme_name: str = "Unknown") -> Dict[str, Any]:
        """
        Generate CTU boundary annotations for a given text.
        
        Args:
            text: Input text to segment
            scheme_name: Name of the welfare scheme
            
        Returns:
            Dictionary with annotations and metadata
        """
        prompt = f"""
You are an expert in discourse analysis and text segmentation. Your task is to identify Coherent Text Units (CTUs) in welfare scheme documents.

A CTU is a segment of text that:
1. Contains related information about a specific aspect of the scheme
2. Has internal coherence and logical flow
3. Can be understood as a complete unit of information
4. Typically contains 3-15 sentences

For the following welfare scheme text, identify CTU boundaries by marking the start and end of each CTU.

Welfare Scheme: {scheme_name}

Text:
{text}

Instructions:
1. Analyze the text for natural topic shifts and information grouping
2. Identify where one coherent topic ends and another begins
3. Mark CTU boundaries at logical break points
4. Each CTU should contain related information (e.g., eligibility criteria, benefits, application process, etc.)
5. Return your analysis as a JSON object with the following structure:
{{
    "ctus": [
        {{
            "ctu_id": 1,
            "start_sentence": 0,
            "end_sentence": 4,
            "text": "full text of the CTU",
            "topic": "brief description of what this CTU covers",
            "confidence": 0.9
        }}
    ],
    "metadata": {{
        "total_sentences": 15,
        "num_ctus": 3,
        "avg_sentences_per_ctu": 5.0
    }}
}}

Return only the JSON object, no additional text.
"""
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,  # Low temperature for consistent results
                max_tokens=2000
            )
            
            # Parse JSON response
            content = response.choices[0].message.content.strip()
            if content.startswith("```json"):
                content = content[7:-3]  # Remove markdown code blocks
            elif content.startswith("```"):
                content = content[3:-3]
            
            annotations = json.loads(content)
            return annotations
            
        except Exception as e:
            print(f"Error generating annotations: {e}")
            return None
    
    def process_scheme_file(self, scheme_path: Path) -> Dict[str, Any]:
        """
        Process a single scheme file and generate annotations.
        
        Args:
            scheme_path: Path to scheme description file
            
        Returns:
            Dictionary with scheme data and annotations
        """
        # Read scheme text
        with open(scheme_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        scheme_name = scheme_path.parent.name
        
        # Generate annotations
        annotations = self.generate_ctu_annotations(text, scheme_name)
        
        if annotations:
            return {
                "scheme_name": scheme_name,
                "file_path": str(scheme_path),
                "original_text": text,
                "annotations": annotations,
                "generation_method": "llm_gpt35"
            }
        else:
            return None
    
    def batch_process_schemes(self, schemes_dir: Path, output_file: Path) -> None:
        """
        Process multiple scheme files and save annotations.
        
        Args:
            schemes_dir: Directory containing scheme files
            output_file: Path to save annotations
        """
        scheme_files = []
        
        # Find all description.txt files
        for scheme_dir in schemes_dir.iterdir():
            if scheme_dir.is_dir():
                desc_file = scheme_dir / "description.txt"
                if desc_file.exists():
                    scheme_files.append(desc_file)
        
        print(f"Found {len(scheme_files)} scheme files to process")
        
        results = []
        for scheme_file in tqdm(scheme_files, desc="Processing schemes"):
            result = self.process_scheme_file(scheme_file)
            if result:
                results.append(result)
                print(f"✅ Processed {result['scheme_name']}: {result['annotations']['metadata']['num_ctus']} CTUs")
            else:
                print(f"❌ Failed to process {scheme_file}")
        
        # Save results
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n✅ Generated annotations for {len(results)} schemes")
        print(f"📁 Saved to: {output_file}")
        
        # Print summary statistics
        total_ctus = sum(r['annotations']['metadata']['num_ctus'] for r in results)
        total_sentences = sum(r['annotations']['metadata']['total_sentences'] for r in results)
        
        print(f"\n📊 Summary:")
        print(f"   Total schemes: {len(results)}")
        print(f"   Total CTUs: {total_ctus}")
        print(f"   Total sentences: {total_sentences}")
        print(f"   Avg CTUs per scheme: {total_ctus/len(results):.1f}")
        print(f"   Avg sentences per CTU: {total_sentences/total_ctus:.1f}")

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