# src/rag/extract_cultural_snippets.py
import json
import re
from pathlib import Path
from typing import List, Dict, Any
import fitz  # PyMuPDF
from tqdm import tqdm

class CulturalSnippetExtractor:
    """Extract cultural snippets from Santali language resources."""
    
    def __init__(self, resources_dir: Path):
        """Initialize with Santali resources directory."""
        self.resources_dir = resources_dir
        self.snippets = []
        
        # Cultural categories for classification
        self.categories = {
            'farming': ['farm', 'agriculture', 'crop', 'seed', 'fertilizer', 'harvest', 'organic'],
            'festival': ['festival', 'celebration', 'dance', 'music', 'tradition', 'ceremony'],
            'gender': ['women', 'men', 'family', 'marriage', 'community', 'role'],
            'housing': ['house', 'home', 'construction', 'bamboo', 'mud', 'traditional'],
            'language': ['santali', 'ol chiki', 'script', 'literature', 'writing'],
            'customs': ['custom', 'tradition', 'ritual', 'belief', 'practice'],
            'economy': ['trade', 'craft', 'livelihood', 'income', 'business'],
            'social': ['community', 'village', 'society', 'relationship', 'cooperation']
        }
    
    def extract_text_from_pdf(self, pdf_path: Path) -> str:
        """Extract text from PDF file."""
        try:
            doc = fitz.open(str(pdf_path))
            text = ""
            
            for page in doc:
                text += page.get_text()
            
            doc.close()
            return text
        except Exception as e:
            print(f"Error extracting text from {pdf_path}: {e}")
            return ""
    
    def extract_snippets_from_text(self, text: str, source_file: str) -> List[Dict[str, Any]]:
        """Extract cultural snippets from text."""
        snippets = []
        
        # Split into sentences
        sentences = re.split(r'[.!?]+', text)
        
        # Cultural keywords to look for
        cultural_keywords = [
            'santali', 'santal', 'tribal', 'traditional', 'indigenous',
            'farm', 'agriculture', 'festival', 'dance', 'music',
            'women', 'family', 'community', 'village', 'house',
            'custom', 'ritual', 'belief', 'practice', 'culture',
            'ol chiki', 'script', 'language', 'literature'
        ]
        
        for i, sentence in enumerate(sentences):
            sentence = sentence.strip()
            if len(sentence) < 20:  # Skip very short sentences
                continue
            
            # Check if sentence contains cultural content
            sentence_lower = sentence.lower()
            cultural_score = sum(1 for keyword in cultural_keywords if keyword in sentence_lower)
            
            if cultural_score >= 1:  # At least one cultural keyword
                # Determine category
                category = self.classify_snippet(sentence)
                
                # Determine region (from filename or content)
                region = self.extract_region(source_file, sentence)
                
                snippet = {
                    'text': sentence,
                    'category': category,
                    'region': region,
                    'language': 'Santali',
                    'source_file': source_file,
                    'cultural_score': cultural_score,
                    'sentence_id': i
                }
                
                snippets.append(snippet)
        
        return snippets
    
    def classify_snippet(self, text: str) -> str:
        """Classify snippet into cultural categories."""
        text_lower = text.lower()
        
        # Calculate scores for each category
        category_scores = {}
        for category, keywords in self.categories.items():
            score = sum(1 for keyword in keywords if keyword in text_lower)
            category_scores[category] = score
        
        # Return category with highest score, default to 'customs'
        best_category = max(category_scores.items(), key=lambda x: x[1])
        return best_category[0] if best_category[1] > 0 else 'customs'
    
    def extract_region(self, source_file: str, text: str) -> str:
        """Extract region information from source file or text."""
        # Common regions in Santali context
        regions = ['Jharkhand', 'Odisha', 'West Bengal', 'Bihar', 'Assam']
        
        # Check text for region mentions
        text_lower = text.lower()
        for region in regions:
            if region.lower() in text_lower:
                return region
        
        # Check source filename for region hints
        source_lower = source_file.lower()
        if 'jharkhand' in source_lower or 'jharkhand' in text_lower:
            return 'Jharkhand'
        elif 'odisha' in source_lower or 'odisha' in text_lower:
            return 'Odisha'
        elif 'bengal' in source_lower or 'bengal' in text_lower:
            return 'West Bengal'
        elif 'bihar' in source_lower or 'bihar' in text_lower:
            return 'Bihar'
        elif 'assam' in source_lower or 'assam' in text_lower:
            return 'Assam'
        
        return 'Unknown'
    
    def process_all_resources(self) -> List[Dict[str, Any]]:
        """Process all Santali resources and extract snippets."""
        pdf_files = list(self.resources_dir.glob("*.pdf"))
        
        print(f"Found {len(pdf_files)} PDF files to process")
        
        all_snippets = []
        
        for pdf_file in tqdm(pdf_files, desc="Processing PDFs"):
            # Extract text from PDF
            text = self.extract_text_from_pdf(pdf_file)
            
            if text:
                # Extract snippets from text
                snippets = self.extract_snippets_from_text(text, pdf_file.name)
                all_snippets.extend(snippets)
                
                print(f"✅ {pdf_file.name}: {len(snippets)} snippets")
        
        return all_snippets
    
    def filter_and_rank_snippets(self, snippets: List[Dict[str, Any]], 
                                min_score: int = 2, max_length: int = 200) -> List[Dict[str, Any]]:
        """Filter and rank snippets by quality."""
        filtered_snippets = []
        
        for snippet in snippets:
            # Filter by cultural score and length
            if (snippet['cultural_score'] >= min_score and 
                len(snippet['text']) <= max_length and
                snippet['category'] != 'Unknown'):
                
                # Add quality score
                quality_score = snippet['cultural_score'] / len(snippet['text']) * 100
                snippet['quality_score'] = quality_score
                
                filtered_snippets.append(snippet)
        
        # Sort by quality score
        filtered_snippets.sort(key=lambda x: x['quality_score'], reverse=True)
        
        return filtered_snippets
    
    def save_snippets_tsv(self, snippets: List[Dict[str, Any]], output_file: Path) -> None:
        """Save snippets to TSV format for cultural retriever."""
        import pandas as pd
        
        # Convert to DataFrame
        df = pd.DataFrame(snippets)
        
        # Select relevant columns
        df_output = df[['text', 'category', 'region', 'language', 'cultural_score', 'quality_score']]
        
        # Save to TSV
        df_output.to_csv(output_file, sep='\t', index=False)
        
        print(f"✅ Saved {len(snippets)} snippets to {output_file}")
    
    def save_snippets_json(self, snippets: List[Dict[str, Any]], output_file: Path) -> None:
        """Save snippets to JSON format with full metadata."""
        with open(output_file, 'w') as f:
            json.dump(snippets, f, indent=2)
        
        print(f"✅ Saved {len(snippets)} snippets to {output_file}")

def main():
    """CLI interface for cultural snippet extraction."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Extract cultural snippets from Santali resources")
    parser.add_argument("--resources-dir", required=True, help="Directory containing Santali PDFs")
    parser.add_argument("--output-tsv", required=True, help="Output TSV file for cultural retriever")
    parser.add_argument("--output-json", help="Output JSON file with full metadata")
    parser.add_argument("--min-score", type=int, default=2, help="Minimum cultural score")
    parser.add_argument("--max-length", type=int, default=200, help="Maximum snippet length")
    
    args = parser.parse_args()
    
    # Initialize extractor
    extractor = CulturalSnippetExtractor(Path(args.resources_dir))
    
    # Process all resources
    print("🔍 Extracting cultural snippets from Santali resources...")
    all_snippets = extractor.process_all_resources()
    
    print(f"\n📊 Found {len(all_snippets)} raw snippets")
    
    # Filter and rank snippets
    print("🎯 Filtering and ranking snippets...")
    filtered_snippets = extractor.filter_and_rank_snippets(
        all_snippets, 
        min_score=args.min_score, 
        max_length=args.max_length
    )
    
    print(f"✅ Filtered to {len(filtered_snippets)} high-quality snippets")
    
    # Save snippets
    extractor.save_snippets_tsv(filtered_snippets, Path(args.output_tsv))
    
    if args.output_json:
        extractor.save_snippets_json(filtered_snippets, Path(args.output_json))
    
    # Print summary statistics
    categories = {}
    regions = {}
    
    for snippet in filtered_snippets:
        cat = snippet['category']
        reg = snippet['region']
        
        categories[cat] = categories.get(cat, 0) + 1
        regions[reg] = regions.get(reg, 0) + 1
    
    print(f"\n📈 Summary:")
    print(f"   Total snippets: {len(filtered_snippets)}")
    print(f"   Categories: {dict(sorted(categories.items(), key=lambda x: x[1], reverse=True))}")
    print(f"   Regions: {dict(sorted(regions.items(), key=lambda x: x[1], reverse=True))}")
    print(f"   Avg quality score: {sum(s['quality_score'] for s in filtered_snippets)/len(filtered_snippets):.2f}")

if __name__ == "__main__":
    main() 