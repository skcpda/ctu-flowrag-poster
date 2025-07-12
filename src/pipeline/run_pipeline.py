# src/pipeline/run_pipeline.py
import json
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
import argparse

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.io.load_scheme import load_scheme
from src.prep.sent_split_lid import sent_split_lid
from src.ctu.segment import segment_scheme
from src.role.tag import tag_ctus
from src.rag.cultural import CulturalRetriever
from src.prompt.synth import PromptSynthesizer
from src.image_gen.generate import ImageGenerator

class CTUFlowRAGPipeline:
    """Main pipeline for CTU-FlowRAG system."""
    
    def __init__(self, 
                 cultural_index_path: Optional[Path] = None,
                 output_dir: Path = Path("output"),
                 tiling_window: int = 6,
                 tiling_thresh: float = 0.15,
                 fallback_sentences: int = 8):
        """Initialize pipeline components."""
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.cultural_retriever = None
        if cultural_index_path and cultural_index_path.exists():
            self.cultural_retriever = CulturalRetriever(cultural_index_path)
        
        self.prompt_synthesizer = PromptSynthesizer()
        self.image_generator = ImageGenerator()

        # Segmentation params
        self.tiling_window = tiling_window
        self.tiling_thresh = tiling_thresh
        self.fallback_sentences = fallback_sentences
        
        # Pipeline results
        self.results = {
            'scheme_data': None,
            'sentences': None,
            'ctus': None,
            'tagged_ctus': None,
            'cultural_contexts': None,
            'prompts': None,
            'posters': None
        }
    
    def load_scheme_data(self, scheme_dir: Path) -> Dict[str, Any]:
        """Load and preprocess scheme data."""
        print("📁 Loading scheme data...")
        
        scheme_data = load_scheme(scheme_dir)
        
        # Combine all text for processing
        combined_text = ""
        if scheme_data.get('description'):
            combined_text += scheme_data['description'] + "\n"
        if scheme_data.get('longDescription'):
            combined_text += scheme_data['longDescription'] + "\n"
        if scheme_data.get('shortDescription'):
            combined_text += scheme_data['shortDescription'] + "\n"
        
        scheme_data['combined_text'] = combined_text.strip()
        self.results['scheme_data'] = scheme_data
        
        print(f"✅ Loaded scheme: {scheme_data.get('schemeName', 'Unknown')}")
        return scheme_data
    
    def split_sentences(self, text: str) -> List[Dict[str, Any]]:
        """Split text into sentences with language detection."""
        print("🔤 Splitting sentences and detecting languages...")
        
        sentences = sent_split_lid(text)
        self.results['sentences'] = sentences
        
        print(f"✅ Split into {len(sentences)} sentences")
        
        # Print language distribution
        lang_counts = {}
        for sent in sentences:
            lang = sent.get('lang', 'unknown')
            lang_counts[lang] = lang_counts.get(lang, 0) + 1
        
        print(f"   Language distribution: {lang_counts}")
        return sentences
    
    def segment_ctus(self, sentences: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Segment sentences into CTUs."""
        print("📝 Segmenting into Coherent Text Units...")
        
        ctus = segment_scheme(sentences,
                              window=self.tiling_window,
                              thresh=self.tiling_thresh,
                              fallback_sentences=self.fallback_sentences)
        self.results['ctus'] = ctus
        
        print(f"✅ Identified {len(ctus)} CTUs")
        for i, ctu in enumerate(ctus):
            print(f"   CTU {i+1}: {ctu['text'][:100]}...")
        
        return ctus
    
    def tag_roles(self, ctus: List[Dict[str, Any]], use_llm: bool = True) -> List[Dict[str, Any]]:
        """Tag CTUs with their roles."""
        print("🏷️ Tagging CTUs with roles...")
        
        tagged_ctus = tag_ctus(ctus, use_llm=use_llm)
        self.results['tagged_ctus'] = tagged_ctus
        
        # Print role distribution
        role_counts = {}
        for ctu in tagged_ctus:
            role = ctu.get('role', 'unknown')
            role_counts[role] = role_counts.get(role, 0) + 1
        
        print(f"✅ Tagged CTUs with roles: {role_counts}")
        return tagged_ctus
    
    def retrieve_cultural_context(self, ctus: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Retrieve cultural context for each CTU."""
        print("🌍 Retrieving cultural context...")
        
        if not self.cultural_retriever:
            print("⚠️ No cultural retriever available, skipping cultural context")
            return [{'ctu_id': ctu['ctu_id'], 'cultural_snippets': []} for ctu in ctus]
        
        cultural_contexts = []
        for ctu in ctus:
            # Query cultural retriever
            snippets = self.cultural_retriever.query_ctu(
                ctu['text'], 
                ctu.get('lang_counts', {}),
                top_k=3
            )
            
            cultural_contexts.append({
                'ctu_id': ctu['ctu_id'],
                'cultural_snippets': snippets
            })
        
        self.results['cultural_contexts'] = cultural_contexts
        print(f"✅ Retrieved cultural context for {len(cultural_contexts)} CTUs")
        return cultural_contexts
    
    def synthesize_prompts(self, tagged_ctus: List[Dict[str, Any]], 
                          cultural_contexts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Synthesize prompts for poster generation."""
        print("🎨 Synthesizing prompts...")
        
        prompts = []
        for i, (ctu, context) in enumerate(zip(tagged_ctus, cultural_contexts)):
            # Combine CTU with cultural context
            ctu_with_context = ctu.copy()
            ctu_with_context['cultural_snippets'] = context['cultural_snippets']
            
            # Generate prompt
            prompt_data = self.prompt_synthesizer.synthesize_poster_data([ctu_with_context])
            
            if prompt_data:
                prompts.extend(prompt_data)
        
        self.results['prompts'] = prompts
        print(f"✅ Generated {len(prompts)} prompts")
        return prompts
    
    def generate_posters(self, prompts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate posters from prompts."""
        print("🖼️ Generating posters...")
        
        posters = []
        for i, prompt_data in enumerate(prompts):
            try:
                # Generate image
                image_path = self.output_dir / f"poster_{i+1}.png"
                image_url = self.image_generator.generate_image(
                    prompt_data['image_prompt'],
                    str(image_path)
                )
                
                poster = {
                    'poster_id': prompt_data['poster_id'],
                    'role': prompt_data['role'],
                    'caption': prompt_data['caption'],
                    'image_path': str(image_path),
                    'image_url': image_url,
                    'prompt': prompt_data['image_prompt']
                }
                
                posters.append(poster)
                print(f"✅ Generated poster {i+1}: {prompt_data['role']}")
                
            except Exception as e:
                print(f"❌ Failed to generate poster {i+1}: {e}")
        
        self.results['posters'] = posters
        print(f"✅ Generated {len(posters)} posters")
        return posters
    
    def run_full_pipeline(self, scheme_dir: Path, use_llm: bool = True) -> Dict[str, Any]:
        """Run the complete CTU-FlowRAG pipeline."""
        print("🚀 Starting CTU-FlowRAG Pipeline")
        print("=" * 50)
        
        try:
            # Step 1: Load scheme data
            scheme_data = self.load_scheme_data(scheme_dir)
            
            # Step 2: Split sentences
            sentences = self.split_sentences(scheme_data['combined_text'])
            
            # Step 3: Segment CTUs
            ctus = self.segment_ctus(sentences)
            
            # Step 4: Tag roles
            tagged_ctus = self.tag_roles(ctus, use_llm=use_llm)
            
            # Step 5: Retrieve cultural context
            cultural_contexts = self.retrieve_cultural_context(tagged_ctus)
            
            # Step 6: Synthesize prompts
            prompts = self.synthesize_prompts(tagged_ctus, cultural_contexts)
            
            # Step 7: Generate posters
            posters = self.generate_posters(prompts)
            
            # Save results
            self.save_results()
            
            print("\n🎉 Pipeline completed successfully!")
            print(f"📊 Summary:")
            print(f"   Sentences: {len(sentences)}")
            print(f"   CTUs: {len(ctus)}")
            print(f"   Posters: {len(posters)}")
            
            return self.results
            
        except Exception as e:
            print(f"❌ Pipeline failed: {e}")
            raise
    
    def save_results(self) -> None:
        """Save pipeline results to files."""
        # Save detailed results
        results_file = self.output_dir / "pipeline_results.json"
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        # Save posters summary
        if self.results['posters']:
            posters_summary = []
            for poster in self.results['posters']:
                posters_summary.append({
                    'poster_id': poster['poster_id'],
                    'role': poster['role'],
                    'caption': poster['caption'],
                    'image_path': poster['image_path']
                })
            
            posters_file = self.output_dir / "posters_summary.json"
            with open(posters_file, 'w') as f:
                json.dump(posters_summary, f, indent=2)
        
        print(f"📁 Results saved to {self.output_dir}")

def main():
    """CLI interface for the pipeline."""
    parser = argparse.ArgumentParser(description="CTU-FlowRAG Pipeline")
    parser.add_argument("--scheme-dir", required=True, help="Directory containing scheme files")
    parser.add_argument("--cultural-index", help="Path to cultural index")
    parser.add_argument("--output-dir", default="output", help="Output directory")
    parser.add_argument("--no-llm", action="store_true", help="Use only SVM for role tagging")
    parser.add_argument("--tiling-window", type=int, default=6, help="TextTiling window size (sentences)")
    parser.add_argument("--tiling-thresh", type=float, default=0.15, help="TextTiling similarity threshold")
    parser.add_argument("--fallback-sentences", type=int, default=8, help="Sentence count for fallback CTU splitting")
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = CTUFlowRAGPipeline(
        cultural_index_path=Path(args.cultural_index) if args.cultural_index else None,
        output_dir=Path(args.output_dir),
        tiling_window=args.tiling_window,
        tiling_thresh=args.tiling_thresh,
        fallback_sentences=args.fallback_sentences
    )
    
    # Run pipeline
    results = pipeline.run_full_pipeline(
        scheme_dir=Path(args.scheme_dir),
        use_llm=not args.no_llm
    )
    
    print(f"\n✅ Pipeline completed! Check {args.output_dir} for results.")

if __name__ == "__main__":
    main() 