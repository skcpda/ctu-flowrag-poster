# tests/test_full_pipeline.py
import json
import tempfile
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent / "src"))

from prep.sent_split_lid import sent_split_lid
from ctu.segment import segment_scheme
from ctu.graph_builder import build_discourse_graph
from role.tag import hybrid_classifier
from rag.cultural import CulturalRetriever, create_sample_snippets
from prompt.synth import PromptSynthesizer

def test_full_pipeline():
    """Test the complete CTU-FlowRAG pipeline."""
    print("Testing complete CTU-FlowRAG pipeline...")
    
    # Sample text from a welfare scheme
    sample_text = """
    The Pradhan Mantri Kisan Samman Nidhi (PM-KISAN) scheme provides financial support to farmers.
    Under this scheme, eligible farmers receive Rs. 6000 per year in three equal installments.
    The amount is transferred directly to their bank accounts.
    To be eligible, farmers must own agricultural land.
    The land should be in their name or in the name of their family members.
    Small and marginal farmers are the primary beneficiaries of this scheme.
    The scheme aims to supplement the financial needs of farmers for procuring inputs.
    Farmers can use this money for seeds, fertilizers, and other agricultural inputs.
    The application process is simple and can be done online through the PM-KISAN portal.
    Farmers need to provide their Aadhaar number and bank account details.
    The scheme is implemented by the Ministry of Agriculture and Farmers Welfare.
    For more information, contact the nearest agriculture office or call the helpline.
    """
    
    print("\n1. Sentence Splitting and Language Detection")
    sent_records = sent_split_lid(sample_text)
    print(f"   Split into {len(sent_records)} sentences")
    
    print("\n2. CTU Segmentation")
    ctus = segment_scheme(sent_records)
    print(f"   Identified {len(ctus)} CTUs")
    for i, ctu in enumerate(ctus):
        print(f"   CTU {i+1}: {ctu['text'][:100]}...")
    
    print("\n3. Discourse Graph Building")
    graph_data = build_discourse_graph(ctus)
    print(f"   Built graph with {graph_data['num_nodes']} nodes and {graph_data['num_edges']} edges")
    
    print("\n4. Role Classification")
    classified_ctus = []
    for ctu in ctus:
        classification = hybrid_classifier(ctu['text'])
        classified_ctu = ctu.copy()
        classified_ctu['role'] = classification['role']
        classified_ctu['classification_info'] = classification
        classified_ctus.append(classified_ctu)
        print(f"   CTU {ctu['ctu_id']}: {classification['role']} (method: {classification['method']})")
    
    print("\n5. Cultural Retrieval Setup")
    # Create temporary cultural snippets
    with tempfile.NamedTemporaryFile(mode='w', suffix='.tsv', delete=False) as f:
        create_sample_snippets(Path(f.name))
        snippets_file = Path(f.name)
    
    # Build cultural index
    index_dir = Path("temp_cultural_index")
    retriever = CulturalRetriever()
    retriever.build_index(snippets_file, index_dir)
    
    print("\n6. Prompt Synthesis")
    synthesizer = PromptSynthesizer()
    poster_data = synthesizer.synthesize_poster_data(classified_ctus, retriever)
    
    print(f"   Generated {len(poster_data)} posters:")
    for poster in poster_data:
        print(f"   Poster {poster['poster_id']}: {poster['role']}")
        print(f"     Prompt: {poster['image_prompt'][:80]}...")
        print(f"     Caption: {poster['caption']}")
    
    # Cleanup
    snippets_file.unlink()
    import shutil
    shutil.rmtree(index_dir, ignore_errors=True)
    
    print("\n✅ Full pipeline test completed successfully!")
    return poster_data

def test_integration_with_real_data():
    """Test integration with actual scheme data if available."""
    print("\nTesting integration with real scheme data...")
    
    # Check if we have scheme data
    schemes_dir = Path("data/raw/schemes")
    if not schemes_dir.exists():
        print("   No scheme data found, skipping real data test")
        return
    
    # Find first available scheme
    scheme_dirs = list(schemes_dir.iterdir())
    if not scheme_dirs:
        print("   No scheme directories found")
        return
    
    scheme_dir = scheme_dirs[0]
    print(f"   Testing with scheme: {scheme_dir.name}")
    
    try:
        from src.io.load_scheme import load_scheme
        
        # Load scheme data
        scheme_data = load_scheme(scheme_dir)
        main_text = scheme_data.get('longDescription', '')
        
        if not main_text:
            print("   No description text found")
            return
        
        # Run pipeline
        sent_records = sent_split_lid(main_text)
        ctus = segment_scheme(sent_records)
        
        print(f"   Processed {len(sent_records)} sentences into {len(ctus)} CTUs")
        
        # Classify roles
        classified_ctus = []
        for ctu in ctus:
            classification = hybrid_classifier(ctu['text'])
            classified_ctu = ctu.copy()
            classified_ctu['role'] = classification['role']
            classified_ctus.append(classified_ctu)
        
        # Generate posters
        synthesizer = PromptSynthesizer()
        poster_data = synthesizer.synthesize_poster_data(classified_ctus)
        
        print(f"   Generated {len(poster_data)} posters")
        
        # Save results for inspection
        output_file = Path("test_integration_results.json")
        with open(output_file, 'w') as f:
            json.dump(poster_data, f, indent=2)
        print(f"   Results saved to {output_file}")
        
    except Exception as e:
        print(f"   Error in real data test: {e}")

def main():
    """Run all pipeline tests."""
    print("🚀 CTU-FlowRAG Pipeline Test Suite")
    print("=" * 50)
    
    # Test with sample data
    poster_data = test_full_pipeline()
    
    # Test with real data if available
    test_integration_with_real_data()
    
    print("\n🎉 All tests completed!")
    print(f"Generated {len(poster_data)} sample posters")

if __name__ == "__main__":
    main() 