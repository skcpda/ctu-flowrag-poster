# tests/test_core_pipeline.py
import json
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent / "src"))

from prep.sent_split_lid import sent_split_lid
from ctu.segment import segment_scheme
from ctu.graph_builder import build_discourse_graph
from role.tag import hybrid_classifier
from prompt.synth import PromptSynthesizer

def test_core_pipeline():
    """Test the core CTU-FlowRAG pipeline without cultural retrieval."""
    print("Testing core CTU-FlowRAG pipeline...")
    
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
    
    print("\n5. Prompt Synthesis (without cultural retrieval)")
    synthesizer = PromptSynthesizer()
    poster_data = synthesizer.synthesize_poster_data(classified_ctus)
    
    print(f"   Generated {len(poster_data)} posters:")
    for poster in poster_data:
        print(f"   Poster {poster['poster_id']}: {poster['role']}")
        print(f"     Prompt: {poster['image_prompt'][:80]}...")
        print(f"     Caption: {poster['caption']}")
    
    print("\n✅ Core pipeline test completed successfully!")
    return poster_data

def test_fact_extraction():
    """Test fact extraction from CTU text."""
    print("\nTesting fact extraction...")
    
    synthesizer = PromptSynthesizer()
    
    test_texts = [
        "Farmers receive Rs. 6000 per year in three installments.",
        "To be eligible, farmers must own agricultural land.",
        "The application process is simple and can be done online.",
        "Contact the nearest agriculture office for help."
    ]
    
    for text in test_texts:
        facts = synthesizer.extract_facts(text)
        print(f"Text: {text}")
        print(f"Facts: {facts}")
        print()

def main():
    """Run core pipeline tests."""
    print("🚀 CTU-FlowRAG Core Pipeline Test")
    print("=" * 40)
    
    # Test core pipeline
    poster_data = test_core_pipeline()
    
    # Test fact extraction
    test_fact_extraction()
    
    print("\n🎉 Core tests completed!")
    print(f"Generated {len(poster_data)} sample posters")
    
    # Save results for inspection
    output_file = Path("test_core_results.json")
    with open(output_file, 'w') as f:
        json.dump(poster_data, f, indent=2)
    print(f"Results saved to {output_file}")

if __name__ == "__main__":
    main() 