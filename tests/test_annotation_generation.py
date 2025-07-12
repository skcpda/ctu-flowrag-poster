# tests/test_annotation_generation.py
import json
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent / "src"))

from ctu.generate_annotations import CTUAnnotationGenerator

def test_single_scheme_annotation():
    """Test annotation generation for a single scheme."""
    print("🧪 Testing LLM-generated CTU annotation...")
    
    # Sample welfare scheme text
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
    
    # Initialize generator
    generator = CTUAnnotationGenerator()
    
    # Generate annotations
    annotations = generator.generate_ctu_annotations(sample_text, "PM-KISAN")
    
    if annotations:
        print("✅ Successfully generated annotations!")
        print(f"   Number of CTUs: {annotations['metadata']['num_ctus']}")
        print(f"   Total sentences: {annotations['metadata']['total_sentences']}")
        
        for ctu in annotations['ctus']:
            print(f"\n   CTU {ctu['ctu_id']}:")
            print(f"     Topic: {ctu['topic']}")
            print(f"     Sentences: {ctu['start_sentence']}-{ctu['end_sentence']}")
            print(f"     Confidence: {ctu['confidence']}")
            print(f"     Text: {ctu['text'][:100]}...")
        
        return annotations
    else:
        print("❌ Failed to generate annotations")
        return None

def test_batch_processing():
    """Test batch processing of multiple schemes."""
    print("\n🧪 Testing batch processing...")
    
    # Initialize generator
    generator = CTUAnnotationGenerator()
    
    # Process a few schemes from the data directory
    schemes_dir = Path("data/raw/schemes")
    output_file = Path("test_annotations.json")
    
    if schemes_dir.exists():
        # Process first 3 schemes for testing
        scheme_files = []
        for scheme_dir in list(schemes_dir.iterdir())[:3]:
            if scheme_dir.is_dir():
                desc_file = scheme_dir / "description.txt"
                if desc_file.exists():
                    scheme_files.append(desc_file)
        
        if scheme_files:
            print(f"Processing {len(scheme_files)} schemes...")
            
            results = []
            for scheme_file in scheme_files:
                result = generator.process_scheme_file(scheme_file)
                if result:
                    results.append(result)
                    print(f"✅ {result['scheme_name']}: {result['annotations']['metadata']['num_ctus']} CTUs")
            
            # Save results
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2)
            
            print(f"\n✅ Batch processing completed!")
            print(f"📁 Results saved to: {output_file}")
            
            # Print summary
            total_ctus = sum(r['annotations']['metadata']['num_ctus'] for r in results)
            print(f"   Total schemes: {len(results)}")
            print(f"   Total CTUs: {total_ctus}")
            
            return results
        else:
            print("❌ No scheme files found")
            return None
    else:
        print("❌ Schemes directory not found")
        return None

def main():
    """Run annotation generation tests."""
    print("🚀 LLM-Generated Annotation Test")
    print("=" * 40)
    
    # Test single scheme
    single_result = test_single_scheme_annotation()
    
    # Test batch processing
    batch_result = test_batch_processing()
    
    if single_result and batch_result:
        print("\n🎉 All tests passed!")
        print("✅ LLM-generated annotations are working")
        print("✅ Ready for evaluation dataset creation")
    else:
        print("\n❌ Some tests failed")

if __name__ == "__main__":
    main() 