#!/usr/bin/env python3
"""
Complete Pipeline Runner - Run the entire CTU relation generation pipeline
"""

import os
import sys
import json
import time
from pathlib import Path
from datetime import datetime

def run_step(script_name: str, description: str) -> bool:
    """Run a pipeline step and return success status"""
    print(f"\n{'='*60}")
    print(f"STEP: {description}")
    print(f"{'='*60}")
    
    start_time = time.time()
    
    try:
        # Import and run the script
        if script_name == "ctu_extractor":
            from ctu_extractor import main as extractor_main
            extractor_main()
        elif script_name == "role_tagger":
            from role_tagger import main as tagger_main
            tagger_main()
        elif script_name == "relation_generator":
            from relation_generator import main as generator_main
            generator_main()
        else:
            print(f"❌ Unknown script: {script_name}")
            return False
        
        elapsed = time.time() - start_time
        print(f"✅ {description} completed in {elapsed:.1f}s")
        return True
        
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"❌ {description} failed after {elapsed:.1f}s: {e}")
        return False

def check_dependencies():
    """Check if all required dependencies are available"""
    print("Checking dependencies...")
    
    # Check if input data exists
    input_dir = Path("input_data/scheme_descriptions")
    if not input_dir.exists():
        print(f"❌ Input directory not found: {input_dir}")
        return False
    
    scheme_files = list(input_dir.glob("*.json"))
    if not scheme_files:
        print(f"❌ No scheme files found in {input_dir}")
        return False
    
    print(f"✅ Found {len(scheme_files)} scheme files")
    
    # Check if model exists
    model_dir = Path("config/fine_tuned_bge_ctu_relations")
    if not model_dir.exists():
        print(f"⚠️  Fine-tuned model not found: {model_dir}")
        print("   Will use base BGE model as fallback")
    else:
        print(f"✅ Fine-tuned model found: {model_dir}")
    
    return True

def generate_final_report():
    """Generate final pipeline report"""
    print(f"\n{'='*60}")
    print("GENERATING FINAL REPORT")
    print(f"{'='*60}")
    
    # Read all summaries
    summaries = {}
    output_dir = Path("output_data")
    
    for summary_file in output_dir.rglob("*_summary.json"):
        try:
            with open(summary_file, 'r') as f:
                summaries[summary_file.stem] = json.load(f)
        except:
            pass
    
    # Generate final report
    final_report = {
        'pipeline_completed': datetime.now().isoformat(),
        'total_schemes_processed': 0,
        'total_ctus_extracted': 0,
        'total_relations_generated': 0,
        'average_semantic_ratio': 0,
        'step_summaries': summaries,
        'pipeline_status': 'COMPLETE'
    }
    
    # Aggregate metrics
    if 'extraction_summary' in summaries:
        final_report['total_schemes_processed'] = summaries['extraction_summary'].get('successful', 0)
        final_report['total_ctus_extracted'] = summaries['extraction_summary'].get('total_ctus', 0)
    
    if 'production_pipeline_summary' in summaries:
        final_report['total_relations_generated'] = summaries['production_pipeline_summary'].get('total_relations', 0)
        final_report['average_semantic_ratio'] = summaries['production_pipeline_summary'].get('average_semantic_ratio', 0)
    
    # Save final report
    with open("output_data/final_pipeline_report.json", 'w') as f:
        json.dump(final_report, f, indent=2)
    
    print(f"✅ Final report saved: output_data/final_pipeline_report.json")
    print(f"📊 Total schemes: {final_report['total_schemes_processed']}")
    print(f"📊 Total CTUs: {final_report['total_ctus_extracted']}")
    print(f"📊 Total relations: {final_report['total_relations_generated']}")
    print(f"📊 Semantic ratio: {final_report['average_semantic_ratio']:.1%}")

def main():
    """Run the complete pipeline"""
    print("🚀 CTU RELATION GENERATION PIPELINE")
    print("=" * 60)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Check dependencies
    if not check_dependencies():
        print("❌ Dependency check failed. Exiting.")
        return False
    
    # Pipeline steps
    steps = [
        ("ctu_extractor", "Extract CTUs from scheme descriptions"),
        ("role_tagger", "Tag CTUs with semantic roles"),
        ("relation_generator", "Generate RCR-GAT/CSRA ready relations")
    ]
    
    success_count = 0
    
    for script_name, description in steps:
        if run_step(script_name, description):
            success_count += 1
        else:
            print(f"❌ Pipeline failed at step: {description}")
            break
    
    # Generate final report
    if success_count == len(steps):
        generate_final_report()
        print(f"\n🎉 PIPELINE COMPLETED SUCCESSFULLY!")
        print(f"All {len(steps)} steps completed successfully.")
        print(f"Results ready for RCR-GAT/CSRA training!")
    else:
        print(f"\n❌ PIPELINE FAILED")
        print(f"Only {success_count}/{len(steps)} steps completed.")
    
    print(f"\nCompleted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == '__main__':
    main()
