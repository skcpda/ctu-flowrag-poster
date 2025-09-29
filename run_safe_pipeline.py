#!/usr/bin/env python3
"""
Safe Pipeline Runner - Ultra-conservative budget settings
"""

import os
import sys
from ctu_relation_labeler_v2 import CTURelationLabelerV2

def run_safe_pipeline():
    """Run pipeline with ultra-conservative settings"""
    print("🔒 SAFE PIPELINE RUNNER")
    print("💰 Budget: $5 max")
    print("🎯 GPT calls: 5 per scheme max")
    print("=" * 40)
    
    # Initialize with fine-tuned model
    labeler = CTURelationLabelerV2()
    
    # Ultra-conservative settings
    labeler.max_gpt_pairs = 5  # Only 5 GPT calls per scheme
    labeler.embedding_threshold = 0.4  # Lower threshold for more BGE usage
    
    # Get remaining schemes
    ctu_dir = "organized_output/outputs/ctu_embedding_labeled"
    relation_dir = "organized_output/outputs/ctu_relations"
    
    if not os.path.exists(ctu_dir):
        print("❌ CTU directory not found!")
        return
    
    # Get all CTU files
    ctu_files = [f for f in os.listdir(ctu_dir) if f.endswith('.json')]
    
    # Get already processed files
    processed_files = set()
    if os.path.exists(relation_dir):
        processed_files = set(f.replace('_relations.json', '_labeled.json') for f in os.listdir(relation_dir) if f.endswith('_relations.json'))
    
    # Get remaining files
    remaining_files = [f for f in ctu_files if f not in processed_files]
    
    print(f"📊 STATUS:")
    print(f"   Total schemes: {len(ctu_files)}")
    print(f"   Processed: {len(processed_files)}")
    print(f"   Remaining: {len(remaining_files)}")
    
    if not remaining_files:
        print("✅ All schemes already processed!")
        return
    
    # Process in small batches
    batch_size = 5
    total_cost = 0.0
    
    for i in range(0, len(remaining_files), batch_size):
        batch_files = remaining_files[i:i+batch_size]
        
        print(f"\n📦 Processing batch {i//batch_size + 1}: {len(batch_files)} schemes")
        
        for j, scheme_file in enumerate(batch_files):
            print(f"   {j+1}. {scheme_file}")
            
            try:
                # Process scheme
                result = labeler.process_scheme_relations_optimized(
                    os.path.join(ctu_dir, scheme_file),
                    relation_dir
                )
                
                # Track cost
                if 'cost' in result:
                    total_cost += result['cost']
                    print(f"      Cost: ${result['cost']:.4f}")
                
            except Exception as e:
                print(f"      ❌ Error: {e}")
                continue
        
        print(f"   📊 Batch cost: ${total_cost:.4f}")
        
        # Safety check
        if total_cost > 3.0:  # Stop at $3 to stay under $5 budget
            print("🛑 Budget limit reached! Stopping.")
            break
    
    print(f"\n🎉 PIPELINE COMPLETE!")
    print(f"💰 Total cost: ${total_cost:.4f}")
    print(f"📊 Schemes processed: {min(len(remaining_files), (i//batch_size + 1) * batch_size)}")

if __name__ == "__main__":
    run_safe_pipeline()
