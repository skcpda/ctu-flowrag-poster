#!/usr/bin/env python3
"""
Process all relations for all schemes using the fixed relation labeler
"""

import os
import sys
import glob
import json
from datetime import datetime
from ctu_relation_labeler_v3_fixed import CTURelationLabelerV3Fixed

def process_all_schemes():
    input_dir = "organized_output/outputs/ctu_embedding_labeled"
    output_dir = "organized_output/outputs/ctu_relations_v3_fixed_all"
    
    print("🚀 PROCESSING ALL SCHEMES FOR RELATIONS")
    print(f"📁 Input: {input_dir}")
    print(f"📁 Output: {output_dir}")
    print(f"⏰ Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all labeled scheme files
    scheme_files = glob.glob(os.path.join(input_dir, "*_labeled.json"))
    total_schemes = len(scheme_files)
    
    print(f"📊 Found {total_schemes} schemes to process")
    
    if total_schemes == 0:
        print("❌ No scheme files found!")
        return
    
    # Initialize the fixed labeler
    print("🔧 Initializing relation labeler...")
    labeler = CTURelationLabelerV3Fixed()
    
    # Process files in batches
    batch_size = 50
    processed_count = 0
    failed_count = 0
    total_cost = 0.0
    all_relation_distributions = {}
    
    for i in range(0, total_schemes, batch_size):
        batch_files = scheme_files[i:i + batch_size]
        batch_num = (i // batch_size) + 1
        total_batches = (total_schemes + batch_size - 1) // batch_size
        
        print(f"\n📦 Processing batch {batch_num}/{total_batches} ({len(batch_files)} files)")
        
        for j, scheme_file in enumerate(batch_files):
            try:
                # Show progress
                file_num = i + j + 1
                progress = (file_num / total_schemes) * 100
                print(f"\r  [{file_num}/{total_schemes}] ({progress:.1f}%) Processing: {os.path.basename(scheme_file)}", end="", flush=True)
                
                # Process the scheme
                result = labeler.process_scheme_relations_optimized(scheme_file, output_dir)
                
                if "error" not in result:
                    processed_count += 1
                    total_cost += result.get("total_cost", 0.0)
                    
                    # Aggregate relation distributions
                    for relation, count in result.get("relation_distribution", {}).items():
                        all_relation_distributions[relation] = all_relation_distributions.get(relation, 0) + count
                else:
                    failed_count += 1
                    print(f"\n    ❌ Error: {result['error']}")
                
            except Exception as e:
                failed_count += 1
                print(f"\n    ❌ Exception: {e}")
        
        # Save progress after each batch
        progress_data = {
            "total_schemes": total_schemes,
            "processed_count": processed_count,
            "failed_count": failed_count,
            "total_cost": total_cost,
            "last_batch": batch_num,
            "last_update": datetime.now().isoformat(),
            "relation_distribution": all_relation_distributions
        }
        
        with open(os.path.join(output_dir, "progress.json"), 'w') as f:
            json.dump(progress_data, f, indent=2)
        
        print(f"\n  ✅ Batch {batch_num} complete: {processed_count} processed, {failed_count} failed")
    
    # Create final summary
    summary = {
        "total_schemes": total_schemes,
        "processed_count": processed_count,
        "failed_count": failed_count,
        "total_cost": total_cost,
        "relation_distribution": all_relation_distributions,
        "processing_timestamp": datetime.now().isoformat()
    }
    
    summary_file = os.path.join(output_dir, "final_summary.json")
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    
    print(f"\n🎉 PROCESSING COMPLETE!")
    print(f"✅ Processed: {processed_count}/{total_schemes} schemes")
    print(f"❌ Failed: {failed_count}")
    print(f"💰 Total cost: ${total_cost:.4f}")
    print(f"📁 Results saved to: {output_dir}")

if __name__ == "__main__":
    process_all_schemes()
