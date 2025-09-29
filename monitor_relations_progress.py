#!/usr/bin/env python3
"""
Monitor relations generation progress
"""

import os
import time
from datetime import datetime

def monitor_progress():
    input_dir = "organized_output/outputs/ctu_embedding_labeled"
    output_dir = "organized_output/outputs/ctu_relations_v3_fixed_all"
    
    # Count total schemes
    total_schemes = len([f for f in os.listdir(input_dir) if f.endswith('_labeled.json')])
    
    print(f"🔍 MONITORING RELATIONS GENERATION PROGRESS")
    print(f"📊 Total schemes to process: {total_schemes}")
    print(f"📁 Input directory: {input_dir}")
    print(f"📁 Output directory: {output_dir}")
    print(f"⏰ Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    
    while True:
        try:
            # Count processed files
            if os.path.exists(output_dir):
                processed_files = len([f for f in os.listdir(output_dir) if f.endswith('_relations_v3_fixed.json')])
            else:
                processed_files = 0
            
            # Calculate progress
            progress_percent = (processed_files / total_schemes) * 100 if total_schemes > 0 else 0
            remaining = total_schemes - processed_files
            
            # Estimate time remaining (rough estimate)
            if processed_files > 0:
                # Assume ~2-3 seconds per file on average
                estimated_remaining_time = (remaining * 2.5) / 60  # minutes
            else:
                estimated_remaining_time = 0
            
            print(f"\r⏳ Progress: {processed_files}/{total_schemes} ({progress_percent:.1f}%) | Remaining: {remaining} | Est. time: {estimated_remaining_time:.1f} min", end="", flush=True)
            
            # Check if complete
            if processed_files >= total_schemes:
                print(f"\n\n✅ COMPLETE! All {total_schemes} schemes processed.")
                break
            
            time.sleep(10)  # Check every 10 seconds
            
        except KeyboardInterrupt:
            print(f"\n\n⏹️  Monitoring stopped. Current progress: {processed_files}/{total_schemes}")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            time.sleep(10)

if __name__ == "__main__":
    monitor_progress()
