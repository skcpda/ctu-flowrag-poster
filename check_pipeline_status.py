#!/usr/bin/env python3
"""
Check Pipeline Status
Show detailed status of all pipeline stages
"""

import os
import json
import glob
from datetime import datetime

def check_status():
    """Check the status of all pipeline stages"""
    print("📊 PIPELINE STATUS CHECK")
    print("=" * 50)
    
    # Stage 1: Processed schemes
    processed_dir = "organized_output/outputs/processed_schemes"
    if os.path.exists(processed_dir):
        processed_files = glob.glob(f"{processed_dir}/*.json")
        processed_count = len(processed_files)
        print(f"✅ Stage 1 - Processed Schemes: {processed_count}")
    else:
        print("❌ Stage 1 - Processed Schemes: Not found")
        processed_count = 0
    
    # Stage 2: GPT Descriptions
    gpt_dir = "organized_output/outputs/gpt_descriptions"
    if os.path.exists(gpt_dir):
        gpt_files = glob.glob(f"{gpt_dir}/*.json")
        gpt_count = len(gpt_files)
        print(f"✅ Stage 2 - GPT Descriptions: {gpt_count}")
    else:
        print("❌ Stage 2 - GPT Descriptions: Not found")
        gpt_count = 0
    
    # Stage 3: Fixed GPT Descriptions (most recent)
    fixed_dir = "organized_output/outputs/gpt_descriptions_fixed"
    if os.path.exists(fixed_dir):
        fixed_files = glob.glob(f"{fixed_dir}/*.json")
        # Exclude summary file
        fixed_count = len([f for f in fixed_files if not f.endswith("summary.json")])
        print(f"✅ Stage 3 - Fixed GPT Descriptions: {fixed_count}")
        
        # Check summary file
        summary_file = f"{fixed_dir}/summary.json"
        if os.path.exists(summary_file):
            with open(summary_file, 'r') as f:
                summary = json.load(f)
            print(f"   📈 Summary: {summary.get('successful', 0)} successful, {summary.get('failed', 0)} failed")
            print(f"   💰 Total cost: ${summary.get('total_cost', 0):.4f}")
    else:
        print("❌ Stage 3 - Fixed GPT Descriptions: Not found")
        fixed_count = 0
    
    # Stage 4: CTU Embedding Labeled
    ctu_dir = "organized_output/outputs/ctu_embedding_labeled"
    if os.path.exists(ctu_dir):
        ctu_files = glob.glob(f"{ctu_dir}/*.json")
        ctu_count = len(ctu_files)
        print(f"✅ Stage 4 - CTU Embedding Labeled: {ctu_count}")
    else:
        print("❌ Stage 4 - CTU Embedding Labeled: Not found")
        ctu_count = 0
    
    # Stage 5: CTU Relations
    relations_dir = "organized_output/outputs/ctu_relations"
    if os.path.exists(relations_dir):
        relation_files = glob.glob(f"{relations_dir}/*.json")
        relation_count = len(relation_files)
        print(f"✅ Stage 5 - CTU Relations: {relation_count}")
        
        # Check summary
        summary_file = f"{relations_dir}/summary.json"
        if os.path.exists(summary_file):
            with open(summary_file, 'r') as f:
                summary = json.load(f)
            print(f"   📈 Relations: {summary.get('total_relations', 0)} total")
    else:
        print("❌ Stage 5 - CTU Relations: Not found")
        relation_count = 0
    
    print("\n🎯 RECOMMENDED NEXT STEPS:")
    if fixed_count > 0 and ctu_count == 0:
        print("   → Run CTU embedding labeling (Stage 4)")
    elif ctu_count > 0 and relation_count == 0:
        print("   → Run CTU relation labeling (Stage 5)")
    elif relation_count > 0:
        print("   → Pipeline complete! Analyze results")
    else:
        print("   → Start from Stage 1 (processed schemes)")
    
    print(f"\n⏰ Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    check_status()
