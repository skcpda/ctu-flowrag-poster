#!/usr/bin/env python3
"""
Budget Monitor - Track costs in real-time
"""

import os
import json
import time
from datetime import datetime

def monitor_budget():
    """Monitor current costs and progress"""
    print("💰 BUDGET MONITOR")
    print("=" * 30)
    
    # Check GPT descriptions cost
    gpt_summary = "organized_output/outputs/gpt_descriptions_fixed/summary.json"
    gpt_cost = 0.0
    if os.path.exists(gpt_summary):
        with open(gpt_summary, 'r') as f:
            summary = json.load(f)
            gpt_cost = summary.get('total_cost', 0.0)
    
    # Count processed schemes
    ctu_dir = "organized_output/outputs/ctu_embedding_labeled"
    relation_dir = "organized_output/outputs/ctu_relations"
    
    total_schemes = 0
    processed_schemes = 0
    
    if os.path.exists(ctu_dir):
        total_schemes = len([f for f in os.listdir(ctu_dir) if f.endswith('.json')])
    
    if os.path.exists(relation_dir):
        processed_schemes = len([f for f in os.listdir(relation_dir) if f.endswith('_relations.json')])
    
    remaining_schemes = total_schemes - processed_schemes
    
    # Estimate remaining cost
    estimated_remaining_cost = remaining_schemes * 0.001  # $0.001 per scheme
    total_estimated_cost = gpt_cost + estimated_remaining_cost
    
    print(f"📊 CURRENT STATUS:")
    print(f"   GPT Descriptions: ${gpt_cost:.4f}")
    print(f"   Total schemes: {total_schemes}")
    print(f"   Processed: {processed_schemes}")
    print(f"   Remaining: {remaining_schemes}")
    print(f"   Estimated remaining cost: ${estimated_remaining_cost:.4f}")
    print(f"   Total estimated cost: ${total_estimated_cost:.4f}")
    
    # Budget status
    budget_limit = 10.0
    if total_estimated_cost > budget_limit:
        print(f"⚠️  WARNING: Estimated cost (${total_estimated_cost:.4f}) exceeds budget (${budget_limit})")
    else:
        print(f"✅ Budget safe: ${total_estimated_cost:.4f} < ${budget_limit}")
    
    return total_estimated_cost

if __name__ == "__main__":
    monitor_budget()
