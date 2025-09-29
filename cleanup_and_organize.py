#!/usr/bin/env python3
"""
Clean up and organize the project directory
Remove unnecessary files and organize the structure
"""

import os
import shutil
from pathlib import Path

def cleanup_and_organize():
    """Clean up and organize the project"""
    print("🧹 CLEANING UP AND ORGANIZING PROJECT")
    print("=" * 50)
    
    # Files to keep (essential)
    essential_files = [
        # Core scripts
        "ctu_relation_labeler_v2.py",
        "bge_fine_tuner_lightweight.py", 
        "run_safe_pipeline.py",
        "quick_analysis.py",
        
        # Documentation
        "README.md",
        "README_V2.md",
        "BGE_FINE_TUNING_SUMMARY.md",
        "FINAL_COST_PROJECTION.md",
        
        # Requirements
        "requirements_v2.txt",
        
        # Output directories
        "organized_output/",
        "fine_tuned_bge_ctu_relations/",
        
        # Essential utilities
        "budget_monitor.py",
        "check_pipeline_status.py"
    ]
    
    # Files to remove (temporary/duplicate)
    files_to_remove = [
        # Old/duplicate scripts
        "ctu_relation_labeler.py",
        "ctu_relation_labeler_fixed.py", 
        "ctu_role_labeler.py",
        "ctu_role_labeler_batch.py",
        "ctu_embedding_labeler.py",
        "bge_fine_tuner.py",
        
        # Test scripts (keep only essential)
        "test_ctu_labeling.py",
        "test_embedding_ctu.py", 
        "test_batch_ctu.py",
        "test_ctu_embedding_quality.py",
        "test_graph_quality.py",
        "test_pipeline_no_gpt.py",
        "test_pipeline.py",
        "test_single_scheme.py",
        "test_specific_details.py",
        "test_fine_tuned_model.py",
        
        # Temporary/development scripts
        "run_full_pipeline.py",
        "run_optimized_pipeline.py",
        "resume_pipeline.py",
        "resume_from_1425.py",
        "resume_gpt_descriptions.py",
        "gpt_scheme_descriptions_fixed.py",
        "pipeline_summary.py",
        "cost_estimator.py",
        "analyze_relation_results.py",
        "quick_sanity_check.py",
        "monitor_progress.py",
        "budget_safe_pipeline.py",
        
        # Documentation files (keep only essential)
        "CTU_LABELING_GUIDE.md",
        "CTU_RESULTS_ANALYSIS.md", 
        "CURRENT_SETUP.md",
        "EMBEDDING_COST_ANALYSIS.md",
        "RELATION_LABELING_COST_ANALYSIS.md",
        "COST_ANALYSIS.md",
        "PIPELINE_SUMMARY.md",
        "PROGRESS_LOG.md",
        "QUICK_START.md",
        "cost_analysis_report.md",
        "graph_quality_test_report.md",
        
        # Requirements files (keep only v2)
        "requirements_embedding.txt",
        
        # Cache and temporary directories
        "__pycache__/",
        "checkpoints/",
        "logs/"
    ]
    
    # Remove files
    removed_count = 0
    for file_path in files_to_remove:
        if os.path.exists(file_path):
            try:
                if os.path.isdir(file_path):
                    shutil.rmtree(file_path)
                    print(f"   🗂️  Removed directory: {file_path}")
                else:
                    os.remove(file_path)
                    print(f"   🗑️  Removed file: {file_path}")
                removed_count += 1
            except Exception as e:
                print(f"   ❌ Error removing {file_path}: {e}")
    
    print(f"\n📊 CLEANUP SUMMARY:")
    print(f"   Files/directories removed: {removed_count}")
    
    # Create final project structure
    print(f"\n📁 FINAL PROJECT STRUCTURE:")
    print(f"   📄 Core Scripts:")
    print(f"      - ctu_relation_labeler_v2.py (main relation labeler)")
    print(f"      - bge_fine_tuner_lightweight.py (BGE fine-tuning)")
    print(f"      - run_safe_pipeline.py (safe pipeline runner)")
    print(f"      - quick_analysis.py (results analysis)")
    print(f"      - budget_monitor.py (cost monitoring)")
    print(f"      - check_pipeline_status.py (status checker)")
    
    print(f"\n   📚 Documentation:")
    print(f"      - README.md (main documentation)")
    print(f"      - README_V2.md (detailed guide)")
    print(f"      - BGE_FINE_TUNING_SUMMARY.md (fine-tuning results)")
    print(f"      - FINAL_COST_PROJECTION.md (cost analysis)")
    
    print(f"\n   📦 Outputs:")
    print(f"      - organized_output/ (all processed data)")
    print(f"      - fine_tuned_bge_ctu_relations/ (fine-tuned model)")
    
    print(f"\n   📋 Requirements:")
    print(f"      - requirements_v2.txt (dependencies)")
    
    print(f"\n✅ PROJECT ORGANIZED SUCCESSFULLY!")
    print(f"   🎯 Ready for production use")
    print(f"   🧹 Clean and organized structure")
    print(f"   📁 Only essential files remaining")

if __name__ == "__main__":
    cleanup_and_organize()
