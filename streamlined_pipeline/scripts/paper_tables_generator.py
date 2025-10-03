#!/usr/bin/env python3
"""
Generate detailed tables and statistics for research paper
"""

import json
import os
import pandas as pd
from collections import defaultdict, Counter
import numpy as np

def load_comprehensive_stats():
    """Load the comprehensive statistics"""
    with open('output_data/ctu_relations_production_ready/comprehensive_paper_statistics.json', 'r') as f:
        return json.load(f)

def generate_latex_tables():
    """Generate LaTeX tables for the paper"""
    stats = load_comprehensive_stats()
    
    # Table 1: Dataset Overview
    print("\\begin{table}[h]")
    print("\\centering")
    print("\\caption{Dataset Overview}")
    print("\\begin{tabular}{|l|r|}")
    print("\\hline")
    print("Metric & Value \\\\")
    print("\\hline")
    print(f"Total Documents & {stats['dataset_overview']['total_files']:,} \\\\")
    print(f"Total CTUs & {stats['dataset_overview']['total_ctus']:,} \\\\")
    print(f"Total Relations & {stats['dataset_overview']['total_relations']:,} \\\\")
    print(f"Structural Relations & {stats['dataset_overview']['structural_relations']:,} \\\\")
    print(f"Semantic Relations & {stats['dataset_overview']['semantic_relations']:,} \\\\")
    print(f"Avg CTUs per Document & {stats['dataset_overview']['average_ctus_per_document']:.1f} \\\\")
    print(f"Avg Relations per Document & {stats['dataset_overview']['average_relations_per_document']:.1f} \\\\")
    print(f"Avg Sections per Document & {stats['dataset_overview']['average_sections_per_document']:.1f} \\\\")
    print("\\hline")
    print("\\end{tabular}")
    print("\\end{table}")
    print()
    
    # Table 2: Relation Type Distribution
    print("\\begin{table}[h]")
    print("\\centering")
    print("\\caption{Relation Type Distribution}")
    print("\\begin{tabular}{|l|r|r|}")
    print("\\hline")
    print("Relation Type & Count & Percentage \\\\")
    print("\\hline")
    for rel_type, count in stats['relation_analysis']['relation_type_distribution'].items():
        pct = stats['relation_analysis']['relation_type_percentages'][rel_type]
        print(f"{rel_type} & {count:,} & {pct:.1f}\\% \\\\")
    
    # Add missing relation types for completeness
    print(f"ELABORATES & 0 & 0.0\\% \\\\")
    print(f"SUPPORTS & 0 & 0.0\\% \\\\")
    print("\\hline")
    print("\\end{tabular}")
    print("\\end{table}")
    print()
    
    # Table 3: Role Distribution
    print("\\begin{table}[h]")
    print("\\centering")
    print("\\caption{Role Distribution}")
    print("\\begin{tabular}{|l|r|r|}")
    print("\\hline")
    print("Role & Count & Percentage \\\\")
    print("\\hline")
    for role, count in stats['role_analysis']['role_distribution'].items():
        pct = stats['role_analysis']['role_percentages'][role]
        print(f"{role} & {count:,} & {pct:.1f}\\% \\\\")
    print("\\hline")
    print("\\end{tabular}")
    print("\\end{table}")
    print()
    
    # Table 4: Graph Structure Metrics
    print("\\begin{table}[h]")
    print("\\centering")
    print("\\caption{Graph Structure Metrics}")
    print("\\begin{tabular}{|l|r|}")
    print("\\hline")
    print("Metric & Value \\\\")
    print("\\hline")
    print(f"Average Density & {stats['graph_structure']['average_density']:.6f} \\\\")
    print(f"Density Std Dev & {stats['graph_structure']['density_std']:.6f} \\\\")
    print(f"Average Semantic Ratio & {stats['graph_structure']['average_semantic_ratio']:.3f} \\\\")
    print(f"Semantic Ratio Std Dev & {stats['graph_structure']['semantic_ratio_std']:.3f} \\\\")
    print(f"Adjacency Complete Files & {stats['graph_structure']['adjacency_complete_files']:,} \\\\")
    print(f"Adjacency Complete \\% & {stats['graph_structure']['adjacency_complete_percentage']:.1f}\\% \\\\")
    print("\\hline")
    print("\\end{tabular}")
    print("\\end{table}")
    print()
    
    # Table 5: Section Analysis
    print("\\begin{table}[h]")
    print("\\centering")
    print("\\caption{Section Analysis}")
    print("\\begin{tabular}{|l|r|}")
    print("\\hline")
    print("Metric & Value \\\\")
    print("\\hline")
    print(f"Single Section Files & {stats['section_analysis']['single_section_files']:,} \\\\")
    print(f"Multi-Section Files & {stats['section_analysis']['multi_section_files']:,} \\\\")
    print(f"Max Sections per File & {stats['section_analysis']['max_sections_per_file']} \\\\")
    print(f"Average Sections per File & {stats['section_analysis']['average_sections_per_file']:.1f} \\\\")
    print(f"Section Std Dev & {stats['section_analysis']['section_std']:.1f} \\\\")
    print("\\hline")
    print("\\end{tabular}")
    print("\\end{table}")
    print()

def generate_csv_summaries():
    """Generate CSV summary files for easy analysis"""
    stats = load_comprehensive_stats()
    
    # Create output directory
    os.makedirs('output_data/ctu_relations_production_ready/paper_tables', exist_ok=True)
    
    # 1. Dataset Summary
    dataset_summary = pd.DataFrame([
        ['Total Documents', stats['dataset_overview']['total_files']],
        ['Total CTUs', stats['dataset_overview']['total_ctus']],
        ['Total Relations', stats['dataset_overview']['total_relations']],
        ['Structural Relations', stats['dataset_overview']['structural_relations']],
        ['Semantic Relations', stats['dataset_overview']['semantic_relations']],
        ['Avg CTUs per Document', stats['dataset_overview']['average_ctus_per_document']],
        ['Avg Relations per Document', stats['dataset_overview']['average_relations_per_document']],
        ['Avg Sections per Document', stats['dataset_overview']['average_sections_per_document']],
        ['Max Sections per Document', stats['section_analysis']['max_sections_per_file']]
    ], columns=['Metric', 'Value'])
    dataset_summary.to_csv('output_data/ctu_relations_production_ready/paper_tables/dataset_summary.csv', index=False)
    
    # 2. Graph Structure Summary
    graph_summary = pd.DataFrame([
        ['Average Density', stats['graph_structure']['average_density']],
        ['Density Std Dev', stats['graph_structure']['density_std']],
        ['Average Semantic Ratio', stats['graph_structure']['average_semantic_ratio']],
        ['Semantic Ratio Std Dev', stats['graph_structure']['semantic_ratio_std']],
        ['Adjacency Complete Files', stats['graph_structure']['adjacency_complete_files']],
        ['Adjacency Complete %', stats['graph_structure']['adjacency_complete_percentage']]
    ], columns=['Metric', 'Value'])
    graph_summary.to_csv('output_data/ctu_relations_production_ready/paper_tables/graph_structure_summary.csv', index=False)
    
    # 3. Relation Type Summary
    rel_summary = pd.DataFrame([
        [rel_type, count, stats['relation_analysis']['relation_type_percentages'][rel_type]]
        for rel_type, count in stats['relation_analysis']['relation_type_distribution'].items()
    ], columns=['Relation_Type', 'Count', 'Percentage'])
    rel_summary.to_csv('output_data/ctu_relations_production_ready/paper_tables/relation_type_summary.csv', index=False)
    
    # 4. Role Summary
    role_summary = pd.DataFrame([
        [role, count, stats['role_analysis']['role_percentages'][role]]
        for role, count in stats['role_analysis']['role_distribution'].items()
    ], columns=['Role', 'Count', 'Percentage'])
    role_summary.to_csv('output_data/ctu_relations_production_ready/paper_tables/role_summary.csv', index=False)
    
    # 5. Section Distribution
    section_dist = pd.DataFrame([
        [sections, count] for sections, count in stats['section_analysis']['section_distribution'].items()
    ], columns=['Sections_Per_File', 'File_Count'])
    section_dist = section_dist.sort_values('Sections_Per_File')
    section_dist.to_csv('output_data/ctu_relations_production_ready/paper_tables/section_distribution.csv', index=False)
    
    # 6. Confidence Analysis
    conf_summary = pd.DataFrame([
        ['Raw Confidence Mean', stats['confidence_analysis']['raw_confidence']['mean']],
        ['Raw Confidence Std', stats['confidence_analysis']['raw_confidence']['std']],
        ['Raw Confidence Min', stats['confidence_analysis']['raw_confidence']['min']],
        ['Raw Confidence Max', stats['confidence_analysis']['raw_confidence']['max']],
        ['Calibrated Confidence Mean', stats['confidence_analysis']['calibrated_confidence']['mean']],
        ['Calibrated Confidence Std', stats['confidence_analysis']['calibrated_confidence']['std']],
        ['Calibrated Confidence Min', stats['confidence_analysis']['calibrated_confidence']['min']],
        ['Calibrated Confidence Max', stats['confidence_analysis']['calibrated_confidence']['max']]
    ], columns=['Metric', 'Value'])
    conf_summary.to_csv('output_data/ctu_relations_production_ready/paper_tables/confidence_summary.csv', index=False)
    
    # 7. Document Length Analysis
    doc_length_stats = stats['document_analysis']['document_length_stats']
    doc_length_summary = pd.DataFrame([
        ['Mean', doc_length_stats['mean']],
        ['Std Dev', doc_length_stats['std']],
        ['Min', doc_length_stats['min']],
        ['Max', doc_length_stats['max']],
        ['Median', doc_length_stats['median']]
    ], columns=['Statistic', 'CTUs_Per_Document'])
    doc_length_summary.to_csv('output_data/ctu_relations_production_ready/paper_tables/document_length_summary.csv', index=False)
    
    print("✅ Generated CSV summary files in paper_tables/ directory")

def generate_role_pair_analysis():
    """Generate detailed role pair analysis"""
    stats = load_comprehensive_stats()
    
    # Convert role pair matrix to DataFrame
    role_pairs = stats['role_analysis']['role_pair_matrix']
    role_pairs_df = pd.DataFrame(role_pairs).fillna(0)
    
    # Save full matrix
    role_pairs_df.to_csv('output_data/ctu_relations_production_ready/paper_tables/role_pair_matrix_full.csv')
    
    # Create top role pairs summary
    top_pairs = []
    for role1 in role_pairs_df.index:
        for role2 in role_pairs_df.columns:
            count = role_pairs_df.loc[role1, role2]
            if count > 0:
                top_pairs.append([role1, role2, int(count)])
    
    top_pairs_df = pd.DataFrame(top_pairs, columns=['Source_Role', 'Target_Role', 'Count'])
    top_pairs_df = top_pairs_df.sort_values('Count', ascending=False)
    top_pairs_df.to_csv('output_data/ctu_relations_production_ready/paper_tables/top_role_pairs.csv', index=False)
    
    # Create role transition probabilities
    transition_probs = stats['role_analysis']['role_transition_matrix']
    transition_df = pd.DataFrame(transition_probs).fillna(0)
    transition_df.to_csv('output_data/ctu_relations_production_ready/paper_tables/role_transition_probabilities.csv')
    
    print("✅ Generated role pair analysis files")

def main():
    print("📊 Generating detailed tables and statistics for research paper...")
    
    # Generate LaTeX tables
    print("\n📝 LaTeX Tables:")
    generate_latex_tables()
    
    # Generate CSV summaries
    print("\n📊 CSV Summaries:")
    generate_csv_summaries()
    
    # Generate role pair analysis
    print("\n🔗 Role Pair Analysis:")
    generate_role_pair_analysis()
    
    print("\n✅ All paper tables and statistics generated!")
    print("📁 Files saved in: output_data/ctu_relations_production_ready/paper_tables/")

if __name__ == "__main__":
    main()
