#!/usr/bin/env python3
"""
Comprehensive Statistics Generator for Research Paper
Generates detailed statistics about relations, labels, and graph structure
"""

import json
import os
import sys
from collections import defaultdict, Counter
from typing import Dict, List, Any, Tuple
import numpy as np
import pandas as pd
from pathlib import Path

class ComprehensiveGraphAnalyzer:
    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        self.stats = {
            # Basic counts
            'total_files': 0,
            'total_ctus': 0,
            'total_relations': 0,
            'structural_relations': 0,
            'semantic_relations': 0,
            
            # Relation analysis
            'relation_types': Counter(),
            'relation_methods': Counter(),
            'relation_confidence_stats': defaultdict(list),
            'relation_distance_stats': defaultdict(list),
            
            # Role analysis
            'role_distribution': Counter(),
            'role_pair_matrix': defaultdict(lambda: defaultdict(int)),
            'role_transition_probs': defaultdict(lambda: defaultdict(float)),
            
            # Graph structure
            'out_degree_dist': Counter(),
            'in_degree_dist': Counter(),
            'graph_density': [],
            'semantic_ratio': [],
            'section_stats': defaultdict(int),
            'max_sections': 0,
            
            # Edge analysis
            'edge_confidence_raw': [],
            'edge_confidence_calibrated': [],
            'edge_logits': [],
            'edge_distances': [],
            
            # Document analysis
            'doc_length_dist': [],
            'doc_section_dist': [],
            'doc_relation_density': [],
            'doc_semantic_ratio_dist': [],
            
            # Quality metrics
            'adjacency_complete': 0,
            'role_coverage': defaultdict(int),
            'relation_coverage': defaultdict(int),
        }
        
    def analyze_file(self, filepath: str):
        """Analyze a single relation JSON file"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            self.stats['total_files'] += 1
            
            # CTU Analysis
            ctus = data.get('ctus', [])
            self.stats['total_ctus'] += len(ctus)
            self.stats['doc_length_dist'].append(len(ctus))
            
            # Role distribution
            roles = [ctu.get('role', 'UNKNOWN') for ctu in ctus]
            for role in roles:
                self.stats['role_distribution'][role] += 1
                self.stats['role_coverage'][role] += 1
            
            # Section analysis
            sections = set(ctu.get('sid', 1) for ctu in ctus)
            num_sections = len(sections)
            self.stats['section_stats'][num_sections] += 1
            self.stats['max_sections'] = max(self.stats['max_sections'], num_sections)
            self.stats['doc_section_dist'].append(num_sections)
            
            # Relations Analysis
            relations = data.get('relations', [])
            self.stats['total_relations'] += len(relations)
            
            # Document-level metrics
            if len(ctus) > 0:
                max_possible_edges = len(ctus) * (len(ctus) - 1)
                if max_possible_edges > 0:
                    density = len(relations) / max_possible_edges
                    self.stats['graph_density'].append(density)
                    self.stats['doc_relation_density'].append(density)
                
                semantic_count = sum(1 for rel in relations if rel.get('relation', '') not in ['PRECEDES', 'SEGMENT_CONTINUATION'])
                semantic_ratio = semantic_count / len(relations) if len(relations) > 0 else 0
                self.stats['semantic_ratio'].append(semantic_ratio)
                self.stats['doc_semantic_ratio_dist'].append(semantic_ratio)
            
            # Relation analysis
            for rel in relations:
                rel_type = rel.get('relation', 'UNKNOWN')
                method = rel.get('method', 'unknown')
                distance = rel.get('distance', 0)
                
                self.stats['relation_types'][rel_type] += 1
                self.stats['relation_methods'][method] += 1
                self.stats['relation_coverage'][rel_type] += 1
                
                # Structural vs Semantic
                if rel_type in ['PRECEDES', 'SEGMENT_CONTINUATION']:
                    self.stats['structural_relations'] += 1
                else:
                    self.stats['semantic_relations'] += 1
                
                # Confidence analysis
                raw_conf = rel.get('edge_confidence_raw', rel.get('confidence', 0))
                calib_conf = rel.get('edge_confidence', rel.get('confidence', 0))
                logit = rel.get('edge_logit', 0)
                
                if raw_conf > 0:
                    self.stats['edge_confidence_raw'].append(raw_conf)
                    self.stats['relation_confidence_stats'][rel_type].append(raw_conf)
                
                if calib_conf > 0:
                    self.stats['edge_confidence_calibrated'].append(calib_conf)
                
                if logit != 0:
                    self.stats['edge_logits'].append(logit)
                
                if distance > 0:
                    self.stats['edge_distances'].append(distance)
                    self.stats['relation_distance_stats'][rel_type].append(distance)
                
                # Role pair analysis
                ctu1_info = rel.get('ctu1', {})
                ctu2_info = rel.get('ctu2', {})
                role1 = ctu1_info.get('role', 'UNKNOWN')
                role2 = ctu2_info.get('role', 'UNKNOWN')
                
                if role1 and role2:
                    self.stats['role_pair_matrix'][role1][role2] += 1
            
            # Adjacency completeness
            metadata = data.get('metadata', {})
            if metadata.get('adjacency_complete', False):
                self.stats['adjacency_complete'] += 1
                
        except Exception as e:
            print(f"Error processing {filepath}: {e}")
    
    def calculate_transition_probabilities(self):
        """Calculate role transition probabilities"""
        total_transitions = sum(sum(role_pairs.values()) for role_pairs in self.stats['role_pair_matrix'].values())
        
        for role1, role_pairs in self.stats['role_pair_matrix'].items():
            role1_total = sum(role_pairs.values())
            for role2, count in role_pairs.items():
                if role1_total > 0:
                    self.stats['role_transition_probs'][role1][role2] = count / role1_total
    
    def generate_comprehensive_report(self) -> Dict[str, Any]:
        """Generate comprehensive statistics report"""
        self.calculate_transition_probabilities()
        
        # Calculate basic statistics
        avg_density = np.mean(self.stats['graph_density']) if self.stats['graph_density'] else 0
        avg_semantic_ratio = np.mean(self.stats['semantic_ratio']) if self.stats['semantic_ratio'] else 0
        avg_doc_length = np.mean(self.stats['doc_length_dist']) if self.stats['doc_length_dist'] else 0
        avg_sections = np.mean(self.stats['doc_section_dist']) if self.stats['doc_section_dist'] else 0
        
        # Confidence statistics
        raw_conf_stats = self._calculate_stats(self.stats['edge_confidence_raw'])
        calib_conf_stats = self._calculate_stats(self.stats['edge_confidence_calibrated'])
        logit_stats = self._calculate_stats(self.stats['edge_logits'])
        
        # Distance statistics
        distance_stats = self._calculate_stats(self.stats['edge_distances'])
        
        # Document statistics
        doc_length_stats = self._calculate_stats(self.stats['doc_length_dist'])
        doc_section_stats = self._calculate_stats(self.stats['doc_section_dist'])
        doc_density_stats = self._calculate_stats(self.stats['doc_relation_density'])
        doc_semantic_stats = self._calculate_stats(self.stats['doc_semantic_ratio_dist'])
        
        # Role pair matrix as DataFrame
        role_pairs_df = pd.DataFrame(self.stats['role_pair_matrix']).fillna(0)
        
        # Relation type confidence by type
        relation_conf_by_type = {}
        for rel_type, confidences in self.stats['relation_confidence_stats'].items():
            if confidences:
                relation_conf_by_type[rel_type] = self._calculate_stats(confidences)
        
        # Relation distance by type
        relation_dist_by_type = {}
        for rel_type, distances in self.stats['relation_distance_stats'].items():
            if distances:
                relation_dist_by_type[rel_type] = self._calculate_stats(distances)
        
        report = {
            'dataset_overview': {
                'total_files': self.stats['total_files'],
                'total_ctus': self.stats['total_ctus'],
                'total_relations': self.stats['total_relations'],
                'structural_relations': self.stats['structural_relations'],
                'semantic_relations': self.stats['semantic_relations'],
                'average_ctus_per_document': round(avg_doc_length, 2),
                'average_relations_per_document': round(self.stats['total_relations'] / self.stats['total_files'], 2),
                'average_sections_per_document': round(avg_sections, 2),
                'max_sections_per_document': self.stats['max_sections']
            },
            
            'graph_structure': {
                'average_density': round(avg_density, 6),
                'density_std': round(np.std(self.stats['graph_density']), 6),
                'density_percentiles': self._percentiles(self.stats['graph_density']),
                'average_semantic_ratio': round(avg_semantic_ratio, 4),
                'semantic_ratio_std': round(np.std(self.stats['semantic_ratio']), 4),
                'semantic_ratio_percentiles': self._percentiles(self.stats['semantic_ratio']),
                'adjacency_complete_files': self.stats['adjacency_complete'],
                'adjacency_complete_percentage': round(self.stats['adjacency_complete'] / self.stats['total_files'] * 100, 2)
            },
            
            'relation_analysis': {
                'relation_type_distribution': dict(self.stats['relation_types'].most_common()),
                'relation_type_percentages': {k: round(v/self.stats['total_relations']*100, 2) for k, v in self.stats['relation_types'].items()},
                'method_distribution': dict(self.stats['relation_methods']),
                'relation_confidence_by_type': relation_conf_by_type,
                'relation_distance_by_type': relation_dist_by_type,
                'average_distance': round(distance_stats['mean'], 2),
                'distance_std': round(distance_stats['std'], 2)
            },
            
            'role_analysis': {
                'role_distribution': dict(self.stats['role_distribution'].most_common()),
                'role_percentages': {k: round(v/self.stats['total_ctus']*100, 2) for k, v in self.stats['role_distribution'].items()},
                'unique_roles': len(self.stats['role_distribution']),
                'role_coverage': dict(self.stats['role_coverage']),
                'top_role_pairs': self._get_top_role_pairs(20),
                'role_transition_matrix': dict(self.stats['role_transition_probs']),
                'role_pair_matrix': role_pairs_df.to_dict()
            },
            
            'confidence_analysis': {
                'raw_confidence': raw_conf_stats,
                'calibrated_confidence': calib_conf_stats,
                'logit_statistics': logit_stats,
                'confidence_ranges': {
                    'raw': self._confidence_ranges(self.stats['edge_confidence_raw']),
                    'calibrated': self._confidence_ranges(self.stats['edge_confidence_calibrated'])
                }
            },
            
            'degree_analysis': {
                'out_degree_distribution': dict(sorted(self.stats['out_degree_dist'].items())),
                'in_degree_distribution': dict(sorted(self.stats['in_degree_dist'].items())),
                'max_out_degree': max(self.stats['out_degree_dist'].keys()) if self.stats['out_degree_dist'] else 0,
                'max_in_degree': max(self.stats['in_degree_dist'].keys()) if self.stats['in_degree_dist'] else 0,
                'average_out_degree': round(sum(k*v for k,v in self.stats['out_degree_dist'].items()) / sum(self.stats['out_degree_dist'].values()), 2) if self.stats['out_degree_dist'] else 0,
                'average_in_degree': round(sum(k*v for k,v in self.stats['in_degree_dist'].items()) / sum(self.stats['in_degree_dist'].values()), 2) if self.stats['in_degree_dist'] else 0
            },
            
            'section_analysis': {
                'section_distribution': dict(sorted(self.stats['section_stats'].items())),
                'single_section_files': self.stats['section_stats'].get(1, 0),
                'multi_section_files': sum(v for k, v in self.stats['section_stats'].items() if k > 1),
                'max_sections_per_file': self.stats['max_sections'],
                'average_sections_per_file': round(avg_sections, 2),
                'section_std': round(np.std(self.stats['doc_section_dist']), 2)
            },
            
            'document_analysis': {
                'document_length_stats': doc_length_stats,
                'document_section_stats': doc_section_stats,
                'document_density_stats': doc_density_stats,
                'document_semantic_ratio_stats': doc_semantic_stats,
                'length_percentiles': self._percentiles(self.stats['doc_length_dist']),
                'section_percentiles': self._percentiles(self.stats['doc_section_dist'])
            },
            
            'quality_metrics': {
                'files_with_high_semantic_ratio': sum(1 for r in self.stats['semantic_ratio'] if r > 0.15),
                'files_with_medium_semantic_ratio': sum(1 for r in self.stats['semantic_ratio'] if 0.05 <= r <= 0.15),
                'files_with_low_semantic_ratio': sum(1 for r in self.stats['semantic_ratio'] if r < 0.05),
                'dense_graphs': sum(1 for d in self.stats['graph_density'] if d > 0.1),
                'sparse_graphs': sum(1 for d in self.stats['graph_density'] if d <= 0.1),
                'role_coverage_percentage': {k: round(v/self.stats['total_files']*100, 2) for k, v in self.stats['role_coverage'].items()},
                'relation_coverage_percentage': {k: round(v/self.stats['total_files']*100, 2) for k, v in self.stats['relation_coverage'].items()}
            }
        }
        
        return report
    
    def _calculate_stats(self, data: List[float]) -> Dict[str, float]:
        """Calculate basic statistics for a list of numbers"""
        if not data:
            return {'min': 0, 'max': 0, 'mean': 0, 'std': 0, 'median': 0}
        
        return {
            'min': round(min(data), 4),
            'max': round(max(data), 4),
            'mean': round(np.mean(data), 4),
            'std': round(np.std(data), 4),
            'median': round(np.median(data), 4)
        }
    
    def _percentiles(self, data: List[float]) -> Dict[str, float]:
        """Calculate percentiles for a list of numbers"""
        if not data:
            return {}
        
        return {
            'p25': round(np.percentile(data, 25), 4),
            'p50': round(np.percentile(data, 50), 4),
            'p75': round(np.percentile(data, 75), 4),
            'p90': round(np.percentile(data, 90), 4),
            'p95': round(np.percentile(data, 95), 4),
            'p99': round(np.percentile(data, 99), 4)
        }
    
    def _confidence_ranges(self, confidences: List[float]) -> Dict[str, int]:
        """Calculate confidence ranges"""
        if not confidences:
            return {}
        
        ranges = {
            'very_low (0.0-0.2)': 0,
            'low (0.2-0.4)': 0,
            'medium (0.4-0.6)': 0,
            'high (0.6-0.8)': 0,
            'very_high (0.8-1.0)': 0
        }
        
        for conf in confidences:
            if conf < 0.2:
                ranges['very_low (0.0-0.2)'] += 1
            elif conf < 0.4:
                ranges['low (0.2-0.4)'] += 1
            elif conf < 0.6:
                ranges['medium (0.4-0.6)'] += 1
            elif conf < 0.8:
                ranges['high (0.6-0.8)'] += 1
            else:
                ranges['very_high (0.8-1.0)'] += 1
        
        return ranges
    
    def _get_top_role_pairs(self, n: int) -> List[Tuple[str, str, int]]:
        """Get top N role pairs by frequency"""
        pairs = []
        for role1, role_pairs in self.stats['role_pair_matrix'].items():
            for role2, count in role_pairs.items():
                pairs.append((role1, role2, count))
        
        return sorted(pairs, key=lambda x: x[2], reverse=True)[:n]

def main():
    if len(sys.argv) != 2:
        print("Usage: python comprehensive_paper_stats.py <data_directory>")
        sys.exit(1)
    
    data_dir = sys.argv[1]
    if not os.path.exists(data_dir):
        print(f"Directory {data_dir} does not exist")
        sys.exit(1)
    
    print("🔍 Starting comprehensive analysis for research paper...")
    analyzer = ComprehensiveGraphAnalyzer(data_dir)
    
    # Process all files
    files = [f for f in os.listdir(data_dir) if f.endswith('.json')]
    print(f"📊 Processing {len(files)} files...")
    
    for i, filename in enumerate(files):
        if i % 500 == 0:
            print(f"  Progress: {i}/{len(files)} files processed")
        analyzer.analyze_file(os.path.join(data_dir, filename))
    
    print("📈 Generating comprehensive report...")
    report = analyzer.generate_comprehensive_report()
    
    # Save detailed report
    output_file = os.path.join(os.path.dirname(data_dir), 'comprehensive_paper_statistics.json')
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    # Save CSV files for easy analysis
    csv_dir = os.path.join(os.path.dirname(data_dir), 'paper_statistics_csvs')
    os.makedirs(csv_dir, exist_ok=True)
    
    # Role pair matrix CSV
    role_pairs_df = pd.DataFrame(report['role_analysis']['role_pair_matrix'])
    role_pairs_df.to_csv(os.path.join(csv_dir, 'role_pair_matrix.csv'))
    
    # Relation type distribution CSV
    rel_df = pd.DataFrame(list(report['relation_analysis']['relation_type_distribution'].items()), 
                         columns=['Relation_Type', 'Count'])
    rel_df['Percentage'] = rel_df['Count'] / rel_df['Count'].sum() * 100
    rel_df.to_csv(os.path.join(csv_dir, 'relation_type_distribution.csv'), index=False)
    
    # Role distribution CSV
    role_df = pd.DataFrame(list(report['role_analysis']['role_distribution'].items()), 
                          columns=['Role', 'Count'])
    role_df['Percentage'] = role_df['Count'] / role_df['Count'].sum() * 100
    role_df.to_csv(os.path.join(csv_dir, 'role_distribution.csv'), index=False)
    
    print(f"✅ Comprehensive analysis complete!")
    print(f"📄 Detailed report: {output_file}")
    print(f"📊 CSV files: {csv_dir}")
    
    # Print summary for paper
    print("\n" + "="*80)
    print("📊 COMPREHENSIVE STATISTICS FOR RESEARCH PAPER")
    print("="*80)
    
    overview = report['dataset_overview']
    print(f"Dataset Scale:")
    print(f"  • Files: {overview['total_files']:,}")
    print(f"  • CTUs: {overview['total_ctus']:,}")
    print(f"  • Relations: {overview['total_relations']:,}")
    print(f"  • Avg CTUs/doc: {overview['average_ctus_per_document']:.1f}")
    print(f"  • Avg relations/doc: {overview['average_relations_per_document']:.1f}")
    print(f"  • Avg sections/doc: {overview['average_sections_per_document']:.1f}")
    
    graph = report['graph_structure']
    print(f"\nGraph Structure:")
    print(f"  • Density: {graph['average_density']:.6f} ± {graph['density_std']:.6f}")
    print(f"  • Semantic ratio: {graph['average_semantic_ratio']:.3f} ± {graph['semantic_ratio_std']:.3f}")
    print(f"  • Adjacency complete: {graph['adjacency_complete_percentage']:.1f}%")
    
    relations = report['relation_analysis']
    print(f"\nRelation Types (Top 5):")
    for i, (rel_type, count) in enumerate(list(relations['relation_type_distribution'].items())[:5]):
        pct = relations['relation_type_percentages'][rel_type]
        print(f"  {i+1}. {rel_type}: {count:,} ({pct:.1f}%)")
    
    roles = report['role_analysis']
    print(f"\nRole Distribution (Top 5):")
    for i, (role, count) in enumerate(list(roles['role_distribution'].items())[:5]):
        pct = roles['role_percentages'][role]
        print(f"  {i+1}. {role}: {count:,} ({pct:.1f}%)")
    
    degrees = report['degree_analysis']
    print(f"\nDegree Analysis:")
    print(f"  • Max out-degree: {degrees['max_out_degree']}")
    print(f"  • Max in-degree: {degrees['max_in_degree']}")
    print(f"  • Avg out-degree: {degrees['average_out_degree']:.2f}")
    print(f"  • Avg in-degree: {degrees['average_in_degree']:.2f}")
    
    conf = report['confidence_analysis']
    print(f"\nConfidence Analysis:")
    print(f"  • Raw: {conf['raw_confidence']['mean']:.3f} ± {conf['raw_confidence']['std']:.3f}")
    print(f"  • Calibrated: {conf['calibrated_confidence']['mean']:.3f} ± {conf['calibrated_confidence']['std']:.3f}")
    
    sections = report['section_analysis']
    print(f"\nSection Analysis:")
    print(f"  • Multi-section files: {sections['multi_section_files']:,} ({sections['multi_section_files']/overview['total_files']*100:.1f}%)")
    print(f"  • Max sections: {sections['max_sections_per_file']}")
    print(f"  • Avg sections: {sections['average_sections_per_file']:.1f}")

if __name__ == "__main__":
    main()
