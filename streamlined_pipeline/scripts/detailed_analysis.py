#!/usr/bin/env python3
"""
Detailed Analysis Script for CTU Relation Graphs
Generates comprehensive statistics about graphs, relations, and labels
"""

import json
import os
import sys
from collections import defaultdict, Counter
from typing import Dict, List, Any
import numpy as np

class GraphAnalyzer:
    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        self.stats = {
            'files_processed': 0,
            'total_ctus': 0,
            'total_relations': 0,
            'structural_relations': 0,
            'semantic_relations': 0,
            'relation_types': Counter(),
            'role_distribution': Counter(),
            'confidence_stats': {'min': float('inf'), 'max': 0, 'mean': 0, 'std': 0},
            'graph_density': [],
            'semantic_ratios': [],
            'adjacency_complete_count': 0,
            'out_degree_distribution': Counter(),
            'role_pair_combinations': Counter(),
            'method_distribution': Counter(),
            'section_stats': {'single_section': 0, 'multi_section': 0, 'max_sections': 0},
            'edge_confidence_ranges': Counter()
        }
        
    def analyze_file(self, filepath: str):
        """Analyze a single relation JSON file"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            self.stats['files_processed'] += 1
            
            # CTU Analysis
            ctus = data.get('ctus', [])
            self.stats['total_ctus'] += len(ctus)
            
            # Role distribution
            for ctu in ctus:
                role = ctu.get('role', 'UNKNOWN')
                self.stats['role_distribution'][role] += 1
            
            # Relations Analysis
            relations = data.get('relations', [])
            self.stats['total_relations'] += len(relations)
            
            # Relation types
            for rel in relations:
                rel_type = rel.get('relation', 'UNKNOWN')
                self.stats['relation_types'][rel_type] += 1
                
                # Structural vs Semantic
                if rel_type in ['PRECEDES', 'SEGMENT_CONTINUATION']:
                    self.stats['structural_relations'] += 1
                else:
                    self.stats['semantic_relations'] += 1
                
                # Method distribution
                method = rel.get('method', 'unknown')
                self.stats['method_distribution'][method] += 1
                
                # Confidence analysis
                confidence = rel.get('edge_confidence', 0)
                if confidence > 0:
                    self.stats['confidence_stats']['min'] = min(self.stats['confidence_stats']['min'], confidence)
                    self.stats['confidence_stats']['max'] = max(self.stats['confidence_stats']['max'], confidence)
                    
                    # Confidence ranges
                    if confidence < 0.3:
                        self.stats['edge_confidence_ranges']['low (0-0.3)'] += 1
                    elif confidence < 0.7:
                        self.stats['edge_confidence_ranges']['medium (0.3-0.7)'] += 1
                    else:
                        self.stats['edge_confidence_ranges']['high (0.7-1.0)'] += 1
                
                # Role pair combinations
                ctu1_info = rel.get('ctu1', {})
                ctu2_info = rel.get('ctu2', {})
                ctu1_role = ctu1_info.get('role', 'UNKNOWN')
                ctu2_role = ctu2_info.get('role', 'UNKNOWN')
                if ctu1_role and ctu2_role:
                    self.stats['role_pair_combinations'][(ctu1_role, ctu2_role)] += 1
            
            # Graph density and semantic ratio
            if len(ctus) > 0:
                max_possible_edges = len(ctus) * (len(ctus) - 1)
                if max_possible_edges > 0:
                    density = len(relations) / max_possible_edges
                    self.stats['graph_density'].append(density)
                
                semantic_count = sum(1 for rel in relations if rel.get('relation', '') not in ['PRECEDES', 'SEGMENT_CONTINUATION'])
                semantic_ratio = semantic_count / len(relations) if len(relations) > 0 else 0
                self.stats['semantic_ratios'].append(semantic_ratio)
            
            # Adjacency completeness
            metadata = data.get('metadata', {})
            if metadata.get('adjacency_complete', False):
                self.stats['adjacency_complete_count'] += 1
            
            # Out-degree analysis
            out_degrees = defaultdict(int)
            for rel in relations:
                ctu1_info = rel.get('ctu1', {})
                ctu1_key = f"{ctu1_info.get('line_idx', '')}_{ctu1_info.get('role', '')}"
                out_degrees[ctu1_key] += 1
            
            for degree in out_degrees.values():
                self.stats['out_degree_distribution'][degree] += 1
            
            # Section analysis
            sections = set(ctu.get('section_id', 'section_0') for ctu in ctus)
            if len(sections) == 1:
                self.stats['section_stats']['single_section'] += 1
            else:
                self.stats['section_stats']['multi_section'] += 1
            self.stats['section_stats']['max_sections'] = max(self.stats['section_stats']['max_sections'], len(sections))
            
        except Exception as e:
            print(f"Error processing {filepath}: {e}")
    
    def _get_ctu_role(self, ctus: List[Dict], ctu_id: str) -> str:
        """Get role for a CTU by ID"""
        for ctu in ctus:
            if ctu.get('ctu_id', '') == ctu_id:
                return ctu.get('role', 'UNKNOWN')
        return 'UNKNOWN'
    
    def finalize_stats(self):
        """Calculate final statistics"""
        if self.stats['total_relations'] > 0:
            # Confidence statistics
            confidences = []
            for filepath in os.listdir(self.data_dir):
                if filepath.endswith('.json'):
                    try:
                        with open(os.path.join(self.data_dir, filepath), 'r', encoding='utf-8') as f:
                            data = json.load(f)
                        for rel in data.get('relations', []):
                            conf = rel.get('edge_confidence', 0)
                            if conf > 0:
                                confidences.append(conf)
                    except:
                        continue
            
            if confidences:
                self.stats['confidence_stats']['mean'] = np.mean(confidences)
                self.stats['confidence_stats']['std'] = np.std(confidences)
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive analysis report"""
        self.finalize_stats()
        
        # Calculate averages
        avg_density = np.mean(self.stats['graph_density']) if self.stats['graph_density'] else 0
        avg_semantic_ratio = np.mean(self.stats['semantic_ratios']) if self.stats['semantic_ratios'] else 0
        
        # Top role pairs
        top_role_pairs = self.stats['role_pair_combinations'].most_common(20)
        
        # Top relation types
        top_relation_types = self.stats['relation_types'].most_common(10)
        
        # Top roles
        top_roles = self.stats['role_distribution'].most_common(15)
        
        report = {
            'overview': {
                'total_files_processed': self.stats['files_processed'],
                'total_ctus': self.stats['total_ctus'],
                'total_relations': self.stats['total_relations'],
                'structural_relations': self.stats['structural_relations'],
                'semantic_relations': self.stats['semantic_relations'],
                'average_graph_density': round(avg_density, 4),
                'average_semantic_ratio': round(avg_semantic_ratio, 4),
                'adjacency_complete_files': self.stats['adjacency_complete_count'],
                'adjacency_complete_percentage': round(self.stats['adjacency_complete_count'] / self.stats['files_processed'] * 100, 2) if self.stats['files_processed'] > 0 else 0
            },
            'relation_analysis': {
                'relation_type_distribution': dict(top_relation_types),
                'method_distribution': dict(self.stats['method_distribution']),
                'confidence_statistics': {
                    'min_confidence': round(self.stats['confidence_stats']['min'], 4),
                    'max_confidence': round(self.stats['confidence_stats']['max'], 4),
                    'mean_confidence': round(self.stats['confidence_stats']['mean'], 4),
                    'std_confidence': round(self.stats['confidence_stats']['std'], 4),
                    'confidence_ranges': dict(self.stats['edge_confidence_ranges'])
                }
            },
            'role_analysis': {
                'role_distribution': dict(top_roles),
                'total_unique_roles': len(self.stats['role_distribution']),
                'top_role_pairs': [(f"{pair[0]} → {pair[1]}", count) for pair, count in top_role_pairs]
            },
            'graph_structure': {
                'out_degree_distribution': dict(sorted(self.stats['out_degree_distribution'].items())),
                'max_out_degree': max(self.stats['out_degree_distribution'].keys()) if self.stats['out_degree_distribution'] else 0,
                'average_out_degree': round(sum(k * v for k, v in self.stats['out_degree_distribution'].items()) / sum(self.stats['out_degree_distribution'].values()), 2) if self.stats['out_degree_distribution'] else 0,
                'section_statistics': self.stats['section_stats']
            },
            'quality_metrics': {
                'files_with_high_semantic_ratio': sum(1 for ratio in self.stats['semantic_ratios'] if ratio > 0.15),
                'files_with_medium_semantic_ratio': sum(1 for ratio in self.stats['semantic_ratios'] if 0.05 <= ratio <= 0.15),
                'files_with_low_semantic_ratio': sum(1 for ratio in self.stats['semantic_ratios'] if ratio < 0.05),
                'density_distribution': {
                    'sparse_graphs': sum(1 for d in self.stats['graph_density'] if d < 0.1),
                    'medium_density_graphs': sum(1 for d in self.stats['graph_density'] if 0.1 <= d < 0.3),
                    'dense_graphs': sum(1 for d in self.stats['graph_density'] if d >= 0.3)
                }
            }
        }
        
        return report

def main():
    if len(sys.argv) != 2:
        print("Usage: python detailed_analysis.py <data_directory>")
        sys.exit(1)
    
    data_dir = sys.argv[1]
    if not os.path.exists(data_dir):
        print(f"Directory {data_dir} does not exist")
        sys.exit(1)
    
    print("🔍 Starting detailed analysis...")
    analyzer = GraphAnalyzer(data_dir)
    
    # Process all files
    files = [f for f in os.listdir(data_dir) if f.endswith('.json')]
    print(f"📊 Processing {len(files)} files...")
    
    for i, filename in enumerate(files):
        if i % 500 == 0:
            print(f"  Progress: {i}/{len(files)} files processed")
        analyzer.analyze_file(os.path.join(data_dir, filename))
    
    print("📈 Generating comprehensive report...")
    report = analyzer.generate_report()
    
    # Save report
    output_file = os.path.join(os.path.dirname(data_dir), 'detailed_graph_analysis.json')
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Analysis complete! Report saved to: {output_file}")
    
    # Print summary
    print("\n" + "="*60)
    print("📊 DETAILED GRAPH ANALYSIS SUMMARY")
    print("="*60)
    
    overview = report['overview']
    print(f"📁 Files Processed: {overview['total_files_processed']:,}")
    print(f"🔗 Total CTUs: {overview['total_ctus']:,}")
    print(f"🔗 Total Relations: {overview['total_relations']:,}")
    print(f"  ├─ Structural: {overview['structural_relations']:,}")
    print(f"  └─ Semantic: {overview['semantic_relations']:,}")
    print(f"📊 Average Graph Density: {overview['average_graph_density']:.4f}")
    print(f"📊 Average Semantic Ratio: {overview['average_semantic_ratio']:.4f}")
    print(f"✅ Adjacency Complete: {overview['adjacency_complete_files']:,} files ({overview['adjacency_complete_percentage']:.1f}%)")
    
    print(f"\n🏷️  Top Relation Types:")
    for rel_type, count in list(report['relation_analysis']['relation_type_distribution'].items())[:5]:
        print(f"  • {rel_type}: {count:,}")
    
    print(f"\n👥 Top Roles:")
    for role, count in list(report['role_analysis']['role_distribution'].items())[:5]:
        print(f"  • {role}: {count:,}")
    
    print(f"\n📈 Quality Metrics:")
    quality = report['quality_metrics']
    print(f"  • High Semantic Ratio (>15%): {quality['files_with_high_semantic_ratio']:,} files")
    print(f"  • Medium Semantic Ratio (5-15%): {quality['files_with_medium_semantic_ratio']:,} files")
    print(f"  • Low Semantic Ratio (<5%): {quality['files_with_low_semantic_ratio']:,} files")

if __name__ == "__main__":
    main()
