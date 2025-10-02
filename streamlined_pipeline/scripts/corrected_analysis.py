#!/usr/bin/env python3
"""
Corrected Analysis Script for CTU Relation Graphs
Fixes confidence reporting, section parsing, and out-degree analysis
"""

import json
import os
import sys
from collections import defaultdict, Counter
from typing import Dict, List, Any
import numpy as np

class CorrectedGraphAnalyzer:
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
            'confidence_stats': {
                'raw_min': float('inf'), 'raw_max': 0, 'raw_mean': 0, 'raw_std': 0,
                'calibrated_min': float('inf'), 'calibrated_max': 0, 'calibrated_mean': 0, 'calibrated_std': 0
            },
            'graph_density': [],
            'semantic_ratios': [],
            'adjacency_complete_count': 0,
            'out_degree_distribution': Counter(),
            'in_degree_distribution': Counter(),
            'role_pair_combinations': Counter(),
            'method_distribution': Counter(),
            'section_stats': {'single_section': 0, 'multi_section': 0, 'max_sections': 0, 'section_distribution': Counter()},
            'confidence_ranges_raw': Counter(),
            'confidence_ranges_calibrated': Counter()
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
            
            # Section analysis - check for 'sid' field
            sections = set()
            for ctu in ctus:
                section_id = ctu.get('sid', ctu.get('section_id', 'section_0'))
                sections.add(section_id)
            
            if len(sections) == 1:
                self.stats['section_stats']['single_section'] += 1
            else:
                self.stats['section_stats']['multi_section'] += 1
            self.stats['section_stats']['max_sections'] = max(self.stats['section_stats']['max_sections'], len(sections))
            self.stats['section_stats']['section_distribution'][len(sections)] += 1
            
            # Relations Analysis
            relations = data.get('relations', [])
            self.stats['total_relations'] += len(relations)
            
            # Confidence analysis - separate raw vs calibrated
            raw_confidences = []
            calibrated_confidences = []
            
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
                
                # Confidence analysis - both raw and calibrated
                raw_conf = rel.get('edge_confidence_raw', rel.get('confidence', 0))
                calibrated_conf = rel.get('edge_confidence', rel.get('confidence', 0))
                
                if raw_conf > 0:
                    raw_confidences.append(raw_conf)
                if calibrated_conf > 0:
                    calibrated_confidences.append(calibrated_conf)
                
                # Role pair combinations
                ctu1_info = rel.get('ctu1', {})
                ctu2_info = rel.get('ctu2', {})
                ctu1_role = ctu1_info.get('role', 'UNKNOWN')
                ctu2_role = ctu2_info.get('role', 'UNKNOWN')
                if ctu1_role and ctu2_role:
                    self.stats['role_pair_combinations'][(ctu1_role, ctu2_role)] += 1
            
            # Update confidence statistics
            if raw_confidences:
                self.stats['confidence_stats']['raw_min'] = min(self.stats['confidence_stats']['raw_min'], min(raw_confidences))
                self.stats['confidence_stats']['raw_max'] = max(self.stats['confidence_stats']['raw_max'], max(raw_confidences))
                
                # Raw confidence ranges
                for conf in raw_confidences:
                    if conf < 0.3:
                        self.stats['confidence_ranges_raw']['low (0-0.3)'] += 1
                    elif conf < 0.7:
                        self.stats['confidence_ranges_raw']['medium (0.3-0.7)'] += 1
                    else:
                        self.stats['confidence_ranges_raw']['high (0.7-1.0)'] += 1
            
            if calibrated_confidences:
                self.stats['confidence_stats']['calibrated_min'] = min(self.stats['confidence_stats']['calibrated_min'], min(calibrated_confidences))
                self.stats['confidence_stats']['calibrated_max'] = max(self.stats['confidence_stats']['calibrated_max'], max(calibrated_confidences))
                
                # Calibrated confidence ranges
                for conf in calibrated_confidences:
                    if conf < 0.3:
                        self.stats['confidence_ranges_calibrated']['low (0-0.3)'] += 1
                    elif conf < 0.7:
                        self.stats['confidence_ranges_calibrated']['medium (0.3-0.7)'] += 1
                    else:
                        self.stats['confidence_ranges_calibrated']['high (0.7-1.0)'] += 1
            
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
            
            # Out-degree and In-degree analysis
            out_degrees = defaultdict(int)
            in_degrees = defaultdict(int)
            
            for rel in relations:
                ctu1_info = rel.get('ctu1', {})
                ctu2_info = rel.get('ctu2', {})
                
                # Use line_idx as CTU identifier
                ctu1_key = ctu1_info.get('line_idx', '')
                ctu2_key = ctu2_info.get('line_idx', '')
                
                out_degrees[ctu1_key] += 1
                in_degrees[ctu2_key] += 1
            
            for degree in out_degrees.values():
                self.stats['out_degree_distribution'][degree] += 1
            for degree in in_degrees.values():
                self.stats['in_degree_distribution'][degree] += 1
            
        except Exception as e:
            print(f"Error processing {filepath}: {e}")
    
    def finalize_stats(self):
        """Calculate final statistics"""
        # Calculate confidence means and stds
        all_raw_confidences = []
        all_calibrated_confidences = []
        
        for filepath in os.listdir(self.data_dir):
            if filepath.endswith('.json'):
                try:
                    with open(os.path.join(self.data_dir, filepath), 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    for rel in data.get('relations', []):
                        raw_conf = rel.get('edge_confidence_raw', rel.get('confidence', 0))
                        calibrated_conf = rel.get('edge_confidence', rel.get('confidence', 0))
                        if raw_conf > 0:
                            all_raw_confidences.append(raw_conf)
                        if calibrated_conf > 0:
                            all_calibrated_confidences.append(calibrated_conf)
                except:
                    continue
        
        if all_raw_confidences:
            self.stats['confidence_stats']['raw_mean'] = np.mean(all_raw_confidences)
            self.stats['confidence_stats']['raw_std'] = np.std(all_raw_confidences)
        
        if all_calibrated_confidences:
            self.stats['confidence_stats']['calibrated_mean'] = np.mean(all_calibrated_confidences)
            self.stats['confidence_stats']['calibrated_std'] = np.std(all_calibrated_confidences)
    
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
            'confidence_analysis': {
                'raw_confidence_statistics': {
                    'min': round(self.stats['confidence_stats']['raw_min'], 4),
                    'max': round(self.stats['confidence_stats']['raw_max'], 4),
                    'mean': round(self.stats['confidence_stats']['raw_mean'], 4),
                    'std': round(self.stats['confidence_stats']['raw_std'], 4),
                    'ranges': dict(self.stats['confidence_ranges_raw'])
                },
                'calibrated_confidence_statistics': {
                    'min': round(self.stats['confidence_stats']['calibrated_min'], 4),
                    'max': round(self.stats['confidence_stats']['calibrated_max'], 4),
                    'mean': round(self.stats['confidence_stats']['calibrated_mean'], 4),
                    'std': round(self.stats['confidence_stats']['calibrated_std'], 4),
                    'ranges': dict(self.stats['confidence_ranges_calibrated'])
                }
            },
            'relation_analysis': {
                'relation_type_distribution': dict(top_relation_types),
                'method_distribution': dict(self.stats['method_distribution'])
            },
            'role_analysis': {
                'role_distribution': dict(top_roles),
                'total_unique_roles': len(self.stats['role_distribution']),
                'top_role_pairs': [(f"{pair[0]} → {pair[1]}", count) for pair, count in top_role_pairs]
            },
            'graph_structure': {
                'out_degree_distribution': dict(sorted(self.stats['out_degree_distribution'].items())),
                'in_degree_distribution': dict(sorted(self.stats['in_degree_distribution'].items())),
                'max_out_degree': max(self.stats['out_degree_distribution'].keys()) if self.stats['out_degree_distribution'] else 0,
                'max_in_degree': max(self.stats['in_degree_distribution'].keys()) if self.stats['in_degree_distribution'] else 0,
                'average_out_degree': round(sum(k * v for k, v in self.stats['out_degree_distribution'].items()) / sum(self.stats['out_degree_distribution'].values()), 2) if self.stats['out_degree_distribution'] else 0,
                'average_in_degree': round(sum(k * v for k, v in self.stats['in_degree_distribution'].items()) / sum(self.stats['in_degree_distribution'].values()), 2) if self.stats['in_degree_distribution'] else 0,
                'section_statistics': {
                    'single_section_files': self.stats['section_stats']['single_section'],
                    'multi_section_files': self.stats['section_stats']['multi_section'],
                    'max_sections_per_file': self.stats['section_stats']['max_sections'],
                    'section_distribution': dict(self.stats['section_stats']['section_distribution'])
                }
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
        print("Usage: python corrected_analysis.py <data_directory>")
        sys.exit(1)
    
    data_dir = sys.argv[1]
    if not os.path.exists(data_dir):
        print(f"Directory {data_dir} does not exist")
        sys.exit(1)
    
    print("🔍 Starting corrected analysis...")
    analyzer = CorrectedGraphAnalyzer(data_dir)
    
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
    output_file = os.path.join(os.path.dirname(data_dir), 'corrected_graph_analysis.json')
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Analysis complete! Report saved to: {output_file}")
    
    # Print summary
    print("\n" + "="*60)
    print("📊 CORRECTED GRAPH ANALYSIS SUMMARY")
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
    
    # Confidence analysis
    conf_analysis = report['confidence_analysis']
    print(f"\n🎯 CONFIDENCE ANALYSIS:")
    print(f"Raw Confidence: {conf_analysis['raw_confidence_statistics']['min']:.3f} - {conf_analysis['raw_confidence_statistics']['max']:.3f} (mean: {conf_analysis['raw_confidence_statistics']['mean']:.3f})")
    print(f"Calibrated Confidence: {conf_analysis['calibrated_confidence_statistics']['min']:.3f} - {conf_analysis['calibrated_confidence_statistics']['max']:.3f} (mean: {conf_analysis['calibrated_confidence_statistics']['mean']:.3f})")
    
    # Section analysis
    section_stats = report['graph_structure']['section_statistics']
    print(f"\n📑 SECTION ANALYSIS:")
    print(f"Single Section Files: {section_stats['single_section_files']:,} ({section_stats['single_section_files']/overview['total_files_processed']*100:.1f}%)")
    print(f"Multi-Section Files: {section_stats['multi_section_files']:,} ({section_stats['multi_section_files']/overview['total_files_processed']*100:.1f}%)")
    print(f"Max Sections per File: {section_stats['max_sections_per_file']}")
    
    # Degree analysis
    graph_structure = report['graph_structure']
    print(f"\n🔗 DEGREE ANALYSIS:")
    print(f"Max Out-Degree: {graph_structure['max_out_degree']}")
    print(f"Max In-Degree: {graph_structure['max_in_degree']}")
    print(f"Avg Out-Degree: {graph_structure['average_out_degree']:.2f}")
    print(f"Avg In-Degree: {graph_structure['average_in_degree']:.2f}")

if __name__ == "__main__":
    main()
