#!/usr/bin/env python3
"""
Batch quality fix for all relations files
"""

import os
import json
from relations_quality_fixer import RelationsQualityFixer

def main():
    input_dir = "organized_output/outputs/ctu_relations_filtered"
    output_dir = "organized_output/outputs/ctu_relations_quality_fixed"
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all filtered relations files
    relations_files = [f for f in os.listdir(input_dir) if f.endswith('_relations_filtered.json')]
    
    print(f"Found {len(relations_files)} relations files to quality fix")
    
    fixer = RelationsQualityFixer()
    all_reports = {}
    
    for filename in relations_files:
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename.replace('_filtered.json', '_quality_fixed.json'))
        
        print(f"\n{'='*60}")
        print(f"Quality fixing {filename}")
        print(f"{'='*60}")
        
        try:
            data = fixer.process_file(input_path, output_path)
            all_reports[filename] = data['quality_report']
        except Exception as e:
            print(f"Error processing {filename}: {e}")
            continue
    
    # Generate summary report
    summary_report = {
        'total_files_processed': len(all_reports),
        'files': all_reports,
        'overall_stats': {
            'total_relations': sum(report['total_pairs'] for report in all_reports.values()),
            'average_density': sum(report['density_percentage'] for report in all_reports.values()) / len(all_reports),
            'average_edges_per_node': sum(report['average_edges_per_node'] for report in all_reports.values()) / len(all_reports),
            'average_structural_edges': sum(report['structural_edges'] for report in all_reports.values()) / len(all_reports)
        }
    }
    
    # Save summary
    with open(os.path.join(output_dir, 'quality_fix_summary.json'), 'w') as f:
        json.dump(summary_report, f, indent=2)
    
    print(f"\n{'='*60}")
    print("QUALITY FIXING COMPLETE")
    print(f"{'='*60}")
    print(f"Processed {len(all_reports)} files")
    print(f"Average density: {summary_report['overall_stats']['average_density']:.1f}%")
    print(f"Average edges per node: {summary_report['overall_stats']['average_edges_per_node']:.1f}")
    print(f"Average structural edges: {summary_report['overall_stats']['average_structural_edges']:.1f}")
    print(f"Results saved to: {output_dir}")

if __name__ == '__main__':
    main()
