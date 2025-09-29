#!/usr/bin/env python3
"""
Batch filter all relations files
"""

import os
import json
from relations_post_filter import RelationsPostFilter

def main():
    input_dir = "organized_output/outputs/ctu_relations"
    output_dir = "organized_output/outputs/ctu_relations_filtered"
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all relations files
    relations_files = [f for f in os.listdir(input_dir) if f.endswith('_relations.json') and f != 'summary.json']
    
    print(f"Found {len(relations_files)} relations files to process")
    
    filter_processor = RelationsPostFilter()
    all_reports = {}
    
    for filename in relations_files:
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename.replace('.json', '_filtered.json'))
        
        print(f"\n{'='*60}")
        print(f"Processing {filename}")
        print(f"{'='*60}")
        
        try:
            data = filter_processor.process_file(input_path, output_path)
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
            'average_edges_per_node': sum(report['average_edges_per_node'] for report in all_reports.values()) / len(all_reports)
        }
    }
    
    # Save summary
    with open(os.path.join(output_dir, 'filtered_summary.json'), 'w') as f:
        json.dump(summary_report, f, indent=2)
    
    print(f"\n{'='*60}")
    print("BATCH PROCESSING COMPLETE")
    print(f"{'='*60}")
    print(f"Processed {len(all_reports)} files")
    print(f"Average density: {summary_report['overall_stats']['average_density']:.1f}%")
    print(f"Average edges per node: {summary_report['overall_stats']['average_edges_per_node']:.1f}")
    print(f"Results saved to: {output_dir}")

if __name__ == '__main__':
    main()
