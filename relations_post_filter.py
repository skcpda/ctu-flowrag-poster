#!/usr/bin/env python3
"""
Relations Post-Filter Script
Applies all the identified fixes to clean up relations data
"""

import json
import re
import argparse
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Any
import math

class RelationsPostFilter:
    def __init__(self):
        # Role-pair whitelist for CONDITIONS (correct direction)
        self.conditions_whitelist = {
            ('Eligibility', 'BenefitsAssistance'),
            ('Eligibility', 'ApplicationProcess'),
            ('Documents', 'ApplicationProcess'),
            ('Eligibility', 'TimelineFrequency'),
            ('AuthoritiesGovernance', 'ApplicationProcess')
        }
        
        # Edge budget per node per relation type
        self.edge_budget = {
            'SUPPORTS': 5,
            'PRECEDES': 2,  # Only adjacent
            'CONDITIONS': 3,
            'ELABORATES': 2,
            'EXAMPLES': 2,
            'CONTRADICTS': 1,
            'CAUSES': 1
        }
        
        # Method calibration weights
        self.method_weights = {
            'gpt': 1.0,
            'rule_based': 0.7,  # Down-weight rule-based for tricky relations
            'embedding': 0.9
        }
    
    def load_relations(self, filepath: str) -> Dict:
        """Load relations from JSON file"""
        with open(filepath, 'r') as f:
            return json.load(f)
    
    def save_relations(self, data: Dict, filepath: str):
        """Save relations to JSON file"""
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    def add_missing_fields(self, data: Dict) -> Dict:
        """Add sid and line_idx fields to relations"""
        print("Adding missing sid/line_idx fields...")
        
        # Create sentence to index mapping
        sentence_to_idx = {}
        for i, rel in enumerate(data['relations']):
            sent1 = rel['ctu1']['sentence']
            sent2 = rel['ctu2']['sentence']
            
            if sent1 not in sentence_to_idx:
                sentence_to_idx[sent1] = len(sentence_to_idx)
            if sent2 not in sentence_to_idx:
                sentence_to_idx[sent2] = len(sentence_to_idx)
        
        # Add fields to each relation
        for rel in data['relations']:
            rel['ctu1']['sid'] = sentence_to_idx[rel['ctu1']['sentence']]
            rel['ctu1']['line_idx'] = sentence_to_idx[rel['ctu1']['sentence']]
            rel['ctu2']['sid'] = sentence_to_idx[rel['ctu2']['sentence']]
            rel['ctu2']['line_idx'] = sentence_to_idx[rel['ctu2']['sentence']]
        
        return data
    
    def enforce_adjacency_for_precedes(self, data: Dict) -> Dict:
        """Enforce adjacency for PRECEDES relations"""
        print("Enforcing adjacency for PRECEDES relations...")
        
        filtered_relations = []
        precedes_count = 0
        removed_precedes = 0
        
        for rel in data['relations']:
            if rel['relation'] == 'PRECEDES':
                precedes_count += 1
                # Check if truly adjacent (line_idx difference = 1)
                line_diff = abs(rel['ctu1']['line_idx'] - rel['ctu2']['line_idx'])
                if line_diff == 1:
                    filtered_relations.append(rel)
                else:
                    removed_precedes += 1
                    # Convert to ELABORATES if not adjacent
                    new_rel = rel.copy()
                    new_rel['relation'] = 'ELABORATES'
                    new_rel['confidence'] *= 0.8  # Reduce confidence
                    filtered_relations.append(new_rel)
            else:
                filtered_relations.append(rel)
        
        data['relations'] = filtered_relations
        print(f"Removed {removed_precedes}/{precedes_count} non-adjacent PRECEDES relations")
        return data
    
    def fix_conditions_direction(self, data: Dict) -> Dict:
        """Fix CONDITIONS relation directionality"""
        print("Fixing CONDITIONS direction...")
        
        filtered_relations = []
        fixed_conditions = 0
        removed_conditions = 0
        
        for rel in data['relations']:
            if rel['relation'] == 'CONDITIONS':
                role1 = rel['ctu1']['role']
                role2 = rel['ctu2']['role']
                
                # Check if direction is correct
                if (role1, role2) in self.conditions_whitelist:
                    filtered_relations.append(rel)
                elif (role2, role1) in self.conditions_whitelist:
                    # Flip direction
                    new_rel = rel.copy()
                    new_rel['ctu1'] = rel['ctu2']
                    new_rel['ctu2'] = rel['ctu1']
                    new_rel['confidence'] *= 0.9  # Slight confidence reduction
                    filtered_relations.append(new_rel)
                    fixed_conditions += 1
                else:
                    # Remove invalid CONDITIONS
                    removed_conditions += 1
            else:
                filtered_relations.append(rel)
        
        data['relations'] = filtered_relations
        print(f"Fixed {fixed_conditions} CONDITIONS directions, removed {removed_conditions} invalid ones")
        return data
    
    def add_contradiction_guardrails(self, data: Dict) -> Dict:
        """Add guardrails to reduce false positive CONTRADICTS"""
        print("Adding contradiction guardrails...")
        
        filtered_relations = []
        removed_contradicts = 0
        converted_contradicts = 0
        
        for rel in data['relations']:
            if rel['relation'] == 'CONTRADICTS':
                sent1 = rel['ctu1']['sentence'].lower()
                sent2 = rel['ctu2']['sentence'].lower()
                
                # Check for discourse markers (likely false positives)
                discourse_markers = ['however', 'but', 'although', 'though', 'despite', 'nevertheless']
                has_discourse_marker = any(marker in sent1 or marker in sent2 for marker in discourse_markers)
                
                # Check for shared key terms
                words1 = set(re.findall(r'\b\w+\b', sent1))
                words2 = set(re.findall(r'\b\w+\b', sent2))
                shared_terms = words1.intersection(words2)
                
                # Check for numeric/date conflicts
                has_numeric_conflict = bool(re.search(r'\d+', sent1) and re.search(r'\d+', sent2))
                
                if has_discourse_marker and not (len(shared_terms) >= 2 and has_numeric_conflict):
                    # Convert to CONTRASTS/EXCEPTION
                    new_rel = rel.copy()
                    new_rel['relation'] = 'ELABORATES'  # Use ELABORATES as fallback
                    new_rel['confidence'] *= 0.6
                    filtered_relations.append(new_rel)
                    converted_contradicts += 1
                elif len(shared_terms) >= 2 and has_numeric_conflict:
                    # Keep as CONTRADICTS
                    filtered_relations.append(rel)
                else:
                    # Remove weak contradictions
                    removed_contradicts += 1
            else:
                filtered_relations.append(rel)
        
        data['relations'] = filtered_relations
        print(f"Converted {converted_contradicts} false CONTRADICTS to ELABORATES, removed {removed_contradicts}")
        return data
    
    def merge_elaborates_examples(self, data: Dict) -> Dict:
        """Merge ELABORATES and EXAMPLES relations"""
        print("Merging ELABORATES and EXAMPLES...")
        
        filtered_relations = []
        merged_count = 0
        
        for rel in data['relations']:
            if rel['relation'] == 'EXAMPLES':
                # Convert EXAMPLES to ELABORATES
                new_rel = rel.copy()
                new_rel['relation'] = 'ELABORATES'
                new_rel['confidence'] *= 0.9  # Slight confidence reduction
                filtered_relations.append(new_rel)
                merged_count += 1
            else:
                filtered_relations.append(rel)
        
        data['relations'] = filtered_relations
        print(f"Merged {merged_count} EXAMPLES into ELABORATES")
        return data
    
    def apply_edge_budget(self, data: Dict) -> Dict:
        """Apply edge budget per node per relation type"""
        print("Applying edge budget per node...")
        
        # Group relations by node and type
        node_relations = defaultdict(lambda: defaultdict(list))
        
        for rel in data['relations']:
            if rel['relation'] != 'NONE':
                # Use sentence as node identifier
                node1 = rel['ctu1']['sentence']
                node2 = rel['ctu2']['sentence']
                rel_type = rel['relation']
                
                # Add confidence-weighted score
                method_weight = self.method_weights.get(rel['method'], 1.0)
                weighted_confidence = rel['confidence'] * method_weight
                
                node_relations[node1][rel_type].append((weighted_confidence, rel))
                node_relations[node2][rel_type].append((weighted_confidence, rel))
        
        # Filter relations based on budget
        filtered_relations = []
        kept_relations = set()
        
        for node, rel_types in node_relations.items():
            for rel_type, relations in rel_types.items():
                if rel_type in self.edge_budget:
                    # Sort by confidence and keep top-k
                    relations.sort(key=lambda x: x[0], reverse=True)
                    budget = self.edge_budget[rel_type]
                    
                    for _, rel in relations[:budget]:
                        rel_id = id(rel)
                        if rel_id not in kept_relations:
                            filtered_relations.append(rel)
                            kept_relations.add(rel_id)
        
        # Add back NONE relations
        for rel in data['relations']:
            if rel['relation'] == 'NONE':
                filtered_relations.append(rel)
        
        data['relations'] = filtered_relations
        print(f"Applied edge budget, kept {len(filtered_relations)} relations")
        return data
    
    def apply_method_calibration(self, data: Dict) -> Dict:
        """Apply method-aware calibration"""
        print("Applying method calibration...")
        
        for rel in data['relations']:
            if rel['relation'] != 'NONE':
                method = rel['method']
                weight = self.method_weights.get(method, 1.0)
                rel['confidence'] *= weight
                rel['calibrated_confidence'] = rel['confidence']
        
        return data
    
    def generate_quality_report(self, data: Dict) -> Dict:
        """Generate quality report after filtering"""
        print("Generating quality report...")
        
        relation_counts = Counter()
        method_counts = Counter()
        role_pairs = Counter()
        
        for rel in data['relations']:
            relation_counts[rel['relation']] += 1
            method_counts[rel['method']] += 1
            
            if rel['relation'] == 'CONDITIONS':
                role1 = rel['ctu1']['role']
                role2 = rel['ctu2']['role']
                role_pairs[f'{role1}->{role2}'] += 1
        
        total_pairs = len(data['relations'])
        non_none = sum(count for rel, count in relation_counts.items() if rel != 'NONE')
        density = (non_none / total_pairs) * 100 if total_pairs > 0 else 0
        
        report = {
            'total_pairs': total_pairs,
            'relation_distribution': dict(relation_counts),
            'method_distribution': dict(method_counts),
            'conditions_role_pairs': dict(role_pairs.most_common(10)),
            'density_percentage': density,
            'average_edges_per_node': non_none / data['total_sentences'] if data['total_sentences'] > 0 else 0
        }
        
        return report
    
    def process_file(self, input_file: str, output_file: str):
        """Process a single relations file with all fixes"""
        print(f"Processing {input_file}...")
        
        # Load data
        data = self.load_relations(input_file)
        
        # Apply all fixes
        data = self.add_missing_fields(data)
        data = self.enforce_adjacency_for_precedes(data)
        data = self.fix_conditions_direction(data)
        data = self.add_contradiction_guardrails(data)
        data = self.merge_elaborates_examples(data)
        data = self.apply_edge_budget(data)
        data = self.apply_method_calibration(data)
        
        # Generate quality report
        quality_report = self.generate_quality_report(data)
        data['quality_report'] = quality_report
        
        # Save processed data
        self.save_relations(data, output_file)
        
        print(f"Processed file saved to {output_file}")
        print(f"Quality report: {quality_report}")
        
        return data

def main():
    parser = argparse.ArgumentParser(description='Post-filter relations data')
    parser.add_argument('input_file', help='Input relations JSON file')
    parser.add_argument('output_file', help='Output filtered relations JSON file')
    
    args = parser.parse_args()
    
    filter_processor = RelationsPostFilter()
    filter_processor.process_file(args.input_file, args.output_file)

if __name__ == '__main__':
    main()
