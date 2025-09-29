#!/usr/bin/env python3
"""
Relations Quality Fixer - Comprehensive post-processor
Fixes all identified issues: structural edges, role-pair gating, distance priors, confidence calibration
"""

import json
import re
import os
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Any
import numpy as np

class RelationsQualityFixer:
    def __init__(self):
        # Role-pair whitelist for CONDITIONS (strict)
        self.conditions_whitelist = {
            ('Eligibility', 'BenefitsAssistance'),
            ('Eligibility', 'ApplicationProcess'),
            ('Eligibility', 'TimelineFrequency'),
            ('ApplicationProcess', 'BenefitsAssistance'),
            ('DefinitionsReferences', 'ApplicationProcess'),
            ('DefinitionsReferences', 'BenefitsAssistance')
        }
        
        # Role-pair whitelist for CAUSES (very restrictive)
        self.causes_whitelist = {
            ('ApplicationProcess', 'BenefitsAssistance'),
            ('TimelineFrequency', 'BenefitsAssistance')
        }
        
        # Role-pair whitelist for ADMINISTERED_BY
        self.administered_by_whitelist = {
            ('ContextObjective', 'AuthoritiesGovernance'),
            ('BenefitsAssistance', 'AuthoritiesGovernance'),
            ('ApplicationProcess', 'AuthoritiesGovernance')
        }
        
        # Confidence thresholds
        self.confidence_thresholds = {
            'CONDITIONS': 0.7,
            'CAUSES': 0.8,
            'CONTRADICTS': 0.75,
            'SUPPORTS': 0.6,
            'ELABORATES': 0.5,
            'PRECEDES': 0.4,  # Lower threshold for structural edges
            'ADMINISTERED_BY': 0.6
        }
        
        # Distance decay parameters
        self.max_distance = 10  # Beyond this, apply decay
        self.distance_decay = 0.1  # Decay factor per distance unit
        
        # Structural edge parameters
        self.structural_confidence = 0.5
        self.segment_continuation_threshold = 0.3
    
    def load_relations(self, filepath: str) -> Dict:
        """Load relations from JSON file"""
        with open(filepath, 'r') as f:
            return json.load(f)
    
    def save_relations(self, data: Dict, filepath: str):
        """Save relations to JSON file"""
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    def is_structural_continuation(self, sentence1: str, sentence2: str, role1: str, role2: str) -> bool:
        """Check if sentences form a structural continuation"""
        # Check for pronoun/connector continuation
        continuation_markers = [
            'this', 'that', 'these', 'those', 'it', 'they', 'such', 'the above',
            'furthermore', 'additionally', 'moreover', 'also', 'in addition',
            'specifically', 'in particular', 'for example', 'for instance'
        ]
        
        sentence2_lower = sentence2.lower()
        has_continuation_marker = any(marker in sentence2_lower for marker in continuation_markers)
        
        # Check for same role continuation
        same_role = role1 == role2
        
        # Check for section continuation (simple heuristic)
        section_continuation = any(word in sentence2_lower for word in ['section', 'paragraph', 'clause', 'article'])
        
        return has_continuation_marker or (same_role and section_continuation)
    
    def calculate_distance_penalty(self, sid1: int, sid2: int) -> float:
        """Calculate distance penalty for edge"""
        distance = abs(sid1 - sid2)
        if distance <= 1:
            return 1.0  # No penalty for adjacent
        elif distance <= self.max_distance:
            return 1.0 - (distance - 1) * self.distance_decay
        else:
            return max(0.1, 1.0 - (distance - 1) * self.distance_decay)
    
    def should_keep_edge(self, relation: str, confidence: float, distance_penalty: float, 
                        role1: str, role2: str, method: str) -> Tuple[bool, str, float]:
        """Determine if edge should be kept and what relation it should have"""
        
        # Apply distance penalty to confidence
        adjusted_confidence = confidence * distance_penalty
        
        # Check confidence thresholds
        threshold = self.confidence_thresholds.get(relation, 0.6)
        if adjusted_confidence < threshold:
            return False, "NONE", 0.0
        
        # Role-pair validation
        if relation == "CONDITIONS":
            if (role1, role2) not in self.conditions_whitelist:
                # Demote to SUPPORTS or ELABORATES
                if adjusted_confidence > 0.5:
                    return True, "SUPPORTS", adjusted_confidence * 0.8
                else:
                    return False, "NONE", 0.0
        
        elif relation == "CAUSES":
            if (role1, role2) not in self.causes_whitelist:
                # Demote to SUPPORTS or ELABORATES
                if adjusted_confidence > 0.5:
                    return True, "SUPPORTS", adjusted_confidence * 0.8
                else:
                    return False, "NONE", 0.0
        
        # Check for ADMINISTERED_BY opportunities
        if (role1, role2) in self.administered_by_whitelist and relation in ["CAUSES", "SUPPORTS"]:
            if adjusted_confidence > 0.6:
                return True, "ADMINISTERED_BY", adjusted_confidence
        
        return True, relation, adjusted_confidence
    
    def add_structural_edges(self, data: Dict) -> Dict:
        """Add missing structural edges (PRECEDES, SEGMENT_CONTINUATION)"""
        print("Adding structural edges...")
        
        relations = data['relations']
        sentences = []
        
        # Extract sentences with their metadata
        for rel in relations:
            ctu1 = rel['ctu1']
            ctu2 = rel['ctu2']
            if ctu1 not in sentences:
                sentences.append(ctu1)
            if ctu2 not in sentences:
                sentences.append(ctu2)
        
        # Sort by sid
        sentences.sort(key=lambda x: x.get('sid', 0))
        
        new_relations = []
        added_structural = 0
        
        # Add PRECEDES for adjacent sentences
        for i in range(len(sentences) - 1):
            sent1 = sentences[i]
            sent2 = sentences[i + 1]
            
            # Check if PRECEDES already exists
            existing_precedes = any(
                rel['relation'] == 'PRECEDES' and 
                rel['ctu1']['sid'] == sent1.get('sid', i) and 
                rel['ctu2']['sid'] == sent2.get('sid', i + 1)
                for rel in relations
            )
            
            if not existing_precedes:
                new_relations.append({
                    'ctu1': sent1,
                    'ctu2': sent2,
                    'relation': 'PRECEDES',
                    'method': 'structural',
                    'confidence': self.structural_confidence
                })
                added_structural += 1
        
        # Add SEGMENT_CONTINUATION for structural continuations
        for i in range(len(sentences)):
            for j in range(i + 1, min(i + 5, len(sentences))):  # Look ahead up to 5 sentences
                sent1 = sentences[i]
                sent2 = sentences[j]
                
                if self.is_structural_continuation(sent1['sentence'], sent2['sentence'], 
                                                sent1['role'], sent2['role']):
                    # Check if already exists
                    existing_continuation = any(
                        rel['relation'] in ['SEGMENT_CONTINUATION', 'ELABORATES'] and 
                        rel['ctu1']['sid'] == sent1.get('sid', i) and 
                        rel['ctu2']['sid'] == sent2.get('sid', j)
                        for rel in relations
                    )
                    
                    if not existing_continuation:
                        new_relations.append({
                            'ctu1': sent1,
                            'ctu2': sent2,
                            'relation': 'SEGMENT_CONTINUATION',
                            'method': 'structural',
                            'confidence': self.segment_continuation_threshold
                        })
                        added_structural += 1
        
        # Add new relations to existing ones
        data['relations'].extend(new_relations)
        print(f"Added {added_structural} structural edges")
        
        return data
    
    def fix_role_drift(self, data: Dict) -> Dict:
        """Fix role drift issues"""
        print("Fixing role drift...")
        
        fixed_roles = 0
        
        for rel in data['relations']:
            ctu1 = rel['ctu1']
            ctu2 = rel['ctu2']
            
            # Fix impact/outcome sentences labeled as Eligibility
            for ctu in [ctu1, ctu2]:
                sentence = ctu['sentence'].lower()
                if ctu['role'] == 'Eligibility' and any(word in sentence for word in 
                    ['impact', 'outcome', 'uptake', 'benefit', 'result', 'achievement']):
                    ctu['role'] = 'ContextObjective'
                    fixed_roles += 1
            
            # Fix first sentences labeled as Eligibility
            for ctu in [ctu1, ctu2]:
                sentence = ctu['sentence'].lower()
                if (ctu['role'] == 'Eligibility' and 
                    any(word in sentence for word in ['scheme', 'program', 'initiative', 'launched', 'introduced'])):
                    ctu['role'] = 'ContextObjective'
                    fixed_roles += 1
        
        print(f"Fixed {fixed_roles} role drift issues")
        return data
    
    def apply_quality_fixes(self, data: Dict) -> Dict:
        """Apply all quality fixes"""
        print("Applying quality fixes...")
        
        # Fix role drift first
        data = self.fix_role_drift(data)
        
        # Add structural edges
        data = self.add_structural_edges(data)
        
        # Process existing relations
        filtered_relations = []
        fixed_relations = 0
        demoted_relations = 0
        
        for rel in data['relations']:
            if rel['relation'] == 'NONE':
                filtered_relations.append(rel)
                continue
            
            ctu1 = rel['ctu1']
            ctu2 = rel['ctu2']
            
            # Calculate distance penalty
            sid1 = ctu1.get('sid', 0)
            sid2 = ctu2.get('sid', 0)
            distance_penalty = self.calculate_distance_penalty(sid1, sid2)
            
            # Determine if edge should be kept
            should_keep, new_relation, new_confidence = self.should_keep_edge(
                rel['relation'], rel['confidence'], distance_penalty,
                ctu1['role'], ctu2['role'], rel['method']
            )
            
            if should_keep:
                rel['relation'] = new_relation
                rel['confidence'] = new_confidence
                filtered_relations.append(rel)
                
                if new_relation != rel['relation']:
                    fixed_relations += 1
            else:
                # Convert to NONE
                rel['relation'] = 'NONE'
                rel['confidence'] = 0.0
                filtered_relations.append(rel)
                demoted_relations += 1
        
        data['relations'] = filtered_relations
        
        print(f"Fixed {fixed_relations} relations, demoted {demoted_relations} to NONE")
        return data
    
    def generate_quality_report(self, data: Dict) -> Dict:
        """Generate quality report after fixes"""
        print("Generating quality report...")
        
        relation_counts = Counter()
        method_counts = Counter()
        role_pairs = Counter()
        distance_stats = defaultdict(list)
        
        for rel in data['relations']:
            relation_counts[rel['relation']] += 1
            method_counts[rel['method']] += 1
            
            if rel['relation'] == 'CONDITIONS':
                role1 = rel['ctu1']['role']
                role2 = rel['ctu2']['role']
                role_pairs[f'{role1}->{role2}'] += 1
            
            # Calculate distance
            sid1 = rel['ctu1'].get('sid', 0)
            sid2 = rel['ctu2'].get('sid', 0)
            distance = abs(sid1 - sid2)
            distance_stats[rel['relation']].append(distance)
        
        total_pairs = len(data['relations'])
        non_none = sum(count for rel, count in relation_counts.items() if rel != 'NONE')
        density = (non_none / total_pairs) * 100 if total_pairs > 0 else 0
        
        # Calculate average distances
        avg_distances = {}
        for relation, distances in distance_stats.items():
            if distances:
                avg_distances[relation] = sum(distances) / len(distances)
        
        report = {
            'total_pairs': total_pairs,
            'relation_distribution': dict(relation_counts),
            'method_distribution': dict(method_counts),
            'conditions_role_pairs': dict(role_pairs.most_common(10)),
            'density_percentage': density,
            'average_edges_per_node': non_none / data['total_sentences'] if data['total_sentences'] > 0 else 0,
            'average_distances': avg_distances,
            'structural_edges': relation_counts.get('PRECEDES', 0) + relation_counts.get('SEGMENT_CONTINUATION', 0)
        }
        
        return report
    
    def process_file(self, input_file: str, output_file: str):
        """Process a single relations file with all quality fixes"""
        print(f"Processing {input_file}...")
        
        # Load data
        data = self.load_relations(input_file)
        
        # Apply all quality fixes
        data = self.apply_quality_fixes(data)
        
        # Generate quality report
        quality_report = self.generate_quality_report(data)
        data['quality_report'] = quality_report
        
        # Save processed data
        self.save_relations(data, output_file)
        
        print(f"Processed file saved to {output_file}")
        print(f"Quality report: {quality_report}")
        
        return data

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Fix relations quality issues')
    parser.add_argument('input_file', help='Input relations JSON file')
    parser.add_argument('output_file', help='Output fixed relations JSON file')
    
    args = parser.parse_args()
    
    fixer = RelationsQualityFixer()
    fixer.process_file(args.input_file, args.output_file)

if __name__ == '__main__':
    main()
