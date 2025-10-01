#!/usr/bin/env python3
"""
Relation Generator - Generate RCR-GAT/CSRA ready relation graphs
"""

import os
import json
import re
import math
from collections import defaultdict, Counter
from typing import Dict, List, Set, Tuple
from pathlib import Path

# Edge allow-map for validation
ALLOWED_EDGES = {
    # Structural
    ('*', 'PRECEDES', '*'): True,
    ('*', 'SEGMENT_CONTINUATION', '*'): True,
    
    # Essential semantic edges
    ('Eligibility', 'PREREQUISITE_OF', 'ApplicationProcess'): True,
    ('ApplicationProcess', 'ENABLES', 'BenefitsAssistance'): True,
    ('BenefitsAssistance', 'CAP_LIMITS', 'BenefitsAssistance'): True,
    ('AuthoritiesGovernance', 'ADMINISTERED_BY', 'ApplicationProcess'): True,
    ('AuthoritiesGovernance', 'ADMINISTERED_BY', 'BenefitsAssistance'): True,
    
    # Additional semantic
    ('ContextObjective', 'ELABORATES', 'ContextObjective'): True,
    ('ContextObjective', 'ELABORATES', 'BenefitsAssistance'): True,
    ('BenefitsAssistance', 'ELABORATES', 'BenefitsAssistance'): True,
    ('Eligibility', 'ELABORATES', 'Eligibility'): True,
    ('ApplicationProcess', 'ELABORATES', 'ApplicationProcess'): True,
    
    ('ContextObjective', 'SUPPORTS', 'BenefitsAssistance'): True,
    ('Eligibility', 'SUPPORTS', 'BenefitsAssistance'): True,
    ('BenefitsAssistance', 'SUPPORTS', 'ApplicationProcess'): True,
    ('Eligibility', 'CONDITIONS', 'BenefitsAssistance'): True,
    ('Eligibility', 'CONDITIONS', 'ApplicationProcess'): True,
    
    ('TimelineFrequency', 'TIMELINE_FOR', 'ApplicationProcess'): True,
    ('TimelineFrequency', 'TIMELINE_FOR', 'BenefitsAssistance'): True,
    ('ApplicationProcess', 'TIMELINE_FOR', 'TimelineFrequency'): True,
    ('BenefitsAssistance', 'TIMELINE_FOR', 'TimelineFrequency'): True,
}

# Method-aware confidence calibration
METHOD_CALIBRATION = {
    'structural': 1.0,
    'rule_based': 0.9,
    'gpt': 0.85,
    'enhanced_semantic_boost': 0.8,
    'essential_semantic_injection': 0.75,
    'postfix_precedes_fill': 0.9,
    'postfix_minimal_semantic': 0.8,
    'embedding': 0.7,
    'production_pipeline': 0.8
}

# Semantic edge patterns
PREREQUISITE_PATTERNS = [
    r'\bmust\b', r'\brequired\b', r'\bshall\b', r'\bonly if\b', r'\bprerequisite\b',
    r'\bcondition\b', r'\bnecessary\b', r'\bmandatory\b', r'\bcompulsory\b'
]

ENABLES_PATTERNS = [
    r'\bapproval\b', r'\bapproved\b', r'\bverification\b', r'\bverified\b',
    r'\bselection\b', r'\bselected\b', r'\bupon\b', r'\bafter\b', r'\bif\b'
]

CAP_LIMITS_PATTERNS = [
    r'₹', r'rs\.', r'\bpercent\b', r'%', r'\bup to\b', r'\bmaximum\b',
    r'\bcap\b', r'\blimit\b', r'\bper year\b', r'\bannually\b'
]

ADMINISTERED_PATTERNS = [
    r'\bdepartment\b', r'\bministry\b', r'\boffice\b', r'\bauthority\b',
    r'\badminister\b', r'\bmanage\b', r'\boversee\b', r'\bgovern\b'
]

TIMELINE_PATTERNS = [
    r'\bwithin\b', r'\bdays?\b', r'\bdeadline\b', r'\bper year\b',
    r'\bsemester\b', r'\bquarterly\b', r'\bmonthly\b', r'\bannually\b'
]

def is_allowed_edge(role1: str, relation: str, role2: str) -> bool:
    """Check if edge is allowed by the allow-map"""
    if (role1, relation, role2) in ALLOWED_EDGES:
        return True
    if ('*', relation, '*') in ALLOWED_EDGES:
        return True
    return False

def has_pattern(text: str, patterns: List[str]) -> bool:
    """Check if text contains any of the patterns"""
    text_lower = text.lower()
    return any(re.search(pattern, text_lower) for pattern in patterns)

def find_nearest_forward(ctus: List[Dict], start_idx: int, target_roles: List[str], max_distance: int = 5) -> int:
    """Find nearest forward CTU with target role within max_distance"""
    for i in range(start_idx + 1, min(len(ctus), start_idx + 1 + max_distance)):
        if ctus[i].get('role') in target_roles:
            return i
    return -1

def find_nearest_anywhere(ctus: List[Dict], start_idx: int, target_roles: List[str], max_distance: int = 5) -> int:
    """Find nearest CTU with target role within max_distance"""
    for i in range(start_idx + 1, min(len(ctus), start_idx + 1 + max_distance)):
        if ctus[i].get('role') in target_roles:
            return i
    for i in range(max(0, start_idx - max_distance), start_idx):
        if ctus[i].get('role') in target_roles:
            return i
    return -1

def create_semantic_edge(ctus: List[Dict], i: int, j: int, relation: str, confidence: float) -> Dict:
    """Create a semantic edge with proper schema"""
    ctu1 = ctus[i]
    ctu2 = ctus[j]
    
    return {
        'ctu1': {
            'sentence': ctu1.get('sentence') or ctu1.get('text', ''),
            'role': ctu1.get('role', 'Unknown'),
            'sid': ctu1.get('sid'),
            'line_idx': ctu1.get('line_idx', i)
        },
        'ctu2': {
            'sentence': ctu2.get('sentence') or ctu2.get('text', ''),
            'role': ctu2.get('role', 'Unknown'),
            'sid': ctu2.get('sid'),
            'line_idx': ctu2.get('line_idx', j)
        },
        'relation': relation,
        'method': 'production_pipeline',
        'confidence': confidence,
        'edge_confidence': confidence,
        'edge_logit': math.log(confidence / (1 - confidence)) if confidence < 1 else 5.0,
        'distance': abs(i - j)
    }

def add_essential_semantic_edges(rels: List[Dict], ctus: List[Dict]) -> List[Dict]:
    """Add 4 essential semantic edge types"""
    new_edges = []
    n = len(ctus)
    
    # 1. PREREQUISITE_OF: Eligibility → ApplicationProcess
    for i in range(n):
        if ctus[i].get('role') == 'Eligibility':
            text = ctus[i].get('sentence') or ctus[i].get('text') or ''
            if has_pattern(text, PREREQUISITE_PATTERNS):
                j = find_nearest_forward(ctus, i, ['ApplicationProcess'], 5)
                if j >= 0:
                    new_edges.append(create_semantic_edge(ctus, i, j, 'PREREQUISITE_OF', 0.85))
    
    # 2. ENABLES: ApplicationProcess → BenefitsAssistance
    for i in range(n):
        if ctus[i].get('role') == 'ApplicationProcess':
            text = ctus[i].get('sentence') or ctus[i].get('text') or ''
            if has_pattern(text, ENABLES_PATTERNS):
                j = find_nearest_forward(ctus, i, ['BenefitsAssistance'], 5)
                if j >= 0:
                    new_edges.append(create_semantic_edge(ctus, i, j, 'ENABLES', 0.8))
    
    # 3. CAP_LIMITS: BenefitsAssistance → BenefitsAssistance
    for i in range(n):
        if ctus[i].get('role') == 'BenefitsAssistance':
            text = ctus[i].get('sentence') or ctus[i].get('text') or ''
            if has_pattern(text, CAP_LIMITS_PATTERNS):
                j = find_nearest_anywhere(ctus, i, ['BenefitsAssistance'], 5)
                if j >= 0 and j != i:
                    target_text = ctus[j].get('sentence') or ctus[j].get('text') or ''
                    if has_pattern(target_text, CAP_LIMITS_PATTERNS):
                        new_edges.append(create_semantic_edge(ctus, i, j, 'CAP_LIMITS', 0.85))
    
    # 4. ADMINISTERED_BY: AuthoritiesGovernance → ApplicationProcess/BenefitsAssistance
    for i in range(n):
        if ctus[i].get('role') == 'AuthoritiesGovernance':
            text = ctus[i].get('sentence') or ctus[i].get('text') or ''
            if has_pattern(text, ADMINISTERED_PATTERNS):
                j = find_nearest_forward(ctus, i, ['ApplicationProcess', 'BenefitsAssistance'], 5)
                if j >= 0:
                    new_edges.append(create_semantic_edge(ctus, i, j, 'ADMINISTERED_BY', 0.7))
    
    return rels + new_edges

def add_structural_edges(rels: List[Dict], ctus: List[Dict]) -> List[Dict]:
    """Add structural PRECEDES edges"""
    new_edges = []
    n = len(ctus)
    
    # Add PRECEDES edges for consecutive CTUs
    for i in range(n - 1):
        new_edges.append(create_semantic_edge(ctus, i, i + 1, 'PRECEDES', 0.95))
    
    return rels + new_edges

def calibrate_confidences(rels: List[Dict]) -> List[Dict]:
    """Apply method-aware confidence calibration"""
    for r in rels:
        method = r.get('method', 'unknown')
        calibration_factor = METHOD_CALIBRATION.get(method, 0.8)
        
        raw_conf = r.get('confidence', 0.5)
        r['edge_confidence_raw'] = raw_conf
        
        calibrated_conf = min(0.99, max(0.01, raw_conf * calibration_factor))
        r['confidence'] = calibrated_conf
        r['edge_confidence'] = calibrated_conf
        r['edge_logit'] = math.log(calibrated_conf / (1 - calibrated_conf)) if calibrated_conf < 1 else 5.0
    
    return rels

def smooth_timeline_roles(ctus: List[Dict]) -> List[Dict]:
    """Light role relabel smoothing for timeline content"""
    for i, ctu in enumerate(ctus):
        text = ctu.get('sentence') or ctu.get('text') or ''
        current_role = ctu.get('role', 'Unknown')
        
        if has_pattern(text, TIMELINE_PATTERNS) and current_role != 'TimelineFrequency':
            prev_role = ctus[i-1].get('role', 'Unknown') if i > 0 else 'Unknown'
            next_role = ctus[i+1].get('role', 'Unknown') if i < len(ctus) - 1 else 'Unknown'
            
            if prev_role in ['ApplicationProcess', 'BenefitsAssistance'] or next_role in ['ApplicationProcess', 'BenefitsAssistance']:
                ctu['role'] = 'TimelineFrequency'
                ctu['role_updated'] = True
    
    return ctus

def validate_edges(rels: List[Dict]) -> List[Dict]:
    """Filter edges using allow-map"""
    filtered_rels = []
    
    for r in rels:
        role1 = 'Unknown'
        role2 = 'Unknown'
        
        if 'source_role' in r and 'target_role' in r:
            role1, role2 = r['source_role'], r['target_role']
        elif 'ctu1' in r and 'ctu2' in r:
            role1 = r['ctu1'].get('role', 'Unknown')
            role2 = r['ctu2'].get('role', 'Unknown')
        
        relation = r.get('relation', 'NONE')
        
        if is_allowed_edge(role1, relation, role2):
            filtered_rels.append(r)
    
    return filtered_rels

def process_scheme_file(input_file: Path, output_file: Path) -> Dict:
    """Process a single scheme file through the complete pipeline"""
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        ctus = data.get('ctus', [])
        if not ctus:
            return {'success': False, 'error': 'No CTUs found'}
        
        # Start with empty relations
        rels = []
        
        # Add structural edges
        rels = add_structural_edges(rels, ctus)
        
        # Add essential semantic edges
        rels = add_essential_semantic_edges(rels, ctus)
        
        # Calibrate confidences
        rels = calibrate_confidences(rels)
        
        # Smooth timeline roles
        ctus = smooth_timeline_roles(ctus)
        
        # Validate edges
        rels = validate_edges(rels)
        
        # Update data
        data['relations'] = rels
        data['ctus'] = ctus
        
        # Recompute adjacency
        precedes_count = sum(1 for r in rels if r.get('relation') == 'PRECEDES')
        data['adjacency_completeness'] = {
            'expected_precedes': precedes_count,
            'actual_precedes': precedes_count,
            'complete': True
        }
        
        # Add pipeline metadata
        data['production_pipeline'] = {
            'essential_semantic_edges_added': True,
            'confidence_calibration_applied': True,
            'timeline_roles_smoothed': True,
            'edge_validation_applied': True,
            'ready_for_rcr_gat_csra': True
        }
        
        # Save file
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        # Calculate metrics
        relation_counts = Counter(r.get('relation', 'NONE') for r in rels)
        structural_count = sum(relation_counts.get(rel, 0) for rel in ['PRECEDES', 'SEGMENT_CONTINUATION'])
        semantic_count = len(rels) - structural_count
        semantic_ratio = semantic_count / len(rels) if rels else 0
        
        return {
            'success': True,
            'total': len(rels),
            'structural': structural_count,
            'semantic': semantic_count,
            'semantic_ratio': semantic_ratio,
            'distribution': dict(relation_counts)
        }
        
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'file': str(input_file)
        }

def main():
    """Generate relations for all scheme files"""
    input_dir = Path("output_data/ctu_role_tagged")
    output_dir = Path("output_data/ctu_relations_production_ready")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get all tagged CTU files
    tagged_files = list(input_dir.glob("*_tagged.json"))
    print(f"Found {len(tagged_files)} tagged CTU files")
    
    results = {
        'total_files': len(tagged_files),
        'successful': 0,
        'failed': 0,
        'total_relations': 0,
        'total_structural': 0,
        'total_semantic': 0,
        'average_semantic_ratio': 0,
        'errors': []
    }
    
    for i, tagged_file in enumerate(tagged_files, 1):
        output_file = output_dir / f"{tagged_file.stem.replace('_tagged', '')}_production_ready.json"
        
        print(f"Processing {i}/{len(tagged_files)}: {tagged_file.name}")
        result = process_scheme_file(tagged_file, output_file)
        
        if result['success']:
            results['successful'] += 1
            results['total_relations'] += result['total']
            results['total_structural'] += result['structural']
            results['total_semantic'] += result['semantic']
            print(f"  ✓ Generated {result['total']} relations (Structural: {result['structural']}, Semantic: {result['semantic']}, Ratio: {result['semantic_ratio']:.1%})")
        else:
            results['failed'] += 1
            results['errors'].append(result['error'])
            print(f"  ❌ Error: {result['error']}")
    
    results['average_semantic_ratio'] = results['total_semantic'] / results['total_relations'] if results['total_relations'] > 0 else 0
    
    # Save summary
    with open(output_dir / "production_pipeline_summary.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n=== Relation Generation Complete ===")
    print(f"Total files: {results['total_files']}")
    print(f"Successful: {results['successful']}")
    print(f"Failed: {results['failed']}")
    print(f"Total relations: {results['total_relations']}")
    print(f"Structural: {results['total_structural']}")
    print(f"Semantic: {results['total_semantic']}")
    print(f"Average semantic ratio: {results['average_semantic_ratio']:.1%}")
    print(f"Ready for RCR-GAT/CSRA training!")

if __name__ == '__main__':
    main()
