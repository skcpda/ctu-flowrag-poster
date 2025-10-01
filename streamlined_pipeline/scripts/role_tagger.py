#!/usr/bin/env python3
"""
Role Tagger - Assign roles to CTUs using fine-tuned BGE model
"""

import os
import json
import re
import torch
from sentence_transformers import SentenceTransformer
from typing import List, Dict
from pathlib import Path
import numpy as np

# Role definitions
ROLES = [
    'ContextObjective',
    'Eligibility', 
    'BenefitsAssistance',
    'ApplicationProcess',
    'AuthoritiesGovernance',
    'TimelineFrequency',
    'DefinitionsReferences'
]

# Role patterns for rule-based tagging
ROLE_PATTERNS = {
    'ContextObjective': [
        r'\bobjective\b', r'\baim\b', r'\bgoal\b', r'\bpurpose\b', r'\btarget\b',
        r'\binitiative\b', r'\bprogram\b', r'\bscheme\b', r'\bpolicy\b'
    ],
    'Eligibility': [
        r'\beligible\b', r'\beligibility\b', r'\bqualify\b', r'\bqualification\b',
        r'\bincome\b', r'\blandholding\b', r'\bresident\b', r'\brequired\b',
        r'\bcriteria\b', r'\bcondition\b', r'\bmust\b', r'\bshall\b'
    ],
    'BenefitsAssistance': [
        r'\bbenefit\b', r'\bassistance\b', r'\bfinancial\b', r'\bgrant\b',
        r'\bsubsidy\b', r'\bamount\b', r'\bmoney\b', r'\b₹\b', r'\bINR\b',
        r'\bdisbursement\b', r'\bpayment\b', r'\btransfer\b'
    ],
    'ApplicationProcess': [
        r'\bapplication\b', r'\bapply\b', r'\bprocess\b', r'\bprocedure\b',
        r'\bform\b', r'\bsubmit\b', r'\bregister\b', r'\bonline\b',
        r'\bportal\b', r'\bwebsite\b', r'\bdocument\b'
    ],
    'AuthoritiesGovernance': [
        r'\bgovernment\b', r'\bministry\b', r'\bdepartment\b', r'\boffice\b',
        r'\bauthority\b', r'\bministry\b', r'\bgovernance\b', r'\badminister\b',
        r'\bmanage\b', r'\boversee\b'
    ],
    'TimelineFrequency': [
        r'\bday\b', r'\bmonth\b', r'\byear\b', r'\bdeadline\b', r'\btimeframe\b',
        r'\bwithin\b', r'\binstallment\b', r'\bquarterly\b', r'\bannually\b',
        r'\bperiod\b', r'\bduration\b'
    ],
    'DefinitionsReferences': [
        r'\bdefinition\b', r'\bmeaning\b', r'\bexplain\b', r'\bclarify\b',
        r'\breference\b', r'\bcontact\b', r'\bwebsite\b', r'\bportal\b'
    ]
}

class RoleTagger:
    def __init__(self, model_path: str = "config/fine_tuned_bge_ctu_relations"):
        """Initialize the role tagger"""
        self.model_path = model_path
        self.model = None
        self.role_embeddings = None
        self.load_model()
        self.prepare_role_embeddings()
    
    def load_model(self):
        """Load the fine-tuned BGE model"""
        try:
            self.model = SentenceTransformer(self.model_path)
            print(f"✓ Loaded model from {self.model_path}")
        except Exception as e:
            print(f"⚠️  Could not load fine-tuned model: {e}")
            print("Using base BGE model as fallback")
            self.model = SentenceTransformer('BAAI/bge-large-en-v1.5')
    
    def prepare_role_embeddings(self):
        """Prepare role embeddings for similarity matching"""
        role_descriptions = {
            'ContextObjective': 'Scheme objectives, goals, aims, and purposes',
            'Eligibility': 'Eligibility criteria, requirements, qualifications',
            'BenefitsAssistance': 'Financial benefits, assistance, grants, subsidies',
            'ApplicationProcess': 'Application procedures, forms, submission process',
            'AuthoritiesGovernance': 'Government departments, authorities, governance',
            'TimelineFrequency': 'Time periods, deadlines, frequencies, schedules',
            'DefinitionsReferences': 'Definitions, explanations, contact information'
        }
        
        role_texts = [role_descriptions[role] for role in ROLES]
        self.role_embeddings = self.model.encode(role_texts)
    
    def tag_with_rules(self, sentence: str) -> str:
        """Tag using rule-based patterns"""
        sentence_lower = sentence.lower()
        role_scores = {}
        
        for role, patterns in ROLE_PATTERNS.items():
            score = 0
            for pattern in patterns:
                if re.search(pattern, sentence_lower):
                    score += 1
            role_scores[role] = score
        
        # Return role with highest score, or Unknown if no matches
        if role_scores:
            best_role = max(role_scores, key=role_scores.get)
            if role_scores[best_role] > 0:
                return best_role
        
        return 'Unknown'
    
    def tag_with_embeddings(self, sentence: str) -> str:
        """Tag using embedding similarity"""
        try:
            sentence_embedding = self.model.encode([sentence])
            similarities = np.dot(sentence_embedding, self.role_embeddings.T)[0]
            best_role_idx = np.argmax(similarities)
            confidence = similarities[best_role_idx]
            
            if confidence > 0.3:  # Threshold for confidence
                return ROLES[best_role_idx]
            else:
                return 'Unknown'
        except Exception as e:
            print(f"Embedding tagging failed: {e}")
            return 'Unknown'
    
    def tag_sentence(self, sentence: str) -> Dict:
        """Tag a sentence with role using hybrid approach"""
        # Try rule-based first
        rule_role = self.tag_with_rules(sentence)
        
        # Try embedding-based
        embedding_role = self.tag_with_embeddings(sentence)
        
        # Choose the best result
        if rule_role != 'Unknown':
            return {
                'role': rule_role,
                'method': 'rule_based',
                'confidence': 0.8
            }
        elif embedding_role != 'Unknown':
            return {
                'role': embedding_role,
                'method': 'embedding',
                'confidence': 0.7
            }
        else:
            return {
                'role': 'ContextObjective',  # Default fallback
                'method': 'fallback',
                'confidence': 0.5
            }
    
    def tag_ctus(self, ctus: List[Dict]) -> List[Dict]:
        """Tag all CTUs in a scheme"""
        for ctu in ctus:
            sentence = ctu.get('sentence', '')
            tag_result = self.tag_sentence(sentence)
            
            ctu['role'] = tag_result['role']
            ctu['confidence'] = tag_result['confidence']
            ctu['method'] = tag_result['method']
        
        return ctus

def process_scheme_file(input_file: Path, output_file: Path, tagger: RoleTagger) -> Dict:
    """Process a single scheme file with role tagging"""
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        ctus = data.get('ctus', [])
        if not ctus:
            return {'success': False, 'error': 'No CTUs found'}
        
        # Tag CTUs
        tagged_ctus = tagger.tag_ctus(ctus)
        
        # Update data
        data['ctus'] = tagged_ctus
        data['role_tagging'] = {
            'applied': True,
            'total_ctus': len(tagged_ctus),
            'role_distribution': {}
        }
        
        # Calculate role distribution
        role_counts = {}
        for ctu in tagged_ctus:
            role = ctu.get('role', 'Unknown')
            role_counts[role] = role_counts.get(role, 0) + 1
        data['role_tagging']['role_distribution'] = role_counts
        
        # Save output
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        return {
            'success': True,
            'ctus_count': len(tagged_ctus),
            'scheme_name': data.get('scheme_name', 'Unknown')
        }
        
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'file': str(input_file)
        }

def main():
    """Tag roles for all CTU files"""
    input_dir = Path("output_data/ctu_extracted")
    output_dir = Path("output_data/ctu_role_tagged")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize tagger
    tagger = RoleTagger()
    
    # Get all CTU files
    ctu_files = list(input_dir.glob("*_ctus.json"))
    print(f"Found {len(ctu_files)} CTU files")
    
    results = {
        'total_files': len(ctu_files),
        'successful': 0,
        'failed': 0,
        'total_ctus': 0,
        'role_distribution': {},
        'errors': []
    }
    
    for i, ctu_file in enumerate(ctu_files, 1):
        output_file = output_dir / f"{ctu_file.stem}_tagged.json"
        
        print(f"Processing {i}/{len(ctu_files)}: {ctu_file.name}")
        result = process_scheme_file(ctu_file, output_file, tagger)
        
        if result['success']:
            results['successful'] += 1
            results['total_ctus'] += result['ctus_count']
            print(f"  ✓ Tagged {result['ctus_count']} CTUs")
        else:
            results['failed'] += 1
            results['errors'].append(result['error'])
            print(f"  ❌ Error: {result['error']}")
    
    # Save summary
    with open(output_dir / "role_tagging_summary.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n=== Role Tagging Complete ===")
    print(f"Total files: {results['total_files']}")
    print(f"Successful: {results['successful']}")
    print(f"Failed: {results['failed']}")
    print(f"Total CTUs: {results['total_ctus']}")

if __name__ == '__main__':
    main()
