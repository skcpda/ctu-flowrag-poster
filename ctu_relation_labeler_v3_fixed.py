#!/usr/bin/env python3
"""
CTU Relation Labeler V3 - Fixed with Upstream Improvements
Implements all the identified fixes at the source to prevent issues:
- Adjacency enforcement for PRECEDES
- Directional constraints for CONDITIONS  
- Contradiction guardrails
- Role-pair validation
- Method-aware calibration
- Edge budget system
"""

import os
import json
import glob
import random
import numpy as np
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.isotonic import IsotonicRegression
from typing import List, Dict, Any, Tuple, Set
import logging
from datetime import datetime
from collections import defaultdict, Counter
import re

# Setup logging
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = os.path.join(log_dir, f"ctu_relation_labeler_v3_fixed_{timestamp}.log")
logging.basicConfig(filename=log_file, level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Enhanced CTU Relation Types with proper constraints
RELATION_TYPES = {
    "PRECEDES": {
        "description": "One CTU comes before another in logical sequence (must be adjacent)",
        "max_edges": 2,
        "requires_adjacency": True
    },
    "CAUSES": {
        "description": "One CTU causes or leads to another",
        "max_edges": 1,
        "requires_adjacency": False
    },
    "CONTRADICTS": {
        "description": "One CTU contradicts or conflicts with another (requires shared terms + numeric conflict)",
        "max_edges": 1,
        "requires_adjacency": False,
        "requires_shared_terms": True,
        "requires_numeric_conflict": True
    },
    "SUPPORTS": {
        "description": "One CTU supports or reinforces another",
        "max_edges": 5,
        "requires_adjacency": False
    },
    "ELABORATES": {
        "description": "One CTU provides more detail about another (includes examples)",
        "max_edges": 2,
        "requires_adjacency": False
    },
    "CONDITIONS": {
        "description": "One CTU sets conditions for another (must follow prerequisite direction)",
        "max_edges": 3,
        "requires_adjacency": False,
        "requires_directional_constraint": True
    },
    "NONE": {
        "description": "No meaningful relation between CTUs",
        "max_edges": 0
    }
}

# Role-pair whitelist for CONDITIONS (correct direction)
CONDITIONS_ROLE_WHITELIST = {
    ('Eligibility', 'BenefitsAssistance'),
    ('Eligibility', 'ApplicationProcess'),
    ('Documents', 'ApplicationProcess'),
    ('Eligibility', 'TimelineFrequency'),
    ('AuthoritiesGovernance', 'ApplicationProcess')
}

# Role compatibility masks for other relations
ROLE_COMPATIBILITY = {
    "SUPPORTS": [
        ("ContextObjective", "BenefitsAssistance"),
        ("Eligibility", "BenefitsAssistance"),
        ("ApplicationProcess", "BenefitsAssistance"),
        ("BenefitsAssistance", "BenefitsAssistance")
    ],
    "ELABORATES": [
        ("ContextObjective", "ContextObjective"),
        ("BenefitsAssistance", "BenefitsAssistance"),
        ("Eligibility", "Eligibility"),
        ("ContextObjective", "BenefitsAssistance")
    ],
    "PRECEDES": [
        ("ContextObjective", "BenefitsAssistance"),
        ("Eligibility", "ApplicationProcess"),
        ("ApplicationProcess", "TimelineFrequency"),
        ("ContextObjective", "ApplicationProcess")
    ]
}

# Method calibration weights
METHOD_CALIBRATION = {
    'gpt': 1.0,
    'rule_based': 0.7,  # Down-weight for tricky relations
    'embedding': 0.9
}

# Relation-specific method weights
RELATION_METHOD_WEIGHTS = {
    'CONTRADICTS': {'gpt': 1.0, 'rule_based': 0.3, 'embedding': 0.7},
    'PRECEDES': {'gpt': 1.0, 'rule_based': 0.8, 'embedding': 0.9},
    'CONDITIONS': {'gpt': 1.0, 'rule_based': 0.6, 'embedding': 0.8},
    'SUPPORTS': {'gpt': 1.0, 'rule_based': 0.9, 'embedding': 0.95},
    'ELABORATES': {'gpt': 1.0, 'rule_based': 0.8, 'embedding': 0.9}
}

class CTURelationLabelerV3Fixed:
    def __init__(self, model_name: str = "fine_tuned_bge_ctu_relations/", fine_tune: bool = True):
        """
        Initialize the fixed CTU relation labeler with upstream improvements
        """
        print(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        
        # Create relation embeddings
        self.relation_embeddings = {}
        for relation, config in RELATION_TYPES.items():
            if relation != "NONE":
                embedding = self.model.encode(config["description"])
                self.relation_embeddings[relation] = embedding
        
        print(f"Loaded {len(self.relation_embeddings)} relation embeddings")
        
        # Fine-tuning data storage
        self.fine_tune_data = []
        self.calibration_data = []
        self.fine_tune = fine_tune
        
        # Confidence calibration
        self.calibration_model = None
        self.calibration_threshold = 0.85
    
    def add_missing_fields(self, labeled_sentences: List[Dict]) -> List[Dict]:
        """Add sid and line_idx fields to sentences"""
        for i, sentence in enumerate(labeled_sentences):
            sentence['sid'] = i
            sentence['line_idx'] = i
        return labeled_sentences
    
    def check_adjacency(self, ctu1: Dict, ctu2: Dict) -> bool:
        """Check if two CTUs are adjacent (line_idx difference = 1)"""
        return abs(ctu1.get('line_idx', 0) - ctu2.get('line_idx', 0)) == 1
    
    def check_shared_terms(self, sentence1: str, sentence2: str, min_terms: int = 2) -> bool:
        """Check if sentences share enough key terms"""
        words1 = set(re.findall(r'\b\w+\b', sentence1.lower()))
        words2 = set(re.findall(r'\b\w+\b', sentence2.lower()))
        shared_terms = words1.intersection(words2)
        return len(shared_terms) >= min_terms
    
    def check_numeric_conflict(self, sentence1: str, sentence2: str) -> bool:
        """Check if sentences have numeric/date conflicts"""
        # Look for numbers, dates, and negations
        has_numeric1 = bool(re.search(r'\d+', sentence1))
        has_numeric2 = bool(re.search(r'\d+', sentence2))
        has_negation1 = bool(re.search(r'\b(not|no|never|none|nothing)\b', sentence1.lower()))
        has_negation2 = bool(re.search(r'\b(not|no|never|none|nothing)\b', sentence2.lower()))
        
        return (has_numeric1 and has_numeric2) or (has_negation1 and has_negation2)
    
    def check_discourse_markers(self, sentence1: str, sentence2: str) -> bool:
        """Check for discourse markers that might indicate false contradictions"""
        discourse_markers = ['however', 'but', 'although', 'though', 'despite', 'nevertheless', 'on the other hand']
        combined_text = (sentence1 + ' ' + sentence2).lower()
        return any(marker in combined_text for marker in discourse_markers)
    
    def validate_conditions_direction(self, ctu1: Dict, ctu2: Dict) -> bool:
        """Validate CONDITIONS relation direction"""
        role1 = ctu1['role']
        role2 = ctu2['role']
        return (role1, role2) in CONDITIONS_ROLE_WHITELIST
    
    def validate_role_compatibility(self, role1: str, role2: str, relation: str) -> bool:
        """Validate role compatibility for relation type"""
        if relation == "CONDITIONS":
            return self.validate_conditions_direction({'role': role1}, {'role': role2})
        
        valid_pairs = ROLE_COMPATIBILITY.get(relation, [])
        return (role1, role2) in valid_pairs or (role2, role1) in valid_pairs
    
    def enhanced_rule_based_classification(self, ctu1: Dict, ctu2: Dict, idx1: int, idx2: int) -> str:
        """Enhanced rule-based classification with all constraints"""
        sentence1 = ctu1['sentence'].lower()
        sentence2 = ctu2['sentence'].lower()
        
        # Check for PRECEDES (must be adjacent)
        if self.check_adjacency(ctu1, ctu2):
            if any(word in sentence1 for word in ['first', 'initially', 'begin', 'start']) and \
               any(word in sentence2 for word in ['then', 'next', 'after', 'follow', 'subsequent']):
                return "PRECEDES"
        
        # Check for CONTRADICTS (with guardrails)
        if self.check_shared_terms(sentence1, sentence2) and self.check_numeric_conflict(sentence1, sentence2):
            # Avoid discourse marker false positives
            if not self.check_discourse_markers(sentence1, sentence2):
                return "CONTRADICTS"
        
        # Check for CONDITIONS (with direction validation)
        if self.validate_conditions_direction(ctu1, ctu2):
            if any(word in sentence1 for word in ['require', 'must', 'need', 'condition', 'eligibility']) and \
               any(word in sentence2 for word in ['benefit', 'assistance', 'support', 'application']):
                return "CONDITIONS"
        
        # Check for SUPPORTS
        if any(word in sentence1 for word in ['support', 'help', 'assist', 'enable', 'facilitate']) and \
           any(word in sentence2 for word in ['benefit', 'advantage', 'help', 'support']):
            return "SUPPORTS"
        
        # Check for ELABORATES (includes examples)
        if any(word in sentence1 for word in ['example', 'instance', 'such as', 'including', 'specifically']) or \
           any(word in sentence2 for word in ['example', 'instance', 'such as', 'including', 'specifically']):
            return "ELABORATES"
        
        return "NONE"
    
    def embedding_based_classification(self, ctu1: Dict, ctu2: Dict) -> Tuple[str, float]:
        """Embedding-based classification with constraints"""
        sentence1 = ctu1['sentence']
        sentence2 = ctu2['sentence']
        
        # Encode sentences
        embeddings = self.model.encode([sentence1, sentence2])
        sentence1_emb, sentence2_emb = embeddings[0], embeddings[1]
        
        # Calculate similarities with relation embeddings
        best_relation = "NONE"
        best_similarity = 0.0
        
        for relation, rel_embedding in self.relation_embeddings.items():
            # Create relation context
            relation_context = f"{sentence1} [RELATION] {sentence2}"
            context_embedding = self.model.encode(relation_context)
            
            # Calculate similarity
            similarity = cosine_similarity([context_embedding], [rel_embedding])[0][0]
            
            if similarity > best_similarity:
                best_similarity = similarity
                best_relation = relation
        
        return best_relation, best_similarity
    
    def apply_relation_constraints(self, relation: str, ctu1: Dict, ctu2: Dict, confidence: float) -> Tuple[str, float]:
        """Apply all relation-specific constraints"""
        if relation == "NONE":
            return relation, confidence
        
        # PRECEDES adjacency constraint
        if relation == "PRECEDES" and not self.check_adjacency(ctu1, ctu2):
            return "NONE", 0.0
        
        # CONTRADICTS guardrails
        if relation == "CONTRADICTS":
            if not (self.check_shared_terms(ctu1['sentence'], ctu2['sentence']) and 
                   self.check_numeric_conflict(ctu1['sentence'], ctu2['sentence'])):
                return "NONE", 0.0
            if self.check_discourse_markers(ctu1['sentence'], ctu2['sentence']):
                return "ELABORATES", confidence * 0.6  # Convert to ELABORATES
        
        # CONDITIONS direction constraint
        if relation == "CONDITIONS" and not self.validate_conditions_direction(ctu1, ctu2):
            return "NONE", 0.0
        
        # Role compatibility check
        if not self.validate_role_compatibility(ctu1['role'], ctu2['role'], relation):
            return "NONE", 0.0
        
        return relation, confidence
    
    def apply_method_calibration(self, relation: str, method: str, confidence: float) -> float:
        """Apply method-aware calibration"""
        base_weight = METHOD_CALIBRATION.get(method, 1.0)
        relation_weight = RELATION_METHOD_WEIGHTS.get(relation, {}).get(method, 1.0)
        return confidence * base_weight * relation_weight
    
    def apply_edge_budget(self, relations: List[Dict]) -> List[Dict]:
        """Apply edge budget per node per relation type"""
        # Group relations by node and type
        node_relations = defaultdict(lambda: defaultdict(list))
        
        for rel in relations:
            if rel['relation'] != 'NONE':
                node1 = rel['ctu1']['sentence']
                node2 = rel['ctu2']['sentence']
                rel_type = rel['relation']
                
                # Add confidence-weighted score
                method_weight = self.apply_method_calibration(
                    rel['relation'], rel['method'], rel['confidence']
                )
                
                node_relations[node1][rel_type].append((method_weight, rel))
                node_relations[node2][rel_type].append((method_weight, rel))
        
        # Filter relations based on budget
        filtered_relations = []
        kept_relations = set()
        
        for node, rel_types in node_relations.items():
            for rel_type, relations_list in rel_types.items():
                if rel_type in RELATION_TYPES:
                    # Sort by confidence and keep top-k
                    relations_list.sort(key=lambda x: x[0], reverse=True)
                    budget = RELATION_TYPES[rel_type]['max_edges']
                    
                    for _, rel in relations_list[:budget]:
                        rel_id = id(rel)
                        if rel_id not in kept_relations:
                            filtered_relations.append(rel)
                            kept_relations.add(rel_id)
        
        # Add back NONE relations
        for rel in relations:
            if rel['relation'] == 'NONE':
                filtered_relations.append(rel)
        
        return filtered_relations
    
    def hybrid_classification_optimized(self, ctu_pairs: List[Tuple[Dict, Dict]], labeled_sentences: List[Dict]) -> List[Dict[str, Any]]:
        """
        Optimized hybrid classification with all upstream fixes
        """
        results = []
        gpt_pairs = []
        gpt_indices = []
        
        # Step 1: Try rule-based classification for all pairs
        for i, (ctu1, ctu2) in enumerate(ctu_pairs):
            # Rule-based classification
            rule_relation = self.enhanced_rule_based_classification(ctu1, ctu2, i, i + 1)
            
            if rule_relation != "NONE":
                # Apply constraints
                final_relation, final_confidence = self.apply_relation_constraints(
                    rule_relation, ctu1, ctu2, 0.9
                )
                
                if final_relation != "NONE":
                    # Apply method calibration
                    calibrated_confidence = self.apply_method_calibration(
                        final_relation, "rule_based", final_confidence
                    )
                    
                    results.append({
                        "relation": final_relation,
                        "method": "rule_based",
                        "confidence": calibrated_confidence,
                        "cost": 0.0
                    })
                else:
                    # Try embedding-based
                    embedding_relation, embedding_confidence = self.embedding_based_classification(ctu1, ctu2)
                    final_relation, final_confidence = self.apply_relation_constraints(
                        embedding_relation, ctu1, ctu2, embedding_confidence
                    )
                    
                    if final_relation != "NONE":
                        calibrated_confidence = self.apply_method_calibration(
                            final_relation, "embedding", final_confidence
                        )
                        
                        results.append({
                            "relation": final_relation,
                            "method": "embedding",
                            "confidence": calibrated_confidence,
                            "cost": 0.0
                        })
                    else:
                        # Queue for GPT
                        gpt_pairs.append((ctu1, ctu2))
                        gpt_indices.append(i)
                        results.append(None)
            else:
                # Try embedding-based
                embedding_relation, embedding_confidence = self.embedding_based_classification(ctu1, ctu2)
                final_relation, final_confidence = self.apply_relation_constraints(
                    embedding_relation, ctu1, ctu2, embedding_confidence
                )
                
                if final_relation != "NONE":
                    calibrated_confidence = self.apply_method_calibration(
                        final_relation, "embedding", final_confidence
                    )
                    
                    results.append({
                        "relation": final_relation,
                        "method": "embedding",
                        "confidence": calibrated_confidence,
                        "cost": 0.0
                    })
                else:
                    # Queue for GPT
                    gpt_pairs.append((ctu1, ctu2))
                    gpt_indices.append(i)
                    results.append(None)
        
        # Step 2: Process GPT batches (limit to reasonable number)
        if gpt_pairs:
            max_gpt_pairs = 10  # Reduced limit with better upstream filtering
            if len(gpt_pairs) > max_gpt_pairs:
                print(f"  Limiting GPT processing to {max_gpt_pairs} pairs (out of {len(gpt_pairs)})")
                gpt_pairs = gpt_pairs[:max_gpt_pairs]
                gpt_indices = gpt_indices[:max_gpt_pairs]
            
            print(f"  Processing {len(gpt_pairs)} pairs with GPT (batched)")
            gpt_results = self.gpt_classification_batch(gpt_pairs, batch_size=5)
            
            for i, gpt_result in enumerate(gpt_results):
                if i < len(gpt_indices):  # Safety check
                    idx = gpt_indices[i]
                    if gpt_result is not None:
                        # Apply constraints to GPT results too
                        final_relation, final_confidence = self.apply_relation_constraints(
                            gpt_result, gpt_pairs[i][0], gpt_pairs[i][1], 0.8
                        )
                        
                        if final_relation != "NONE":
                            calibrated_confidence = self.apply_method_calibration(
                                final_relation, "gpt", final_confidence
                            )
                            
                            results[idx] = {
                                "relation": final_relation,
                                "method": "gpt",
                                "confidence": calibrated_confidence,
                                "cost": 0.001
                            }
                        else:
                            results[idx] = {
                                "relation": "NONE",
                                "method": "gpt_filtered",
                                "confidence": 0.0,
                                "cost": 0.001
                            }
                    else:
                        results[idx] = {
                            "relation": "NONE",
                            "method": "gpt_failed",
                            "confidence": 0.0,
                            "cost": 0.0
                        }
        
        # Fill any remaining None results
        for i, result in enumerate(results):
            if result is None:
                results[i] = {
                    "relation": "NONE",
                    "method": "none",
                    "confidence": 0.0,
                    "cost": 0.0
                }
        
        return results
    
    def gpt_classification_batch(self, ctu_pairs: List[Tuple[Dict, Dict]], batch_size: int = 5) -> List[str]:
        """GPT classification in batches with relation constraints"""
        results = []
        
        for i in range(0, len(ctu_pairs), batch_size):
            batch = ctu_pairs[i:i + batch_size]
            
            # Create batch prompt
            batch_prompt = self.create_batch_prompt(batch)
            
            try:
                response = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": batch_prompt}],
                    temperature=0.1,
                    max_tokens=500
                )
                
                batch_results = self.parse_batch_response(response.choices[0].message.content, len(batch))
                results.extend(batch_results)
                
            except Exception as e:
                logging.error(f"GPT batch error: {e}")
                results.extend(["NONE"] * len(batch))
        
        return results
    
    def create_batch_prompt(self, ctu_pairs: List[Tuple[Dict, Dict]]) -> str:
        """Create batch prompt for GPT classification"""
        prompt = """Analyze the following CTU pairs and determine their relations. Consider adjacency for PRECEDES, direction for CONDITIONS, and shared terms for CONTRADICTS.

Relation Types:
- PRECEDES: One CTU comes before another (must be adjacent)
- CONDITIONS: One CTU sets conditions for another (Eligibility->Benefits direction)
- CONTRADICTS: One CTU contradicts another (requires shared terms + numeric conflict)
- SUPPORTS: One CTU supports another
- ELABORATES: One CTU provides more detail about another
- NONE: No meaningful relation

CTU Pairs:
"""
        
        for i, (ctu1, ctu2) in enumerate(ctu_pairs):
            prompt += f"\nPair {i+1}:\n"
            prompt += f"CTU1: {ctu1['sentence']}\n"
            prompt += f"CTU2: {ctu2['sentence']}\n"
        
        prompt += "\nRespond with only the relation types, one per line, in order."
        
        return prompt
    
    def parse_batch_response(self, response: str, expected_count: int) -> List[str]:
        """Parse GPT batch response"""
        lines = response.strip().split('\n')
        results = []
        
        for line in lines[:expected_count]:
            line = line.strip().upper()
            if line in RELATION_TYPES:
                results.append(line)
            else:
                results.append("NONE")
        
        # Pad with NONE if needed
        while len(results) < expected_count:
            results.append("NONE")
        
        return results
    
    def process_scheme_relations_optimized(self, scheme_file: str, output_dir: str) -> Dict[str, Any]:
        """
        Optimized processing with all upstream fixes
        """
        try:
            with open(scheme_file, 'r', encoding='utf-8') as f:
                scheme_data = json.load(f)
            
            scheme_name = scheme_data.get('scheme_name', 'Unknown Scheme')
            labeled_sentences = scheme_data.get('labeled_sentences', [])
            
            if len(labeled_sentences) < 2:
                return {"error": "Not enough sentences for relation analysis"}
            
            print(f"  Processing {len(labeled_sentences)} sentences for relations")
            
            # Step 1: Add missing fields
            labeled_sentences = self.add_missing_fields(labeled_sentences)
            
            # Step 2: Extract CTU pairs
            ctu_pairs = []
            for i in range(len(labeled_sentences)):
                for j in range(i + 1, len(labeled_sentences)):
                    ctu_pairs.append((labeled_sentences[i], labeled_sentences[j]))
            
            print(f"  Found {len(ctu_pairs)} CTU pairs to analyze")
            
            # Step 3: Classify relations with all fixes
            relation_results = self.hybrid_classification_optimized(ctu_pairs, labeled_sentences)
            
            # Step 4: Create relations list
            relations = []
            total_cost = 0.0
            method_counts = {"rule_based": 0, "embedding": 0, "gpt": 0}
            
            for i, (ctu1, ctu2) in enumerate(ctu_pairs):
                relation_result = relation_results[i]
                relations.append({
                    "ctu1": ctu1,
                    "ctu2": ctu2,
                    "relation": relation_result["relation"],
                    "method": relation_result["method"],
                    "confidence": relation_result["confidence"]
                })
                
                total_cost += relation_result["cost"]
                method_counts[relation_result["method"]] += 1
            
            # Step 5: Apply edge budget
            relations = self.apply_edge_budget(relations)
            
            # Create output structure
            output_data = {
                "scheme_name": scheme_name,
                "total_sentences": len(labeled_sentences),
                "total_pairs": len(ctu_pairs),
                "relations": relations,
                "relation_distribution": {},
                "method_distribution": method_counts,
                "total_cost": total_cost,
                "processing_timestamp": datetime.now().isoformat()
            }
            
            # Calculate relation distribution
            relation_counts = {}
            for relation_data in relations:
                relation = relation_data['relation']
                relation_counts[relation] = relation_counts.get(relation, 0) + 1
            
            output_data["relation_distribution"] = relation_counts
            
            # Save individual file
            scheme_name_slug = os.path.basename(scheme_file).replace('_labeled.json', '')
            output_file = os.path.join(output_dir, f"{scheme_name_slug}_relations_v3_fixed.json")
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, ensure_ascii=False, indent=2)
            
            return output_data
            
        except Exception as e:
            logging.error(f"Error processing {scheme_file}: {e}")
            return {"error": str(e)}

def process_all_scheme_relations_v3_fixed(input_dir: str, output_dir: str, sample_size: int = 3):
    """
    Process all scheme files with V3 fixed relation labeler
    """
    print("=== CTU RELATION LABELER V3 - FIXED ===")
    print(f"Input: {input_dir}")
    print(f"Output: {output_dir}")
    print(f"Sample size: {sample_size}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all labeled scheme files
    scheme_files = glob.glob(os.path.join(input_dir, "*_labeled.json"))
    
    if not scheme_files:
        print("No labeled scheme files found!")
        return
    
    print(f"Found {len(scheme_files)} scheme files")
    
    # Initialize the fixed labeler
    labeler = CTURelationLabelerV3Fixed()
    
    # Process files
    processed_count = 0
    total_cost = 0.0
    all_relation_distributions = {}
    
    for scheme_file in scheme_files[:sample_size]:
        print(f"\nProcessing: {os.path.basename(scheme_file)}")
        
        result = labeler.process_scheme_relations_optimized(scheme_file, output_dir)
        
        if "error" not in result:
            processed_count += 1
            total_cost += result.get("total_cost", 0.0)
            
            # Aggregate relation distributions
            for relation, count in result.get("relation_distribution", {}).items():
                all_relation_distributions[relation] = all_relation_distributions.get(relation, 0) + count
            
            print(f"  Relations: {result.get('total_pairs', 0)} pairs")
            print(f"  Cost: ${result.get('total_cost', 0.0):.4f}")
        else:
            print(f"  Error: {result['error']}")
    
    # Create summary
    summary = {
        "total_files_processed": processed_count,
        "total_cost": total_cost,
        "relation_distribution": all_relation_distributions,
        "processing_timestamp": datetime.now().isoformat()
    }
    
    summary_file = os.path.join(output_dir, "v3_fixed_summary.json")
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    
    print(f"\n=== PROCESSING COMPLETE ===")
    print(f"Processed: {processed_count} files")
    print(f"Total cost: ${total_cost:.4f}")
    print(f"Results saved to: {output_dir}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python ctu_relation_labeler_v3_fixed.py <input_dir> <output_dir> [sample_size]")
        sys.exit(1)
    
    input_dir = sys.argv[1]
    output_dir = sys.argv[2]
    sample_size = int(sys.argv[3]) if len(sys.argv) > 3 else 3
    
    process_all_scheme_relations_v3_fixed(input_dir, output_dir, sample_size)
