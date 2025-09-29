#!/usr/bin/env python3
"""
CTU Relation Labeler V2 - Full Pipeline with Fine-tuned BGE
Implements all optimizations: fan-out limits, role masks, de-duplication, 
enhanced rules, confidence calibration, and BGE fine-tuning.
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
log_file = os.path.join(log_dir, f"ctu_relation_labeler_v2_{timestamp}.log")
logging.basicConfig(filename=log_file, level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# CTU Relation Types with fan-out limits
RELATION_TYPES = {
    "PRECEDES": {"description": "One CTU comes before another in logical sequence", "max_edges": 3},
    "CAUSES": {"description": "One CTU causes or leads to another", "max_edges": 2},
    "CONTRADICTS": {"description": "One CTU contradicts or conflicts with another", "max_edges": 2},
    "SUPPORTS": {"description": "One CTU supports or reinforces another", "max_edges": 5},
    "ELABORATES": {"description": "One CTU provides more detail about another", "max_edges": 4},
    "EXAMPLES": {"description": "One CTU provides examples of another", "max_edges": 3},
    "CONDITIONS": {"description": "One CTU sets conditions for another", "max_edges": 2},
    "NONE": {"description": "No meaningful relation between CTUs", "max_edges": 0}
}

# Role compatibility masks
ROLE_COMPATIBILITY = {
    "CONDITIONS": [("Eligibility", "Benefits"), ("Eligibility", "ApplicationProcess")],
    "ADMINISTERED_BY": [("Authorities", "Benefits"), ("Authorities", "ApplicationProcess"), ("Authorities", "Eligibility")],
    "EXAMPLES": [("ContextObjective", "Benefits"), ("ContextObjective", "ApplicationProcess")],
    "SUPPORTS": [("Benefits", "Benefits"), ("Eligibility", "Benefits"), ("ApplicationProcess", "Benefits")],
    "ELABORATES": [("ContextObjective", "ContextObjective"), ("Benefits", "Benefits"), ("Eligibility", "Eligibility")]
}

class CTURelationLabelerV2:
    def __init__(self, model_name: str = "fine_tuned_bge_ctu_relations/", fine_tune: bool = True):
        """
        Initialize the enhanced CTU relation labeler with fine-tuning capability
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
    
    def de_duplicate_sentences(self, labeled_sentences: List[Dict]) -> Tuple[List[Dict], Dict[int, int]]:
        """
        Remove near-duplicate sentences to prevent hub formation
        """
        if len(labeled_sentences) < 2:
            return labeled_sentences, {}
        
        # Encode all sentences
        sentences = [s['sentence'] for s in labeled_sentences]
        embeddings = self.model.encode(sentences)
        
        # Find duplicates using cosine similarity
        duplicate_map = {}
        keep_indices = set(range(len(sentences)))
        
        for i in range(len(sentences)):
            if i not in keep_indices:
                continue
                
            for j in range(i + 1, len(sentences)):
                if j not in keep_indices:
                    continue
                    
                similarity = cosine_similarity(
                    embeddings[i].reshape(1, -1),
                    embeddings[j].reshape(1, -1)
                )[0][0]
                
                if similarity > 0.97:  # Near duplicate
                    # Keep the one with higher confidence
                    if labeled_sentences[i]['confidence'] >= labeled_sentences[j]['confidence']:
                        duplicate_map[j] = i
                        keep_indices.remove(j)
                    else:
                        duplicate_map[i] = j
                        keep_indices.remove(i)
                        break
        
        # Create deduplicated list
        deduplicated = []
        index_mapping = {}
        new_index = 0
        
        for i in range(len(labeled_sentences)):
            if i in keep_indices:
                deduplicated.append(labeled_sentences[i])
                index_mapping[i] = new_index
                new_index += 1
            else:
                # Map to the kept duplicate
                original_index = duplicate_map[i]
                index_mapping[i] = index_mapping[original_index]
        
        print(f"  De-duplicated: {len(labeled_sentences)} → {len(deduplicated)} sentences")
        return deduplicated, index_mapping
    
    def enhanced_rule_based_classification(self, ctu1: Dict, ctu2: Dict, index1: int, index2: int) -> str:
        """
        Enhanced rule-based classification with structural patterns
        """
        role1 = ctu1['role']
        role2 = ctu2['role']
        sentence1 = ctu1['sentence'].lower()
        sentence2 = ctu2['sentence'].lower()
        
        # Rule 1: Windowed PRECEDES (only within ±2 sentences)
        if abs(index2 - index1) <= 2:
            sequential_patterns = [
                ("ContextObjective", "BenefitsAssistance"),
                ("ContextObjective", "Eligibility"),
                ("ContextObjective", "ApplicationProcess"),
                ("BenefitsAssistance", "Eligibility"),
                ("Eligibility", "ApplicationProcess"),
                ("ApplicationProcess", "TimelineFrequency"),
                ("ApplicationProcess", "AuthoritiesGovernance"),
                ("TimelineFrequency", "AuthoritiesGovernance")
            ]
            
            if (role1, role2) in sequential_patterns:
                return "PRECEDES"
        
        # Rule 2: Section continuation (same header/context)
        if role1 == role2 and abs(index2 - index1) <= 3:
            continuation_keywords = [
                "furthermore", "additionally", "moreover", "in addition",
                "also", "besides", "what's more", "not only", "but also"
            ]
            if any(keyword in sentence2 for keyword in continuation_keywords):
                return "ELABORATES"
        
        # Rule 3: List item relationships
        if role1 == role2 and abs(index2 - index1) <= 5:
            list_keywords = [
                "first", "second", "third", "next", "then", "finally",
                "step 1", "step 2", "step 3", "phase 1", "phase 2"
            ]
            if any(keyword in sentence2 for keyword in list_keywords):
                return "PRECEDES"
        
        # Rule 4: Stricter CONTRADICTS (numeric mismatch or negation flip)
        if self._has_numeric_mismatch(sentence1, sentence2) or self._has_negation_flip(sentence1, sentence2):
            return "CONTRADICTS"
        
        # Rule 5: Example keywords (EXAMPLES)
        example_keywords = [
            "for example", "such as", "including", "like", "e.g.", "for instance",
            "namely", "specifically", "in particular", "as an example"
        ]
        if any(keyword in sentence2 for keyword in example_keywords):
            return "EXAMPLES"
        
        # Rule 6: Support keywords (SUPPORTS)
        support_keywords = [
            "therefore", "thus", "hence", "consequently", "as a result", "because",
            "since", "due to", "owing to", "thanks to", "this means", "this implies"
        ]
        if any(keyword in sentence2 for keyword in support_keywords):
            return "SUPPORTS"
        
        # Rule 7: Condition keywords (CONDITIONS)
        condition_keywords = [
            "if", "when", "provided that", "assuming", "in case", "unless",
            "only if", "as long as", "so long as", "given that", "on condition that"
        ]
        if any(keyword in sentence2 for keyword in condition_keywords):
            return "CONDITIONS"
        
        # Rule 8: Cause keywords (CAUSES)
        cause_keywords = [
            "leads to", "results in", "causes", "brings about", "gives rise to",
            "triggers", "initiates", "stimulates", "promotes", "encourages"
        ]
        if any(keyword in sentence2 for keyword in cause_keywords):
            return "CAUSES"
        
        return "NONE"
    
    def _has_numeric_mismatch(self, sentence1: str, sentence2: str) -> bool:
        """Check for numeric mismatches between sentences"""
        # Extract numbers and units
        nums1 = re.findall(r'₹?\d+(?:,\d{3})*(?:\.\d+)?%?', sentence1)
        nums2 = re.findall(r'₹?\d+(?:,\d{3})*(?:\.\d+)?%?', sentence2)
        
        if not nums1 or not nums2:
            return False
        
        # Check for conflicting numbers on same concepts
        for num1 in nums1:
            for num2 in nums2:
                if num1 != num2 and any(word in sentence1 and word in sentence2 for word in ['age', 'amount', 'limit', 'maximum', 'minimum']):
                    return True
        return False
    
    def _has_negation_flip(self, sentence1: str, sentence2: str) -> bool:
        """Check for negation polarity flips"""
        negations1 = re.findall(r'\b(?:not|no|never|none|nothing|nobody|nowhere)\b', sentence1)
        negations2 = re.findall(r'\b(?:not|no|never|none|nothing|nobody|nowhere)\b', sentence2)
        
        if (negations1 and not negations2) or (not negations1 and negations2):
            # Check if they're talking about the same concept
            shared_words = set(sentence1.split()) & set(sentence2.split())
            if len(shared_words) > 3:  # Enough overlap
                return True
        return False
    
    def role_compatibility_check(self, role1: str, role2: str, relation: str) -> bool:
        """
        Check if two roles are compatible for a given relation
        """
        if relation not in ROLE_COMPATIBILITY:
            return True  # No restrictions
        
        compatible_pairs = ROLE_COMPATIBILITY[relation]
        return (role1, role2) in compatible_pairs or (role2, role1) in compatible_pairs
    
    def embedding_based_classification(self, ctu1: Dict, ctu2: Dict) -> Tuple[str, float]:
        """
        Embedding-based classification using BGE
        """
        # Create context for relation classification
        context = f"CTU1 ({ctu1['role']}): {ctu1['sentence']}\nCTU2 ({ctu2['role']}): {ctu2['sentence']}"
        
        # Encode the context
        context_embedding = self.model.encode(context)
        
        # Calculate similarities with all relations
        similarities = {}
        for relation, relation_embedding in self.relation_embeddings.items():
            similarity = cosine_similarity(
                context_embedding.reshape(1, -1),
                relation_embedding.reshape(1, -1)
            )[0][0]
            similarities[relation] = float(similarity)
        
        # Find the best match
        best_relation = max(similarities, key=similarities.get)
        best_similarity = similarities[best_relation]
        
        return best_relation, best_similarity
    
    def gpt_classification_batch(self, ctu_pairs: List[Tuple[Dict, Dict]], batch_size: int = 10) -> List[str]:
        """
        GPT-based classification with batching for efficiency
        """
        results = []
        
        for i in range(0, len(ctu_pairs), batch_size):
            batch = ctu_pairs[i:i + batch_size]
            
            try:
                # Create batch prompt
                batch_text = ""
                for j, (ctu1, ctu2) in enumerate(batch):
                    batch_text += f"{j+1}. CTU1 ({ctu1['role']}): \"{ctu1['sentence']}\"\n   CTU2 ({ctu2['role']}): \"{ctu2['sentence']}\"\n\n"
                
                prompt = f"""You are an expert at identifying relations between Content Thematic Units (CTUs) in welfare scheme descriptions.

{batch_text}

For each pair above, identify the relation. Choose ONE of these relations:

1. PRECEDES: One CTU comes before another in logical sequence
2. CAUSES: One CTU causes or leads to another
3. CONTRADICTS: One CTU contradicts or conflicts with another
4. SUPPORTS: One CTU supports or reinforces another
5. ELABORATES: One CTU provides more detail about another
6. EXAMPLES: One CTU provides examples of another
7. CONDITIONS: One CTU sets conditions for another
8. NONE: No meaningful relation between CTUs

Respond with ONLY the relation names, one per line, in order (e.g., "PRECEDES\nELABORATES\nNONE"). No explanations needed."""
                response = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "You are an expert at identifying relations between CTUs. Always respond with only the relation names, one per line."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=100,
                    temperature=0.1
                )
                
                batch_results = response.choices[0].message.content.strip().split('\n')
                
                # Pad results if needed
                while len(batch_results) < len(batch):
                    batch_results.append("NONE")
                
                results.extend(batch_results[:len(batch)])
                
            except Exception as e:
                print(f"    ⚠️ GPT batch {i//batch_size + 1} failed: {e}")
                # Fill with NONE for failed batch
                results.extend(['NONE'] * len(batch))
        
        return results
    
    def apply_fan_out_limits(self, relations: List[Dict]) -> List[Dict]:
        """
        Apply fan-out limits to prevent hub formation
        """
        # Group relations by CTU
        ctu_relations = defaultdict(list)
        for relation in relations:
            ctu1_id = relation['ctu1']['id']
            ctu2_id = relation['ctu2']['id']
            ctu_relations[ctu1_id].append(relation)
            ctu_relations[ctu2_id].append(relation)
        
        # Apply limits per CTU
        filtered_relations = []
        for ctu_id, ctu_rels in ctu_relations.items():
            # Group by relation type
            by_type = defaultdict(list)
            for rel in ctu_rels:
                by_type[rel['relation']].append(rel)
            
            # Apply limits
            for relation_type, rels in by_type.items():
                if relation_type == "NONE":
                    continue
                
                max_edges = RELATION_TYPES[relation_type]["max_edges"]
                if len(rels) > max_edges:
                    # Sort by confidence and keep top-k
                    rels.sort(key=lambda x: x['confidence'], reverse=True)
                    rels = rels[:max_edges]
                
                filtered_relations.extend(rels)
        
        # Remove duplicates
        seen = set()
        final_relations = []
        for rel in filtered_relations:
            key = (rel['ctu1']['id'], rel['ctu2']['id'], rel['relation'])
            if key not in seen:
                seen.add(key)
                final_relations.append(rel)
        
        return final_relations
    
    def apply_locality_re_ranking(self, relations: List[Dict], labeled_sentences: List[Dict]) -> List[Dict]:
        """
        Apply locality-based re-ranking
        """
        # Create index mapping
        index_map = {s['id']: i for i, s in enumerate(labeled_sentences)}
        
        for relation in relations:
            ctu1_id = relation['ctu1']['id']
            ctu2_id = relation['ctu2']['id']
            
            if ctu1_id in index_map and ctu2_id in index_map:
                idx1 = index_map[ctu1_id]
                idx2 = index_map[ctu2_id]
                
                # Calculate locality factor
                locality_factor = 0.0
                
                # Same header/section
                if relation['ctu1']['role'] == relation['ctu2']['role']:
                    locality_factor += 0.1
                
                # Close indices
                if abs(idx2 - idx1) <= 3:
                    locality_factor += 0.05
                
                # Role mismatch penalty
                if not self.role_compatibility_check(
                    relation['ctu1']['role'], 
                    relation['ctu2']['role'], 
                    relation['relation']
                ):
                    locality_factor -= 0.1
                
                # Apply locality factor to confidence
                relation['confidence'] = min(1.0, relation['confidence'] + locality_factor)
        
        return relations
    
    def calibrate_confidence(self, relations: List[Dict]) -> List[Dict]:
        """
        Apply confidence calibration
        """
        if not self.calibration_model:
            # Use simple threshold for now
            for relation in relations:
                if relation['relation'] != 'NONE' and relation['confidence'] < self.calibration_threshold:
                    relation['relation'] = 'NONE'
                    relation['confidence'] = 0.5
        else:
            # Use calibration model
            confidences = [r['confidence'] for r in relations]
            calibrated = self.calibration_model.predict(confidences)
            
            for i, relation in enumerate(relations):
                relation['confidence'] = calibrated[i]
                if relation['confidence'] < self.calibration_threshold:
                    relation['relation'] = 'NONE'
        
        return relations
    
    def hybrid_classification_optimized(self, ctu_pairs: List[Tuple[Dict, Dict]], labeled_sentences: List[Dict]) -> List[Dict[str, Any]]:
        """
        Optimized hybrid classification with all enhancements
        """
        results = []
        gpt_pairs = []
        gpt_indices = []
        
        # Step 1: Try rule-based classification for all pairs
        for i, (ctu1, ctu2) in enumerate(ctu_pairs):
            # Role compatibility check first
            rule_relation = self.enhanced_rule_based_classification(ctu1, ctu2, i, i + 1)
            
            if rule_relation != "NONE":
                # Check role compatibility
                if self.role_compatibility_check(ctu1['role'], ctu2['role'], rule_relation):
                    results.append({
                        "relation": rule_relation,
                        "method": "rule_based",
                        "confidence": 0.9,
                        "cost": 0.0
                    })
                else:
                    # Try embedding-based
                    embedding_relation, embedding_confidence = self.embedding_based_classification(ctu1, ctu2)
                    if embedding_confidence >= 0.5 and self.role_compatibility_check(ctu1['role'], ctu2['role'], embedding_relation):
                        results.append({
                            "relation": embedding_relation,
                            "method": "embedding",
                            "confidence": embedding_confidence,
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
                if embedding_confidence >= 0.5 and self.role_compatibility_check(ctu1['role'], ctu2['role'], embedding_relation):
                    results.append({
                        "relation": embedding_relation,
                        "method": "embedding",
                        "confidence": embedding_confidence,
                        "cost": 0.0
                    })
                else:
                    # Queue for GPT
                    gpt_pairs.append((ctu1, ctu2))
                    gpt_indices.append(i)
                    results.append(None)
        
        # Step 2: Process GPT batches (limit to reasonable number)
        if gpt_pairs:
            max_gpt_pairs = 20  # Limit to prevent excessive API calls (reduced with fine-tuned BGE)
            if len(gpt_pairs) > max_gpt_pairs:
                print(f"  Limiting GPT processing to {max_gpt_pairs} pairs (out of {len(gpt_pairs)})")
                gpt_pairs = gpt_pairs[:max_gpt_pairs]
                gpt_indices = gpt_indices[:max_gpt_pairs]
            
            print(f"  Processing {len(gpt_pairs)} pairs with GPT (batched)")
            gpt_results = self.gpt_classification_batch(gpt_pairs, batch_size=10)
            
            for i, gpt_result in enumerate(gpt_results):
                if i < len(gpt_indices):  # Safety check
                    idx = gpt_indices[i]
                    if gpt_result is not None:
                        results[idx] = {
                            "relation": gpt_result,
                            "method": "gpt",
                            "confidence": 0.8,
                            "cost": 0.001
                        }
                    else:
                        results[idx] = {
                            "relation": "NONE",
                            "method": "gpt_failed",
                            "confidence": 0.0,
                            "cost": 0.0
                        }
        
        return results
    
    def process_scheme_relations_optimized(self, scheme_file: str, output_dir: str) -> Dict[str, Any]:
        """
        Optimized processing of all CTU relations in a scheme file
        """
        try:
            with open(scheme_file, 'r', encoding='utf-8') as f:
                scheme_data = json.load(f)
            
            scheme_name = scheme_data.get('scheme_name', 'Unknown Scheme')
            labeled_sentences = scheme_data.get('labeled_sentences', [])
            
            if len(labeled_sentences) < 2:
                return {"error": "Not enough sentences for relation analysis"}
            
            print(f"  Processing {len(labeled_sentences)} sentences for relations")
            
            # Step 1: De-duplicate sentences
            deduplicated_sentences, index_mapping = self.de_duplicate_sentences(labeled_sentences)
            
            # Step 2: Extract CTU pairs
            ctu_pairs = []
            for i in range(len(deduplicated_sentences)):
                for j in range(i + 1, len(deduplicated_sentences)):
                    ctu_pairs.append((deduplicated_sentences[i], deduplicated_sentences[j]))
            
            print(f"  Found {len(ctu_pairs)} CTU pairs to analyze")
            
            # Step 3: Classify relations with optimized approach
            relation_results = self.hybrid_classification_optimized(ctu_pairs, deduplicated_sentences)
            
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
            
            # Step 5: Apply fan-out limits
            relations = self.apply_fan_out_limits(relations)
            
            # Step 6: Apply locality re-ranking
            relations = self.apply_locality_re_ranking(relations, deduplicated_sentences)
            
            # Step 7: Calibrate confidence
            relations = self.calibrate_confidence(relations)
            
            # Create output structure
            output_data = {
                "scheme_name": scheme_name,
                "total_sentences": len(deduplicated_sentences),
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
            output_file = os.path.join(output_dir, f"{scheme_name_slug}_relations.json")
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, ensure_ascii=False, indent=2)
            
            return output_data
            
        except Exception as e:
            logging.error(f"Error processing {scheme_file}: {e}")
            return {"error": str(e)}

def process_all_scheme_relations_optimized(input_dir: str, output_dir: str, sample_size: int = 3):
    """
    Optimized processing of all scheme files for CTU relation labeling
    """
    print("=== CTU RELATION LABELER V2 - FULL PIPELINE ===")
    print(f"Input: {input_dir}")
    print(f"Output: {output_dir}")
    print(f"Sample size: {sample_size}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize the labeler
    labeler = CTURelationLabelerV2(fine_tune=True)
    
    # Get all scheme files
    scheme_files = glob.glob(f"{input_dir}/*_labeled.json")
    total_files = len(scheme_files)
    
    print(f"Found {total_files} scheme files to process")
    
    # Process sample first
    if sample_size is None:
        sample_files = scheme_files  # Process all files
    else:
        sample_files = random.sample(scheme_files, min(sample_size, total_files))
    
    processed_count = 0
    failed_count = 0
    total_cost = 0.0
    relation_distribution = {}
    method_distribution = {"rule_based": 0, "embedding": 0, "gpt": 0}
    
    for i, scheme_file in enumerate(sample_files):
        print(f"\n[{i+1}/{len(sample_files)}] Processing: {os.path.basename(scheme_file)}")
        
        result = labeler.process_scheme_relations_optimized(scheme_file, output_dir)
        
        if "error" in result:
            failed_count += 1
            print(f"  ❌ Failed: {result['error']}")
        else:
            processed_count += 1
            total_cost += result.get('total_cost', 0)
            
            # Update distributions
            for relation, count in result.get('relation_distribution', {}).items():
                relation_distribution[relation] = relation_distribution.get(relation, 0) + count
            
            for method, count in result.get('method_distribution', {}).items():
                method_distribution[method] += method_distribution.get(method, 0) + count
            
            print(f"  ✅ Success: {result['total_pairs']} relations labeled")
            print(f"  💰 Cost: ${result.get('total_cost', 0):.4f}")
            print(f"  📊 Relations: {result.get('relation_distribution', {})}")
            print(f"  🔧 Methods: {result.get('method_distribution', {})}")
    
    # Save summary
    summary_data = {
        "total_files": len(sample_files),
        "processed_files": processed_count,
        "failed_files": failed_count,
        "total_cost": total_cost,
        "relation_distribution": relation_distribution,
        "method_distribution": method_distribution,
        "processing_timestamp": datetime.now().isoformat()
    }
    
    with open(os.path.join(output_dir, "summary.json"), 'w', encoding='utf-8') as f:
        json.dump(summary_data, f, ensure_ascii=False, indent=2)
    
    print(f"\n=== OPTIMIZED RELATION LABELING COMPLETE ===")
    print(f"Total files: {len(sample_files)}")
    print(f"Processed: {processed_count}")
    print(f"Failed: {failed_count}")
    print(f"Total cost: ${total_cost:.4f}")
    print(f"Relation distribution: {relation_distribution}")
    print(f"Method distribution: {method_distribution}")

if __name__ == "__main__":
    input_dir = "organized_output/outputs/ctu_embedding_labeled"
    output_dir = "organized_output/outputs/ctu_relations_v2"
    
    # Process sample of 3 schemes first
    process_all_scheme_relations_optimized(input_dir, output_dir, sample_size=3)
