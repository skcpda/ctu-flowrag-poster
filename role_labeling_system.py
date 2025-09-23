#!/usr/bin/env python3
"""
Role Labeling System for Government Schemes
Labels sentences with primary roles and extracts structured slots
"""

import json
import os
import time
import requests
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import re

@dataclass
class PricingInfo:
    """Pricing information for different models"""
    model: str
    input_cost_per_1k: float
    output_cost_per_1k: float
    estimated_total_cost: float

class RoleLabelingSystem:
    """System for labeling sentences with roles and extracting slots"""
    
    ALLOWED_ROLES = [
        "ProblemContext", "Objective", "Benefit", "Eligibility", "ApplicationProcess", 
        "Timeline", "ContactsGovernance", "Exclusion", "Definition", 
        "ImplementingAgencyJurisdiction", "FinancialDetails", "RequiredDocuments",
        "VerificationInspection", "DisbursalComputation", "ComplianceConditions", 
        "AppealsGrievance", "TargetBeneficiariesSector", "GeographyScope", 
        "GovernanceBodies", "Mode", "FrequencyCycle", "FootnoteLegalBasis", "Misc"
    ]
    
    def __init__(self, api_key: str, model: str = "gpt-3.5-turbo"):
        self.api_key = api_key
        self.model = model
        self.base_url = "https://api.openai.com/v1/chat/completions"
        
    def create_prompt(self, sentences: List[Dict]) -> str:
        """Create the prompt for role labeling"""
        sentences_text = "\n".join([f'{{"sid":"{s["sid"]}","text":"{s["text"]}"}}' for s in sentences])
        
        return f"""You are an expert at analyzing government welfare schemes. For each sentence, assign exactly ONE primary role from the allowed set and extract relevant slots.

ALLOWED ROLES:
- ProblemContext, Objective, Benefit, Eligibility, ApplicationProcess, Timeline, ContactsGovernance,
- Exclusion, Definition, ImplementingAgencyJurisdiction, FinancialDetails, RequiredDocuments,
- VerificationInspection, DisbursalComputation, ComplianceConditions, AppealsGrievance,
- TargetBeneficiariesSector, GeographyScope, GovernanceBodies, Mode, FrequencyCycle, FootnoteLegalBasis, Misc

RULES:
- Exactly ONE primary role per sentence (best fit). Use "Misc" only if none apply.
- Compute role_probs for TOP-3 roles (sum ≤ 1.0).
- Extract slots depending on role:
  * Benefit: {{amount, rate_percent, cap_amount, periodicity, included_costs[]}}
  * Eligibility: {{subject, age, income_cap, unit_type, registration, geo_scope, exceptions[]}}
  * ApplicationProcess: {{steps[], channel, office, form, fee}}
  * Definition: {{term, gloss}}
  * FinancialDetails: {{cost_heads[], gst_included: bool}}
  * Exclusion: {{negative_list[]}}
  * Timeline: {{cycle, opens_on, closes_on}}
  * ContactsGovernance: {{authority, committee, address, phone, email}}
  * ImplementingAgencyJurisdiction: {{agency, jurisdiction}}
- Normalize currency to INR plain numbers (e.g., "₹6.25 lakh" → 625000.0), percents to 0–100 floats, ages to ints.

SENTENCES:
{sentences_text}

OUTPUT STRICT JSON:
{{
  "labels": [
    {{
      "sid": "S1",
      "role": "<one role>",
      "role_probs": {{"<role1>": 0.xx, "<role2>": 0.xx, "<role3>": 0.xx}},
      "slots": {{ ... role-specific fields ... }},
      "has_numbers": true|false
    }}
  ]
}}"""

    def call_openai_api(self, prompt: str) -> Dict:
        """Call OpenAI API with the prompt"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "You are an expert at analyzing government welfare schemes and extracting structured information."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.1,
            "max_tokens": 4000
        }
        
        try:
            response = requests.post(self.base_url, headers=headers, json=data)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"API Error: {e}")
            return None

    def process_sentences(self, doc_id: str, sentences: List[Dict]) -> Dict:
        """Process sentences and return labeled results"""
        prompt = self.create_prompt(sentences)
        
        # Call OpenAI API
        response = self.call_openai_api(prompt)
        if not response:
            return {"error": "API call failed"}
        
        try:
            # Extract the content from the response
            content = response['choices'][0]['message']['content']
            
            # Parse JSON from the content
            result = json.loads(content)
            result["doc_id"] = doc_id
            
            return result
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Error parsing response: {e}")
            return {"error": "Failed to parse response"}

    def batch_process(self, input_file: str, output_file: str, batch_size: int = 10):
        """Process sentences in batches"""
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        doc_id = data.get('doc_id', 'unknown')
        sentences = data.get('sentences', [])
        
        results = {
            "doc_id": doc_id,
            "labels": []
        }
        
        # Process in batches
        for i in range(0, len(sentences), batch_size):
            batch = sentences[i:i + batch_size]
            print(f"Processing batch {i//batch_size + 1}/{(len(sentences) + batch_size - 1)//batch_size}")
            
            batch_result = self.process_sentences(doc_id, batch)
            
            if "error" not in batch_result:
                results["labels"].extend(batch_result.get("labels", []))
            else:
                print(f"Error in batch {i//batch_size + 1}: {batch_result['error']}")
            
            # Rate limiting
            time.sleep(1)
        
        # Save results
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"Processed {len(sentences)} sentences for {doc_id}")
        return results

def calculate_pricing_analysis():
    """Calculate pricing for different models and batch sizes"""
    
    # Current pricing (as of September 2024)
    pricing = {
        "gpt-3.5-turbo": {
            "input_cost_per_1k": 0.0015,
            "output_cost_per_1k": 0.002
        },
        "gpt-4o": {
            "input_cost_per_1k": 0.0025,
            "output_cost_per_1k": 0.01
        }
    }
    
    # Estimate tokens per sentence
    avg_tokens_per_sentence = 25  # Input tokens
    avg_output_tokens = 30  # Output tokens (including JSON structure)
    total_sentences = 23761
    
    print("=== PRICING ANALYSIS FOR ROLE LABELING ===")
    print(f"Total sentences to process: {total_sentences:,}")
    print(f"Average tokens per sentence: {avg_tokens_per_sentence}")
    print(f"Average output tokens per sentence: {avg_output_tokens}")
    print()
    
    for model, costs in pricing.items():
        input_cost = (total_sentences * avg_tokens_per_sentence * costs["input_cost_per_1k"]) / 1000
        output_cost = (total_sentences * avg_output_tokens * costs["output_cost_per_1k"]) / 1000
        total_cost = input_cost + output_cost
        
        print(f"📊 {model.upper()}:")
        print(f"  Input cost:  ${input_cost:.2f}")
        print(f"  Output cost: ${output_cost:.2f}")
        print(f"  Total cost:  ${total_cost:.2f}")
        print()
    
    # Batch processing recommendations
    print("=== BATCH PROCESSING RECOMMENDATIONS ===")
    print("Recommended batch sizes:")
    print("  • Small batches (5-10 sentences): Better accuracy, higher cost")
    print("  • Medium batches (15-25 sentences): Balanced cost/accuracy")
    print("  • Large batches (30-50 sentences): Lower cost, potential accuracy loss")
    print()
    
    # Cost optimization strategies
    print("=== COST OPTIMIZATION STRATEGIES ===")
    print("1. Use GPT-3.5-turbo for initial processing (~$1.66 total)")
    print("2. Use GPT-4o only for complex/ambiguous cases")
    print("3. Implement caching for similar sentence patterns")
    print("4. Pre-filter sentences to avoid processing 'Misc' categories")
    print("5. Batch similar sentence types together")
    print()
    
    return pricing

def create_sample_implementation():
    """Create a sample implementation"""
    sample_code = '''
# Example usage:
import os
from role_labeling_system import RoleLabelingSystem

# Initialize the system
api_key = os.getenv("OPENAI_API_KEY")
labeler = RoleLabelingSystem(api_key=api_key, model="gpt-3.5-turbo")

# Process a single document
input_file = "organized_output/outputs/split_sentences/sample_sentences.json"
output_file = "organized_output/outputs/labeled_sentences/sample_labeled.json"
labeler.batch_process(input_file, output_file, batch_size=10)

# Process all documents
import glob
input_dir = "organized_output/outputs/split_sentences/"
output_dir = "organized_output/outputs/labeled_sentences/"

os.makedirs(output_dir, exist_ok=True)

for input_file in glob.glob(f"{input_dir}/*.json"):
    filename = os.path.basename(input_file)
    output_file = os.path.join(output_dir, filename.replace("_sentences.json", "_labeled.json"))
    labeler.batch_process(input_file, output_file, batch_size=15)
'''
    
    return sample_code

def main():
    """Main function to demonstrate the system"""
    print("=== GOVERNMENT SCHEME ROLE LABELING SYSTEM ===")
    print()
    
    # Calculate pricing
    pricing = calculate_pricing_analysis()
    
    # Create sample implementation
    sample_code = create_sample_implementation()
    
    print("=== SAMPLE IMPLEMENTATION ===")
    print(sample_code)
    
    print("=== NEXT STEPS ===")
    print("1. Set up OpenAI API key: export OPENAI_API_KEY='your-key-here'")
    print("2. Install dependencies: pip install requests")
    print("3. Run the labeling system on your sentence files")
    print("4. Monitor costs and adjust batch sizes as needed")
    print()
    
    print("=== RECOMMENDED APPROACH ===")
    print("1. Start with GPT-3.5-turbo for cost efficiency (~$1.66 total)")
    print("2. Use batch size of 15-25 sentences for optimal balance")
    print("3. Process in phases to monitor quality and costs")
    print("4. Consider hybrid approach: GPT-3.5 for simple cases, GPT-4o for complex ones")

if __name__ == "__main__":
    main()
