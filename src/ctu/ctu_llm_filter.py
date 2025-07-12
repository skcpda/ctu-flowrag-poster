# src/ctu/ctu_llm_filter.py
import json
import openai
from pathlib import Path
from typing import List, Dict, Tuple
import time

# Configure OpenAI client
client = openai.OpenAI()

def create_boundary_prompt(sentences: List[str], boundary_idx: int, window: int = 5) -> str:
    """Create prompt for boundary validation."""
    start = max(0, boundary_idx - window)
    end = min(len(sentences), boundary_idx + window)
    
    before_text = " ".join(sentences[start:boundary_idx])
    after_text = " ".join(sentences[boundary_idx:end])
    
    prompt = f"""You are evaluating whether a sentence boundary represents a coherent thematic unit (CTU) break.

Context before boundary:
"{before_text}"

Context after boundary:
"{after_text}"

Question: Does this boundary represent a clear thematic shift? Consider if the topics, entities, or discourse flow change significantly.

Answer with only: YES or NO"""
    
    return prompt

def validate_boundary_with_llm(sentences: List[str], boundary_idx: int) -> bool:
    """Use OpenAI to validate if boundary_idx represents a real CTU boundary."""
    prompt = create_boundary_prompt(sentences, boundary_idx)
    
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=10
        )
        
        answer = response.choices[0].message.content.strip().upper()
        return answer == "YES"
    
    except Exception as e:
        print(f"API error for boundary {boundary_idx}: {e}")
        return True  # Default to keeping boundary if API fails

def batch_validate_boundaries(sentences: List[str], boundaries: List[int], 
                            batch_size: int = 16) -> List[bool]:
    """Validate multiple boundaries in batches to reduce API costs."""
    results = []
    
    for i in range(0, len(boundaries), batch_size):
        batch = boundaries[i:i + batch_size]
        batch_results = []
        
        for boundary in batch:
            result = validate_boundary_with_llm(sentences, boundary)
            batch_results.append(result)
            time.sleep(0.1)  # Rate limiting
        
        results.extend(batch_results)
        print(f"Processed batch {i//batch_size + 1}/{(len(boundaries) + batch_size - 1)//batch_size}")
    
    return results

def filter_ctu_boundaries(sentences: List[str], raw_boundaries: List[int]) -> List[int]:
    """Filter CTU boundaries using LLM validation."""
    print(f"Validating {len(raw_boundaries)} boundaries with LLM...")
    
    # Validate boundaries
    validations = batch_validate_boundaries(sentences, raw_boundaries)
    
    # Keep only validated boundaries
    filtered_boundaries = [boundary for boundary, is_valid in zip(raw_boundaries, validations) if is_valid]
    
    print(f"Kept {len(filtered_boundaries)}/{len(raw_boundaries)} boundaries after LLM validation")
    return filtered_boundaries

def main():
    """CLI interface for boundary filtering."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Filter CTU boundaries using LLM validation")
    parser.add_argument("--input", required=True, help="JSON file with sentences and boundaries")
    parser.add_argument("--output", required=True, help="Output file for filtered boundaries")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size for API calls")
    
    args = parser.parse_args()
    
    # Load input data
    with open(args.input, 'r') as f:
        data = json.load(f)
    
    sentences = data['sentences']
    raw_boundaries = data['boundaries']
    
    # Filter boundaries
    filtered_boundaries = filter_ctu_boundaries(sentences, raw_boundaries)
    
    # Save results
    output_data = {
        'sentences': sentences,
        'original_boundaries': raw_boundaries,
        'filtered_boundaries': filtered_boundaries
    }
    
    with open(args.output, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"Results saved to {args.output}")

if __name__ == "__main__":
    main() 