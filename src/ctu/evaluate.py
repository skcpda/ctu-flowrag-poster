# src/ctu/evaluate.py
import numpy as np
from typing import List, Dict, Tuple
from scipy.spatial.distance import cosine

def pk_score(gold_boundaries: List[int], pred_boundaries: List[int], 
             num_sentences: int, k: float = 0.5) -> float:
    """
    Calculate Pk score for boundary detection.
    
    Args:
        gold_boundaries: List of gold standard boundary indices
        pred_boundaries: List of predicted boundary indices  
        num_sentences: Total number of sentences
        k: Window size as fraction of total sentences
    
    Returns:
        Pk score (lower is better)
    """
    window_size = int(k * num_sentences)
    
    def count_errors(boundaries1: List[int], boundaries2: List[int]) -> int:
        """Count boundary errors between two boundary sets."""
        errors = 0
        for i in range(num_sentences - window_size + 1):
            # Check if boundaries1 and boundaries2 disagree on whether
            # there's a boundary in window [i, i+window_size)
            b1_in_window = any(i <= b < i + window_size for b in boundaries1)
            b2_in_window = any(i <= b < i + window_size for b in boundaries2)
            if b1_in_window != b2_in_window:
                errors += 1
        return errors
    
    # Count errors in both directions
    errors_forward = count_errors(gold_boundaries, pred_boundaries)
    errors_backward = count_errors(pred_boundaries, gold_boundaries)
    
    # Normalize by number of possible windows
    total_windows = 2 * (num_sentences - window_size + 1)
    pk = (errors_forward + errors_backward) / total_windows
    
    return pk

def windowdiff_score(gold_boundaries: List[int], pred_boundaries: List[int], 
                    num_sentences: int, k: float = 0.5) -> float:
    """
    Calculate WindowDiff score for boundary detection.
    
    Args:
        gold_boundaries: List of gold standard boundary indices
        pred_boundaries: List of predicted boundary indices
        num_sentences: Total number of sentences  
        k: Window size as fraction of total sentences
    
    Returns:
        WindowDiff score (lower is better)
    """
    window_size = int(k * num_sentences)
    
    def count_boundaries_in_window(boundaries: List[int], start: int, end: int) -> int:
        """Count boundaries in window [start, end)."""
        return sum(1 for b in boundaries if start <= b < end)
    
    total_diff = 0
    for i in range(num_sentences - window_size + 1):
        gold_count = count_boundaries_in_window(gold_boundaries, i, i + window_size)
        pred_count = count_boundaries_in_window(pred_boundaries, i, i + window_size)
        total_diff += abs(gold_count - pred_count)
    
    # Normalize by number of windows
    num_windows = num_sentences - window_size + 1
    windowdiff = total_diff / num_windows
    
    return windowdiff

def f1_score(gold_boundaries: List[int], pred_boundaries: List[int], 
             tolerance: int = 1) -> Tuple[float, float, float]:
    """
    Calculate F1 score for boundary detection with tolerance.
    
    Args:
        gold_boundaries: List of gold standard boundary indices
        pred_boundaries: List of predicted boundary indices
        tolerance: Number of sentences tolerance for matching boundaries
    
    Returns:
        Tuple of (precision, recall, f1)
    """
    def is_close(b1: int, b2: int, tol: int) -> bool:
        return abs(b1 - b2) <= tol
    
    # Count true positives
    tp = 0
    matched_gold = set()
    matched_pred = set()
    
    for i, gold_b in enumerate(gold_boundaries):
        for j, pred_b in enumerate(pred_boundaries):
            if is_close(gold_b, pred_b, tolerance):
                if i not in matched_gold and j not in matched_pred:
                    tp += 1
                    matched_gold.add(i)
                    matched_pred.add(j)
    
    # Calculate precision and recall
    precision = tp / len(pred_boundaries) if pred_boundaries else 0.0
    recall = tp / len(gold_boundaries) if gold_boundaries else 0.0
    
    # Calculate F1
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return precision, recall, f1

def evaluate_segmentation(gold_data: List[Dict], pred_data: List[Dict]) -> Dict[str, float]:
    """
    Evaluate segmentation performance.
    
    Args:
        gold_data: List of gold standard CTUs with 'start' and 'end' keys
        pred_data: List of predicted CTUs with 'start' and 'end' keys
    
    Returns:
        Dictionary with evaluation metrics
    """
    # Convert CTU data to boundary lists
    def ctu_to_boundaries(ctus: List[Dict]) -> List[int]:
        boundaries = []
        for ctu in ctus:
            boundaries.append(ctu['start'])
        return sorted(boundaries)
    
    gold_boundaries = ctu_to_boundaries(gold_data)
    pred_boundaries = ctu_to_boundaries(pred_data)
    
    # Get total number of sentences (assuming last CTU ends at max sentence index)
    num_sentences = max(ctu['end'] for ctu in gold_data + pred_data)
    
    # Calculate metrics
    pk = pk_score(gold_boundaries, pred_boundaries, num_sentences)
    windowdiff = windowdiff_score(gold_boundaries, pred_boundaries, num_sentences)
    precision, recall, f1 = f1_score(gold_boundaries, pred_boundaries)
    
    return {
        'pk': pk,
        'windowdiff': windowdiff, 
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'num_gold_boundaries': len(gold_boundaries),
        'num_pred_boundaries': len(pred_boundaries)
    }

def main():
    """CLI interface for evaluation."""
    import argparse
    import json
    
    parser = argparse.ArgumentParser(description="Evaluate CTU segmentation")
    parser.add_argument("--gold", required=True, help="Gold standard JSON file")
    parser.add_argument("--pred", required=True, help="Predicted JSON file")
    parser.add_argument("--output", help="Output file for results")
    
    args = parser.parse_args()
    
    # Load data
    with open(args.gold, 'r') as f:
        gold_data = json.load(f)
    
    with open(args.pred, 'r') as f:
        pred_data = json.load(f)
    
    # Evaluate
    results = evaluate_segmentation(gold_data, pred_data)
    
    # Print results
    print("Segmentation Evaluation Results:")
    print(f"Pk Score: {results['pk']:.4f}")
    print(f"WindowDiff: {results['windowdiff']:.4f}")
    print(f"Precision: {results['precision']:.4f}")
    print(f"Recall: {results['recall']:.4f}")
    print(f"F1 Score: {results['f1']:.4f}")
    print(f"Gold boundaries: {results['num_gold_boundaries']}")
    print(f"Predicted boundaries: {results['num_pred_boundaries']}")
    
    # Save results if output specified
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {args.output}")

if __name__ == "__main__":
    main() 