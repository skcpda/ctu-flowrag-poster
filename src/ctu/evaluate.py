# src/ctu/evaluate.py
from __future__ import annotations

from typing import List, Tuple, Dict

__all__ = ["pk_score", "windowdiff_score", "f1_score"]


def _segment_vector(boundaries: List[int], n_sentences: int) -> List[int]:
    """Return a binary vector of length *n_sentences* where 1 marks a boundary."""
    vec = [0] * n_sentences
    for b in boundaries:
        if 0 <= b < n_sentences:
            vec[b] = 1
    return vec


def pk_score(gold: List[int], pred: List[int], n_sentences: int, k: int | None = None) -> float:
    """Compute Pk segmentation metric (Beeferman 1999).

    Simplified: default window *k* = half the average gold segment length.
    """
    if k is None:
        avg_seg = n_sentences / (len(gold) + 1)
        k = int(round(avg_seg / 2)) or 1

    gold_vec = _segment_vector(gold, n_sentences)
    pred_vec = _segment_vector(pred, n_sentences)

    disagreements = 0
    total = 0
    for i in range(n_sentences - k):
        total += 1
        same_gold = any(gold_vec[i : i + k])
        same_pred = any(pred_vec[i : i + k])
        if same_gold != same_pred:
            disagreements += 1
    return disagreements / total if total else 0.0


def windowdiff_score(gold: List[int], pred: List[int], n_sentences: int, k: int | None = None) -> float:
    """Compute WindowDiff metric (Pevzner & Hearst 2002).
    """
    if k is None:
        avg_seg = n_sentences / (len(gold) + 1)
        k = int(round(avg_seg / 2)) or 1

    gold_vec = _segment_vector(gold, n_sentences)
    pred_vec = _segment_vector(pred, n_sentences)

    wd = 0
    total = 0
    for i in range(n_sentences - k):
        total += 1
        gold_count = sum(gold_vec[i : i + k])
        pred_count = sum(pred_vec[i : i + k])
        if gold_count != pred_count:
            wd += 1
    return wd / total if total else 0.0


def f1_score(gold: List[int], pred: List[int]) -> Tuple[float, float, float]:
    """Compute precision, recall, F1 for boundary detection."""
    gold_set = set(gold)
    pred_set = set(pred)
    tp = len(gold_set & pred_set)
    fp = len(pred_set - gold_set)
    fn = len(gold_set - pred_set)

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
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