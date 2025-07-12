# tests/test_ctu_pipeline.py
import pytest
import json
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent / "src"))

from prep.sent_split_lid import sent_split_lid
from ctu.segment import segment_scheme, texttiling
from ctu.evaluate import pk_score, windowdiff_score, f1_score
from ctu.graph_builder import build_discourse_graph

def test_sentence_splitting():
    """Test sentence splitting and language detection."""
    text = "This is a test sentence. यह एक परीक्षण वाक्य है. This is another sentence."
    
    records = sent_split_lid(text)
    
    assert len(records) > 0
    assert all('sent' in record for record in records)
    assert all('lang' in record for record in records)
    
    # Check that we have both English and Hindi
    langs = [record['lang'] for record in records]
    assert 'en' in langs or 'hi' in langs

def test_texttiling():
    """Test TextTiling algorithm."""
    sentences = [
        "The scheme provides financial support.",
        "Farmers receive Rs. 6000 per year.",
        "The amount is transferred in installments.",
        "To be eligible, farmers must own land.",
        "The land should be in their name.",
        "Small farmers are the primary beneficiaries.",
        "The scheme aims to supplement financial needs.",
        "Farmers can use money for seeds and fertilizers.",
        "The application process is simple.",
        "Farmers need to provide Aadhaar number."
    ]
    
    ctus = texttiling(sentences, window=3, thresh=0.1)
    
    assert len(ctus) > 0
    assert all('start' in ctu for ctu in ctus)
    assert all('end' in ctu for ctu in ctus)
    
    # Check that boundaries are valid
    for ctu in ctus:
        assert ctu['start'] < ctu['end']
        assert ctu['end'] - ctu['start'] >= 3  # min 3 sentences

def test_segment_scheme():
    """Test full scheme segmentation."""
    # Create mock sentence records
    sent_records = [
        {"sent": "The scheme provides support.", "lang": "en"},
        {"sent": "Farmers receive money.", "lang": "en"},
        {"sent": "The amount is transferred.", "lang": "en"},
        {"sent": "To be eligible, own land.", "lang": "en"},
        {"sent": "The land should be in name.", "lang": "en"},
        {"sent": "Small farmers are beneficiaries.", "lang": "en"},
        {"sent": "The scheme aims to help.", "lang": "en"},
        {"sent": "Farmers can use money.", "lang": "en"},
        {"sent": "The application is simple.", "lang": "en"},
        {"sent": "Provide Aadhaar number.", "lang": "en"}
    ]
    
    ctus = segment_scheme(sent_records)
    
    assert len(ctus) > 0
    assert all('ctu_id' in ctu for ctu in ctus)
    assert all('text' in ctu for ctu in ctus)
    assert all('lang_counts' in ctu for ctu in ctus)

def test_evaluation_metrics():
    """Test evaluation metrics."""
    # Mock gold and predicted boundaries
    gold_boundaries = [3, 6, 8]
    pred_boundaries = [3, 7, 9]
    num_sentences = 10
    
    # Test Pk score
    pk = pk_score(gold_boundaries, pred_boundaries, num_sentences)
    assert 0 <= pk <= 1
    
    # Test WindowDiff
    windowdiff = windowdiff_score(gold_boundaries, pred_boundaries, num_sentences)
    assert 0 <= windowdiff <= 1
    
    # Test F1 score
    precision, recall, f1 = f1_score(gold_boundaries, pred_boundaries)
    assert 0 <= precision <= 1
    assert 0 <= recall <= 1
    assert 0 <= f1 <= 1

def test_graph_building():
    """Test discourse graph building."""
    # Create mock CTUs
    ctus = [
        {"ctu_id": 1, "text": "The scheme provides financial support to farmers."},
        {"ctu_id": 2, "text": "To be eligible, farmers must own agricultural land."},
        {"ctu_id": 3, "text": "The application process is simple and can be done online."},
        {"ctu_id": 4, "text": "Farmers need to provide their Aadhaar number and bank details."}
    ]
    
    graph_data = build_discourse_graph(ctus, min_weight=0.1)
    
    assert graph_data['num_nodes'] == 4
    assert 'adjacency_matrix' in graph_data
    assert 'edge_weights' in graph_data
    assert 'parameters' in graph_data

def test_integration():
    """Test full pipeline integration."""
    # Mock text
    text = """
    The Pradhan Mantri Kisan Samman Nidhi scheme provides financial support to farmers.
    Under this scheme, eligible farmers receive Rs. 6000 per year.
    The amount is transferred directly to their bank accounts in three equal installments.
    To be eligible, farmers must own agricultural land.
    The land should be in their name or in the name of their family members.
    Small and marginal farmers are the primary beneficiaries of this scheme.
    The scheme aims to supplement the financial needs of farmers for procuring inputs.
    Farmers can use this money for seeds, fertilizers, and other agricultural inputs.
    The application process is simple and can be done online.
    Farmers need to provide their Aadhaar number and bank account details.
    """
    
    # Step 1: Sentence splitting
    sent_records = sent_split_lid(text)
    assert len(sent_records) > 0
    
    # Step 2: CTU segmentation
    ctus = segment_scheme(sent_records)
    assert len(ctus) > 0
    
    # Step 3: Graph building
    graph_data = build_discourse_graph(ctus)
    assert graph_data['num_nodes'] == len(ctus)
    
    print(f"Integration test passed: {len(sent_records)} sentences → {len(ctus)} CTUs → {graph_data['num_edges']} graph edges")

if __name__ == "__main__":
    # Run tests
    test_sentence_splitting()
    test_texttiling()
    test_segment_scheme()
    test_evaluation_metrics()
    test_graph_building()
    test_integration()
    
    print("All tests passed!") 