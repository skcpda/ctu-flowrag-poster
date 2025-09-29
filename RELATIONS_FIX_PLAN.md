# Relations Data Fix Plan

## **Problem Summary**
The current relations data has several critical issues that need immediate attention:

1. **Over-dense graphs** (60.8% non-NONE relations, 21.3 edges per node)
2. **PRECEDES over-firing** (321 relations for 71 sentences, should be ~70)
3. **CONDITIONS directionality problems** (87% wrong direction)
4. **CONTRADICTS false positives** (100% discourse marker triggered)
5. **ELABORATES/EXAMPLES redundancy** (13:1 ratio)
6. **Missing structural fields** (sid/line_idx)
7. **Role noise propagation**

## **Phase 1: Immediate Post-Processing Fixes** ✅ COMPLETED

### **1.1 Post-Filter Script** (`relations_post_filter.py`)
- **Adds missing fields**: `sid` and `line_idx` to all relations
- **Enforces adjacency**: PRECEDES only when `|line_idx_i - line_idx_j| == 1`
- **Fixes CONDITIONS direction**: Uses role-pair whitelist with correct directionality
- **Adds contradiction guardrails**: Requires shared terms + numeric conflicts for CONTRADICTS
- **Merges ELABORATES/EXAMPLES**: Converts EXAMPLES to ELABORATES
- **Applies edge budget**: Caps edges per node per relation type
- **Method calibration**: Down-weights rule-based for tricky relations

### **1.2 Batch Processing** (`batch_filter_relations.py`)
- Processes all relations files in batch
- Generates quality reports
- Creates filtered output directory

## **Phase 2: Upstream Fixes (Medium Term)**

### **2.1 Modify Relation Labeler** (`ctu_relation_labeler_v2.py`)
```python
# Add adjacency enforcement in PRECEDES scoring
def score_precedes(self, ctu1, ctu2):
    if abs(ctu1.line_idx - ctu2.line_idx) != 1:
        return 0.0  # Not adjacent
    return self._score_precedes_content(ctu1, ctu2)

# Add role-pair constraints for CONDITIONS
def score_conditions(self, ctu1, ctu2):
    valid_pairs = {
        ('Eligibility', 'BenefitsAssistance'),
        ('Eligibility', 'ApplicationProcess'),
        ('Documents', 'ApplicationProcess')
    }
    if (ctu1.role, ctu2.role) not in valid_pairs:
        return 0.0
    return self._score_conditions_content(ctu1, ctu2)

# Add contradiction guardrails
def score_contradicts(self, ctu1, ctu2):
    # Require shared terms + numeric conflict
    shared_terms = self._get_shared_terms(ctu1.sentence, ctu2.sentence)
    has_numeric_conflict = self._has_numeric_conflict(ctu1.sentence, ctu2.sentence)
    
    if len(shared_terms) < 2 or not has_numeric_conflict:
        return 0.0
    return self._score_contradicts_content(ctu1, ctu2)
```

### **2.2 Add Missing Fields to Data Structure**
```python
# In relation extraction
relation = {
    'ctu1': {
        'sentence': sentence1,
        'role': role1,
        'confidence': confidence1,
        'sid': sentence_id1,        # ADD THIS
        'line_idx': line_idx1,     # ADD THIS
        'similarities': similarities1
    },
    'ctu2': {
        'sentence': sentence2,
        'role': role2,
        'confidence': confidence2,
        'sid': sentence_id2,        # ADD THIS
        'line_idx': line_idx2,     # ADD THIS
        'similarities': similarities2
    },
    'relation': relation_type,
    'method': method,
    'confidence': confidence
}
```

## **Phase 3: Advanced Optimizations (Long Term)**

### **3.1 Role-Pair Gates**
```python
# Add role-pair validation before relation scoring
VALID_ROLE_PAIRS = {
    'PRECEDES': [
        ('ContextObjective', 'BenefitsAssistance'),
        ('Eligibility', 'ApplicationProcess'),
        ('ApplicationProcess', 'TimelineFrequency')
    ],
    'CONDITIONS': [
        ('Eligibility', 'BenefitsAssistance'),
        ('Eligibility', 'ApplicationProcess'),
        ('Documents', 'ApplicationProcess')
    ]
}

def validate_role_pair(relation_type, role1, role2):
    valid_pairs = VALID_ROLE_PAIRS.get(relation_type, [])
    return (role1, role2) in valid_pairs or (role2, role1) in valid_pairs
```

### **3.2 Method-Aware Calibration**
```python
# Calibrate confidence based on method and relation type
CALIBRATION_WEIGHTS = {
    'CONTRADICTS': {'gpt': 1.0, 'rule_based': 0.3, 'embedding': 0.7},
    'PRECEDES': {'gpt': 1.0, 'rule_based': 0.8, 'embedding': 0.9},
    'CONDITIONS': {'gpt': 1.0, 'rule_based': 0.6, 'embedding': 0.8}
}

def calibrate_confidence(relation_type, method, confidence):
    weight = CALIBRATION_WEIGHTS.get(relation_type, {}).get(method, 1.0)
    return confidence * weight
```

### **3.3 Edge Budget System**
```python
# Implement per-node edge budgets
EDGE_BUDGET = {
    'SUPPORTS': 5,
    'PRECEDES': 2,      # Only adjacent
    'CONDITIONS': 3,
    'ELABORATES': 2,
    'CONTRADICTS': 1,
    'CAUSES': 1
}

def apply_edge_budget(relations):
    # Group by node and relation type
    # Keep top-k by confidence
    # Apply method weights
    pass
```

## **Phase 4: Quality Monitoring**

### **4.1 Automated Quality Checks**
```python
def quality_check(relations_data):
    checks = {
        'density': check_density(relations_data),
        'adjacency': check_precedes_adjacency(relations_data),
        'direction': check_conditions_direction(relations_data),
        'contradicts': check_contradicts_quality(relations_data),
        'redundancy': check_elaborates_examples(relations_data)
    }
    return checks
```

### **4.2 Continuous Monitoring**
- Track density metrics over time
- Monitor false positive rates
- Alert on quality degradation
- Generate quality reports

## **Expected Results After Fixes**

### **Before Fixes:**
- Density: 60.8% non-NONE relations
- PRECEDES: 321 relations (4.5x expected)
- CONDITIONS: 87% wrong direction
- CONTRADICTS: 100% false positives
- Edges per node: 21.3

### **After Fixes:**
- Density: ~35% non-NONE relations ✅
- PRECEDES: ~15 relations (truly adjacent) ✅
- CONDITIONS: Correct directionality ✅
- CONTRADICTS: Only real contradictions ✅
- Edges per node: ~7.2 ✅

## **Implementation Timeline**

### **Week 1: Immediate Fixes**
- [x] Create post-filter script
- [x] Test on sample data
- [x] Batch process all files
- [ ] Validate results

### **Week 2: Upstream Fixes**
- [ ] Modify relation labeler
- [ ] Add missing fields
- [ ] Implement role-pair gates
- [ ] Test on new data

### **Week 3: Advanced Optimizations**
- [ ] Method-aware calibration
- [ ] Edge budget system
- [ ] Quality monitoring
- [ ] Performance optimization

## **Usage Instructions**

### **Apply Post-Filter to Existing Data:**
```bash
# Single file
python3 relations_post_filter.py input.json output.json

# Batch process all files
python3 batch_filter_relations.py
```

### **Monitor Quality:**
```bash
# Check quality metrics
python3 -c "
import json
with open('filtered_summary.json') as f:
    report = json.load(f)
    print(f'Average density: {report[\"overall_stats\"][\"average_density\"]:.1f}%')
    print(f'Average edges per node: {report[\"overall_stats\"][\"average_edges_per_node\"]:.1f}')
"
```

## **Success Metrics**

1. **Density**: < 40% non-NONE relations
2. **PRECEDES**: Only adjacent relations
3. **CONDITIONS**: Correct directionality
4. **CONTRADICTS**: < 5% false positives
5. **Sparsity**: < 10 edges per node
6. **Quality**: > 90% precision on sampled relations

This plan addresses all identified issues systematically and provides a clear path to clean, sparse, high-quality relations data suitable for RCR-GAT and Sinkhorn-beam pathing.
