# CTU Relation Generation Pipeline

**Production-ready pipeline for generating RCR-GAT/CSRA training data from government scheme descriptions.**

## Overview

This streamlined pipeline converts raw government scheme descriptions into structured relation graphs suitable for training Role-Conditioned Relational Graph Attention Networks (RCR-GAT) and Contextualized Semantic Retrieval Agents (CSRA).

## Pipeline Steps

1. **CTU Extraction** (`ctu_extractor.py`) - Extract Content Thematic Units from scheme descriptions
2. **Role Tagging** (`role_tagger.py`) - Assign semantic roles to CTUs using fine-tuned BGE model
3. **Relation Generation** (`relation_generator.py`) - Generate structural and semantic relations

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run complete pipeline
python scripts/run_pipeline.py
```

## Directory Structure

```
streamlined_pipeline/
├── input_data/
│   └── scheme_descriptions/          # Input: GPT-4o mini descriptions
├── output_data/
│   ├── ctu_extracted/                # Step 1 output: Extracted CTUs
│   ├── ctu_role_tagged/              # Step 2 output: Role-tagged CTUs
│   └── ctu_relations_production_ready/ # Step 3 output: Final relations
├── config/
│   └── fine_tuned_bge_ctu_relations/ # Fine-tuned BGE model
├── scripts/
│   ├── ctu_extractor.py             # Step 1: Extract CTUs
│   ├── role_tagger.py               # Step 2: Tag roles
│   ├── relation_generator.py        # Step 3: Generate relations
│   └── run_pipeline.py              # Complete pipeline runner
├── requirements.txt                  # Python dependencies
└── README.md                         # This file
```

## Input Format

**Scheme Description Files** (`input_data/scheme_descriptions/*.json`):
```json
{
  "scheme_name": "Scheme Name",
  "sentences": [
    "Sentence 1",
    "Sentence 2",
    "..."
  ],
  "timestamp": "2024-01-01T00:00:00Z",
  "model": "gpt-4o-mini"
}
```

## Output Format

**Final Relation Files** (`output_data/ctu_relations_production_ready/*.json`):
```json
{
  "scheme_name": "Scheme Name",
  "ctus": [
    {
      "sentence": "CTU text",
      "role": "Eligibility",
      "confidence": 0.85,
      "sid": 1,
      "line_idx": 0
    }
  ],
  "relations": [
    {
      "ctu1": {...},
      "ctu2": {...},
      "relation": "PRECEDES",
      "edge_confidence": 0.95,
      "edge_logit": 2.94,
      "method": "production_pipeline"
    }
  ],
  "adjacency_completeness": {
    "complete": true
  },
  "production_pipeline": {
    "ready_for_rcr_gat_csra": true
  }
}
```

## Key Features

### ✅ Production Ready
- **Essential semantic edges**: PREREQUISITE_OF, ENABLES, CAP_LIMITS, ADMINISTERED_BY
- **Structural backbone**: Complete PRECEDES chains
- **Confidence calibration**: Method-aware confidence scoring
- **Edge validation**: Allow-map filtering for quality control

### ✅ RCR-GAT/CSRA Compatible
- **Role-conditioned relations**: 7 semantic roles with proper edge types
- **Confidence scores**: [0,1] range with logit conversion
- **Adjacency completeness**: Verified structural chains
- **Production metadata**: Ready for model training

### ✅ Quality Metrics
- **Semantic ratio**: 30-40% semantic edges
- **Role distribution**: Balanced across 7 roles
- **Edge validation**: Only allowed role-relation-role triplets
- **Confidence calibration**: Method-aware scoring

## Usage Examples

### Run Individual Steps
```bash
# Step 1: Extract CTUs
python scripts/ctu_extractor.py

# Step 2: Tag roles
python scripts/role_tagger.py

# Step 3: Generate relations
python scripts/relation_generator.py
```

### Monitor Progress
```bash
# Check extraction progress
ls output_data/ctu_extracted/ | wc -l

# Check role tagging progress
ls output_data/ctu_role_tagged/ | wc -l

# Check final relations
ls output_data/ctu_relations_production_ready/ | wc -l
```

## Configuration

### Model Path
Update the model path in `role_tagger.py` if using a different fine-tuned model:
```python
tagger = RoleTagger("path/to/your/model")
```

### Edge Allow-Map
Modify `ALLOWED_EDGES` in `relation_generator.py` to add/remove allowed relation types.

### Confidence Calibration
Adjust `METHOD_CALIBRATION` in `relation_generator.py` for different confidence scoring.

## Troubleshooting

### Common Issues

1. **"No CTUs found"**
   - Check input file format
   - Ensure sentences array is not empty

2. **"Model not found"**
   - Verify model path in config/
   - Pipeline will fallback to base BGE model

3. **Low semantic ratio**
   - Check role tagging quality
   - Verify pattern matching in relation generator

### Performance Tips

- **Batch processing**: Pipeline processes files in batches
- **Memory usage**: Large schemes may require more RAM
- **Model loading**: Fine-tuned model loads once per pipeline run

## Output Quality

### Expected Metrics
- **Total relations**: ~2-3x number of CTUs
- **Semantic ratio**: 30-40%
- **Adjacency complete**: 100% of files
- **Role distribution**: Balanced across 7 roles

### Quality Checks
- All edges have confidence scores
- All edges pass allow-map validation
- Structural chains are complete
- Semantic edges are meaningful

## Next Steps

After running the pipeline:

1. **Verify output quality** using the generated summary files
2. **Train RCR-GAT** using the relation graphs
3. **Deploy CSRA** with the structured data
4. **Monitor performance** using the confidence scores

## Support

For issues or questions:
1. Check the generated summary files for error details
2. Verify input file format matches expected schema
3. Ensure all dependencies are installed correctly

---

**Ready for RCR-GAT/CSRA training! 🚀**
