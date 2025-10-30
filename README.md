## CTU-FlowRAG

Python package and experiments for CTU Flow Retrieval-Augmented Generation.

### Repository Layout
- `ctu_flowrag/`: Core Python package (models, retrieval, data IO, training utils).
- `experiments/`: Training entrypoints, evaluation harness, cluster jobs, and the streamlined pipeline.
  - `experiments/training/`
  - `experiments/cluster/`
  - `experiments/streamlined_pipeline/`
- `examples/`: Quickstart demos and simple scripts (e.g., `quick_start.sh`, `path_search_demo.py`).
- `results/`: Consolidated results and paper artifacts.
  - `results/paper/`
  - `results/evaluation/`
  - `results/legacy_root_exports/` (kept for traceability, ignored by git)
- `docs/`: Documentation (experiment guide, results, cluster notes).
- `tests/`: Integration tests colocated at repo root; package-level tests remain under `ctu_flowrag/tests/`.

### Getting Started
1. Install requirements:
   ```bash
   pip install -e . -r requirements.txt
   ```
2. Try a demo:
   ```bash
   bash examples/quick_start.sh
   ```
3. Run evaluations:
   ```bash
   python experiments/run_comprehensive_evaluation.py
   ```

### Docs
- Experiments: see `docs/EXPERIMENTS.md`.
- Results: see `docs/RESULTS.md`.
- Additional project docs moved under `docs/`.


