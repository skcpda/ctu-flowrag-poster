## Results and Artifacts

This repository consolidates all evaluation and paper artifacts under `results/`:

- `results/paper/`
  - Canonical paper tables/CSVs and the zipped bundle `paper_results.zip`.
- `results/evaluation/`
  - Outputs produced by evaluation runs (e.g., JSON/CSV summaries, LaTeX tables). Previously top-level `evaluation_results/` is now here at `results/evaluation/evaluation_results/`.
- `results/legacy_root_exports/`
  - Legacy CSV exports that were previously in the repo root (e.g., `main_results.csv`, `ablation_results.csv`). These are preserved for reference but excluded from version control via `.gitignore` to avoid duplication with the canonical results above.

### Notes
- Prefer using files in `results/paper/` and `results/evaluation/` for any analysis or reporting. The `legacy_root_exports/` content is retained purely for traceability.
- If a script references an old root-level CSV path, update it to the equivalent file in `results/paper/` or `results/evaluation/`.


