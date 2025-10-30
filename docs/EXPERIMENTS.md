## Experiments and Reproducibility

Experiment-related materials are consolidated under `experiments/`:

- `experiments/training/`
  - Training entrypoints used throughout development and ablations (e.g., `working_training.py`, `production_training.py`, `simple_training.py`).
  - Tests closely tied to these flows live in `tests/integration/`.

- `experiments/cluster/`
  - SLURM `.sbatch` job scripts and cluster helpers (e.g., `train_rcrgat_gpu_long.sbatch`, `setup_cluster_env.sh`).

- `experiments/streamlined_pipeline/`
  - A self-contained pipeline and scripts for demos, analysis, and paper table generation originally under `streamlined_pipeline/`.

- Root of `experiments/`
  - Evaluation harness and commands: `evaluation_framework.py`, `run_comprehensive_evaluation.py`, `evaluation_commands.sh`.

### Running
- Demos and quick starts: see `examples/` (e.g., `quick_start.sh`, `path_search_demo.py`).
- Evaluation: use `experiments/run_comprehensive_evaluation.py` and consult `experiments/evaluation_commands.sh`.
- Cluster jobs: submit `.sbatch` scripts from `experiments/cluster/` after running `setup_cluster_env.sh`.

### Migration Notes
- If any script referenced old top-level paths, update them to the new locations shown above.
- The Python package code remains in `ctu_flowrag/` and should be imported the same way as before.


