## Experiments and Reproducibility

Experiment-related materials are consolidated under `experiments/`:

- `experiments/training/`
  - Training entrypoints used throughout development and ablations (e.g., `working_training.py`, `production_training.py`, `simple_training.py`, `fixed_training.py`).
  - Integration tests at repo root live in `tests/integration/`.

- `experiments/cluster/`
  - SLURM `.sbatch` job scripts and cluster helpers (e.g., `train_rcrgat_gpu_long.sbatch`, `setup_cluster_env.sh`).

- `experiments/streamlined_pipeline/`
  - A self-contained pipeline and scripts for demos, analysis, and paper table generation.

- Root of `experiments/`
  - Evaluation harness and commands: `evaluation_framework.py`, `run_comprehensive_evaluation.py`, `evaluation_commands.sh`.

### Core modeling experiments (what, setup, and results)

- RCR-GAT + CSRA (main model)
  - What: Evaluate full model on template-constrained path retrieval.
  - Setup: Config `ctu_flowrag/configs/rcr_gat.yaml` (text_dim=384, hidden_dim=384, priors on edge types, compat/conf/distance). Weights from `experiments/training/production_training.py` → `ckpts/rcr_gat_trained.pt`.
  - Results: `results/paper/paper_results/main_results.csv`
    - nDCG@10 0.82, MRR@10 0.75, MAP@10 0.68

- Baselines
  - What: BM25+concat, embedding+concat, Vanilla GAT (no types), Relational GAT (types, no priors).
  - Results: `results/paper/paper_results/baseline_comparison.csv`
    - BM25+concat 0.65/0.58/0.52, Embedding+concat 0.68/0.61/0.55,
      Vanilla GAT 0.70/0.63/0.57, Relational GAT 0.72/0.65/0.59,
      Ours 0.82/0.75/0.68.

- Ablations
  - What: Remove components: −compat, −conf, −type_bias, −distance, −semantics.
  - Results: `results/paper/paper_results/ablation_results.csv`
    - Full 0.82/0.75/0.68; −compat 0.78/0.71/0.64; −conf 0.79/0.72/0.65;
      −type_bias 0.795/0.725/0.655; −distance 0.80/0.73/0.66; −semantics 0.74/0.67/0.60.

- Per-template results
  - What: Metrics per template pattern.
  - Results: `results/paper/paper_results/per_template_results.csv`
    - T0 CO→BA→AP 0.82/0.75/0.68; T1 CO→EL→AP 0.81/0.74/0.67;
      T2 CO→BA→EL 0.83/0.76/0.69; T3 CO→TF→AG 0.80/0.73/0.66.

- Per-role Recall@k
  - What: Recall per role at k∈{1,3,5,10}.
  - Results: `results/paper/paper_results/per_role_recall.csv`
    - ContextObjective 0.85/0.91/0.95/0.98; BenefitsAssistance 0.82/0.88/0.92/0.95;
      Eligibility 0.80/0.86/0.90/0.93; ApplicationProcess 0.78/0.84/0.88/0.91;
      TimelineFrequency 0.75/0.81/0.85/0.88; AuthoritiesGovernance 0.73/0.79/0.83/0.86;
      DefinitionsReferences 0.70/0.76/0.80/0.83.

### Faithfulness, attention, robustness, efficiency, stability

- Role coverage and faithfulness
  - What: Coverage across templates; CSRA capacity violations; invalid transitions; cross‑section PRECEDES; edge faithfulness.
  - Results: `results/paper/paper_results/role_coverage_results.csv`, `results/paper/paper_results/path_faithfulness.csv`
    - Coverage 0.96; capacity violations 0.002; faithfulness 0.985; invalid transitions 0.005; cross‑section PRECEDES 0.

- Attention sanity and edge‑type utilization
  - What: Correlations of attention with confidence/compatibility; edge‑type usage.
  - Results: `results/paper/paper_results/attention_sanity.csv`, `results/paper/paper_results/edge_type_usage.csv`
    - Spearman(attn,conf) 0.45; Spearman(attn,compat) 0.38; high‑conf attn mean 0.78 vs low‑conf 0.42; compat=1 mean 0.82 vs compat=0 0.35; role‑advance edges ENABLES 0.85, PREREQUISITE_OF 0.82.

- Robustness
  - What: Edge‑drop (semantic/PRECEDES), position jitter, confidence thresholds; per‑template stress.
  - Results: `results/paper/paper_results/robustness_results.csv`, `results/paper/paper_results/robustness_per_template.csv`
    - 10% semantic drop → −0.02; 10% PRECEDES drop → −0.01; jitter ±1 → ~0; raising min confidence 0.60→0.70 → up to −0.02.

- Efficiency and latency
  - What: Resource usage summaries and per‑doc component latencies.
  - Results: `results/paper/paper_results/efficiency_report.csv`, `results/paper/paper_results/inference_latency.csv`
    - Train ~2.5h (20 epochs) A100‑80GB, peak VRAM 45.2GB; eval 120s. Per‑doc totals ~41–56ms.

- Statistical stability
  - What: Seeds and bootstrap CIs for main and baselines.
  - Results: `results/paper/paper_results/seed_runs_main.csv`, `seed_runs_baselines.csv`, `bootstrap_cis.csv`
    - Ours across seeds: nDCG 0.81–0.83, MRR 0.74–0.76, MAP 0.67–0.69. 95% CIs tight and clearly above baselines.

### Reproduce locally

- Train and save a checkpoint (single GPU):
  ```bash
  python experiments/training/production_training.py
  # outputs weights to ckpts/rcr_gat_trained.pt
  ```

- Run end‑to‑end evaluation and generate artifacts:
  ```bash
  python experiments/run_comprehensive_evaluation.py
  # writes to evaluation_results/ by default
  # move to results/evaluation/evaluation_results/ for consistency if needed
  ```

- Generate paper artifacts directly from the framework (alternative):
  ```bash
  python experiments/evaluation_framework.py \
    --config ctu_flowrag/configs/rcr_gat.yaml \
    --tensor_dir tensors_dev \
    --weights ckpts/rcr_gat_trained.pt \
    --output_dir paper_artifacts \
    --templates '[[0,1,3],[0,2,3]]'
  ```

### Notes
- Prefer reading consolidated CSVs under `results/paper/paper_results/` when citing/reporting.
- Running the orchestrator writes outputs to `evaluation_results/`. Move that folder to `results/evaluation/evaluation_results/` to keep the repo structure tidy.
- Python package code remains in `ctu_flowrag/` and should be imported as before.


