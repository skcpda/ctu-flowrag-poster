# CTU-FlowRAG Results Checklist

This document provides a comprehensive checklist for validating that the CTU-FlowRAG model meets all targets for paper submission.

## 🎯 Core Metrics (Targets to call it "final")

### A. Path Quality (End-to-End; Main Metric Block)

**Targets:**
- **path-nDCG@10** (↑): **+7–10 pts** over strongest non-graph baseline
- **MRR@10** (↑): **+5–8 pts** over strongest non-graph baseline  
- **MAP@10** (↑): **+5–8 pts** over strongest non-graph baseline

**Baseline Performance:**
- BM25 + concat: nDCG@10 ≈ 0.65, MRR@10 ≈ 0.58, MAP@10 ≈ 0.52
- Embedding retriever + concat: nDCG@10 ≈ 0.68, MRR@10 ≈ 0.61, MAP@10 ≈ 0.55
- Vanilla GAT: nDCG@10 ≈ 0.70, MRR@10 ≈ 0.63, MAP@10 ≈ 0.57
- Relational GAT: nDCG@10 ≈ 0.72, MRR@10 ≈ 0.65, MAP@10 ≈ 0.59

**Our Target Performance:**
- **nDCG@10 ≥ 0.82** (+10 pts over best baseline)
- **MRR@10 ≥ 0.75** (+10 pts over best baseline)
- **MAP@10 ≥ 0.68** (+9 pts over best baseline)

### B. Role Coverage & Capacity Fit

**Targets:**
- **Role coverage** (requested roles filled per template): **≥ 95%**
- **Capacity violations** (rows > κ in CSRA): **≤ 0.5%** (soft with tolerance)
- **Per-role Recall@10** (slot recall): **≥ 90%** for core roles {Context, Benefits, Eligibility, Application}

### C. Graph Faithfulness & Logic

**Targets:**
- **Invalid role transitions** in final paths: **< 1%**
- **Edge faithfulness** (edges used are in frozen graph): **≥ 98%**
- **No cross-section PRECEDES** in paths: **0**

### D. Calibration & Interpretability Sanity

**Targets:**
- **Attention–confidence correlation** (Spearman ρ): **≥ 0.40**
- **Edge-type utilization** aligns with priors: ENABLES/PREREQ edges used in role-advancing steps ≥ **80%** of the time

## 📊 Baselines & Ablations (Must Include)

### Baselines (Comparative)

1. **BM25 + top-k concat** (flat)
2. **Sentence-embedding retriever** (e.g., e5/bge) + top-k concat
3. **Vanilla GAT** (no edge types, no priors; same graph)
4. **Relational GAT** (types but **no** compat or confidence priors)

**Expected Performance Deltas:**
- Our model should **beat all** baselines on all core metrics
- Minimum improvement: +5 pts nDCG, +5 pts MRR, +5 pts MAP

### Ablations (Internal)

**Expected Performance Drops:**
- **–compat** (γ=0): **−3 to −5** pts path-nDCG
- **–conf** (β=0): **−2 to −4** pts
- **–type bias** (α_t=0): **−2 to −3** pts
- **–distance** (λ=0): longer hops, **coherence drop** and small nDCG hit
- **–semantics** (structural edges only): big drop on role coverage & MRR

## 📋 Paper-Ready Artifacts (Tables & Figures)

### Table 1 — Main Results

| Model                        | path-nDCG@10 | MRR@10 | MAP@10 |
| ---------------------------- | ------------ | ------ | ------ |
| BM25 + concat                | **0.65**     | 0.58   | 0.52   |
| Embedding retriever + concat | **0.68**     | 0.61   | 0.55   |
| Vanilla GAT (no types)       | **0.70**     | 0.63   | 0.57   |
| Relational GAT (no priors)  | **0.72**     | 0.65   | 0.59   |
| **RCR-GAT + CSRA (ours)**   | **0.82**     | 0.75   | 0.68   |
| **Δ vs best baseline**       | **+0.10**    | +0.10  | +0.09  |

**Include 95% bootstrap CIs** (±) for each cell.

### Table 2 — Role Coverage & Faithfulness

| Metric                             | Value  |
| ---------------------------------- | ------ |
| Role coverage (all templates)      | ≥ 95%  |
| Capacity violations (CSRA rows)    | ≤ 0.5% |
| Edge faithfulness (in-graph)        | ≥ 98%  |
| Invalid role transitions           | < 1%   |
| No cross-section PRECEDES in paths | 0      |

### Table 3 — Ablations

| Variant         | nDCG@10 | Δ vs ours |
| --------------- | ------- | --------- |
| **Ours (full)** | 0.82    | –         |
| –compat         | 0.78    | −0.04     |
| –conf           | 0.79    | −0.03     |
| –type bias      | 0.795   | −0.025    |
| –distance       | 0.80    | −0.02     |
| –semantics      | 0.74    | −0.08     |

### Figure 1 — Per-role Recall@k

Small multiples for {Context, Benefits, Eligibility, Application} comparing **ours vs baselines**.

### Figure 2 — Attention Sanity

Histogram/violin: attention weights for semantic edges split by **compat=1 vs 0**, and by **high vs low confidence**.

### Figure 3 — Capacity Utilization (CSRA)

Row-sum distribution per role; show it respects κ and actually uses capacity (not always at max).

### Figure 4 — Qualitative Case Studies (2–3)

For each: **CTU subgraph** (colored by role, typed edges), **selected CTUs** per role, **final path**, and a short human-readable poster.

## 🔍 QA Battery (Non-Negotiables Before "Final")

### Robustness

- **Edge-drop test**: drop **10%** semantic edges at random → nDCG drop **≤ 3 pts**; drop **10%** PRECEDES edges → nDCG **≤ 2 pts**
- **Position jitter**: shuffle positions within ±1 inside a section → negligible metric change
- **Confidence threshold sweep** (min calibrated p=0.60/0.65/0.70): graceful monotonic tradeoff

### Calibration

- **ECE** for edge confidences (already calibrated) reported
- **Attention–confidence Spearman ≥ 0.40**

### Efficiency

- **Training**: wall-time, #epochs, GPU model (A100-80GB), peak VRAM
- **Inference**: per-doc latency for (encode → CSRA → path), and memory footprint
- **Complexity**: note linearity in edges for attention (sparse ops)

### Reproducibility

- **3 seeds**: report mean ± std; significance (paired bootstrap p<0.05) vs best baseline
- **Exact config & commit hash**: include YAML and git SHA in appendix
- **Deterministic toggles**: list environment variables and torch flags used

## 📦 Deliverables List (What to Send)

### 1. CSV/JSON with per-doc results

- **nDCG@10, MRR@10, MAP@10** for all baselines + ours
- **All ablations** with deltas
- Include 95% CIs, seeds, and commit SHA

### 2. Role Coverage & Faithfulness Report

- role coverage %, CSRA violations, invalid transitions, edge faithfulness %

### 3. Per-role Recall@k CSV

- k∈{1,3,5,10} for core roles

### 4. Attention Sanity Dump

- top-k edges per node with (type, compat, conf, dist, weight)
- correlation summary (Spearman, Kendall)

### 5. Robustness Sweeps

- edge-drop, position-jitter, confidence threshold; tabulate deltas

### 6. Efficiency

- training log excerpt (epochs, time, VRAM), inference latency table

### 7. Figures (PNG/SVG) and Table LaTeX Sources

- Tables 1–3, Figures 1–4

### 8. Qualitative Bundle

- 2–3 case study PDFs (graph visualization + final poster)

## 🚀 Quick Command Reference

### Main Eval (Ours vs Baselines)
```bash
python -m ctu_flowrag.eval.eval_paths \
  --config configs/rcr_gat.yaml \
  --tensor_dir ./tensors_dev \
  --weights ./ckpts/rcr_gat_best.pt \
  --templates "[[0,1,3],[0,2,3]]" \
  --topk 10 \
  --out_csv logs/main_ours.csv
```

### Ablations (Loop Over Flags)
```bash
# --no_compat / --no_conf / --no_typebias / --no_distance / --no_semantics
python -m ctu_flowrag.eval.eval_paths \
  --config configs/rcr_gat.yaml \
  --tensor_dir ./tensors_dev \
  --weights ./ckpts/rcr_gat_best.pt \
  --templates "[[0,1,3],[0,2,3]]" \
  --topk 10 \
  --no_compat \
  --out_csv logs/ablation_no_compat.csv
```

### Per-Role Recall
```bash
python -m ctu_flowrag.eval.eval_paths \
  --config configs/rcr_gat.yaml \
  --tensor_dir ./tensors_dev \
  --weights ./ckpts/rcr_gat_best.pt \
  --templates "[[0,1,3],[0,2,3]]" \
  --topk 10 \
  --per_role \
  --out_csv logs/per_role_recall.csv
```

### Robustness Tests
```bash
# Edge drop
python -m ctu_flowrag.eval.eval_paths \
  --config configs/rcr_gat.yaml \
  --tensor_dir ./tensors_dev \
  --weights ./ckpts/rcr_gat_best.pt \
  --templates "[[0,1,3],[0,2,3]]" \
  --topk 10 \
  --edge_drop 0.1 \
  --edge_types semantic \
  --out_csv logs/robustness_edge_drop.csv

# Position jitter
python -m ctu_flowrag.eval.eval_paths \
  --config configs/rcr_gat.yaml \
  --tensor_dir ./tensors_dev \
  --weights ./ckpts/rcr_gat_best.pt \
  --templates "[[0,1,3],[0,2,3]]" \
  --topk 10 \
  --pos_jitter 1 \
  --out_csv logs/robustness_pos_jitter.csv

# Confidence threshold
python -m ctu_flowrag.eval.eval_paths \
  --config configs/rcr_gat.yaml \
  --tensor_dir ./tensors_dev \
  --weights ./ckpts/rcr_gat_best.pt \
  --templates "[[0,1,3],[0,2,3]]" \
  --topk 10 \
  --min_conf 0.65 \
  --out_csv logs/robustness_conf_threshold.csv
```

### Attention Inspector
```bash
python -m ctu_flowrag.retrieval.attention_inspector \
  --doc_id aa_ctus_production_ready \
  --topk 5 \
  --out_json logs/attn_aa.json
```

### Figures/Tables Exporter
```bash
python scripts/plot_per_role.py  # → fig
python scripts/make_tables.py    # → LaTeX
```

## ✅ Final Validation Checklist

- [ ] **nDCG@10 ≥ 0.82** (+10 pts over best baseline)
- [ ] **MRR@10 ≥ 0.75** (+10 pts over best baseline)  
- [ ] **MAP@10 ≥ 0.68** (+9 pts over best baseline)
- [ ] **Role coverage ≥ 95%**
- [ ] **Capacity violations ≤ 0.5%**
- [ ] **Edge faithfulness ≥ 98%**
- [ ] **Invalid transitions < 1%**
- [ ] **Cross-section PRECEDES = 0**
- [ ] **Attention-confidence correlation ≥ 0.40**
- [ ] **Edge-type utilization ≥ 80%** for ENABLES/PREREQ
- [ ] **All baselines beaten** on all metrics
- [ ] **Ablation drops** as expected
- [ ] **Robustness tests** passed
- [ ] **All 8 deliverables** generated
- [ ] **Paper artifacts** ready (Tables 1-3, Figures 1-4)
- [ ] **3 seeds** with significance testing
- [ ] **Reproducibility** ensured (config, commit hash)

**If you hit these numbers and artifacts, we can confidently call the model "final" and paper-ready! 🎉**
