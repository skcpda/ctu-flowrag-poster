# Implementation Status – July 2025

| Component | Status |
|-----------|--------|
| CTU segmentation + LLM shrink (≤6 sentences) | ✅ merged |
| Role classifier (LLM + SVM) calibrated | ✅ confidence stored |
| Storyboard builder & pipeline wiring | ✅ build_storyboard + prompts |
| Style carry-over across posters | ✅ style_prefix cached |
| Retrieval fusion (graph-augmented BGE + optional BM25) | ✅ dense index + fusion |
| IR metrics util (nDCG, MRR, MAP) | ✅ `src/metrics/retrieval.py` |
| C-FlowGain composite metric | ➡ planned Q3 |
| Docker & GitHub Actions CI | 🚧 (CI workflow added, docker pending) | 