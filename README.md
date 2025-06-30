<h1 align="center">
  CTU-FlowRAG-Poster
</h1>

<p align="center">
  <em>Discourse-aware, low-resource pipeline that turns mixed-language welfare-scheme documents into an ordered series of language-light posters.</em>
</p>

<p align="center">
  <a href="https://github.com/skcpda/ctu-flowrag-poster/actions">
    <img src="https://github.com/skcpda/ctu-flowrag-poster/workflows/Lint/badge.svg" alt="lint status">
  </a>
  <a href="https://opensource.org/licenses/MIT">
    <img src="https://img.shields.io/badge/license-MIT-green.svg" alt="license badge">
  </a>
</p>

---

## ✨ Project Highlights

| Module | What it Does |
|--------|--------------|
| **CTU Segmentation** | Splits long, mixed-code PDFs/TXT into *Coherent Thematic Units* using TextTiling + LLM boundary verification. |
| **Discourse Graph** | Builds a lightweight graph (nodes = CTUs, edges = semantic/temporal links). |
| **Role Tagger** | Zero-shot LLM + SVM hybrid labels each CTU (Target / Eligibility / Benefits / Procedure / Timeline / Contact). |
| **Local-Context RAG** | Retrieves culturally relevant snippets (Santali lifestyle, icons) to ground images. |
| **Prompt Synthesiser** | Templated prompt → refined with a Contextual Thompson Sampling + PRISM loop. |
| **Poster Renderer** | Calls an image-generation backend (DALL·E 3, or local Stable Diffusion) and outputs a sequential poster set with bilingual captions. |
| **Explainability Store** | Maps each poster back to its CTU text + cultural cues for auditing. |

---

## 🗂 Repository Layout

