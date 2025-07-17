# CTU-FlowRAG 🪴

End-to-end pipeline that turns verbose welfare-scheme PDFs into bilingual, storyboard-style poster sequences optimised for low-literacy audiences.

## Quick start

```bash
# 1. create env (conda or venv)
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt  # core deps
# (optional) heavy extras: BGE sentence-transformer, FAISS, BM25, etc.
pip install -r requirements-extra.txt

# 2. set your OpenAI key if you plan to generate images or use GPT helpers
export OPENAI_API_KEY="sk-..."

# 3. run the sample pipeline
python -m src.pipeline.run_pipeline \
       --scheme-dir data/raw/schemes/25-ciss \
       --cultural-index data/culture/index_bge

# outputs → ./output/ (posters*, pipeline_results.json)
```

### Docker (lightweight runtime)

```
docker build -t ctu-flowrag -f docker/Dockerfile .
docker run --rm -v $PWD/output:/app/output ctu-flowrag \
       python scripts/run_full_pipeline.py \
              --scheme-dir data/raw/schemes/25-ciss --run-id demo
```

The Docker image installs only the **core requirements** for a slim footprint (<250 MB). Heavy extras can be added at runtime with:

```
docker run --rm -it ctu-flowrag pip install -r requirements-extra.txt
```

## Evaluation

* `scripts/eval_retriever.py` – computes nDCG@10 / MRR@10 / MAP@10 using silver-standard `data/evaluation/qrels.tsv`.
* `scripts/eval_seg_role.py` – computes Pk / WindowDiff and macro-F1 against gold JSON files and logs them.

Run `python scripts/eval_seg_role.py --run-id demo` after the pipeline to log segmentation/role scores.

---

## Repo layout

| Path                         | Purpose                                   |
| ---------------------------- | ----------------------------------------- |
| `src/ctu/`                   | Segmentation, summariser, role tagging    |
| `src/flow/storyboard.py`     | Builds precedence-graph storyboard        |
| `src/prompt/`                | Prompt synthesiser with style carry-over  |
| `src/retrieval/`             | Graph-augmented dense retriever (BGE + CDGE) |
| `src/metrics/`               | IR metrics (nDCG, MAP, MRR, C-FlowGain)   |
| `src/image_gen/`             | DALL-E / SDXL wrappers + downloader       |
| `tests/`                     | 14-test suite covering full pipeline      |

For full details see `IMPLEMENTATION_STATUS.md`.

