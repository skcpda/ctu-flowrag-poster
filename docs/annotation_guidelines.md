# CTU Boundary Annotation Guidelines

This document outlines the process and best-practices for creating a **gold-standard** dataset of Coherent Text Units (CTUs) in welfare-scheme documents.

---
## 📋 Annotation Checklist

- [ ] Import scheme text into Doccano (longDescription only)
- [ ] Read full document once before annotating
- [ ] Insert a boundary **after the last sentence that still belongs to the current coherent topic**
- [ ] Typical CTU length: 3 – 15 sentences
- [ ] Label the CTU with its **primary role** *(optional but recommended)*
  - target_pop, eligibility, benefits, procedure, timeline, contact, misc
- [ ] Ensure every sentence is in **exactly one** CTU
- [ ] Save and export as JSONL (Doccano default)

## 🔧 Running Doccano via Docker

```bash
# 1. Start Doccano
docker run -d --name doccano -p 8000:8000 doccano/doccano:latest

# 2. Create super-user (first time only)
docker exec -it doccano doccano createuser \
  --username annotator --password secret --email a@b.com

# 3. Open http://localhost:8000  → login
```

## 🚀 Exporting Annotations

Use the helper script:
```bash
python scripts/doccano_export.py \
  --project-id 1 \
  --doccano-url http://localhost:8000 \
  --username annotator --password secret \
  --output data/annotations/ctu_gold.json
```

The exported file is compatible with `src/ctu/evaluate.py`.

---
## 📝 Tips
* Break long numeric/bullet lists into separate CTUs if the topic changes.
* Ignore minor grammatical errors; focus on semantic coherence.
* If unsure, **merge rather than split** – evaluation tolerates slight over-segmentation.

Happy annotating! 🎉 