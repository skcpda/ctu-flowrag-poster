from __future__ import annotations
"""Train an SVM role classifier on the merged CTU corpus.

Usage::
    python scripts/train_role_svm.py --input output/pipeline_results_dedup.json

Saves scikit-learn `LinearSVC` with TF-IDF 1–2-gram features to
`models/role_svm.pkl`.
"""

import argparse
import json
import pickle
from pathlib import Path
from typing import List, Dict

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_class_weight
import numpy as np


def load_ctus(path: Path) -> List[Dict]:
    data = json.loads(path.read_text())
    return data.get("ctus", data)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, default=Path("models/role_svm.pkl"))
    args = p.parse_args()

    ctus = load_ctus(args.input)
    texts = [c["text"] for c in ctus]
    labels = [c.get("role", "misc") for c in ctus]
    classes = np.array(sorted(set(labels)))
    y = np.array(labels)
    class_weights = compute_class_weight(class_weight="balanced", classes=classes, y=y)
    cw = {cls: wt for cls, wt in zip(classes.tolist(), class_weights)}

    pipe: Pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=50000, sublinear_tf=True)),
        ("clf", LinearSVC(class_weight=cw)),
    ])
    print("Training on", len(texts), "CTUs …")
    pipe.fit(texts, y)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "wb") as f:
        pickle.dump(pipe, f)
    print("✅ saved model →", args.output)


if __name__ == "__main__":
    main() 