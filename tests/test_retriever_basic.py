from src.retriever.dense import encode as encode_dense
from src.retriever.index import DenseIndex


def test_retriever_basic():
    texts = ["apple orange banana", "car bus train", "cat dog mouse"]
    vecs = encode_dense(texts)
    index = DenseIndex(vecs.shape[1])
    index.add(vecs)
    scores, idxs = index.search(vecs[:1], k=2)
    assert idxs.shape == (1, 2) 