from pathlib import Path
import urllib.request
import fasttext
import spacy

# ---------- 1. Resolve repo root ----------
REPO_ROOT = Path(__file__).resolve().parents[2]

# ---------- 2. FastText LID model (.ftz, 916 KB) ----------
FT_MODEL_PATH = REPO_ROOT / "models" / "lid.176.ftz"
FT_URL = (
    "https://dl.fbaipublicfiles.com/fasttext/"
    "supervised-models/lid.176.ftz"
)

if not FT_MODEL_PATH.exists():
    print("Downloading fastText LID model …")
    FT_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(FT_URL, FT_MODEL_PATH)

_FT = fasttext.load_model(str(FT_MODEL_PATH))

# ---------- 3. spaCy sentence splitter ----------
_NLP = spacy.load("xx_ent_wiki_sm", disable=["ner", "tagger", "parser"])
# Add sentencizer for sentence boundary detection
if "sentencizer" not in _NLP.pipe_names:
    _NLP.add_pipe("sentencizer")

# ---------- 4. Helpers ----------
def detect_lang(text: str) -> str:
    """Return ISO-639 language code (e.g. 'en', 'hi')."""
    label, _ = _FT.predict(text.replace("\n", " ")[:200], k=1)
    return label[0].split("__")[-1]


def sent_split_lid(raw_text: str) -> list[dict]:
    """Split text into sentences + detect language."""
    doc = _NLP(raw_text)
    records = []
    for s in doc.sents:
        txt = s.text.strip()
        if txt:
            records.append({"sent": txt, "lang": detect_lang(txt)})
    return records
