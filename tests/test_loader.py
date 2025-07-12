# tests/test_loader.py
from src.io.load_scheme import load_scheme

def test_loader_ok(tmp_path):
    # create a mock scheme folder
    folder = tmp_path / "mock"
    folder.mkdir()
    for stem in [
        "schemeName",
        "shortDescription",
        "description",
        "longDescription",
    ]:
        (folder / f"{stem}.txt").write_text("dummy")

    data = load_scheme(folder)
    assert len(data) == 4
    assert set(data) == {
        "schemeName",
        "shortDescription",
        "description",
        "longDescription",
    }
