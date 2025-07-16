from pathlib import Path

from src.utils.exp_logger import ExpLogger


def test_logger_columns(tmp_path: Path):
    logger = ExpLogger(root=tmp_path)
    logger.log("T1_segmentation", {"run_id": "t", "pk": 0, "windowdiff": 0, "graph_cov@6": 0})
    csv_path = tmp_path / "T1_segmentation" / "T1_segmentation.csv"
    assert csv_path.exists()
    header = csv_path.read_text().splitlines()[0]
    assert "graph_cov@6" in header
    logger.log("T3_single_ret", {"run_id": "t", "variant": "bge", "nDCG@10": 0, "MRR@10": 0, "MAP@10": 0, "salience_nDCG": 0})
    csv_path2 = tmp_path / "T3_single_ret" / "T3_single_ret.csv"
    header2 = csv_path2.read_text().splitlines()[0]
    assert "salience_nDCG" in header2 