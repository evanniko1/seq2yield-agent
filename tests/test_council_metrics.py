"""RESEARCH_AGENDA II-6 (cost per accepted claim) + II-7 (reviewer discrimination/calibration)."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from agents import council_metrics as M  # noqa: E402
from orchestration import store  # noqa: E402


def test_cost_per_claim(tmp_path):
    con = store.connect(tmp_path / "db.sqlite")
    con.execute("INSERT INTO model_call VALUES (?,?,?,?,?,?,?,?,?)",
                ("t", "T1", "chair", "anthropic", "claude", 100, 50, 0.20, 1))
    con.execute("INSERT INTO claim (run_id,status,claim) VALUES ('R1','accepted','cnn wins')")
    con.execute("INSERT INTO claim (run_id,status,claim) VALUES ('R2','rejected',NULL)")
    con.commit()
    c = M.cost_per_claim(con)
    assert c["n_accepted_claims"] == 1 and c["cost_per_accepted_claim"] == 0.20
    assert c["by_status"]["accepted"] == 1 and c["by_status"]["rejected"] == 1


def test_reviewer_discrimination_gap():
    cycles = [
        {"outcome": "accepted", "reviews": [
            {"role": "sharp", "score_feasibility": 5, "score_scientific_value": 5,
             "score_confoundedness": 5, "score_reproducibility": 5},
            {"role": "flat", "score_feasibility": 3, "score_scientific_value": 3,
             "score_confoundedness": 3, "score_reproducibility": 3}]},
        {"outcome": "rejected", "reviews": [
            {"role": "sharp", "score_feasibility": 1, "score_scientific_value": 2,
             "score_confoundedness": 1, "score_reproducibility": 2},
            {"role": "flat", "score_feasibility": 3, "score_scientific_value": 3,
             "score_confoundedness": 3, "score_reproducibility": 3}]},
    ]
    d = M.reviewer_discrimination(cycles)
    assert d["sharp"]["discrimination_gap"] > 0 and d["sharp"]["calibrated"]   # separates good/bad
    assert d["flat"]["discrimination_gap"] == 0 and not d["flat"]["calibrated"]  # no signal
