"""G2 — joint multiple-comparison correction. The claim registry AND tournament headline tests are
corrected as ONE BH-FDR family (previously each in isolation → under-corrected)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from seq2yield.statistics import multiple_comparisons as MC  # noqa: E402


def _seed(cd: Path):
    cd.mkdir(parents=True, exist_ok=True)
    (cd / "registry.jsonl").write_text("\n".join(json.dumps(r) for r in [
        {"run_id": "C1", "p_value": 0.001, "status": "accepted"},
        {"run_id": "C2", "p_value": 0.04, "status": "accepted"},
        {"run_id": "C3", "p_value": 0.9, "status": "inconclusive"},
    ]))
    (cd / "tournaments.jsonl").write_text("\n".join(json.dumps(t) for t in [
        {"run_id": "T1", "winner_significant": True,
         "leaderboard": [{"model": "cnn", "rank": 1}, {"model": "rf", "rank": 2, "p_value": 0.002}]},
        {"run_id": "T2", "winner_significant": False,
         "leaderboard": [{"model": "rf", "rank": 1}, {"model": "ridge", "rank": 2, "p_value": 0.3}]},
    ]))


def test_family_combines_claims_and_tournaments(tmp_path):
    _seed(tmp_path)
    fam = MC.gather_family(tmp_path)
    sources = sorted(f["source"] for f in fam)
    assert sources.count("claim") == 3 and sources.count("tournament") == 2


def test_joint_fdr_corrects_across_both_sources(tmp_path):
    _seed(tmp_path)
    res = MC.correct_all(tmp_path)
    assert res["n_comparisons"] == 5 and res["by_source"] == {"claim": 3, "tournament": 2}
    surv = {a["id"] for a in res["items"] if a["survives_correction"]}
    assert "C1" in surv and "T1" in surv          # strong tests survive the joint correction
    assert "C3" not in surv and "T2" not in surv  # weak ones do not
    # the joint family is larger than either alone -> a claim near the edge can drop out
    assert res["n_after_correction"] <= res["n_raw_discoveries"]
