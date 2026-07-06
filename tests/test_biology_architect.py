"""C3 — the proposing Biologist. Verifies that the architecture prior matches the biological length
scale per modality (codon-width for E. coli coding, TF-motif width for yeast promoters, wider for
enhancers), that the narrowed C2 search region is centred on that scale and stays valid, that the
seeds warm-start C2, and that the priors/seeds flow through the C10 gate into a schema-valid,
biology-informed RunSpec.
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agents import biology_architect as B  # noqa: E402
from agents.council import Council  # noqa: E402
from agents.schemas import CouncilProposal  # noqa: E402
from seq2yield.models import registry as reg  # noqa: E402
from seq2yield.search import sample_config  # noqa: E402
from seq2yield.experiments.run_spec import validate_runspec  # noqa: E402


def _proposal(dataset, model="cnn"):
    return CouncilProposal(
        proposal_id="H1", title=f"{model} on {dataset}",
        scientific_hypothesis=f"A biology-matched {model} improves R² on {dataset}.",
        model_family=model, comparator_model="rf", intervention_type="model_architecture",
        dataset=dataset, maturity_tier="tier_0", scope="global", train_sizes=[500, 1000])


# ---- architecture priors match the biology ----
def test_ecoli_coding_gets_codon_scale_filters():
    ap = B.architecture_prior("ecoli")
    assert ap["config"]["kernel_sizes"] == [3, 3, 3]      # codon (3 bp)
    assert ap["scale"] == 3


def test_yeast_promoter_gets_tf_motif_scale_filters():
    ap = B.architecture_prior("yeast")
    assert ap["config"]["kernel_sizes"] == [8, 6, 4]      # TF motif (~8 bp), multi-scale


def test_enhancer_is_wider_than_coding():
    assert max(B.architecture_prior("deng_2023")["config"]["kernel_sizes"]) \
        > max(B.architecture_prior("ecoli")["config"]["kernel_sizes"])


def test_prior_is_a_valid_c1_point():
    cfg = B.architecture_prior("yeast")["config"]
    assert cfg == reg.clean_hyperparameters("cnn", cfg)   # already whitelist-valid


# ---- narrowed search region is centred on the motif scale + stays valid ----
def test_search_region_narrows_kernel_sizes_around_scale():
    region = B.search_region("ecoli", "cnn")
    ks = region["kernel_sizes"]
    assert ks["range"][0] >= 2 and ks["range"][1] <= 7    # tight around the 3 bp codon scale
    # a yeast promoter region is centred higher (TF motif ~8 bp)
    assert B.search_region("yeast", "cnn")["kernel_sizes"]["range"][1] > ks["range"][1]


def test_sampling_the_region_yields_in_region_configs():
    region = B.search_region("ecoli", "cnn")
    rng = np.random.default_rng(0)
    lo, hi = region["kernel_sizes"]["range"]
    for _ in range(15):
        c = sample_config("cnn", rng, space=region)
        assert all(lo <= k <= hi for k in c["kernel_sizes"])   # exploration honours the prior


def test_non_cnn_region_is_the_full_space():
    assert B.search_region("ecoli", "rf") == reg.search_space("rf")


# ---- seeds warm-start C2 ----
def test_seed_configs_are_valid_and_biology_led():
    seeds = B.seed_configs("yeast", "cnn")
    assert seeds and all(s == reg.clean_hyperparameters("cnn", s) for s in seeds)
    assert seeds[0]["kernel_sizes"] == [8, 6, 4]          # the prior is the first seed
    tfm = B.seed_configs("deng_2023", "transformer")[0]   # long region -> sinusoidal positions
    assert tfm["pos_encoding"] == "sinusoidal"


# ---- flows through the C10 gate into a runnable RunSpec ----
def test_biology_runspec_is_valid_and_carries_the_prior():
    c = Council(use_planner=False)
    dec = SimpleNamespace(max_runtime_minutes=20)
    spec, info = c.biology_runspec(_proposal("ecoli"), dec, execute_search=False)
    assert spec.hyperparameters["kernel_sizes"] == [3, 3, 3]
    assert spec.hyperparameters_source == "biology_prior"
    assert info["gate_action"] in ("skip", "light", "full")
    vr = validate_runspec(spec, unlocked_tier="tier_1")
    assert vr.ok, vr.errors


def test_biology_runspec_uses_search_winner_when_executed(monkeypatch):
    # stub the gated search so the winner (not the prior) becomes the RunSpec's hyperparameters
    from agents import search_gate
    winner = SimpleNamespace(best_config={"kernel_sizes": [4, 4, 4], "dropout": 0.25},
                             best_score=0.61, n_evals=8, strategy="bandit")

    def _fake_run_gated(ctx, **kw):
        return search_gate.GatedOutcome(
            decision=search_gate.GateDecision("full", "stub", 0.9, 0.1), result=winner)

    monkeypatch.setattr(search_gate, "run_gated", _fake_run_gated)
    c = Council(use_planner=False)
    spec, info = c.biology_runspec(_proposal("yeast"), SimpleNamespace(max_runtime_minutes=20),
                                   execute_search=True)
    assert spec.hyperparameters["kernel_sizes"] == [4, 4, 4]
    assert spec.hyperparameters_source == "search:bandit"
    assert info["search_best_r2"] == 0.61
