"""Design-family grouping and group-aware holdouts (audit SA-1 / ML-03).

A pooled library's members are often not independent: SNV pairs, motif-perturbation sets, tiling
ladders and mutational series all put near-copies of a sequence on both sides of a random split.
These tests pin the three pieces of the fix: deriving families, holding whole families out, and
MEASURING the near-duplicate exposure so the gate can see what the exact-duplicate check cannot.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from seq2yield.data import families  # noqa: E402
from seq2yield.data.cleaning import SEQ_COL, TARGET_COL  # noqa: E402
from seq2yield.experiments import pooled_runner as PR  # noqa: E402


def _parent(rng, n=40):
    return "".join(rng.choice(list("ACGT"), n))


def _mutate(seq, k):
    s = list(seq)
    s[k] = "A" if s[k] != "A" else "C"
    return "".join(s)


def _library(n_parents=4, n_children=5, seed=0):
    """A designed library: each parent plus single mutants of it — one family per parent."""
    rng = np.random.default_rng(seed)
    rows = []
    for p in range(n_parents):
        parent = _parent(rng)
        rows.append({SEQ_COL: parent, TARGET_COL: float(p), "parent": p})
        for k in range(n_children):
            rows.append({SEQ_COL: _mutate(parent, k), TARGET_COL: float(p) + 0.1 * k, "parent": p})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------- deriving families
def test_components_group_a_parent_with_its_mutants():
    df = _library()
    labels = families.components(df[SEQ_COL], max_dist=2)
    assert len(set(labels)) == 4                                  # one family per parent
    for p, grp in df.assign(f=labels).groupby("parent"):
        assert grp["f"].nunique() == 1


def test_components_leave_unrelated_sequences_alone():
    rng = np.random.default_rng(7)
    seqs = [_parent(rng) for _ in range(30)]
    assert len(set(families.components(seqs, max_dist=2))) == 30   # a random library has no families


def test_components_are_transitive_along_a_tiling_ladder():
    """A ladder of single-step neighbours is one family even though its ends are far apart — which
    is the point: neither end is independent of the middle."""
    base = "A" * 30
    ladder = [base] + ["C" * k + "A" * (30 - k) for k in range(1, 8)]
    labels = families.components(ladder, max_dist=1)
    assert len(set(labels)) == 1
    assert (ladder[0] != ladder[-1]) and sum(a != b for a, b in zip(ladder[0], ladder[-1])) == 7


def test_constant_flanks_do_not_collapse_everything_into_one_family():
    """DREAM's 110-mers carry ~30 constant positions. If those were kept, one pigeonhole block
    could be entirely constant and every sequence would land in a single bucket."""
    rng = np.random.default_rng(3)
    flank = "G" * 30
    seqs = [flank + _parent(rng, 30) + flank for _ in range(25)]
    assert len(set(families.components(seqs, max_dist=2))) == 25


def test_max_dist_zero_is_exact_deduplication():
    labels = families.components(["AAAA", "AAAA", "CCCC"], max_dist=0)
    assert labels[0] == labels[1] != labels[2]


def test_components_require_equal_lengths():
    with pytest.raises(ValueError):
        families.components(["AAAA", "AAAAA"], max_dist=1)


def test_family_labels_prefers_a_declared_column_over_derivation():
    df = _library()
    spec = type("S", (), {"id": "x", "family_col": "parent", "family_distance": 2})()
    labels, how = families.family_labels(df, spec, seq_col=SEQ_COL)
    assert how == "column:parent" and labels.nunique() == 4
    spec_dist = type("S", (), {"id": "x", "family_col": None, "family_distance": 2})()
    _, how_dist = families.family_labels(df, spec_dist, seq_col=SEQ_COL)
    assert how_dist == "hamming<=2"
    spec_none = type("S", (), {"id": "x", "family_col": None, "family_distance": None})()
    labels_none, how_none = families.family_labels(df, spec_none, seq_col=SEQ_COL)
    assert labels_none is None and how_none == "none"


# --------------------------------------------------------------- measuring the exposure
def test_near_duplicate_fraction_sees_what_the_exact_check_misses():
    from seq2yield.diagnostics import signals as S
    df = _library(n_parents=3, n_children=3)
    # The classic failure: a slice out of every family, so each held-out row has a train sibling.
    test = df.groupby("parent").head(1)
    train = df.drop(index=test.index)
    out = S.sequence_leakage(train[SEQ_COL], test[SEQ_COL], near_dist=2)
    assert out["leak_frac"] == 0.0                                 # no exact duplicate anywhere
    assert out["near_dup_frac"] == 1.0                             # yet every test row has a sibling


def test_near_duplicate_fraction_is_zero_for_a_random_library():
    rng = np.random.default_rng(11)
    seqs = [_parent(rng) for _ in range(60)]
    out = families.near_duplicate_fraction(seqs[:40], seqs[40:], max_dist=2)
    assert out["near_dup_frac"] == 0.0 and out["n_identical"] == 0


def test_near_duplicate_fraction_reports_mixed_lengths_instead_of_guessing():
    out = families.near_duplicate_fraction(["AAAA"], ["AAAAA"], max_dist=1)
    assert out["near_dup_frac"] is None and "note" in out


def test_near_duplicate_pitfall_rule_trips_on_the_measured_signal():
    from seq2yield.diagnostics import critic
    flags = critic.evaluate({"leakage": {"leak_frac": 0.0, "near_dup_frac": 0.75}})
    ids = [f["id"] for f in flags]
    assert "near_duplicate_leakage" in ids and "train_test_leakage" not in ids
    flag = next(f for f in flags if f["id"] == "near_duplicate_leakage")
    assert flag["severity"] == "high" and flag["blocking"] is False   # advisory: regroup, don't withhold


# ------------------------------------------------------------------ the split itself
def test_grouped_holdout_never_shares_a_family_across_the_split():
    df = _library(n_parents=20, n_children=4)
    fam = pd.Series(families.components(df[SEQ_COL], max_dist=2), index=df.index)
    train, test = PR.grouped_holdout(df.assign(**{PR.FAMILY_COL: fam}), fam, frac=0.2, seed=0)
    assert not set(train[PR.FAMILY_COL]) & set(test[PR.FAMILY_COL])
    assert 0 < len(test) < len(df)
    nd = families.near_duplicate_fraction(train[SEQ_COL], test[SEQ_COL], max_dist=2)
    assert nd["near_dup_frac"] == 0.0


def test_the_old_stratified_holdout_is_what_leaks():
    """Contrast, so the regression is visible: the per-group stratified split this replaces puts a
    sibling of every held-out sequence in train."""
    df = _library(n_parents=20, n_children=4)
    df = df.rename(columns={"parent": "mut_series"})               # SERIES_COL triggers per-group
    train, test = PR.holdout_frame(df, frac=0.2, seed=0)
    assert families.near_duplicate_fraction(train[SEQ_COL], test[SEQ_COL], max_dist=2)["near_dup_frac"] == 1.0


def test_grouped_holdout_is_deterministic_and_roughly_the_right_size():
    df = _library(n_parents=30, n_children=3)
    fam = pd.Series(families.components(df[SEQ_COL], max_dist=2), index=df.index)
    a = PR.grouped_holdout(df, fam, frac=0.2, seed=5)[1][SEQ_COL].tolist()
    b = PR.grouped_holdout(df, fam, frac=0.2, seed=5)[1][SEQ_COL].tolist()
    c = PR.grouped_holdout(df, fam, frac=0.2, seed=6)[1][SEQ_COL].tolist()
    assert a == b and a != c
    assert 0.15 <= len(a) / len(df) <= 0.30                        # a family is indivisible


def test_grouped_holdout_falls_back_when_one_family_is_everything():
    df = _library(n_parents=1, n_children=6)
    fam = pd.Series(["only"] * len(df), index=df.index)
    train, test = PR.grouped_holdout(df, fam, frac=0.2, seed=0)
    assert len(train) and len(test)                                 # degenerate: must not return empty


def test_holdout_frame_honours_a_family_column_and_still_prefers_a_provided_split():
    df = _library(n_parents=10, n_children=3)
    fam = pd.Series(families.components(df[SEQ_COL], max_dist=2), index=df.index)
    train, test = PR.holdout_frame(df.assign(**{PR.FAMILY_COL: fam}), frac=0.2, seed=1)
    assert not set(train[PR.FAMILY_COL]) & set(test[PR.FAMILY_COL])
    provided = df.assign(**{PR.FAMILY_COL: fam},
                         split=["train"] * (len(df) - 4) + ["test"] * 4)
    tr2, te2 = PR.holdout_frame(provided, frac=0.2, seed=1)
    assert len(te2) == 4                                            # provided split wins


# ------------------------------------------------- the two datasets this was found on
def test_the_affected_datasets_declare_their_families():
    """Guards the fix itself: if either spec silently reverts to a stratified holdout, the published
    R² goes back to measuring interpolation between siblings."""
    from seq2yield.data import datasets
    yeast, dream = datasets.spec("yeast"), datasets.spec("dream2022")
    assert yeast.split_strategy == "grouped_holdout" and yeast.family_col == "mut_series"
    assert dream.split_strategy == "grouped_holdout" and dream.family_distance == 2
    assert datasets.spec("ecoli").structure == "per_series"          # untouched: provided splits


def test_transfer_flags_a_comparison_across_two_split_protocols():
    """A 'replication' of a stratified-holdout finding by a grouped-holdout run is not comparing the
    same quantity. Old runs carry no split block at all, which must also be flagged, not assumed."""
    from seq2yield.experiments import transfer
    sig = {"mean_delta": 0.05, "ci_excludes_zero": True}
    same = transfer.concordance({**sig, "split": {"split_strategy": "grouped_holdout"}},
                                {**sig, "split": {"split_strategy": "grouped_holdout"}})
    assert "split_mismatch" not in same and same["verdict"] == "concordant"
    mixed = transfer.concordance({**sig, "split": {"split_strategy": "stratified_holdout"}},
                                 {**sig, "split": {"split_strategy": "grouped_holdout"}})
    assert mixed["split_mismatch"]["source"] == "stratified_holdout"
    legacy = transfer.concordance(sig, {**sig, "split": {"split_strategy": "grouped_holdout"}})
    assert legacy["split_mismatch"]["source"] is None


def test_real_grouped_split_closes_the_hole(require_data):
    """On the real frames: no family spans the split and the near-duplicate fraction goes to 0
    (measured 1.000 -> 0.000 on yeast, 0.749 -> 0.000 on dream2022)."""
    require_data("yeast")
    from types import SimpleNamespace
    spec = SimpleNamespace(dataset="yeast", seed=1)
    rep = PR.split_report(spec)
    assert rep["split_strategy"] == "grouped_holdout" and rep["families_shared"] == 0
    train, test = PR.holdout(spec)
    nd = families.near_duplicate_fraction(train[SEQ_COL], test[SEQ_COL], max_dist=2)
    assert nd["near_dup_frac"] == 0.0
