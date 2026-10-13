"""Invariants of the label-audit draw (src/build_audit_batches.py).

Synthetic frames only: no network, no data files. The things locked down are the
ones a silent mistake would corrupt -- which points count as disagreements, that
the stratified draw reports honest inclusion probabilities, and that nothing
which would tell the reader the answer reaches the app.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

import build_audit_batches as bab  # noqa: E402
import build_label_batches as blb  # noqa: E402

CANNOT = bab.CANNOT


# ---------------------------------------------------------------------------
# round1_disagreements
# ---------------------------------------------------------------------------
def test_round1_disagreements():
    wide = pd.DataFrame(
        {"e1": ["A -> A", "A -> A", CANNOT, CANNOT, "A -> A", np.nan],
         "e2": ["B -> B", "A -> A", "A -> A", CANNOT, np.nan, "A -> A"]},
        index=["differ", "agree", "one_cannot", "both_cannot", "only_e1", "only_e2"])
    got = set(bab.round1_disagreements(wide))
    assert got == {"differ", "one_cannot"}


# ---------------------------------------------------------------------------
# round1_readings
# ---------------------------------------------------------------------------
def _export(rows):
    return pd.DataFrame(rows, columns=["campaign", "batch_id", "point_id",
                                       "expert_id", "labeller", "labelled_at",
                                       "transition"])


def test_round1_readings_newest_wins_blank_is_cannot_and_filters():
    c = "recover-habloss"
    df = _export([
        (c, "cov001", "p1", "e1", "Dina", "2026-10-01T10:00:00", "A -> A"),
        (c, "cov001", "p1", "e1", "Dina", "2026-10-02T10:00:00", "B -> B"),  # newer
        (c, "cov001", "p1", "e2", "Oy", "2026-10-01T11:00:00", "B -> B"),
        (c, "rar001", "p2", "e1", "Dina", "2026-10-01T10:00:00", ""),        # blank
        (c, "rar001", "p2", "e2", "Oy", "2026-10-01T10:00:00", "C -> C"),
        (c, "b001", "p3", "e1", "Dina", "2026-10-01T10:00:00", "A -> A"),    # other batch
        (c, "cov001", "p4", "e3", "Zander", "2026-10-01T10:00:00", "A -> A"),  # other expert
    ])
    wide = bab.round1_readings(df)
    assert list(wide.columns) == ["e1", "e2"]
    assert set(wide.index) == {"p1", "p2"}
    assert wide.at["p1", "e1"] == "B -> B"        # the newest reading, not the first
    assert wide.at["p1", "e2"] == "B -> B"
    assert wide.at["p2", "e1"] == CANNOT
    assert wide.at["p2", "e2"] == "C -> C"
    assert list(bab.round1_disagreements(wide)) == ["p2"]


def test_round1_readings_missing_transition_is_cannot():
    df = _export([("c", "cov001", "p1", "e1", "D", "2026-10-01", None),
                  ("c", "cov001", "p1", "e2", "O", "2026-10-01", "A -> A")])
    assert bab.round1_readings(df).at["p1", "e1"] == CANNOT


# ---------------------------------------------------------------------------
# orig_conflicts
# ---------------------------------------------------------------------------
def _orig(rows):
    df = pd.DataFrame(rows, columns=["PLOTID", "lc_2018", "lc_2024"])
    df["t"] = [bab.transition(a, b) for a, b in zip(df.lc_2018, df.lc_2024)]
    return df


def test_orig_conflicts():
    orig = _orig([
        (1, "Nature - forest", "Nature - forest"),   # dup, same coarse: not a conflict
        (1, "Nature - other", "Nature - other"),
        (2, "Nature - forest", "Nature - forest"),   # dup, differ coarsely: conflict
        (2, "Nature - forest", "Artificial"),
        (3, "Cropland", "Cropland"),                  # singleton
        (4, "Artificial", "Artificial"),              # singleton
    ])
    assert bab.transition("Nature - forest", "Nature - other") == "Nature -> Nature"
    got = bab.orig_conflicts(orig)
    assert set(got.PLOTID) == {2}
    assert len(got) == 2


def test_orig_conflicts_none():
    assert bab.orig_conflicts(_orig([(1, "Cropland", "Cropland"),
                                     (2, "Artificial", "Artificial")])).empty


# ---------------------------------------------------------------------------
# stratified
# ---------------------------------------------------------------------------
def _pool():
    sizes = {"big": 200, "mid": 60, "small": 5}
    rows = [(f"{k}{i}", k) for k, n in sizes.items() for i in range(n)]
    return pd.DataFrame(rows, columns=["PLOTID", "s"]), sizes


def test_stratified_total_floor_and_inclusion():
    frame, sizes = _pool()
    out = bab.stratified(frame, 50, "s", 10, np.random.default_rng(0))
    assert len(out) == 50
    assert out.PLOTID.is_unique
    drawn = out.groupby("s").size()
    for s, size in sizes.items():
        assert drawn[s] >= min(10, size)
        assert out.loc[out.s == s, "incl_prob"].nunique() == 1
        assert out.loc[out.s == s, "incl_prob"].iloc[0] == pytest.approx(drawn[s] / size)
    # the stratum smaller than the floor is taken whole
    assert drawn["small"] == 5
    assert (out.loc[out.s == "small", "incl_prob"] == 1.0).all()


def test_stratified_deterministic():
    frame, _ = _pool()
    a = bab.stratified(frame, 50, "s", 10, np.random.default_rng(7))
    b = bab.stratified(frame, 50, "s", 10, np.random.default_rng(7))
    assert list(a.PLOTID) == list(b.PLOTID)


def test_stratified_floor_too_large():
    frame, _ = _pool()
    with pytest.raises(SystemExit):
        bab.stratified(frame, 20, "s", 10, np.random.default_rng(0))  # 3 x 10 > 20


# ---------------------------------------------------------------------------
# uncertainty_score
# ---------------------------------------------------------------------------
def test_uncertainty_score_is_scale_free():
    rng = np.random.default_rng(1)
    f = pd.DataFrame({"entropy": rng.random(30), "mutual_info": rng.random(30),
                      "p_assigned": rng.random(30)})
    base = bab.uncertainty_score(f, ["entropy", "mutual_info"])
    g = f.assign(mutual_info=f.mutual_info * 100)
    assert (bab.uncertainty_score(g, ["entropy", "mutual_info"]) == base).all()


def test_uncertainty_score_one_minus_p_assigned():
    f = pd.DataFrame({"p_assigned": [0.9, 0.1, 0.5]})
    s = bab.uncertainty_score(f, ["1-p_assigned"])
    assert s.idxmax() == 1 and s.idxmin() == 0


# ---------------------------------------------------------------------------
# batch_frame -- the blinding guarantee
# ---------------------------------------------------------------------------
def _sup_frame():
    return pd.DataFrame({
        "id": ["orig_1", "orig_2", "b1_3"],
        "lon": [10.0, 10.1, 10.2], "lat": [60.0, 60.1, 60.2],
        "assigned": ["A -> A|B -> B", "A -> A", "C -> C"],
        "assigned_note": ["n1", "n2", "n3"],
        # everything below tells the reader something before they call it
        "stratum": ["disagreement", "uncertain", "uncertain"],
        "score": [1.0, 0.9, 0.8], "entropy": [1.0, 0.9, 0.8],
        "mutual_info": [0.1, 0.1, 0.1], "p_assigned": [0.2, 0.3, 0.4],
        "source": ["x", "y", "z"], "pool": ["orig", "orig", "b1"],
        "incl_prob": [1.0, 1.0, 1.0], "stratum_t": ["a", "b", "c"],
    })


def test_batch_frame_leaks_nothing():
    out = bab.batch_frame(_sup_frame())
    # BLINDING: only these columns may reach the app. Stratum, scores, entropy
    # and source would each tell the reader how hard / how doubtful the point is
    # before they have made their own call.
    assert list(out.columns) == ["id", "lon", "lat", "assigned", "assigned_note",
                                 "channel", "rank"]
    assert (out.channel == "audit").all()
    assert list(out["rank"]) == [1, 2, 3]


# ---------------------------------------------------------------------------
# write_batches integration
# ---------------------------------------------------------------------------
def test_write_batches_audit_payload(tmp_path):
    frame = bab.batch_frame(_sup_frame())
    entries = blb.write_batches(
        frame, campaign="recover-habloss", channel="audit", size=100,
        prefix="aud_sup", outdir=tmp_path, instructions="x",
        feedback="immediate", stage="audit", experts=["e3", "e4"],
        double_frac=1.0, audit=True)
    assert entries[0]["audit"] is True
    payload = json.loads((tmp_path / "aud_sup001.json").read_text())
    assert payload["audit"] is True
    assert payload["stage"] == "audit"
    pts = payload["points"]
    assert len(pts) == 3
    for p in pts:
        assert sorted(p["required_readers"]) == ["e3", "e4"]
        # blinding: the label on file is first-class, never rendered via meta
        assert "assigned" not in p["meta"]
        assert "assigned_note" not in p["meta"]
    two = next(p for p in pts if p["id"] == "orig_1")
    assert two["assigned"] == ["A -> A", "B -> B"]
    assert isinstance(two["assigned"], list) and len(two["assigned"]) == 2
