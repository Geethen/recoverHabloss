"""Audit report: verdicts, blind agreement, kappa, HT estimate. Synthetic only."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import audit_report as ar  # noqa: E402
import label_rounds as lr  # noqa: E402

NC, NA, CA = "Nature -> Cropland", "Nature -> Artificial", "Cropland -> Artificial"
CANNOT = ar.CANNOT


@pytest.mark.parametrize("f1,f2,assigned,want", [
    (NC, NC, [NC], ("confirmed", NC)),
    (NC, NC, [NC, NA], ("resolved", NC)),
    (NA, NA, [NC], ("corrected", NA)),
    (CANNOT, CANNOT, [NC], ("uninterpretable", "")),
    (CANNOT, CANNOT, [CANNOT], ("uninterpretable", "")),
    (NC, NA, [NC], ("disputed", "")),
    (NC, CANNOT, [NC], ("disputed", "")),
    (NC, None, [NC], ("disputed", "")),
    (None, None, [NC], ("disputed", "")),
])
def test_verdict(f1, f2, assigned, want):
    assert ar.verdict(f1, f2, assigned) == want


def test_blank_final_is_cannot_interpret():
    assert ar.norm_call("") == CANNOT and ar.norm_call(np.nan) == CANNOT
    assert ar.norm_call(" " + NC) == NC


def _rows(spec):
    """spec: (point, expert, blind, final, assigned, reason)."""
    d = pd.DataFrame(spec, columns=["point_id", "expert_id", "blind_transition",
                                    "transition", "assigned", "mismatch_reason"])
    d["campaign"] = "c"
    d["batch_id"] = "aud_sup001"
    d["labelled_at"] = "2026-10-13T10:00:00Z"
    return ar.audit_rows(d).assign(set="sup")


def test_blind_agreement_two_assigned():
    rows = _rows([("p1", "e3", NC, NC, f"{NC}|{NA}", ""),
                  ("p2", "e3", NA, NA, f"{NC}|{NA}", ""),
                  ("p3", "e3", CA, CA, f"{NC}|{NA}", "ambiguous"),
                  ("p4", "e3", NC, NC, NA, "assigned_wrong")])
    got = ar.blind_agreement(rows).iloc[0]
    assert got["n"] == 4 and got["agree"] == 0.5
    # only single-assigned points feed the confusion pairs
    assert ar.top_confusions(rows) == [(NA, NC, 1)]


def test_dedupe_keeps_last_call():
    d = pd.DataFrame({"campaign": "c", "batch_id": "aud_sup001", "point_id": "p",
                      "expert_id": "e3", "blind_transition": [NC, NC],
                      "transition": [NC, NA], "assigned": NC, "mismatch_reason": "",
                      "labelled_at": ["2026-10-13T10:00", "2026-10-13T11:00"]})
    assert ar.audit_rows(d)["final"].tolist() == [NA]


def test_kappa_hand_case():
    a = ["x", "x", "y", "y"]
    b = ["x", "y", "y", "y"]
    # po = .75; pe = .5*.25 + .5*.75 = .5 -> kappa .5
    assert ar.cohen_kappa(a, b) == pytest.approx(0.5)
    assert ar.cohen_kappa(a, a) == pytest.approx(1.0)


def test_kappa_degenerate_does_not_crash():
    assert np.isnan(ar.cohen_kappa(["x"] * 5, ["x"] * 5))
    assert np.isnan(ar.cohen_kappa([], []))


def test_reason_of():
    assert ar.reason_of(NC, "", [NC]) == ""
    assert ar.reason_of(NA, None, [NC]) == "not_given"
    assert ar.reason_of(NA, "ambiguous", [NC]) == "ambiguous"


def _sets():
    return pd.DataFrame({
        "id": ["a", "b", "c", "d", "e"], "set": ["sup", "sup", "sup", "sup", "reg"],
        "stratum": "disagreement", "stratum_t": None, "incl_prob": 1.0,
        "assigned": [NC, f"{NC}|{NA}", NC, NC, NC]})


def test_per_point_verdicts_and_missing_reader():
    rows = _rows([
        ("a", "e3", NC, NC, NC, ""), ("a", "e4", NC, NC, NC, ""),
        ("b", "e3", NA, NA, f"{NC}|{NA}", ""), ("b", "e4", NA, NA, f"{NC}|{NA}", ""),
        ("c", "e3", NA, NA, NC, "assigned_wrong"), ("c", "e4", NA, NA, NC, "assigned_wrong"),
        ("d", "e3", NC, NC, NC, ""),  # e4 missing
    ])
    pts = ar.per_point(rows, _sets()).set_index("point_id")
    assert pts.loc["a", "verdict"] == "confirmed"
    assert pts.loc["b", "verdict"] == "resolved" and pts.loc["b", "final_label"] == NA
    assert pts.loc["c", "verdict"] == "corrected" and pts.loc["c", "final_label"] == NA
    assert pts.loc["d", "verdict"] == "disputed" and pts.loc["d", "final_label"] == ""
    assert "e" not in pts.index  # no readings, not in the output


def test_uninterpretable_point():
    rows = _rows([("a", "e3", CANNOT, "", NC, ""), ("a", "e4", CANNOT, np.nan, NC, "")])
    pts = ar.per_point(rows, _sets())
    assert pts.iloc[0]["verdict"] == "uninterpretable"


def test_reason_counts_and_revision():
    rows = _rows([("a", "e3", NA, NC, NC, "revised"),   # moved to file after reveal
                  ("b", "e3", NC, NC, NC, ""),
                  ("c", "e3", NA, NA, NC, "")])
    c = ar.reason_counts(rows).loc["e3"]
    assert c["revised"] == 1 and c["match"] == 1 and c["not_given"] == 1


def test_ht_hand_case_unequal_weights():
    # pi = 1 (w=1) corrected; pi = .25 (w=4) confirmed; pi=.5 (w=2) disputed
    pts = pd.DataFrame({"verdict": ["corrected", "confirmed", "disputed"],
                        "incl_prob": [1.0, 0.25, 0.5],
                        "stratum_t": "s"})
    assert ar.error_rate(pts) == pytest.approx(1 / 7)
    assert ar.error_rate(pts, 0.5) == pytest.approx(2 / 7)
    lo, hi = ar.bootstrap_ci(pts, n_boot=200, seed=1)
    assert 0 <= lo <= hi <= 1
    assert ar.bootstrap_ci(pts, n_boot=200, seed=1) == (lo, hi)  # seeded


def test_split_audit():
    f = pd.DataFrame({"batch_id": ["b001", "aud_reg001", "aud_sup002", "cov001"],
                      "point_id": list("wxyz")})
    keep, aud = lr.split_audit(f)
    assert keep["point_id"].tolist() == ["w", "z"]
    assert aud["point_id"].tolist() == ["x", "y"]
