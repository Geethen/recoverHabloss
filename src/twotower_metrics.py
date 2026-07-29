"""Per-class and per-regime metrics for the two-tower merged2 predictions.

``twotower_lab`` was built to answer one question -- binary change-F1 -- and
every idea in the ledger was judged on it. Two error modes the deployed maps
actually show are invisible to that number:

*Stable Artificial read as stable Vegetation.* Both are "no change", so the
binary gate scores them identically; but on the maps a built-up block that comes
back as stable nature is the most visible failure there is. Measured here as
``art_stable_recall`` and ``art_stable_as_veg`` (the exact confusion), plus
``macro_f1`` over the four merged2 transitions so a gain is not allowed to come
from the 71%-prevalent stable-Vegetation class alone.

*Omission where Tessera fires.* On the deploy read Tessera is present for 36% of
plots, and a fusion that costs change recall exactly there is a fusion that will
under-call change over any AOI with full Tessera coverage (Oslo) while looking
fine over one without it (Johannesburg). ``change_recall_tess`` /
``change_recall_notess`` split the deploy read on the availability mask so that
trade is visible in the ledger rather than only in a map.

Every function takes label arrays, never a model, so it applies equally to a
fresh fit and to the cached out-of-fold probabilities of an idea run days ago.
"""
from __future__ import annotations

import numpy as np

from model_zoo import is_change_label

# Readable ledger-column stems for the merged2 transitions. Anything outside the
# map (the rare pool, or a future legend) falls back to a generated slug.
CLASS_SLUGS = {
    "Artificial -> Artificial": "art_stable",
    "Vegetation -> Vegetation": "veg_stable",
    "Vegetation -> Artificial": "veg_to_art",
    "Artificial -> Vegetation": "art_to_veg",
}
ART_STABLE = "Artificial -> Artificial"
VEG_STABLE = "Vegetation -> Vegetation"


def class_slug(label: str) -> str:
    return CLASS_SLUGS.get(str(label),
                           str(label).lower().replace(" -> ", "_to_").replace(" ", "_"))


def prf(t: np.ndarray, p: np.ndarray) -> tuple[float, float, float]:
    """Precision / recall / F1 from two boolean masks."""
    tp = float((t & p).sum())
    prec = tp / max(float(p.sum()), 1.0)
    rec = tp / max(float(t.sum()), 1.0)
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return prec, rec, f1


def binary_change(truth: np.ndarray, pred: np.ndarray) -> dict:
    """Binary change P/R/F1 -- same definition as ``twotower_lab.change_metrics``."""
    t = np.array([is_change_label(x) for x in truth])
    p = np.array([is_change_label(x) for x in pred])
    prec, rec, f1 = prf(t, p)
    return {"change_f1": f1, "change_precision": prec, "change_recall": rec}


def per_class(truth: np.ndarray, pred: np.ndarray, classes: list) -> dict:
    """Per-merged2-class P/R/F1, macro-F1, and the stable-Artificial error mode.

    ``macro_f1`` is the unweighted mean over the classes actually present in the
    truth, so it cannot be inflated by the dominant stable-Vegetation class the
    way accuracy or the binary gate can.
    """
    out, f1s = {}, []
    for c in classes:
        present = int((truth == c).sum())
        if not present:
            continue
        prec, rec, f1 = prf(truth == c, pred == c)
        slug = class_slug(c)
        out[f"prec_{slug}"] = prec
        out[f"recall_{slug}"] = rec
        out[f"f1_{slug}"] = f1
        f1s.append(f1)
    out["macro_f1"] = float(np.mean(f1s)) if f1s else 0.0

    art = truth == ART_STABLE
    if art.any():
        out["art_stable_recall"] = float((pred[art] == ART_STABLE).mean())
        # The user-reported failure: built-up that comes back as stable nature.
        out["art_stable_as_veg"] = float((pred[art] == VEG_STABLE).mean())
        # ... and the reverse leak, so a "fix" that just floods Artificial shows.
        veg = truth == VEG_STABLE
        out["veg_stable_as_art"] = float((pred[veg] == ART_STABLE).mean()) if veg.any() else np.nan
    return out


def by_tessera(truth: np.ndarray, pred: np.ndarray, tess_present: np.ndarray) -> dict:
    """Change metrics split on Tessera availability -- the omission diagnostic.

    NaN on a read where the split is degenerate (the covered subset, where every
    row has Tessera), so the columns stay comparable across reads.
    """
    out = {}
    for tag, mask in (("tess", tess_present.astype(bool)),
                      ("notess", ~tess_present.astype(bool))):
        if mask.sum() < 20:
            out[f"change_f1_{tag}"] = np.nan
            out[f"change_recall_{tag}"] = np.nan
            continue
        m = binary_change(truth[mask], pred[mask])
        out[f"change_f1_{tag}"] = m["change_f1"]
        out[f"change_recall_{tag}"] = m["change_recall"]
    # Positive = Tessera rows are recalled better than the rest. Negative is the
    # symptom the deployed Oslo map shows: the fusion omits where it fires.
    out["tess_recall_gap"] = out["change_recall_tess"] - out["change_recall_notess"]
    return out


def extended_metrics(truth: np.ndarray, pred: np.ndarray, classes: list,
                     tess_present: np.ndarray | None = None) -> dict:
    """Every non-threshold metric for one label vector."""
    out = dict(binary_change(truth, pred))
    out.update(per_class(truth, pred, classes))
    if tess_present is not None:
        out.update(by_tessera(truth, pred, tess_present))
    return out
