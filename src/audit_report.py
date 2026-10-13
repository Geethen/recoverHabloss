"""Read the label-audit round: blind agreement with the file, and the verdicts.

Audit batches (`aud_sup*` read by e3/e4, `aud_reg*` read by e1/e2) re-read points
that already carry labels. Each sheet row carries `assigned` (the label(s) on
file, '|'-separated), `blind_transition` (the reader's first call, made before
`assigned` was shown) and `mismatch_reason`. `transition` is the FINAL call.

Verdict per point, from the two readers' FINAL calls:

* both equal and in `assigned`      -> confirmed (resolved if two were on file)
* both equal, not in `assigned`     -> corrected (final_label = that call)
* both 'cannot interpret'           -> uninterpretable
* anything else (differ / missing)  -> disputed (final_label '')

Printed: (a) blind agreement with the file per set x expert and the top
confusions; (b) inter-reader blind agreement and Cohen's kappa; (c) verdicts per
set x stratum; (d) a Horvitz-Thompson (ratio) estimate of the frame's label error
rate from the reader set, with a stratified percentile bootstrap; (e) mismatch
reasons and how many blind calls were revised after the reveal.

Usage::

    G=/home/geethen.singh/.pixi/envs/geo/bin/python
    $G src/audit_report.py --readings export.csv      # or --url <exec-url>

Refuses to overwrite its output.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

import label_rounds as lr
from project_paths import project_data_dir

AUDIT_DIR = project_data_dir("label_audit")
DEFAULT_SETS = AUDIT_DIR / "audit_sets_20261013_nature4.csv"
CANNOT = "cannot interpret"
REASONS = ("assigned_wrong", "ambiguous", "revised", "not_given")
#: Who reads which set; the first is reader 1 in the output.
SET_READERS = {"sup": ("e3", "e4"), "reg": ("e1", "e2")}


def parse_assigned(value) -> list[str]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    return [a.strip() for a in str(value).split("|") if a.strip()]


def norm_call(value) -> str | None:
    """A call as text; '' / NaN (uninterpretable final) -> CANNOT."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return CANNOT
    s = str(value).strip()
    return s if s and s.lower() != "nan" else CANNOT


def verdict(final1, final2, assigned: list[str]) -> tuple[str, str]:
    """(verdict, final_label). A reader's final of None means not yet read."""
    if final1 is None or final2 is None or final1 != final2:
        return "disputed", ""
    if final1 == CANNOT:
        return "uninterpretable", ""
    if final1 in assigned:
        return ("resolved" if len(assigned) > 1 else "confirmed"), final1
    return "corrected", final1


def reason_of(final, reason, assigned: list[str]) -> str:
    """The reason as reported: '' stays '' when the call matches the file."""
    r = "" if reason is None or (isinstance(reason, float) and np.isnan(reason)) \
        else str(reason)
    if r:
        return r
    return "" if final in assigned else "not_given"


def cohen_kappa(a, b) -> float:
    a, b = list(a), list(b)
    n = len(a)
    if n == 0:
        return float("nan")
    po = sum(x == y for x, y in zip(a, b)) / n
    ca, cb = Counter(a), Counter(b)
    pe = sum(ca[k] * cb.get(k, 0) for k in ca) / n ** 2
    if pe >= 1.0:  # both readers used one class throughout: kappa undefined
        return float("nan")
    return (po - pe) / (1 - pe)


def audit_rows(readings: pd.DataFrame) -> pd.DataFrame:
    """Deduped audit rows only, with normalised blind/final calls."""
    f = readings.loc[readings["batch_id"].astype(str).str.startswith("aud_")].copy()
    if f.empty:
        return f
    f = lr.dedupe(f)
    who = "expert_id" if "expert_id" in f.columns else "labeller"
    f["expert"] = f[who].astype(str)
    f["final"] = f["transition"].map(norm_call)
    f["blind"] = f["blind_transition"].map(norm_call)
    f["assigned_list"] = f["assigned"].map(parse_assigned)
    return f


def per_point(rows: pd.DataFrame, sets: pd.DataFrame) -> pd.DataFrame:
    """One row per point of the draw that has at least one reading."""
    out = []
    by_point = {k: g for k, g in rows.groupby("point_id")}
    for s in sets.itertuples(index=False):
        g = by_point.get(s.id)
        if g is None:
            continue
        assigned = parse_assigned(s.assigned)
        readers = SET_READERS.get(s.set, ())
        rec = {"point_id": s.id, "set": s.set, "stratum": s.stratum,
               "stratum_t": s.stratum_t if isinstance(s.stratum_t, str) else s.stratum,
               "incl_prob": s.incl_prob, "assigned": " | ".join(assigned)}
        finals = []
        for i, e in enumerate(readers, 1):
            r = g.loc[g["expert"] == e]
            if r.empty:
                rec.update({f"r{i}_expert": e, f"r{i}_blind": None,
                            f"r{i}_final": None, f"r{i}_reason": None})
                finals.append(None)
                continue
            r = r.iloc[-1]
            rec.update({f"r{i}_expert": e, f"r{i}_blind": r["blind"],
                        f"r{i}_final": r["final"],
                        f"r{i}_reason": reason_of(r["final"], r.get("mismatch_reason"),
                                                  assigned)})
            finals.append(r["final"])
        rec["verdict"], rec["final_label"] = verdict(finals[0], finals[1], assigned) \
            if len(finals) == 2 else ("disputed", "")
        out.append(rec)
    return pd.DataFrame(out)


def blind_agreement(rows: pd.DataFrame) -> pd.DataFrame:
    """Per set x expert: n, share of blind calls inside `assigned`."""
    rows = rows.assign(hit=[b in a for b, a in zip(rows["blind"], rows["assigned_list"])])
    return (rows.groupby(["set", "expert"])["hit"].agg(n="size", agree="mean")
            .reset_index())


def top_confusions(rows: pd.DataFrame, k: int = 5) -> list[tuple[str, str, int]]:
    c = Counter((a[0], b) for a, b in zip(rows["assigned_list"], rows["blind"])
                if len(a) == 1 and b != a[0])
    return [(a, b, n) for (a, b), n in c.most_common(k)]


def inter_reader(points: pd.DataFrame) -> tuple[int, float, float]:
    both = points.dropna(subset=["r1_blind", "r2_blind"])
    if both.empty:
        return 0, float("nan"), float("nan")
    agree = float((both["r1_blind"] == both["r2_blind"]).mean())
    return len(both), agree, cohen_kappa(both["r1_blind"], both["r2_blind"])


def error_rate(points: pd.DataFrame, disputed_weight: float = 0.0) -> float:
    """Ratio form of the Horvitz-Thompson estimate: sum(w*e)/sum(w), w=1/pi."""
    if points.empty:
        return float("nan")
    w = 1.0 / points["incl_prob"].to_numpy(float)
    e = (points["verdict"] == "corrected").to_numpy(float) \
        + disputed_weight * (points["verdict"] == "disputed").to_numpy(float)
    return float((w * e).sum() / w.sum())


def bootstrap_ci(points: pd.DataFrame, disputed_weight: float = 0.0,
                 n_boot: int = 2000, seed: int = 0) -> tuple[float, float]:
    """Percentile CI, resampling points within stratum_t."""
    rng = np.random.default_rng(seed)
    groups = [g.reset_index(drop=True) for _, g in points.groupby("stratum_t")]
    if not groups:
        return float("nan"), float("nan")
    w = [1.0 / g["incl_prob"].to_numpy(float) for g in groups]
    e = [(g["verdict"] == "corrected").to_numpy(float)
         + disputed_weight * (g["verdict"] == "disputed").to_numpy(float)
         for g in groups]
    est = np.empty(n_boot)
    for b in range(n_boot):
        num = den = 0.0
        for wi, ei in zip(w, e):
            idx = rng.integers(0, len(wi), len(wi))
            num += (wi[idx] * ei[idx]).sum()
            den += wi[idx].sum()
        est[b] = num / den
    lo, hi = np.percentile(est, [2.5, 97.5])
    return float(lo), float(hi)


def reason_counts(rows: pd.DataFrame) -> pd.DataFrame:
    def one(r):
        if r["blind"] != r["final"] and r["final"] in r["assigned_list"]:
            return "revised"
        return reason_of(r["final"], r.get("mismatch_reason"), r["assigned_list"]) or "match"
    rows = rows.assign(reason=rows.apply(one, axis=1))
    return rows.groupby(["expert", "reason"]).size().unstack(fill_value=0)


def print_report(rows: pd.DataFrame, points: pd.DataFrame) -> None:
    print(f"\n{len(rows)} audit readings, {len(points)} points with a reading")
    print("\n(a) blind agreement with the label on file")
    ba = blind_agreement(rows)
    for r in ba.itertuples():
        print(f"  {r.set:<4}{r.expert:<5} n={r.n:<4} blind agrees {r.agree:.1%}")
        sub = rows[(rows["set"] == r.set) & (rows["expert"] == r.expert)]
        for a, b, n in top_confusions(sub):
            print(f"        {a}  ->  {b}: {n}")
    print("\n(b) inter-reader blind agreement")
    for s, g in points.groupby("set"):
        n, ag, k = inter_reader(g)
        print(f"  {s:<4} n={n:<4} agree {ag:.1%}  kappa {k:.3f}")
    print("\n(c) verdicts per set x stratum")
    print(points.groupby(["set", "stratum", "verdict"]).size()
          .unstack(fill_value=0).to_string())
    print("\n(d) reader set: estimated label error rate (HT, w = 1/incl_prob)")
    reg = points[(points["set"] == "reg")
                 & points["r1_final"].notna() & points["r2_final"].notna()]
    if reg.empty:
        print("  no fully read reader-set points yet")
    else:
        for name, dw in (("corrected only", 0.0), ("disputed counted 0.5", 0.5)):
            est = error_rate(reg, dw)
            lo, hi = bootstrap_ci(reg, dw)
            print(f"  {name:<22} {est:.1%}  (95% CI {lo:.1%} - {hi:.1%}, n={len(reg)})")
    print("\n(e) mismatch reasons per expert (match = on file, unrevised)")
    print(reason_counts(rows).to_string())
    rev = rows[rows["blind"] != rows["final"]]
    print(f"  blind calls revised after the reveal: {len(rev)} of {len(rows)}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--readings", type=Path, help="sheet export CSV")
    ap.add_argument("--url", help="Apps Script /exec URL")
    ap.add_argument("--sets", type=Path, default=DEFAULT_SETS)
    ap.add_argument("--out", type=Path,
                    default=AUDIT_DIR / f"audit_result_{date.today():%Y%m%d}.csv")
    args = ap.parse_args()
    if not args.readings and not args.url:
        ap.error("pass --readings or --url")
    if args.out.exists():
        raise SystemExit(f"{args.out} exists; refusing to overwrite")
    frames = []
    if args.readings:
        frames.append(lr.read_files([args.readings]))
    if args.url:
        frames.append(lr.fetch_sheet(args.url, token=lr.read_config_token()))
    readings = pd.concat(frames, ignore_index=True)
    rows = audit_rows(readings)
    if rows.empty:
        print("no audit rows (batch_id starting 'aud_') in the readings yet")
        return
    sets = pd.read_csv(args.sets)
    rows = rows.merge(sets[["id", "set"]].rename(columns={"id": "point_id"}),
                      on="point_id", how="inner")
    points = per_point(rows, sets)
    if points.empty:
        print("audit rows do not match any point in the draw")
        return
    print_report(rows, points)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    points.to_csv(args.out, index=False)
    print(f"\nwrote {len(points)} points -> {args.out}")


if __name__ == "__main__":
    main()
