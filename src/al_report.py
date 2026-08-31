"""Read `al_lab_ledger.csv` **paired**, and quote everything against the floor.

The one rule this file exists to enforce
----------------------------------------
An arm's raw mean is not a result. Held-out spatial folds differ enormously in
difficulty here -- the same `random` campaign scores 0.42 on one fold and 0.64 on
another -- so an unpaired mean over five folds is dominated by which folds an arm
happened to run on, and a per-arm seed sd is a mixture of fold variance and
method variance in unknown proportion.

Everything below is therefore a **paired delta**: arm minus `random` at the same
(seed, fold, round, tag, n_seed_set), and the summary is the mean and the sign
count of those pairs. `STATE_PRETRAIN_RESEARCH.md` made exactly this correction
once already ("seed sd is not an error bar, compare paired").

The floor is `random_b` and `random_c` -- the same strategy on a different RNG
stream. Their paired delta against `random` is the smallest gap this harness can
resolve; an arm inside it has not been shown to do anything.

Run
---
    python src/al_report.py                       # everything, latest tag
    python src/al_report.py --tag AL1_main --metric change_f1
    python src/al_report.py --tradeoff            # accuracy vs retrieval
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from al_lab import CHANGE_CLASSES, LEDGER, ledger

KEY = ["seed", "fold", "round", "tag", "n_seed_set", "batch", "n_rounds",
       "seed_mode"]
BASE = "random"
FLOOR_ARMS = ("random_b", "random_c")


def paired(df: pd.DataFrame, metric: str, base: str = BASE) -> pd.DataFrame:
    """One row per (arm, round): mean paired delta, sign count, n pairs."""
    ref = df[df["arm"] == base].set_index(KEY)[metric]
    out = []
    for arm, g in df[df["arm"] != base].groupby("arm"):
        g = g.set_index(KEY)
        common = g.index.intersection(ref.index)
        if not len(common):
            continue
        d = (g.loc[common, metric] - ref.loc[common]).reset_index()
        d.columns = list(KEY) + ["delta"]
        for rnd, gr in d.groupby("round"):
            v = gr["delta"].to_numpy(float)
            v = v[np.isfinite(v)]
            if not len(v):
                continue
            out.append(dict(arm=arm, round=int(rnd), n=len(v),
                            mean=float(v.mean()), sd=float(v.std(ddof=1))
                            if len(v) > 1 else np.nan,
                            wins=int((v > 0).sum()),
                            se=float(v.std(ddof=1) / np.sqrt(len(v)))
                            if len(v) > 1 else np.nan))
    return pd.DataFrame(out)


#: The tag AL0 wrote. The floor is a property of the harness and the design, not
#: of an experiment, so it is measured once and every later tag is quoted against
#: it. Reading it from the filtered frame instead returns NaN for every tag that
#: did not re-run the null arms -- which is every real experiment.
FLOOR_TAG = "AL0_floor"


def floor(df: pd.DataFrame, metric: str, full: pd.DataFrame | None = None,
          floor_tag: str = FLOOR_TAG) -> float:
    """The paired noise floor: largest |paired delta| of the null arms.

    Matched to ``df``'s design on the axes that move it (batch, rounds, seed-set
    size); an experiment that changes the design and does not re-run the null
    arms gets NaN rather than a floor borrowed from a different campaign shape.
    """
    src = df if (df["arm"].isin(FLOOR_ARMS)).any() else full
    if src is None:
        return np.nan
    if floor_tag in set(src["tag"].dropna()) and not \
            (df["arm"].isin(FLOOR_ARMS)).any():
        want = {c: df[c].iloc[0] for c in ("batch", "n_rounds", "n_seed_set",
                                           "seed_mode") if c in df}
        cand = src[src["tag"] == floor_tag]
        for c, v in want.items():
            if c in cand:
                cand = cand[cand[c] == v]
        if cand.empty:
            return np.nan
        src = cand
    p = paired(src, metric)
    null = p[p["arm"].isin(FLOOR_ARMS)]
    if null.empty:
        return np.nan
    return float(np.abs(null["mean"]).max())


def table(df: pd.DataFrame, metric: str, last_round_only: bool = True,
          full: pd.DataFrame | None = None) -> pd.DataFrame:
    p = paired(df, metric)
    if p.empty:
        return p
    if last_round_only:
        p = p[p["round"] == p["round"].max()]
    f = floor(df, metric, full)
    p = p.sort_values("mean", ascending=False).copy()
    p["vs_floor"] = np.where(p["arm"].isin(FLOOR_ARMS), "(floor)",
                             np.where(p["mean"].abs() > f, "", "inside floor"))
    return p


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--metric", nargs="+",
                    default=["change_f1", "macro_f1", "change_macro_f1",
                             "natStab_as_art", "natStab_as_crop", "acq_change_n",
                             "vendi_state"])
    ap.add_argument("--arms", nargs="+", default=None,
                    help="restrict the tables to these arms")
    ap.add_argument("--curve", action="store_true",
                    help="paired delta at every round, not just the last")
    ap.add_argument("--tradeoff", action="store_true")
    args = ap.parse_args()

    full = ledger()
    if full.empty:
        raise SystemExit(f"no ledger at {LEDGER}")
    df = full[full["tag"] == args.tag] if args.tag else full
    if args.arms:
        df = df[df["arm"].isin(list(args.arms) + [BASE] + list(FLOOR_ARMS))]
    tags = sorted(df["tag"].dropna().unique())
    print(f"{len(df):,} rows | tags {tags} | arms {df['arm'].nunique()} | "
          f"seeds {sorted(df['seed'].unique())} | "
          f"rounds 0..{df['round'].max()}\n")

    for m in args.metric:
        if m not in df:
            continue
        f = floor(df, m, full)
        print(f"=== {m}   (paired vs {BASE}; floor = {f:.4f}) ===")
        t = table(df, m, last_round_only=not args.curve, full=full)
        if t.empty:
            print("  (no pairs)\n")
            continue
        cols = ["arm", "round", "n", "mean", "se", "wins", "vs_floor"]
        print(t[cols].to_string(index=False,
                                float_format=lambda x: f"{x: .4f}"))
        print()

    if args.tradeoff:
        print("=== the tradeoff: accuracy bought vs rare-class plots bought ===")
        last = df[df["round"] == df["round"].max()]
        g = last.groupby("arm").agg(
            change_f1=("change_f1", "mean"),
            macro_f1=("macro_f1", "mean"),
            change_n=("acq_change_n", "mean"),
            vendi=("vendi_state", "mean"),
            natStab_as_art=("natStab_as_art", "mean"),
            n=("change_f1", "size"))
        base = g.loc[BASE]
        g["dF1"] = g["change_f1"] - base["change_f1"]
        g["retrieval_x"] = g["change_n"] / base["change_n"]
        print(g.sort_values("dF1", ascending=False).to_string(
            float_format=lambda x: f"{x: .4f}"))


if __name__ == "__main__":
    main()
