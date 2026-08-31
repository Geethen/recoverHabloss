"""Does a 2018-only state pool say anything about the 2024 endpoint? (P0)

The gate on the whole "auxiliary single-date path" idea, and the one N14 left
standing as a structural limit: **GLanCE ends in 2020, so it can only ever
supervise the 2018 endpoint** -- while the target classes are ``from -> to`` and
half of what the user asked for (``-> Cropland`` in 2024) lives on the other
side. N14 recorded that as a reason any serious retry needs a pool at *both*
endpoints.

That conclusion assumes a 2018 label cannot reach the 2024 read. On a **shared**
encoder that is an empirical question, not a structural one: ``f`` is one map
applied to both dates, and N2/N3 spend an entire auxiliary objective pushing it
toward year-invariance. If the AlphaEarth embedding space is year-stable, a
state model fitted at 2018 scores the 2024 block just as well, no 2024 labels
are needed, and the "``-> Cropland``" half is reachable from the pool that is
already built.

Five reads, all at the **state** level (never the transition level -- a
single-date pool can say nothing about change), all on the same linear probe so
nothing but the year varies.

1. **self-floor 2018** -- RECOVER 2018 block -> ``lc_2018``, blocked CV. The
   number every row below is read against; reproduces N14a's 0.751 / 0.740.
2. **self-floor 2024** -- RECOVER 2024 block -> ``lc_2024``, blocked CV. Says
   whether the 2024 endpoint is *intrinsically* as separable as the 2018 one,
   which has to be known before any cross-year number is readable.
3. **cross-year, RECOVER's own labels** -- fit on the 2018 block, predict the
   held-out plots' **2024** block, score against ``lc_2024``. The mechanism test
   with the pool removed entirely: it isolates "is the embedding space
   year-stable" from "does GLanCE's legend agree", which read 5 confounds.
4. **pool -> RECOVER 2018** -- N14a's cleared row, recomputed here so 5 has a
   same-script comparison rather than a quoted one.
5. **pool -> RECOVER 2024** -- the question. GLanCE 2018 labels, applied to the
   2024 block, scored against ``lc_2024``.

Reads 3 and 5 are additionally split by whether the plot **changed**. A stable
plot's 2024 state is its 2018 state, so a cross-year model can score it by
memorising 2018 and the aggregate would flatter itself; the changed plots are
where a 2024 read is actually doing 2024 work, and they are where the
commissioned transitions live.

Usage::

    python diagnose_state_year_transfer.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from diagnose_state_pools import (
    RECOVER_FRAME,
    STATES,
    fit_predict,
    metrics,
    show,
)
from experiment_merged_legend import LEGENDS
from project_paths import project_data_dir

POOL_YEAR = 2018
YEARS = (2018, 2024)


def bands(year: int) -> list[str]:
    return [f"A{i:02d}_{year}" for i in range(64)]


def load_recover(path: Path) -> dict:
    """RECOVER plots as both years' blocks plus both years' coarse3 states.

    Completeness is required at **both** endpoints, unlike
    ``diagnose_state_pools.load_recover`` which needs only 2018 -- a cross-year
    read has to score the same plots in both columns or the two rows are not
    comparable.
    """
    frame = pd.read_parquet(path).drop_duplicates("PLOTID").reset_index(drop=True)
    cols = bands(2018) + bands(2024)
    frame = frame.loc[frame[cols].notna().all(axis=1)].reset_index(drop=True)

    legend = LEGENDS["coarse3"]
    out = {"blocks": frame["block_id"].to_numpy()}
    for year in YEARS:
        raw = frame[f"lc_{year}"].astype(str).str.strip().str.lower()
        unknown = sorted(set(raw) - set(legend))
        if unknown:
            raise ValueError(f"lc_{year} values outside the coarse3 legend: {unknown}")
        out[f"y{year}"] = raw.map(legend).str.lower().to_numpy()
        # pandas 3.x hands parquet floats back as nullable extension dtypes,
        # which reach sklearn as object arrays (CLAUDE.md).
        out[f"X{year}"] = frame[bands(year)].astype("float64").to_numpy()
    out["changed"] = out["y2018"] != out["y2024"]
    return out


def oof_cross_year(X_fit, y_fit, X_score, groups, n_splits: int = 5) -> np.ndarray:
    """Blocked-CV predictions where the fit and the scored block differ.

    ``X_fit``/``y_fit`` train on the training blocks; the held-out block is
    predicted from ``X_score``. Passing the same array for both recovers the
    ordinary self-floor, so reads 1-3 come off one function and cannot drift
    apart in their fold construction.
    """
    pred = np.empty(len(y_fit), dtype=object)
    for tr, te in GroupKFold(n_splits=n_splits).split(X_fit, y_fit, groups):
        pred[te] = fit_predict(X_fit[tr], y_fit[tr], X_score[te])
    return pred


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embeddings-dir", type=Path,
                        default=project_data_dir("embeddings"))
    parser.add_argument("--pool", default="glance_strict")
    parser.add_argument("--output", type=Path,
                        default=project_data_dir("analysis_results")
                        / "state_year_transfer.json")
    args = parser.parse_args()

    rec = load_recover(args.embeddings_dir / RECOVER_FRAME)
    n = len(rec["y2018"])
    print(f"RECOVER: {n:,} plots complete at both endpoints, "
          f"{int(rec['changed'].sum()):,} changed at coarse3")
    for year in YEARS:
        counts = ", ".join(f"{s}={int((rec[f'y{year}'] == s).sum()):,}" for s in STATES)
        print(f"  {year}: {counts}")

    keys = ["n", "accuracy", "macro_f1", "f1_artificial", "f1_cropland",
            "f1_nature", "cropland_as_nature", "nature_as_cropland",
            "artificial_as_nature"]
    header = ("  " + " " * 26 + "".join(f"{k.replace('_', ' ')[:9]:>10s}"
                                        for k in keys))
    results: dict[str, dict] = {}

    def record(title, key, y_true, pred, mask=None):
        if mask is not None:
            y_true, pred = y_true[mask], pred[mask]
        row = metrics(y_true, pred)
        results[key] = row
        show(title, row, keys)
        return row

    print("\n=== 1-3: RECOVER's own labels (the pool is not involved) ===")
    print(header)
    pred_18 = oof_cross_year(rec["X2018"], rec["y2018"], rec["X2018"], rec["blocks"])
    record("1 self-floor 2018", "self_2018", rec["y2018"], pred_18)
    pred_24 = oof_cross_year(rec["X2024"], rec["y2024"], rec["X2024"], rec["blocks"])
    record("2 self-floor 2024", "self_2024", rec["y2024"], pred_24)
    # Fitted on 2018 embeddings with 2018 labels, then asked the same question of
    # the 2024 block. Nothing about 2024 is in the fit.
    pred_x = oof_cross_year(rec["X2018"], rec["y2018"], rec["X2024"], rec["blocks"])
    record("3 fit 2018 -> read 2024", "cross_year", rec["y2024"], pred_x)
    record("  ... stable plots", "cross_year_stable", rec["y2024"], pred_x,
           ~rec["changed"])
    record("  ... CHANGED plots", "cross_year_changed", rec["y2024"], pred_x,
           rec["changed"])
    record("  self-floor 2024, changed", "self_2024_changed", rec["y2024"],
           pred_24, rec["changed"])

    path = args.embeddings_dir / f"state_labels_{args.pool}_{POOL_YEAR}.parquet"
    if not path.exists():
        raise SystemExit(f"{path} not found -- run build_state_labels.py first")
    pool = pd.read_parquet(path).reset_index(drop=True)
    Xp = pool[bands(POOL_YEAR)].astype("float64").to_numpy()
    yp = pool["state"].to_numpy()
    print(f"\n=== 4-5: {args.pool} ({len(yp):,} units at {POOL_YEAR}) -> RECOVER ===")
    print(header)
    # One fit on the whole pool: it carries no RECOVER label, so there is nothing
    # for a fold split to protect against here (N14a's construction).
    from diagnose_state_pools import new_model
    model = new_model().fit(Xp, yp)
    p18 = model.predict(rec["X2018"])
    p24 = model.predict(rec["X2024"])
    record("4 pool -> RECOVER 2018", "pool_2018", rec["y2018"], p18)
    record("5 pool -> RECOVER 2024", "pool_2024", rec["y2024"], p24)
    record("  ... stable plots", "pool_2024_stable", rec["y2024"], p24, ~rec["changed"])
    record("  ... CHANGED plots", "pool_2024_changed", rec["y2024"], p24, rec["changed"])

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(
        {"pool": args.pool, "n_plots": int(n),
         "n_changed": int(rec["changed"].sum()), "reads": results},
        indent=2), encoding="utf-8")
    print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
