"""Do LUCAS/GLanCE state labels mean the same thing as RECOVER's? (N13a)

The gate before any of this reaches the siamese encoder. A state head trained on
an external pool only helps if that pool's Nature/Cropland/Artificial boundary is
the one the RECOVER interpreters drew. If it is not, the pool does not add noise
-- it adds a **systematic offset**, and on the Cropland/Nature boundary
specifically, which is where this project's change-F1 ceiling already sits
(``analyse_label_noise.py``, ``cropland-nature-label-noise``). Averaging over more
labels does not fix a systematic offset; it entrenches it.

Three reads, all on the same AlphaEarth 2018 64-D block so nothing else varies.

1. **Centroid displacement.** Per state, the distance between the pool's centroid
   and RECOVER's, in units of RECOVER's own within-class spread. A pool whose
   Cropland centroid sits further from RECOVER's Cropland than RECOVER's Cropland
   plots typically sit from their own centre is describing something else. The
   hard failure is a pool centroid that is closer to a *different* RECOVER class
   -- the same test ``diagnose_stable_artificial.py`` used to establish that
   stable-Artificial is a label problem.

2. **Transfer, both directions.** Train a linear state model on the pool, score it
   on the RECOVER plots' ``lc_2018``; then the reverse. Asymmetry is diagnostic:
   a pool that predicts RECOVER well but is not predicted by it is *finer* than
   RECOVER's legend, which is usable. The opposite means the pool is coarser or
   drawn differently, which is not.

3. **The self-floor, and it is the reason the other two are readable.** RECOVER
   predicting its own ``lc_2018`` under blocked CV. A transfer accuracy of 0.72
   means nothing until you know whether RECOVER's own states are 0.95 or 0.75
   separable from these embeddings. The ledger has been wrong in exactly this way
   before (S18: a subset was called non-replicating before anyone computed what
   the model scored against *itself*), and CLAUDE.md now requires the floor first.

Everything is scored at the **state** level, never the transition level. These
pools carry one date and can say nothing about change.

Usage::

    python diagnose_state_pools.py
    python diagnose_state_pools.py --pools glance_strict
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from experiment_merged_legend import LEGENDS
from project_paths import project_data_dir

YEAR = 2018
BANDS = [f"A{i:02d}_{YEAR}" for i in range(64)]
STATES = ("artificial", "cropland", "nature")
RECOVER_FRAME = "embeddings_habloss_recover_annual.parquet"


def load_recover(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """RECOVER plots as (X, state at 2018, block id), deduped on PLOTID."""
    frame = pd.read_parquet(path).drop_duplicates("PLOTID").reset_index(drop=True)
    # Same completeness filter the lab applies (experiment_merged_legend.build_frame):
    # a plot whose 2018 embedding did not resolve is dropped rather than imputed.
    frame = frame.loc[frame[BANDS].notna().all(axis=1)].reset_index(drop=True)
    legend = LEGENDS["coarse3"]
    raw = frame["lc_2018"].astype(str).str.strip().str.lower()
    unknown = sorted(set(raw) - set(legend))
    if unknown:
        raise ValueError(f"lc_2018 values outside the coarse3 legend: {unknown}")
    state = raw.map(legend).str.lower().to_numpy()
    # pandas 3.x hands parquet floats back as nullable extension dtypes, which
    # reach sklearn as object arrays (CLAUDE.md).
    X = frame[BANDS].astype("float64").to_numpy()
    return X, state, frame["block_id"].to_numpy()


def load_pool(path: Path) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    frame = pd.read_parquet(path).reset_index(drop=True)
    X = frame[BANDS].astype("float64").to_numpy()
    return X, frame["state"].to_numpy(), frame


def centroid_report(Xp, yp, Xr, yr) -> list[dict]:
    """Pool-vs-RECOVER centroid distance, scaled by RECOVER's own spread."""
    r_centroids = {s: Xr[yr == s].mean(0) for s in STATES if (yr == s).any()}
    r_spread = {
        s: float(np.linalg.norm(Xr[yr == s] - c, axis=1).mean())
        for s, c in r_centroids.items()
    }
    rows = []
    for s in STATES:
        if not (yp == s).any() or s not in r_centroids:
            continue
        p_centroid = Xp[yp == s].mean(0)
        dists = {t: float(np.linalg.norm(p_centroid - c))
                 for t, c in r_centroids.items()}
        nearest = min(dists, key=dists.get)
        rows.append({
            "state": s,
            "n_pool": int((yp == s).sum()),
            "n_recover": int((yr == s).sum()),
            "dist_to_own": dists[s],
            "recover_spread": r_spread[s],
            "ratio": dists[s] / r_spread[s],
            "nearest_recover_class": nearest,
            "misplaced": nearest != s,
            **{f"dist_to_{t}": d for t, d in dists.items()},
        })
    return rows


def metrics(y_true, y_pred) -> dict:
    labels = [s for s in STATES if s in set(y_true)]
    per_class = f1_score(y_true, y_pred, labels=labels, average=None,
                         zero_division=0)
    out = {
        "n": int(len(y_true)),
        "accuracy": float((y_pred == y_true).mean()),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels,
                                   average="macro", zero_division=0)),
        **{f"f1_{s}": float(v) for s, v in zip(labels, per_class)},
    }
    # The confusions the project is actually judged on: the Cropland/Nature
    # boundary that caps change-F1, and built-up leaking into vegetation.
    #
    # `artificial_as_cropland` is the fourth and it is a *constraint*, not a
    # score to trade against the others (added 2026-08-03, from the user's read
    # of the Oslo map). Cropland has room to grow there, but it must grow into
    # Nature: a cropland gain paid for out of built-up is the stable-Artificial
    # regression this project has spent ~45 ideas failing to move, arriving from
    # a new direction. Section X's pool is the worked example -- it lifted
    # cropland recall 0.7704 -> 0.7739 and paid for it with
    # `artificial_as_cropland` 0.0720 -> 0.0782 while handing pixels *back* to
    # Nature (0.1592 -> 0.1571), which is the trade backwards on both counts.
    for src, dst, name in (("cropland", "nature", "cropland_as_nature"),
                           ("nature", "cropland", "nature_as_cropland"),
                           ("artificial", "nature", "artificial_as_nature"),
                           ("artificial", "cropland", "artificial_as_cropland")):
        mask = y_true == src
        out[name] = float((y_pred[mask] == dst).mean()) if mask.any() else float("nan")
    return out


def new_model():
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))


def fit_predict(Xtr, ytr, Xte):
    model = new_model()
    model.fit(Xtr, ytr)
    return model.predict(Xte)


def self_floor_oof(X, y, groups, n_splits: int = 5) -> np.ndarray:
    """Out-of-fold RECOVER-on-RECOVER predictions under blocked CV.

    Returned as predictions rather than a score so the *same* fitted floor can be
    read on any subset of plots. That matters for the in-block comparison below:
    re-fitting a floor on 778 European plots would confound "which plots are
    scored" with "how much training data the floor had", and the difference
    between those two is exactly the question LUCAS raises.
    """
    pred = np.empty(len(y), dtype=object)
    for tr, te in GroupKFold(n_splits=n_splits).split(X, y, groups):
        pred[te] = fit_predict(X[tr], y[tr], X[te])
    return pred


def show(title: str, row: dict, keys: list[str]) -> None:
    cells = []
    for key in keys:
        value = row.get(key, float("nan"))
        cells.append(f"{value:>10,}" if key == "n" else f"{value:>10.3f}")
    print(f"  {title:<26s}" + "".join(cells))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embeddings-dir", type=Path,
                        default=project_data_dir("embeddings"))
    parser.add_argument("--pools", nargs="+",
                        default=["lucas", "glance_strict"])
    parser.add_argument("--output", type=Path,
                        default=project_data_dir("analysis_results")
                        / "state_pool_diagnostic.json")
    args = parser.parse_args()

    Xr, yr, blocks = load_recover(args.embeddings_dir / RECOVER_FRAME)
    print(f"RECOVER: {len(yr):,} plots, states "
          + ", ".join(f"{s}={int((yr == s).sum()):,}" for s in STATES))

    keys = ["n", "accuracy", "macro_f1", "f1_artificial", "f1_cropland",
            "f1_nature", "cropland_as_nature", "nature_as_cropland",
            "artificial_as_nature"]
    header = ("  " + " " * 26 + "".join(f"{k.replace('_', ' ')[:9]:>10s}"
                                        for k in keys))

    oof = self_floor_oof(Xr, yr, blocks)
    floor = metrics(yr, oof)
    print("\n=== self-floor: RECOVER -> RECOVER, blocked CV ===")
    print(header)
    show("recover self", floor, keys)

    results = {"self_floor": floor, "pools": {}}
    for name in args.pools:
        path = args.embeddings_dir / f"state_labels_{name}_{YEAR}.parquet"
        if not path.exists():
            print(f"\n!! {name}: {path} not found -- run build_state_labels.py")
            continue
        Xp, yp, frame = load_pool(path)
        pool_blocks = set(frame["block_id"].unique())
        inblock = np.isin(blocks, list(pool_blocks))
        print(f"\n=== {name}: {len(yp):,} rows, {len(pool_blocks)} blocks; "
              f"{int(inblock.sum()):,}/{len(yr):,} RECOVER plots share a block ===")

        cents = centroid_report(Xp, yp, Xr, yr)
        print("  centroid displacement (ratio = pool-to-RECOVER / RECOVER spread)")
        for row in cents:
            flag = "  <-- NEAREST IS " + row["nearest_recover_class"].upper() \
                if row["misplaced"] else ""
            print(f"    {row['state']:<11s} n={row['n_pool']:>6,}  "
                  f"dist={row['dist_to_own']:.2f}  spread={row['recover_spread']:.2f}"
                  f"  ratio={row['ratio']:.2f}{flag}")

        pred_r = fit_predict(Xp, yp, Xr)
        fwd = metrics(yr, pred_r)
        rev = metrics(yp, fit_predict(Xr, yr, Xp))
        # Restricted to the blocks the pool actually covers, against the *same*
        # globally-fitted floor read on the same plots. Separates "this pool
        # disagrees about the legend" from "this pool has never seen these
        # continents", which the global row cannot tell apart.
        fwd_in = metrics(yr[inblock], pred_r[inblock])
        floor_in = metrics(yr[inblock], oof[inblock])
        print(header)
        show(f"{name} -> RECOVER", fwd, keys)
        show(f"RECOVER -> {name}", rev, keys)
        show("-- in shared blocks --", floor_in, keys)
        show(f"{name} -> RECOVER", fwd_in, keys)
        results["pools"][name] = {
            "centroids": cents, "pool_to_recover": fwd, "recover_to_pool": rev,
            "pool_to_recover_inblock": fwd_in, "self_floor_inblock": floor_in,
            "n_rows": int(len(yp)), "n_blocks": len(pool_blocks),
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
