"""Run dataset x architecture under LLTO and append to the ledger.

One row per (dataset, architecture, split, weight, seed, read) in
``data/analysis_results/statepre_ledger.csv``, append-only, so a cold start
resumes from disk and two sessions cannot overwrite each other's numbers.

The project's standing rules apply here unchanged (``docs/research/AUTORESEARCH.md``):
**3 seeds minimum before a verdict, 5 before a win**, and a difference under
0.005 on the headline read is noise until more seeds say otherwise. Seeds move
the k-means fold geography as well as the model init, so a seed here is a
genuinely different experiment rather than a different initialisation of the
same one -- which is the conservative choice and makes the spread wider than a
fixed-fold run would report.

Usage::

    P=/home/geethen.singh/.pixi/envs/geo/bin/python

    $P -m statepre.run --list
    $P -m statepre.run --datasets endpoints stable_years --archs linear --seeds 3
    $P -m statepre.run --datasets stable_years --archs mlp --splits random loc llto
    $P -m statepre.run --report                      # read the ledger back
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from project_paths import project_data_dir
from statepre import data as sp_data
from statepre import llto as sp_llto
from statepre import models as sp_models

LEDGER = project_data_dir("analysis_results") / "statepre_ledger.csv"
#: Per-fold detail, kept in its own file because it is what the arms must
#: actually be compared on. K-means over the RECOVER plots' coordinates returns
#: the *same* five continental folds at every seed -- Asia/Oceania 2,210,
#: Europe/W-Asia 1,765, N.America 853, Africa 809, S.America 777 -- so the seed
#: spread on an LLTO run is model-init noise and nothing else, and reading it as
#: an error bar for "would this hold in a region we did not sample" is wrong. The
#: honest test is **paired**: the same five folds, arm A against arm B, fold by
#: fold. That is what this file is for.
FOLD_LEDGER = project_data_dir("analysis_results") / "statepre_folds.csv"

#: Reads carried into the ledger, and the columns kept off each one.
METRIC_KEYS = ("n", "accuracy", "macro_f1", "f1_artificial", "f1_cropland",
               "f1_nature", "cropland_as_nature", "nature_as_cropland",
               "artificial_as_nature", "artificial_as_cropland")


def one(dataset: str, archname: str, *, seed: int, n_folds: int, split: str,
        space: str, weight: str, fold_ref: str, test_year: int, plots,
        test_frame, verbose: bool, **arch_kw) -> list[dict]:
    frame = sp_data.build(dataset, plots, seed=seed)
    factory = sp_models.factory(archname, **arch_kw)
    start = time.time()
    result = sp_llto.run(frame, factory, test_frame=test_frame, n_folds=n_folds,
                         seed=seed, split=split, space=space,
                         test_year=test_year, weight=weight, fold_ref=fold_ref,
                         verbose=verbose)
    elapsed = time.time() - start

    tag = {"dataset": dataset, "arch": archname, "split": split,
           "weight": weight, "fold_ref": fold_ref, "n_folds": n_folds,
           "seed": seed}
    append(FOLD_LEDGER, [{**tag, **fold} for fold in result["per_fold"]])

    rows = []
    for read, metrics in result["reads"].items():
        rows.append({
            "dataset": dataset, "arch": archname, "split": split, "space": space,
            "weight": weight, "fold_ref": fold_ref, "seed": seed,
            "n_folds": n_folds, "test_year": test_year, "read": read,
            "n_rows_train_pool": int(len(frame)),
            "n_locations": int(frame["loc_id"].nunique()),
            "leaked_locations": int(result["leaked_locations"]),
            "seconds": round(elapsed, 1),
            "timestamp": pd.Timestamp.now().isoformat(timespec="seconds"),
            **{k: metrics.get(k, float("nan")) for k in METRIC_KEYS},
        })
    return rows


def append(path: Path, rows: list[dict]) -> None:
    """Append, reconciled against the header already on disk.

    A headerless append writes the *new* frame's column order, so adding a field
    to a row dict silently shifts every value in the file one column left from
    that point on. The ledgers are the only record of runs that are not cheap to
    repeat, so the reconciliation is a hard check rather than a convenience:
    missing columns are filled with NA, and a genuinely new column stops the run
    and asks for an explicit migration (``--migrate``).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    if path.exists():
        existing = list(pd.read_csv(path, nrows=0).columns)
        extra = [c for c in frame.columns if c not in existing]
        if extra:
            raise SystemExit(
                f"{path.name} has no column(s) {extra}. Appending would corrupt "
                f"every later row -- run `python -m statepre.run --migrate` first.")
        frame = frame.reindex(columns=existing)
    frame.to_csv(path, mode="a", header=not path.exists(), index=False)


def _infer_n_folds(frame: pd.DataFrame) -> np.ndarray:
    """Recover each fold row's run fold-count from the file order.

    The fold ledger was written without one, so a 10-fold run's folds 0-4 and a
    5-fold run's folds 0-4 are indistinguishable by key -- and ``paired`` dedupes
    on that key, so the 5-fold run was being *discarded* rather than pooled with
    the 10-fold one, which is not what section U3 says it did.

    ``run`` emits a run's folds in ascending order, so a fold id that does not
    exceed its predecessor within a cell starts a new run. The count is the
    segment's highest id plus one; a run whose last fold held no test rows would
    be under-counted, which cannot happen on this file (every segment is a clean
    0..4 or 0..9).
    """
    keys = ["dataset", "arch", "split", "weight", "seed"]
    out = np.zeros(len(frame), dtype=int)
    for _, index in frame.groupby(keys, sort=False).groups.items():
        folds = frame.loc[index, "fold"].to_numpy()
        starts = np.flatnonzero(np.r_[True, np.diff(folds) <= 0])
        for begin, end in zip(starts, np.r_[starts[1:], len(folds)]):
            out[np.asarray(index)[begin:end]] = folds[begin:end].max() + 1
    return out


def migrate() -> None:
    """Rewrite both ledgers with today's columns, backfilling the old defaults.

    Only ever adds columns. ``fold_ref`` is ``union`` on every pre-existing row
    because that is what those runs did -- see ``llto``'s module docstring; the
    distinction is load-bearing and must not be backfilled as the new default.
    """
    for path in (LEDGER, FOLD_LEDGER):
        if not path.exists():
            print(f"{path.name}: does not exist, nothing to migrate")
            continue
        frame = pd.read_csv(path)
        added = []
        if "fold_ref" not in frame.columns:
            frame["fold_ref"] = "union"
            added.append("fold_ref")
        # A metric added to `diagnose_state_pools.metrics` reaches both ledgers
        # -- the run ledger through METRIC_KEYS and the fold ledger through the
        # whole `_read` dict -- and `append` refuses to write until the column
        # exists. NA rather than a value: the older runs did not compute it and a
        # backfilled zero would read as "this arm leaked no built-up".
        for key in METRIC_KEYS:
            if key not in frame.columns:
                frame[key] = pd.NA
                added.append(key)
        if path is FOLD_LEDGER:
            # Also repairs rows written by a build that had the column but was
            # not yet filling it -- an empty n_folds silently drops the row out
            # of every paired read, which is a quieter failure than a bad number.
            blank = (frame["n_folds"].isna() if "n_folds" in frame.columns
                     else pd.Series(True, index=frame.index))
            if blank.any():
                inferred = pd.Series(_infer_n_folds(frame), index=frame.index)
                frame["n_folds"] = (inferred if blank.all()
                                    else frame["n_folds"].where(~blank, inferred))
                frame["n_folds"] = frame["n_folds"].astype(int)
                added.append(f"n_folds({int(blank.sum())} rows)")
        if not added:
            print(f"{path.name}: already current")
            continue
        backup = path.with_suffix(".pre_migration.csv")
        if not backup.exists():
            path.rename(backup)
        frame.to_csv(path, index=False)
        print(f"{path.name}: added {added} to {len(frame):,} rows "
              f"(backup at {backup.name})")


#: Fold identity. ``n_folds`` and ``fold_ref`` belong here because fold 3 of a
#: 5-fold run and fold 3 of a 20-fold run are different ground, as are the same
#: fold id cut on the union cloud and on the reference cloud -- pairing across
#: either is pairing two different partitions of the world.
PAIR_KEYS = ["arch", "split", "weight", "fold_ref", "n_folds", "seed", "fold"]


def paired(value_a: str, value_b: str, arch: str = "linear",
           split: str = "llto", weight: str = "none",
           fold_ref: str = "reference", n_folds: int | None = None,
           metric: str = "macro_f1", by: str = "dataset",
           dataset: str = "glance_endpoints") -> pd.DataFrame:
    """A minus B, fold by fold, on the folds they share.

    The comparison the fixed fold geography demands. A mean difference is
    meaningful only if it survives being read against the *fold* spread, and the
    five continental folds differ from each other by far more than any arm
    differs from another -- fold 3 (N.America) scores 0.81 where fold 1 (Africa)
    scores 0.65, which is 16 points of between-fold variation swamping a 1-point
    arm effect in any unpaired read.

    ``by`` picks the axis being compared. ``"dataset"`` puts two pools against
    each other on one architecture (sections U-X); ``"arch"`` puts two
    architectures against each other on one pool, which is what section Y's
    capacity sweep needs and what this function could not express before. The
    *other* axis becomes a filter -- ``arch=`` for the first, ``dataset=`` for the
    second -- because a fold id is a shared key only when everything except the
    compared axis is held fixed.
    """
    if by not in ("dataset", "arch"):
        raise ValueError(f"by must be 'dataset' or 'arch', got {by!r}")
    if not FOLD_LEDGER.exists():
        raise SystemExit(f"{FOLD_LEDGER} does not exist yet")
    frame = pd.read_csv(FOLD_LEDGER)
    missing = [c for c in PAIR_KEYS if c not in frame.columns]
    if missing:
        raise SystemExit(f"{FOLD_LEDGER.name} lacks {missing} -- run --migrate")
    frame = frame.drop_duplicates(subset=["dataset"] + PAIR_KEYS, keep="last")
    # The join index is every identifying column except the one being compared;
    # the compared axis's partner is pinned to a single value instead.
    keys = [k for k in PAIR_KEYS + ["dataset"] if k != by]
    held = {"arch": arch} if by == "dataset" else {"dataset": dataset}
    sel = frame.loc[(frame["split"] == split) & (frame["weight"] == weight)
                    & (frame["fold_ref"] == fold_ref)]
    for column, value in held.items():
        sel = sel.loc[sel[column] == value]
    if n_folds is not None:
        sel = sel.loc[sel["n_folds"] == n_folds]
    a = sel.loc[sel[by] == value_a].set_index(keys)[metric]
    b = sel.loc[sel[by] == value_b].set_index(keys)[metric]
    joined = pd.concat([a.rename(value_a), b.rename(value_b)], axis=1,
                       join="inner")
    joined["diff"] = joined[value_a] - joined[value_b]
    return joined.reset_index()


def confusion(dataset: str, archname: str, *, seeds, n_folds: int, split: str,
              fold_ref: str, weight: str, read: str, plots, test_frame,
              **arch_kw) -> tuple[pd.DataFrame, dict]:
    """Pooled out-of-fold confusion matrix for one arm, over ``seeds``.

    Deliberately **not** written to the ledger: a 3x3 table per (arm, seed, read)
    does not fit a flat run log, and the row-normalised version is the only form
    anyone reads it in anyway. Pooling the seeds rather than averaging their
    matrices keeps the cell counts integers and weights each seed by the rows it
    actually scored, which differ when a fold holds no test plots.
    """
    factory = sp_models.factory(archname, **arch_kw)
    truth, pred = [], []
    for seed in seeds:
        frame = sp_data.build(dataset, plots, seed=seed)
        result = sp_llto.run(frame, factory, test_frame=test_frame,
                             n_folds=n_folds, seed=seed, split=split,
                             weight=weight, fold_ref=fold_ref, verbose=False)
        keep = {"t1_all": slice(None)}.get(read)
        mask = (np.ones(len(result["truth"]), bool) if keep is not None
                else (result["changed"] if read == "t1_changed"
                      else ~result["changed"]))
        truth.append(result["truth"][mask])
        pred.append(result["pred"][mask])
    truth, pred = np.concatenate(truth), np.concatenate(pred)
    states = [s for s in sp_data.STATES if s in set(truth)]
    table = pd.crosstab(pd.Series(truth, name="truth"),
                        pd.Series(pred, name="predicted")).reindex(
        index=states, columns=states, fill_value=0)
    return table, sp_llto._read(truth, pred)


def print_confusion(name: str, table: pd.DataFrame, scores: dict,
                    seeds: int) -> None:
    total = table.to_numpy().sum()
    rates = table.div(table.sum(axis=1), axis=0)
    print(f"\n{name}   n={total:,} over {seeds} seeds   "
          f"accuracy={scores['accuracy']:.4f}   macro_f1={scores['macro_f1']:.4f}")
    print("  per-class F1: " + "  ".join(
        f"{s}={scores[f'f1_{s}']:.4f}" for s in table.index))
    header = "".join(f"{c[:9]:>11s}" for c in table.columns)
    print(f"  {'truth \\ pred':<14s}{header}{'recall':>11s}")
    for state in table.index:
        cells = "".join(f"{table.loc[state, c]:>6,d}{rates.loc[state, c]:>5.0%}"
                        for c in table.columns)
        print(f"  {state:<14s}{cells}{rates.loc[state, state]:>11.3f}")
    precision = table.div(table.sum(axis=0), axis=1)
    print(f"  {'precision':<14s}" + "".join(
        f"{precision.loc[c, c]:>11.3f}" for c in table.columns))


def region_folds(pool: str, n_folds: int, seeds, split: str = "llto",
                 fold_ref: str = "reference", space: str = "sphere",
                 min_rows: int = 100) -> dict[tuple[int, int], int]:
    """``(seed, fold) -> rows of ``pool`` inside that fold``.

    A geographically narrow pool -- LUCAS is EU-27 and nothing else -- is
    *removed entirely* whenever a fold it lives in is held out, and is fully
    present, but far from the test plots, whenever one it does not is. Those are
    two different experiments and an aggregate over both answers neither. This
    is the key that separates them.

    Only meaningful under ``fold_ref="reference"``, where the partition does not
    depend on which arm is being scored: the pool's fold assignment is then the
    same for the arm that contains it and the arm that does not, so the two are
    comparable fold by fold.
    """
    if fold_ref != "reference":
        raise SystemExit("--by-region needs fold_ref=reference; under 'union' "
                         "each arm cuts its own folds and a fold id is not "
                         "shared between the arm with the pool and the one "
                         "without")
    cols = ["loc_id", "lon", "lat", "block_id"]
    test_frame = sp_llto.test_pool_frame(sp_data.load_plots())
    pool_frame = sp_data.load_glance(pool)
    everything = pd.concat([pool_frame[cols], test_frame[cols]], ignore_index=True)
    out = {}
    for seed in seeds:
        mapping = sp_llto.location_folds(everything, n_folds, int(seed), split,
                                         space, reference=test_frame[cols])
        counts = pool_frame["loc_id"].map(mapping).value_counts()
        for fold, n in counts.items():
            if n >= min_rows:
                out[(int(seed), int(fold))] = int(n)
    return out


def report(read: str = "t1_all", sort: str = "macro_f1") -> pd.DataFrame:
    """Seed-averaged ledger, which is the only way these numbers may be read."""
    if not LEDGER.exists():
        raise SystemExit(f"{LEDGER} does not exist yet -- run something first")
    frame = pd.read_csv(LEDGER)
    frame = frame.loc[frame["read"] == read]
    # The ledger is append-only, so re-running a cell leaves both rows in it.
    # The later one is the one that was meant; averaging the pair would silently
    # weight whichever cell happened to be run twice.
    keys = ["dataset", "arch", "split", "weight", "fold_ref", "n_folds", "seed"]
    frame = frame.drop_duplicates(subset=keys + ["read"], keep="last")
    # n_folds is a grouping key, not a covariate: a 5-fold and a 10-fold run hold
    # out different amounts of the world and their means are not comparable.
    # fold_ref likewise -- it changes where the fold boundaries are.
    group = frame.groupby(["dataset", "arch", "split", "weight", "fold_ref",
                           "n_folds"])
    out = group.agg(
        seeds=("seed", "nunique"),
        macro_f1=("macro_f1", "mean"), macro_f1_sd=("macro_f1", "std"),
        accuracy=("accuracy", "mean"),
        f1_artificial=("f1_artificial", "mean"),
        f1_cropland=("f1_cropland", "mean"),
        f1_nature=("f1_nature", "mean"),
        cropland_as_nature=("cropland_as_nature", "mean"),
        seconds=("seconds", "mean"),
    ).reset_index()
    return out.sort_values(sort, ascending=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", default=["endpoints"])
    parser.add_argument("--archs", nargs="+", default=["linear"])
    parser.add_argument("--splits", nargs="+", default=["llto"],
                        choices=list(sp_llto.SPLITS))
    parser.add_argument("--weights", nargs="+", default=["none"],
                        choices=list(sp_llto.WEIGHTS))
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--seed0", type=int, default=0)
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--fold-ref", default="reference",
                        choices=list(sp_llto.FOLD_REFS),
                        help="whose coordinates place the folds (see llto.py)")
    parser.add_argument("--space", default="sphere", choices=["sphere", "lonlat"])
    parser.add_argument("--test-year", type=int, default=2024)
    parser.add_argument("--epochs", type=int, default=None,
                        help="override the MLP epoch count (default 30)")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--report", action="store_true")
    parser.add_argument("--migrate", action="store_true",
                        help="add today's columns to the ledgers on disk")
    parser.add_argument("--confusion", action="store_true",
                        help="accuracy and the 3x3 confusion matrix per arm; "
                             "re-runs the arms, does not touch the ledger")
    parser.add_argument("--paired", nargs=2, metavar=("A", "B"),
                        help="fold-by-fold difference between two arms")
    parser.add_argument("--paired-by", default="dataset",
                        choices=["dataset", "arch"],
                        help="which axis --paired compares. 'arch' holds the "
                             "pool fixed at --datasets[0] and is how section Y "
                             "reads the capacity sweep")
    parser.add_argument("--metric", default="macro_f1",
                        help="the fold-ledger column --paired differences; "
                             "artificial_as_cropland is the Oslo constraint")
    parser.add_argument("--by-region", action="store_true",
                        help="with --paired, split the folds by whether the "
                             "regional pool reaches them (see --region-pool)")
    parser.add_argument("--region-pool", default="lucas",
                        help="the pool whose footprint --by-region splits on")
    parser.add_argument("--read", default="t1_all")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    if args.list:
        print("datasets:")
        for name, (_, desc) in sp_data.ARMS.items():
            print(f"  {name:<22s} {desc}")
        print("\narchitectures:")
        for name, (_, desc) in sp_models.ARCHS.items():
            print(f"  {name:<22s} {desc}")
        print("\nsplits:")
        print("  llto | loc | random | block   (see statepre/llto.py)")
        return

    if args.migrate:
        migrate()
        return

    if args.paired:
        a, b = args.paired
        table = paired(a, b, arch=args.archs[0], split=args.splits[0],
                       weight=args.weights[0], fold_ref=args.fold_ref,
                       n_folds=args.n_folds, metric=args.metric,
                       by=args.paired_by, dataset=args.datasets[0])
        if not len(table):
            raise SystemExit(f"no folds shared by {a} and {b} at "
                             f"n_folds={args.n_folds}, fold_ref={args.fold_ref}")
        if args.by_region:
            reach = region_folds(args.region_pool, args.n_folds,
                                 sorted(table["seed"].unique()),
                                 split=args.splits[0], fold_ref=args.fold_ref)
            table["pool_rows"] = [reach.get((s, f), 0) for s, f
                                  in zip(table["seed"], table["fold"])]
            table["region"] = np.where(table["pool_rows"] > 0,
                                       f"in_{args.region_pool}", "elsewhere")

        def summarise(label: str, diff: pd.Series) -> None:
            print(f"{label:<22s} mean {diff.mean():+.4f}  sd {diff.std():.4f}  "
                  f"{int((diff > 0).sum())}/{len(diff)} folds positive")

        with pd.option_context("display.width", 220, "display.max_rows", 300):
            print(table.to_string(index=False,
                                  float_format=lambda v: f"{v:.4f}"))
        print(f"\n{a} - {b}, n_folds={args.n_folds}, fold_ref={args.fold_ref}")
        summarise("  all folds", table["diff"])
        if args.by_region:
            # The split is the point: a regional pool is *absent* from training
            # on exactly the folds where it would have been nearest the test
            # plots, so an aggregate mixes "helps at home" with "transfers".
            for region, part in table.groupby("region"):
                summarise(f"  {region}", part["diff"])
        return

    if args.report:
        with pd.option_context("display.width", 200,
                               "display.max_columns", 40):
            print(report(args.read).to_string(index=False,
                                              float_format=lambda v: f"{v:.4f}"))
        return

    arch_kw = {} if args.epochs is None else {"epochs": args.epochs}
    plots = sp_data.load_plots()
    test_frame = sp_llto.test_pool_frame(plots)

    if args.confusion:
        seeds = range(args.seed0, args.seed0 + args.seeds)
        print(f"read={args.read}  split={args.splits[0]}  "
              f"n_folds={args.n_folds}  fold_ref={args.fold_ref}")
        for dataset in args.datasets:
            for archname in args.archs:
                kw = arch_kw if archname.startswith("mlp") else {}
                table, scores = confusion(
                    dataset, archname, seeds=seeds, n_folds=args.n_folds,
                    split=args.splits[0], fold_ref=args.fold_ref,
                    weight=args.weights[0], read=args.read, plots=plots,
                    test_frame=test_frame, **kw)
                print_confusion(f"{dataset} / {archname}", table, scores,
                                args.seeds)
        return
    print(f"plots: {len(plots):,} complete at all 7 years, "
          f"{int(plots['changed'].sum()):,} changed at coarse3\n"
          f"test pool: RECOVER observed rows at {args.test_year}\n", flush=True)

    for dataset in args.datasets:
        for archname in args.archs:
            for split in args.splits:
                for weight in args.weights:
                    scores = []
                    for seed in range(args.seed0, args.seed0 + args.seeds):
                        kw = arch_kw if archname.startswith("mlp") else {}
                        rows = one(dataset, archname, seed=seed,
                                   n_folds=args.n_folds, split=split,
                                   space=args.space, weight=weight,
                                   fold_ref=args.fold_ref,
                                   test_year=args.test_year, plots=plots,
                                   test_frame=test_frame,
                                   verbose=not args.quiet, **kw)
                        append(LEDGER, rows)
                        by_read = {r["read"]: r for r in rows}
                        scores.append(by_read["t1_all"]["macro_f1"])
                        print(f"  {dataset:<20s} {archname:<13s} {split:<7s} "
                              f"{weight:<8s} seed {seed}  "
                              f"t1_all={by_read['t1_all']['macro_f1']:.4f}  "
                              f"changed={by_read['t1_changed']['macro_f1']:.4f}  "
                              f"stable={by_read['t1_stable']['macro_f1']:.4f}  "
                              f"({by_read['t1_all']['seconds']:.0f}s)", flush=True)
                    print(f"  {'':<20s} {'':<13s} {split:<7s} {weight:<8s} "
                          f"MEAN      t1_all={np.mean(scores):.4f} "
                          f"+/-{np.std(scores):.4f} over {len(scores)} seeds\n",
                          flush=True)

    print(f"appended to {LEDGER}")


if __name__ == "__main__":
    main()
