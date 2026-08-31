"""UMAP of the AlphaEarth embedding space, with a clickable map link per plot.

Four projections of the same 6,492 labelled plots, so the question "what does
this structure mean" can be asked of each block the model actually reads:

``2018`` / ``2024``  one endpoint's 64-D embedding on its own -- state structure.
``diff``             the 64-D difference block -- change structure. This is the
                     one to look at when a transition class behaves oddly.
``both``             2018 and 2024 concatenated (128-D), the model's own view.

Cosine is the metric on the endpoint and concatenated blocks because that is the
distance the change scalars already use (``twotower_lab.change_scalar_arrays``);
the ``diff`` block is not unit-scaled, so it gets Euclidean.

The projection is *unsupervised* -- labels are attached afterwards for colour
only, never fed to UMAP. A class that separates here separated without being
told to.

External state pools
--------------------
``--pools lucas glance_strict`` adds the single-date label pools to the **2018**
view, projected jointly with the RECOVER plots so all three sources land in one
space. This is the picture behind ``STATE_PRETRAIN_RESEARCH.md`` section V1b: the
question is not whether LUCAS is noisy but whether its *Nature* sits where
RECOVER's *Nature* sits, and the answer is visible by colouring on
**Source x state** and toggling one source off at a time.

A pool row has no 2024 embedding, so it exists in the ``2018`` view and nowhere
else. Rather than drop those rows from the table -- which would make a point mean
different things in different views -- they carry ``null`` coordinates in the
views they cannot appear in, and the page hides them there. The transition-only
fields (``state_2024``, ``transition``, ``changed``) carry an explicit
``single date`` level for the same reason: a pool row is not a stable plot, and
coding it as one would put 25,000 fake Stables in the legend.

Writes a coordinate table and a self-contained interactive page: hover for the
plot, click for its transition and links out to Google Earth / Maps / Timelapse
at that exact point, which is how a suspicious plot gets adjudicated.

Usage::

    /home/geethen.singh/.pixi/envs/geo/bin/python src/umap_embedding.py
    ... --views diff,both --n-neighbors 50
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from experiment_merged_legend import LEGENDS
from model_zoo import DEFAULT_INPUT
from project_paths import REPO_ROOT, project_data_dir

SEED = 20250717
OUT_PARQUET = project_data_dir("analysis_results") / "umap_embedding.parquet"
OUT_HTML = REPO_ROOT / "outputs" / "umap_embedding.html"

#: view name -> (column suffix(es), UMAP metric). Order is the page's order, so
#: the model's own view lands first and the change block second.
VIEWS: dict[str, tuple[tuple[str, ...], str]] = {
    "both": (("2018", "2024"), "cosine"),
    "diff": (("diff",), "euclidean"),
    "2018": (("2018",), "cosine"),
    "2024": (("2024",), "cosine"),
}
VIEW_DESC = {
    "2018": "AlphaEarth 2018 embedding (64-D, cosine)",
    "2024": "AlphaEarth 2024 embedding (64-D, cosine)",
    "diff": "2024 minus 2018 difference block (64-D, Euclidean)",
    "both": "2018 and 2024 concatenated (128-D, cosine) — the model's view",
}

#: The external single-date pools, as written by ``build_state_labels.py``.
#: ``glance_strict`` rather than ``glance_broad``: broad is 75.8% MapBiomas over
#: one continent and would colour the page with one product's decision boundary.
POOLS = ("lucas", "glance_strict")
#: What a transition-only field says about a row that has only one date. Not
#: "Stable" -- a pool row is not a plot that failed to change, it is a plot
#: nobody looked at twice.
SINGLE_DATE = "single date"


def embedding_columns(frame: pd.DataFrame, suffixes: tuple[str, ...]) -> list[str]:
    """The AlphaEarth columns for one or more year suffixes, in band order."""
    cols = [c for c in frame.columns
            if re.match(r"A\d\d_", c) and c.rsplit("_", 1)[1] in suffixes]
    if not cols:
        raise ValueError(f"No embedding columns for suffixes {suffixes}")
    # sort by suffix first so a concatenated view keeps its blocks contiguous
    return sorted(cols, key=lambda c: (suffixes.index(c.rsplit("_", 1)[1]), c))


def coarsen(series: pd.Series) -> pd.Series:
    """Map a raw legend value onto the 3-class legend shared by all sources."""
    cleaned = series.astype(str).str.strip().str.lower()
    unknown = sorted(set(cleaned) - set(LEGENDS["coarse3"]))
    if unknown:
        raise ValueError(f"Values outside the coarse3 legend: {unknown}")
    return cleaned.map(LEGENDS["coarse3"])


def project(values: np.ndarray, metric: str, n_neighbors: int,
            min_dist: float, seed: int) -> np.ndarray:
    import umap

    reducer = umap.UMAP(n_components=2, n_neighbors=n_neighbors,
                        min_dist=min_dist, metric=metric, random_state=seed,
                        verbose=False)
    return reducer.fit_transform(values).astype("float32")


def load_pool(name: str, directory: Path | None = None) -> pd.DataFrame:
    """One external state pool in the plot frame's column vocabulary.

    Only the 2018 block exists, so the returned frame carries ``A00_2018..`` and
    nothing else. ``PLOTID`` is prefixed with the pool name because a GLanCE
    segment id and a RECOVER PLOTID are both small integers and a collision would
    silently merge two points.
    """
    directory = directory or project_data_dir("embeddings")
    path = directory / f"state_labels_{name}_2018.parquet"
    if not path.exists():
        raise SystemExit(f"{path} not found — run build_state_labels.py first")
    pool = pd.read_parquet(path).reset_index(drop=True)
    out = pd.DataFrame({
        "PLOTID": name + ":" + pool["sid"].astype(str),
        "lon": pool["lon"].astype("float64").to_numpy(),
        "lat": pool["lat"].astype("float64").to_numpy(),
        "source": name.split("_")[0],
        "block_id": pool["block_id"].to_numpy(),
        "lc_2018": pool["state"].astype(str).str.lower().to_numpy(),
        "lc_2024": SINGLE_DATE,
    })
    bands = [f"A{i:02d}_2018" for i in range(64)]
    out[bands] = pool[bands].astype("float64").to_numpy()
    return out


def build_table(frame: pd.DataFrame, views: list[str], n_neighbors: int,
                min_dist: float, seed: int,
                pools: list[str] | None = None) -> pd.DataFrame:
    pool_frames = [load_pool(name) for name in (pools or [])]
    for name, pool in zip(pools or [], pool_frames):
        print(f"  + {name}: {len(pool):,} single-date rows")
    # One concat, so every downstream column is computed once over all sources
    # and a pool row cannot take a different code path than a plot.
    full = (pd.concat([frame] + pool_frames, ignore_index=True)
            if pool_frames else frame)
    is_pool = np.r_[np.zeros(len(frame), bool),
                    *[np.ones(len(p), bool) for p in pool_frames]] \
        if pool_frames else np.zeros(len(frame), bool)

    # `coarsen` refuses values outside the legend, which is the check that keeps
    # a mis-joined label from being silently plotted, so the pool rows are held
    # out of it rather than given a placeholder to pass it with.
    state18 = coarsen(full["lc_2018"])
    state24 = pd.Series(SINGLE_DATE, index=full.index, dtype=object)
    state24.loc[~is_pool] = coarsen(full.loc[~is_pool, "lc_2024"]).to_numpy()
    out = pd.DataFrame({
        "PLOTID": full["PLOTID"].to_numpy(),
        "lon": full["lon"].to_numpy("float64"),
        "lat": full["lat"].to_numpy("float64"),
        "source": full["source"].to_numpy(),
        "block_id": full["block_id"].to_numpy(),
        "lc_2018": full["lc_2018"].to_numpy(),
        "lc_2024": full["lc_2024"].to_numpy(),
        "state_2018": state18.to_numpy(),
        "state_2024": state24.to_numpy(),
        "transition": np.where(is_pool, SINGLE_DATE,
                               state18 + " → " + state24),
        "changed": np.where(is_pool, SINGLE_DATE,
                            np.where(state18 != state24, "Changed", "Stable")),
        "is_pool": is_pool,
    })

    for view in views:
        suffixes, metric = VIEWS[view]
        # A pool row has no 2024 block, so it can only join the 2018 view. Rows
        # that cannot are projected as NaN rather than dropped, so a row index
        # means the same point in every view and the page just hides them.
        cols = embedding_columns(full, suffixes)
        usable = full[cols].notna().all(axis=1).to_numpy()
        values = full.loc[usable, cols].to_numpy("float64")
        print(f"  {view:5s} {values.shape[1]:3d} columns, {usable.sum():,} rows, "
              f"metric={metric} ...", flush=True)
        xy = project(values, metric, n_neighbors, min_dist, seed)
        full_xy = np.full((len(full), 2), np.nan, dtype="float32")
        full_xy[usable] = xy
        out[f"umap_{view}_x"], out[f"umap_{view}_y"] = full_xy[:, 0], full_xy[:, 1]
    return out


# ---------------------------------------------------------------------------
# page
# ---------------------------------------------------------------------------
def _norm(values: np.ndarray) -> list[float | None]:
    """Centre and scale one axis to roughly [-1, 1] so the page needs no ranges.

    A row absent from this view is ``None``, and the range is taken over the rows
    that *are* present -- a NaN in the min/max would flatten the whole axis.
    """
    present = np.isfinite(values)
    if not present.any():
        raise ValueError("a view has no finite coordinates")
    lo, hi = float(values[present].min()), float(values[present].max())
    span = max(hi - lo, 1e-9)
    scaled = 2.0 * (values - lo) / span - 1.0
    return [round(float(v), 4) if ok else None for v, ok in zip(scaled, present)]


def page_payload(table: pd.DataFrame, views: list[str]) -> dict:
    """Everything the page draws, as parallel arrays of small ints and floats."""
    def factor(name: str) -> tuple[list[str], list[int]]:
        # SINGLE_DATE sorts to the end rather than alphabetically among the real
        # classes, so the legend reads as "the classes, then the rows that have
        # no such class" instead of hiding it between Cropland and Nature.
        values = pd.unique(table[name]).tolist()
        levels = sorted(v for v in values if v != SINGLE_DATE)
        if SINGLE_DATE in values:
            levels.append(SINGLE_DATE)
        index = {lab: i for i, lab in enumerate(levels)}
        return levels, [index[v] for v in table[name]]

    table = table.copy()
    # Coordinates cached before the pools existed hold `changed` as a bool.
    if table["changed"].dtype == bool:
        table["changed"] = np.where(table["changed"], "Changed", "Stable")
    # The read section V1b is about: does LUCAS's Nature sit where RECOVER's
    # Nature sits? Derived here rather than stored, so it stays fixable without
    # re-projecting. RECOVER's own sub-sources (`habloss_main`,
    # `habloss_landwater`) collapse to one label -- splitting them would make 15
    # legend entries out of a three-way comparison and overflow the palette,
    # which wraps by modulo and would hand two classes the same colour.
    table["source_state"] = np.where(
        table["is_pool"], table["source"].astype(str), "recover"
    ) + " · " + table["state_2018"].astype(str)

    fields = {}
    names = ["state_2018", "state_2024", "transition", "source",
             "lc_2018", "lc_2024"]
    if table["is_pool"].any():
        names.append("source_state")
    for name in names:
        levels, codes = factor(name)
        fields[name] = {"levels": levels, "codes": codes}
    # Fixed order, not alphabetical: the palette is indexed by code and Stable
    # has always been the recessive first colour. Sorting would swap the two.
    changed_levels = ["Stable", "Changed"] + (
        [SINGLE_DATE] if (table["changed"] == SINGLE_DATE).any() else [])
    fields["changed"] = {
        "levels": changed_levels,
        "codes": [changed_levels.index(v) for v in table["changed"]]}

    return {
        "n": int(len(table)),
        "plotid": table["PLOTID"].tolist(),
        "lon": [round(float(v), 6) for v in table["lon"]],
        "lat": [round(float(v), 6) for v in table["lat"]],
        "is_pool": [bool(v) for v in table["is_pool"]],
        "views": {v: {"x": _norm(table[f"umap_{v}_x"].to_numpy("float64")),
                      "y": _norm(table[f"umap_{v}_y"].to_numpy("float64")),
                      "desc": VIEW_DESC[v]}
                  for v in views},
        "view_order": views,
        "fields": fields,
    }


def write_page(payload: dict, path: Path, template: Path) -> None:
    html = template.read_text()
    blob = json.dumps(payload, separators=(",", ":"))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html.replace("__UMAP_DATA__", blob))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--views", default=",".join(VIEWS),
                        help=f"comma-separated subset of {','.join(VIEWS)}")
    parser.add_argument("--n-neighbors", type=int, default=30,
                        help="UMAP neighbourhood; larger favours global structure")
    parser.add_argument("--min-dist", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--parquet", type=Path, default=None)
    parser.add_argument("--html", type=Path, default=None)
    parser.add_argument("--from-parquet", action="store_true",
                        help="re-render the page from cached coordinates; "
                             "edit the template and use this, projecting again "
                             "would move every point for no reason")
    parser.add_argument("--pools", nargs="*", default=[], choices=list(POOLS),
                        help="external single-date state pools to project "
                             "alongside the plots (2018 view only)")
    args = parser.parse_args()

    views = [v.strip() for v in args.views.split(",") if v.strip()]
    unknown = [v for v in views if v not in VIEWS]
    if unknown:
        raise SystemExit(f"Unknown views {unknown}; choose from {list(VIEWS)}")

    # A pooled run writes beside the plots-only page rather than over it. The
    # 2018 manifold is fitted on whatever is handed to it, so adding 25,000 pool
    # rows *moves every plot* in that view -- the two pages are different
    # projections and overwriting one with the other would look like a bug in
    # whichever the reader saw second. Pass --html explicitly to replace it.
    suffix = "_pools" if args.pools else ""
    parquet = args.parquet or OUT_PARQUET.with_name(
        f"{OUT_PARQUET.stem}{suffix}{OUT_PARQUET.suffix}")
    html = args.html or OUT_HTML.with_name(
        f"{OUT_HTML.stem}{suffix}{OUT_HTML.suffix}")
    args.parquet, args.html = parquet, html

    template = Path(__file__).resolve().parent / "umap_page_template.html"
    if args.from_parquet:
        table = pd.read_parquet(args.parquet)
        views = [v for v in views if f"umap_{v}_x" in table.columns]
        # Coordinates cached before the pools existed have no such column.
        if "is_pool" not in table.columns:
            table["is_pool"] = False
        write_page(page_payload(table, views), args.html, template)
        print(f"rewrote {args.html} from {args.parquet}")
        return

    frame = pd.read_parquet(args.input)
    duplicated = int(frame["PLOTID"].duplicated().sum())
    if duplicated:
        print(f"dropping {duplicated} repeated PLOTIDs")
        frame = frame.drop_duplicates("PLOTID").reset_index(drop=True)
    # Same completeness filter the modelling frame uses, applied over *every*
    # block rather than per view, so all four projections share one row set and
    # a point means the same plot whichever view is on screen.
    every = [c for c in frame.columns if re.match(r"A\d\d_", c)]
    complete = frame[every].notna().all(axis=1)
    if not complete.all():
        print(f"dropping {int((~complete).sum())} plots with an incomplete embedding")
        frame = frame.loc[complete].reset_index(drop=True)
    print(f"{len(frame):,} plots, projecting {len(views)} view(s)")
    if args.pools and "2018" not in views:
        raise SystemExit("--pools needs the 2018 view; the pools have no other "
                         "date to be projected in")

    table = build_table(frame, views, args.n_neighbors, args.min_dist, args.seed,
                        pools=args.pools)
    args.parquet.parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(args.parquet, index=False)
    print(f"wrote {args.parquet}")

    write_page(page_payload(table, views), args.html, template)
    print(f"wrote {args.html}")


if __name__ == "__main__":
    main()
