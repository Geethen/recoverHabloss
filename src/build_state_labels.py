"""Single-date land-cover state pools for the siamese encoder (section N13).

Every label in the project is a *transition* -- ``lc_2018 -> lc_2024`` on 6,416
plots. The siamese shared encoder ``f`` is the first structure in the ledger
that is a function of **one** date, so a single-date state label is a valid
input to it and can supervise ``f`` directly through an auxiliary head. The flat
``wide`` trunk has no such entry point: its first layer eats a 192-column
``[2018 | 2024 | diff]`` vector.

This script builds those single-date pools from three public sources and
harmonises all of them onto the project's three-state ``coarse3`` legend (Nature
/ Cropland / Artificial, ``experiment_merged_legend.LEGENDS``).

Why 2018 and not any year
-------------------------
AlphaEarth annual embeddings start in 2017 and the plot extraction already holds
2018..2024 (``metadata_embeddings_habloss_recover_annual.json``). 2018 is the
year LUCAS and GLanCE are densest in *and* an endpoint of the project's
transition, so a state prior learned here lands on the ``from`` side of every
commissioned ``from -> to`` class. ``hcropland`` is the exception and is
extracted at **2020**, the year its map is valid for; the pool frame carries a
``year`` column for exactly this reason and the phase's encoder is single-date,
so a 2020 row is as usable as a 2018 one.

LUCAS
-----
EU-27 in-situ field survey, 337,854 points in 2018. Labels come from the ELC10
reference set (``projects/nina/ELC10/LUCAS_combined_reference_final``, 71,485
points), which is the vote-cleaned 8-class aggregation of LUCAS LC1 already used
in this group. The quality filters replicate the LUCAS notebook exactly: keep
only >=50% cover, and drop the classes with known thematic/spectral ambiguity
(A22 linear artificial features, B55 temporary grassland, E30 spontaneously
re-vegetated surfaces, F40 other bare land).

**LUCAS reaches 8 of the 83 spatial blocks the folds are built on** -- only 778
of 6,416 RECOVER plots (12.1%) fall in a coarse Europe bbox. It is dense but
geographically narrow.

GLanCE
------
Global, 1,874,995 training units, 1984-2020, interpreted as CCDC time segments
(Stanimirova et al. 2023). 1,137,216 units have a segment covering 2018.

Three restrictions matter and are applied here:

* **Stable segments only.** A transitional segment is land cover *mid-change*;
  it has no single state at 2018, which is the one thing this pool is for.
  **``Segment_Type`` is null on 95.7% of the units covering 2018** -- it is
  populated only for Dataset_Codes 1 (STEP), 2 (CLUSTERING) and 999
  (Training_augment), and so is ``LC_Confidence``. Filtering on it therefore
  silently selects the in-house interpreted subset, which is why ``--glance-
  quality`` makes the choice explicit rather than letting a null do it. ``strict``
  keeps ``Segment_Type == 0``; ``broad`` uses ``Change == false``, which *is*
  populated everywhere.
* **Herbaceous with no level-2 class is dropped.** ``Glance_Class_ID_level1 ==
  7`` splits into Grassland (11) and Agriculture (12) at level 2, and 88k units
  carry level2 == 0. That split *is* the Nature/Cropland boundary this project's
  change-F1 ceiling sits on (``analyse_label_noise.py``); guessing it would
  import exactly the systematic offset the diagnostic downstream is built to
  detect.
* **There is no Cropland at level 1.** GLanCE level 1 is Water, Ice/snow,
  Developed, Barren, Trees, Shrub, Herbaceous. Agriculture is level *2* only, so
  ``Glance_Class_ID_level2`` is the field to read for the class this whole line
  of work is about.

hcropland30
-----------
A 100,000-point sample of a **hybrid 30 m global cropland map at 2020**, held
locally at ``data/hcropland30/point_samples.shp``. Two columns carry the label:
``type`` is 1 for cropland and 0 for everything else, and ``uncertaint`` is the
sample standard deviation of the six contributing maps' binary votes -- 0.0 when
all six agree, then 0.408 / 0.516 / 0.548 for 1, 2 and 3 dissenters. That vote
structure is the whole reason this source is worth trying: ``--hcrop-quality
strict`` keeps only the unanimous points, which is the closest thing a map
product offers to an interpreted label.

**Only the cropland points are used, and the 67,657 non-cropland points are
dropped.** ``type == 0`` means "not cropland", which in coarse3 is
``{artificial, nature}`` *plus* the classes coarse3 has no home for -- water,
ice, barren -- and GLanCE drops those rather than force them into Nature. A
negative therefore cannot be turned into a state, not even a set-valued one,
without a second source to say which. A pool of cropland alone is degenerate on
its own and is only ever concatenated with one that carries the other two
states; ``statepre.data`` has no ``hcropland``-only arm for that reason.

The source is aimed at a named failure: ``STATE_PRETRAIN_RESEARCH.md`` section
V1b diagnosed LUCAS as negative because it returns 66% of RECOVER's Cropland as
Nature, on the same Cropland/Nature boundary the project's change-F1 ceiling
sits on. This pool is 32,343 cropland labels spread over 60 of the 83 blocks --
global reach, unlike LUCAS's 8 -- pointed at exactly that boundary. It is also,
unavoidably, **another model's decision boundary** rather than an interpreted
label, which is the concern this file already records against GLanCE's GLC30 and
MapBiomas components; the unanimity filter is the mitigation, and the
``glance_cropdup_endpoints`` control is what separates "these labels are
informative" from "the pool got 32k more cropland rows".

``Dataset_Code`` rides along on every row because GLanCE is not one thing:
791,177 of the 1,137,216 units covering 2018 (70%) are MapBiomas over northern
South America, and several components are sampled from *map products* rather
than interpreted -- GLC30 (705), Training_augment from the World Settlement
Footprint and Global Surface Water (999), MODIS_algo (700), MapBiomas (5).
Training on those imports another model's decision boundary, which is a live
concern here because Training_augment's built-up is the World Settlement
Footprint and stable built-up is the deployed model's remaining win. Keeping the
code as a column is what makes that ablatable instead of baked in.

**The two quality settings differ far more in geography than in size, and it is
the reason ``strict`` is the default.** The RECOVER plots are globally spread
(83 blocks of 20 degrees, lon -147..158). ``strict`` matches that -- Oceania
28.6%, Europe 21.1%, Africa 19.0%, Asia 11.6%, N.America 10.8%, S.America 8.9%.
``broad`` is 75.8% South America because MapBiomas is 785,005 of its 1,057,821
rows. Twenty-three times more labels, drawn almost entirely from one continent
and one map product, is not obviously the better pool.

Usage::

    # counts only, no extraction
    python build_state_labels.py --dry-run

    # the two GEE pools, 5,000 per state per source
    python build_state_labels.py --per-class 5000

    # the local cropland map sample, every cropland point, at 2020
    python build_state_labels.py --sources hcropland
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import ee
import pandas as pd

from extract_embeddings_gee import (
    EMBEDDING_ASSET,
    assign_blocks,
    build_embedding_stack,
    fetch_dataframe,
    init_gee,
    point_collection,
    run_extraction,
)
from project_paths import project_data_dir

PROJECT = "ee-gsingh"
YEAR = 2018
#: The year each source's labels are valid for, and therefore the year its
#: embeddings are pulled at. Not a free choice: a state label read against the
#: wrong year's embedding is a mislabelled row wherever the ground changed.
SOURCE_YEAR = {"lucas": 2018, "glance": 2018, "hcropland": 2020}
DEFAULT_OUTPUT = project_data_dir("embeddings")
HCROPLAND_SHAPEFILE = project_data_dir("hcropland30") / "point_samples.shp"

GLANCE_ASSET = "projects/sat-io/open-datasets/GLANCE/GLANCE_TRAINING_DATA_V1"
LUCAS_ASSET = "JRC/LUCAS_HARMO/THLOC/V1"
ELC10_ASSET = "projects/nina/ELC10/LUCAS_combined_reference_final"

#: The project legend these pools must land on. Lower-case to match
#: ``experiment_merged_legend.target_for_legend``, which lower-cases before
#: mapping -- the pools are written in the same convention as ``lc_2018`` so a
#: state head reads one vocabulary.
STATES = ("artificial", "cropland", "nature")

#: ELC10's 8 classes -> coarse3. Bare land and Water have no home in a legend of
#: Nature/Cropland/Artificial and are dropped rather than forced into Nature:
#: 2,884 of 71,485 ELC10 points (4%). Wetland, Woodland, Shrubland and Grassland
#: are all Nature, which is what the project's own `nature - forest` /
#: `nature - other` sub-labels collapse to.
ELC10_TO_STATE = {
    "Artificial land": "artificial",
    "Cropland": "cropland",
    "Woodland": "nature",
    "Shrubland": "nature",
    "Grassland": "nature",
    "Wetland": "nature",
}

#: LUCAS LC1 codes dropped for thematic/spectral ambiguity, from the notebook.
LUCAS_DROP_LC1 = ["A22", "B55", "E30", "F40", ""]
LUCAS_KEEP_PERC = ["> 75 %", "50 - 75 %"]

#: GLanCE level-2 codes -> coarse3. Level 1 is used only for Developed (3), the
#: classes with no level-2 split (Trees 5, Shrub 6), and the drops.
GLANCE_L2_TO_STATE = {
    11: "nature",     # Grassland
    12: "cropland",   # Agriculture
    13: "nature",     # Moss/lichen
}
GLANCE_L1_TO_STATE = {
    3: "artificial",  # Developed
    5: "nature",      # Trees
    6: "nature",      # Shrub
}
#: Water (1), Ice/snow (2) and Barren (4) have no coarse3 home and are dropped.
GLANCE_DROP_L1 = (1, 2, 4)


def glance_pool(quality: str = "strict") -> ee.FeatureCollection:
    """GLanCE units whose *stable* segment covers 2018, tagged with a state.

    ``quality='strict'`` keeps ``Segment_Type == 0``, which -- because that field
    is null outside STEP/CLUSTERING/Training_augment -- restricts the pool to the
    in-house interpreted components. ``quality='broad'`` uses ``Change == false``
    instead, which is populated on every row and admits MapBiomas, LUCAS, GLC30
    and the rest.
    """
    stable = (ee.Filter.eq("Segment_Type", 0) if quality == "strict"
              else ee.Filter.eq("Change", False))
    fc = (
        ee.FeatureCollection(GLANCE_ASSET)
        .filter(ee.Filter.lte("Start_Year", YEAR))
        .filter(ee.Filter.gte("End_Year", YEAR))
        .filter(stable)
        .filter(ee.Filter.inList("Glance_Class_ID_level1", list(GLANCE_DROP_L1)).Not())
    )

    l2_from = list(GLANCE_L2_TO_STATE)
    l2_to = [GLANCE_L2_TO_STATE[k] for k in l2_from]
    l1_from = list(GLANCE_L1_TO_STATE)
    l1_to = [GLANCE_L1_TO_STATE[k] for k in l1_from]

    def tag(feature):
        l1 = feature.getNumber("Glance_Class_ID_level1")
        l2 = feature.getNumber("Glance_Class_ID_level2")
        # Level 2 wins where it resolves the class (Grassland vs Agriculture is
        # the whole reason this pool exists); level 1 covers Developed, Trees
        # and Shrub, which have no level-2 split that matters here.
        state = ee.Algorithms.If(
            ee.List(l2_from).contains(l2),
            ee.Dictionary.fromLists(
                [ee.Number(k).format("%d") for k in l2_from], l2_to
            ).get(l2.format("%d")),
            ee.Dictionary.fromLists(
                [ee.Number(k).format("%d") for k in l1_from], l1_to
            ).get(l1.format("%d"), ""),
        )
        return feature.set({"state": state, "lon": feature.get("Lon"),
                            "lat": feature.get("Lat")})

    # Herbaceous (7) with an unresolved level 2 falls through to "" and is cut
    # here -- the drop is explicit rather than silent.
    return fc.map(tag).filter(ee.Filter.neq("state", ""))


def lucas_pool() -> ee.FeatureCollection:
    """LUCAS 2018 points carrying their ELC10 class, tagged with a state."""
    lucas = (
        ee.FeatureCollection(LUCAS_ASSET)
        .filter(ee.Filter.eq("year", YEAR))
        .filter(ee.Filter.inList("lc1_perc", LUCAS_KEEP_PERC))
        .filter(ee.Filter.inList("lc1", LUCAS_DROP_LC1).Not())
    )
    elc10 = ee.FeatureCollection(ELC10_ASSET)

    joined = ee.Join.inner("lucas", "elc10").apply(
        lucas, elc10,
        ee.Filter.equals(leftField="point_id", rightField="POINT_ID"),
    )

    keep = list(ELC10_TO_STATE)
    values = [ELC10_TO_STATE[k] for k in keep]
    lookup = ee.Dictionary.fromLists(keep, values)

    def tag(pair):
        left = ee.Feature(pair.get("lucas"))
        right = ee.Feature(pair.get("elc10"))
        coords = left.geometry().coordinates()
        return left.set({
            "state": lookup.get(right.get("LC"), ""),
            "elc10_class": right.get("LC"),
            "lon": coords.get(0),
            "lat": coords.get(1),
        })

    # Bare land and Water fall through to "" and are cut here.
    return ee.FeatureCollection(joined).map(tag).filter(ee.Filter.neq("state", ""))


def hcropland_points(quality: str = "strict",
                     path: Path | None = None) -> pd.DataFrame:
    """The cropland points of the hybrid map sample, as a local lon/lat table.

    Returns a DataFrame rather than an ``ee.FeatureCollection`` because this
    source is a file on disk, not a hosted asset -- the extraction tail is shared
    either way, since both GEE pools are already sent up through
    ``point_collection`` from a client-side table.

    ``quality='strict'`` keeps ``uncertaint == 0``, the points all six
    contributing maps call cropland; ``'all'`` keeps every ``type == 1`` point,
    including the 2,785 whose uncertainty is null. The uncertainty is carried
    through on both so the filter can be re-cut downstream without a
    re-extraction.
    """
    import geopandas as gpd

    frame = gpd.read_file(path or HCROPLAND_SHAPEFILE)
    if frame.crs is not None and frame.crs.to_epsg() != 4326:
        frame = frame.to_crs(4326)
    frame = frame.loc[frame["type"] == 1].copy()
    if quality == "strict":
        # Null uncertainty is not unanimity; it is an unknown and is cut here.
        frame = frame.loc[frame["uncertaint"] == 0.0]
    elif quality != "all":
        raise ValueError(f"hcrop quality must be 'strict' or 'all', got {quality!r}")

    out = pd.DataFrame({
        "lon": frame.geometry.x.to_numpy(),
        "lat": frame.geometry.y.to_numpy(),
        "state": "cropland",
        "uncertainty": frame["uncertaint"].to_numpy(),
        "map_id": frame["Id"].to_numpy(),
    })
    # One pair of coordinates is duplicated in the sample file. Two identical
    # points are one label with two votes, and the fold map would collapse them
    # onto one location anyway, so the duplicate goes now rather than silently.
    return out.drop_duplicates(["lon", "lat"]).reset_index(drop=True)


def stratified_sample(fc: ee.FeatureCollection, per_class: int, seed: int,
                      keep: list[str]) -> pd.DataFrame:
    """``per_class`` rows per coarse3 state, drawn on a seeded random column.

    Sampling is per state rather than uniform because the class this project
    needs most is the rarest: GLanCE has 600k Trees covering 2018 against 12,676
    Developed, and a uniform draw would return a pool that is 90% Nature -- the
    one state the RECOVER plots are already rich in.
    """
    fc = fc.randomColumn("rand", seed)
    frames = []
    for state in STATES:
        subset = (fc.filter(ee.Filter.eq("state", state))
                    .sort("rand")
                    .limit(per_class)
                    .select(keep))
        frame = fetch_dataframe(subset)
        print(f"  {state:11s} {len(frame):>7,}", flush=True)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def report_counts(name: str, fc: ee.FeatureCollection, year: int) -> dict:
    counts = fc.aggregate_histogram("state").getInfo()
    total = sum(counts.values())
    print(f"\n{name}: {total:,} usable single-date state labels at {year}")
    for state in STATES:
        n = counts.get(state, 0)
        print(f"  {state:11s} {n:>9,}  ({n / max(total, 1):.1%})")
    return counts


def chunk_blocks(points: pd.DataFrame, max_rows: int) -> dict[str, pd.DataFrame]:
    """``block_id -> rows``, splitting any block over ``max_rows`` into parts.

    One block is one Earth Engine request whose feature collection is built
    client-side, so a block's size is a request payload size. The GEE pools are
    ~150 points a block and never came near a limit; hcropland30 puts 2,255 into
    its densest, which is where a single request starts to time out.

    The part suffix lives in the *shard name only* -- ``run_extraction`` writes
    the key it is given into ``block_id``, and that column is a fold unit
    (``llto`` ``block``) and a weighting unit (``per_cell``), so it is stripped
    again before the frame is written. ``max_rows <= 0`` disables the split, so
    the shards already on disk for lucas and glance stay addressable.
    """
    blocks = dict(tuple(points.groupby("block_id")))
    if max_rows <= 0:
        return blocks
    out = {}
    for name, block in blocks.items():
        if len(block) <= max_rows:
            out[name] = block
            continue
        for part, start in enumerate(range(0, len(block), max_rows)):
            out[f"{name}#{part:02d}"] = block.iloc[start:start + max_rows]
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", default=PROJECT)
    parser.add_argument("--sources", nargs="+", default=["lucas", "glance"],
                        choices=["lucas", "glance", "hcropland"])
    parser.add_argument("--per-class", type=int, default=5000,
                        help="rows per coarse3 state per source (GEE pools only)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--block-size", type=float, default=20.0)
    parser.add_argument("--max-block-rows", type=int, default=0,
                        help="split blocks larger than this into separate "
                             "requests; 0 disables (see chunk_blocks)")
    parser.add_argument("--max-workers", type=int, default=8)
    parser.add_argument("--scale", type=int, default=10)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--glance-quality", default="strict",
                        choices=["strict", "broad"],
                        help="strict = in-house interpreted (Segment_Type==0); "
                             "broad = every dataset with Change==false")
    parser.add_argument("--hcrop-quality", default="all",
                        choices=["strict", "all"],
                        help="strict = all six maps voted cropland; all = every "
                             "cropland point. 'all' is the default because the "
                             "uncertainty rides along and the cut is cheaper to "
                             "make downstream than a re-extraction is")
    parser.add_argument("--dry-run", action="store_true",
                        help="print pool sizes per state and stop")
    args = parser.parse_args()

    init_gee(args.project)
    builders = {"lucas": lucas_pool,
                "glance": lambda: glance_pool(args.glance_quality)}
    extras = {"lucas": ["elc10_class", "point_id"],
              "glance": ["Dataset_Code", "Continent_Code", "Start_Year",
                         "End_Year", "LC_Confidence", "Glance_ID"],
              "hcropland": ["uncertainty", "map_id"]}
    tags = {"glance": f"glance_{args.glance_quality}",
            "hcropland": f"hcropland_{args.hcrop_quality}"}

    for source in args.sources:
        year = SOURCE_YEAR[source]
        tag = tags.get(source, source)

        if source == "hcropland":
            # A local file: no Earth Engine query to count, and no stratified
            # draw to make -- the pool is one state and every point of it is kept.
            points = hcropland_points(args.hcrop_quality)
            counts = points["state"].value_counts().to_dict()
            print(f"\n{tag}: {len(points):,} cropland points at {year} "
                  f"({args.hcrop_quality})")
            if args.dry_run:
                continue
        else:
            fc = builders[source]()
            counts = report_counts(tag, fc, year)
            if args.dry_run:
                continue
            keep = ["state", "lon", "lat", *extras[source]]
            print(f"\nsampling {source} at {args.per_class:,} per state:",
                  flush=True)
            points = stratified_sample(fc, args.per_class, args.seed, keep)

        points["source"] = source
        points["sid"] = [f"{source}_{i:07d}" for i in range(len(points))]

        points = assign_blocks(points, args.block_size)
        blocks = chunk_blocks(points, args.max_block_rows)
        print(f"{source}: {len(points):,} points in "
              f"{points['block_id'].nunique()} blocks, "
              f"{len(blocks)} requests", flush=True)

        stack = build_embedding_stack((year,))
        carried = ["sid", "state", "lon", "lat", "source", *extras[source]]
        frame = run_extraction(
            blocks, stack, carried,
            shard_dir=args.output_dir / f"state_shards_{tag}",
            scale=args.scale, max_workers=args.max_workers,
            years=(year,), build_collection=point_collection,
        )
        # Undo the request split: the part suffix is a shard name, not a block.
        frame["block_id"] = frame["block_id"].str.split("#").str[0]

        bands = [f"A{i:02d}_{year}" for i in range(64)]
        complete = frame[bands].notna().all(axis=1)
        dropped = int((~complete).sum())
        frame = frame.loc[complete].reset_index(drop=True)

        out = args.output_dir / f"state_labels_{tag}_{year}.parquet"
        frame.to_parquet(out, index=False)
        meta = {
            "source": source, "year": year, "embedding_asset": EMBEDDING_ASSET,
            "scale_m": args.scale, "per_class": args.per_class,
            "seed": args.seed, "pool_counts": counts,
            "glance_quality": args.glance_quality if source == "glance" else None,
            "hcrop_quality": args.hcrop_quality if source == "hcropland" else None,
            "n_sampled": int(len(points)), "n_written": int(len(frame)),
            "n_dropped_no_embedding": dropped,
            "state_counts": frame["state"].value_counts().to_dict(),
            "n_blocks": int(frame["block_id"].nunique()),
        }
        (args.output_dir / f"metadata_state_labels_{tag}_{year}.json").write_text(
            json.dumps(meta, indent=2), encoding="utf-8")
        print(f"{source}: wrote {len(frame):,} rows to {out} "
              f"({dropped:,} dropped for missing embedding)", flush=True)


if __name__ == "__main__":
    main()
