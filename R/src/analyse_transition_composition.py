"""Reproduce weighted RECOVER and main-frame HABLOSS compositions."""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from compare_ppi_subsets import (  # noqa: E402
    build_lookup,
    read_design_strata,
)
from design_analysis import (  # noqa: E402
    read_shapefile_table,
    read_stratum_areas_parallel,
    stratified_ratio,
)


DEFAULT_RECOVER = Path("P:/155020_recover/WP1/R")
DEFAULT_HABLOSS = Path("P:/155069_habloss/R")
DEFAULT_OUTPUT = Path("data/analysis_results")


def habloss_lookup(habloss_root: Path) -> dict[int, str]:
    biomes = pd.read_csv(habloss_root / "data/biomeLookup.csv")
    classes = [
        "stable_nature",
        "stable_built",
        "stable_crop",
        "buffer_forest",
        "buffer_crop",
        "buffer_built",
        "loss_forest",
        "loss_crop",
        "loss_built_crop",
        "loss_built_nature",
    ]
    lookup: dict[int, str] = {}
    for biome in biomes.itertuples(index=False):
        abbreviation = "".join(
            word[:2] for word in re.findall(r"[A-Za-z]+", biome.BIOME_NAME2)
        )
        for class_id, label in enumerate(classes, start=1):
            stratum_id = class_id + (int(biome.BIOME_NUM2) - 1) * 10
            lookup[stratum_id] = f"{abbreviation}_{label}"
    return lookup


def read_habloss_membership(habloss_root: Path) -> pd.DataFrame:
    rows = []
    for filename in [
        "samples_biomes_cleaned.geojson",
        "samples_biomes_extra_cleaned.geojson",
    ]:
        path = habloss_root / "data/from_gee" / filename
        with path.open(encoding="utf-8") as handle:
            collection = json.load(handle)
        rows.extend(feature["properties"] for feature in collection["features"])
    return pd.DataFrame(rows)[["PLOTID", "stratum"]].drop_duplicates("PLOTID")


def classify_recover_reference(frame: pd.DataFrame) -> pd.Series:
    before = frame["lc_2018"].astype(str).str.strip().str.lower()
    after = frame["lc_2024"].astype(str).str.strip().str.lower()
    return pd.Series(
        [
            "uncertain"
            if a == "uncertain" or b == "uncertain"
            else (
                "built_loss"
                if a == "artificial" and b == "nature (not cropland / artificial)"
                else "stable_stable"
            )
            for a, b in zip(before, after)
        ],
        index=frame.index,
    )


def first_mode(series: pd.Series):
    """Match the R workflow: first value wins when modal counts tie."""
    values = list(pd.unique(series.dropna()))
    if not values:
        return None
    counts = series.value_counts(dropna=True)
    maximum = counts.max()
    return next(value for value in values if counts[value] == maximum)


def collapse_recover_labels(frame: pd.DataFrame) -> pd.DataFrame:
    data = frame.copy()
    data["r"] = classify_recover_reference(data)
    data = data.dropna(subset=["r"])
    return (
        data.groupby("REFID", sort=False)
        .agg(
            r=("r", first_mode),
            lc_2018=("lc_2018", first_mode),
            lc_2024=("lc_2024", first_mode),
        )
        .reset_index()
        .rename(columns={"REFID": "PLOTID"})
        .query("r != 'uncertain'")
    )


def read_clean_recover_labels(
    recover_root: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Return script-exact and corrected-unique RECOVER label tables.

    The current R script replaces only IDs flagged by its binary target label.
    The redo file also contains disagreements within the binary non-target
    class, so 54 plot IDs are subsequently bound twice. ``script_exact``
    preserves that behaviour for provenance; ``unique`` makes every available
    reverified record replace its original record.
    """
    sampler_dir = recover_root / "data/from_samplers"
    raw_path = sampler_dir / "public_recover_collection_2 - Sheet1 (3).csv"
    redo_path = sampler_dir / "public_recover_collection_2_redo - Sheet1.csv"
    raw = pd.read_csv(raw_path)
    raw["r"] = classify_recover_reference(raw)

    uncertain_text = (
        raw["lc_2018_uncertain"].astype("string").str.contains("if it", na=False)
        | raw["lc_2024_uncertain"].astype("string").str.contains("if it", na=False)
    )
    uncertain_ids = set(raw.loc[(raw["r"] == "uncertain") & uncertain_text, "REFID"])
    counts = raw.groupby("REFID", sort=False).size()
    double_ids = set(counts[counts == 2].index)
    disagreement_counts = (
        raw[raw["REFID"].isin(double_ids)]
        .groupby(["REFID", "r"], sort=False)
        .size()
        .reset_index(name="n")
    )
    disagree_ids = set(
        disagreement_counts.loc[disagreement_counts["n"] == 1, "REFID"]
    ) - uncertain_ids
    redo_ids = uncertain_ids | disagree_ids

    cleaned = collapse_recover_labels(raw)
    reverified = collapse_recover_labels(pd.read_csv(redo_path))
    script_exact = pd.concat(
        [cleaned[~cleaned["PLOTID"].isin(redo_ids)], reverified],
        ignore_index=True,
    )
    all_replaced_ids = redo_ids | set(reverified["PLOTID"])
    unique = pd.concat(
        [cleaned[~cleaned["PLOTID"].isin(all_replaced_ids)], reverified],
        ignore_index=True,
    )
    if unique["PLOTID"].duplicated().any():
        raise ValueError("Duplicate RECOVER plot IDs remain in corrected table")
    diagnostics = {
        "raw_rows": len(raw),
        "raw_unique_plotids": raw["REFID"].nunique(),
        "redo_ids": len(redo_ids),
        "reverified_ids_not_flagged_by_binary_rule": len(
            set(reverified["PLOTID"]) - redo_ids
        ),
        "script_exact_rows": len(script_exact),
        "script_exact_unique_plotids": script_exact["PLOTID"].nunique(),
        "corrected_unique_plotids": len(unique),
    }
    return script_exact, unique, diagnostics


def add_result(rows, analysis, frame_name, result):
    rows.append({"analysis": analysis, "frame": frame_name, **result})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recover-root", type=Path, default=DEFAULT_RECOVER)
    parser.add_argument("--habloss-root", type=Path, default=DEFAULT_HABLOSS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--area-workers", type=int, default=16)
    args = parser.parse_args()

    labels_path = args.habloss_root / "data/samples_habloss_recover_for_geethen.shp"
    labels = read_shapefile_table(labels_path)
    rows = []
    stratum_outputs = []

    # RECOVER: Artificial -> Cropland as a share of all observed exits from Artificial.
    recover_lookup = build_lookup(args.recover_root)
    recover_id_to_label = dict(
        zip(recover_lookup["stratum_id"], recover_lookup["stratum"])
    )
    recover_areas = read_stratum_areas_parallel(
        args.recover_root / "data/from_gee/areas_recover",
        recover_id_to_label,
        workers=args.area_workers,
    )
    script_labels, unique_labels, recover_diagnostics = read_clean_recover_labels(
        args.recover_root
    )
    design_strata = read_design_strata(args.recover_root)
    for frame_name, recover_labels in [
        ("recover_script_exact", script_labels),
        ("recover_corrected_unique", unique_labels),
    ]:
        recover = recover_labels.merge(
            design_strata,
            on="PLOTID",
            how="left",
            validate="many_to_one",
        )
        before = recover["lc_2018"].astype(str).str.lower()
        after = recover["lc_2024"].astype(str).str.lower()
        art_crop = (before == "artificial") & (after == "cropland")
        art_nature = (before == "artificial") & after.str.startswith("nature")
        result, detail = stratified_ratio(
            recover, recover_areas, art_crop, art_crop | art_nature
        )
        add_result(rows, "artificial_exit_to_cropland", frame_name, result)
        detail.insert(0, "analysis", "artificial_exit_to_cropland")
        detail.insert(1, "frame", frame_name)
        stratum_outputs.append(detail)

    # HABLOSS main frame only. The land-water augmentation has a separate frame.
    membership = read_habloss_membership(args.habloss_root)
    habloss = labels[labels["source"] == "habloss_main"].merge(
        membership,
        on="PLOTID",
        how="left",
        validate="one_to_one",
    )
    h_lookup = habloss_lookup(args.habloss_root)
    habloss_areas = read_stratum_areas_parallel(
        args.habloss_root / "data/from_gee/areas_biomes_with_buffer",
        h_lookup,
        workers=args.area_workers,
    )
    before = habloss["lc_2018"].astype(str)
    after = habloss["lc_2024"].astype(str)
    nature = before.str.startswith("Nature")
    other_nature = before == "Nature - other"
    forest = before == "Nature - forest"
    cropland = before == "Cropland"
    to_artificial = after == "Artificial"

    analyses = [
        (
            "new_artificial_from_nature",
            nature & to_artificial,
            (nature | cropland) & to_artificial,
        ),
        (
            "nature_loss_from_nonforest",
            other_nature & to_artificial,
            (other_nature | forest) & to_artificial,
        ),
    ]
    for analysis, numerator, denominator in analyses:
        result, detail = stratified_ratio(
            habloss, habloss_areas, numerator, denominator
        )
        add_result(rows, analysis, "habloss_main", result)
        detail.insert(0, "analysis", analysis)
        detail.insert(1, "frame", "habloss_main")
        stratum_outputs.append(detail)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = pd.DataFrame(rows)
    summary.to_csv(args.output_dir / "transition_composition.csv", index=False)
    pd.concat(stratum_outputs, ignore_index=True).to_csv(
        args.output_dir / "transition_composition_by_stratum.csv", index=False
    )
    pd.DataFrame([recover_diagnostics]).to_csv(
        args.output_dir / "recover_cleaning_diagnostics.csv", index=False
    )
    print(summary.to_string(index=False))
    print(f"RECOVER cleaning: {recover_diagnostics}")
    print(f"Wrote results to {args.output_dir}")


if __name__ == "__main__":
    main()
