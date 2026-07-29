"""Compare Olofsson, fixed PPI/PTD, and tuned PPI++ at 10k/100k."""
from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import shapefile

sys.path.insert(0, str(Path(__file__).resolve().parent))
from estimators import (  # noqa: E402
    optimal_lam_stratified,
    stratified_ppi_prop,
    stratified_prop,
)
from project_paths import default_recover_root, project_data_dir  # noqa: E402


DEFAULT_INPUT = project_data_dir("ppi_gee")
DEFAULT_RECOVER = default_recover_root()
PROXIES = ["yhat_hard", "yhat_score"]


def abbreviation(name: str) -> str:
    return "".join(word[:2] for word in re.findall(r"[A-Za-z]+", name))


def build_lookup(recover_root: Path) -> pd.DataFrame:
    biomes = pd.read_csv(recover_root / "data/biomeLookup.csv")
    classes = pd.DataFrame(
        {
            "map_class": [1, 2, 3, 4, 5],
            "map_label": [
                "stable_stable",
                "crop_buffer",
                "built_buffer",
                "crop_loss",
                "built_loss",
            ],
        }
    )
    lookup = biomes.merge(classes, how="cross")
    lookup["stratum_id"] = lookup["map_class"] + (lookup["BIOME_NUM2"] - 1) * 5
    lookup["stratum"] = lookup["BIOME_NAME2"].map(abbreviation) + "_" + lookup["map_label"]
    return lookup[["stratum_id", "stratum", "map_label", "BIOME_NAME2"]]


def read_areas(
    recover_root: Path, lookup: pd.DataFrame, cache_path: Path | None = None
) -> dict[str, float]:
    if cache_path is not None and cache_path.exists():
        cached = pd.read_csv(cache_path)
        area = cached.rename(columns={"area_m2": "area"})
        area["area_km2"] = area["area"] / 1e6
        area = area.merge(lookup, on="stratum_id", how="left", validate="one_to_one")
        return dict(zip(area["stratum"], area["area_km2"]))
    totals: dict[int, float] = {}
    area_dir = recover_root / "data/from_gee/areas_recover"
    for path in area_dir.glob("*.csv"):
        with path.open(newline="", encoding="utf-8-sig") as handle:
            for row in csv.DictReader(handle):
                if not row.get("stratum") or not row.get("area"):
                    continue
                h = int(float(row["stratum"]))
                totals[h] = totals.get(h, 0.0) + float(row["area"]) / 1e6
    area = pd.DataFrame({"stratum_id": totals.keys(), "area_km2": totals.values()})
    area = area.merge(lookup, on="stratum_id", how="left", validate="one_to_one")
    if area["stratum"].isna().any():
        raise ValueError("Some GEE area strata are absent from the RECOVER lookup")
    return dict(zip(area["stratum"], area["area_km2"]))


def truth_indicator(frame: pd.DataFrame) -> np.ndarray:
    if "lc_2018" not in frame or "lc_2024" not in frame:
        return frame["r"].astype(str).str.strip().str.lower().eq("built_loss").astype(float).to_numpy()
    before = frame["lc_2018"].astype(str).str.strip().str.lower()
    after = frame["lc_2024"].astype(str).str.strip().str.lower()
    return ((before == "artificial") & after.str.startswith("nature")).astype(float).to_numpy()


def read_design_strata(recover_root: Path) -> pd.DataFrame:
    path = recover_root / "data/for_gee/samples_recover_v2.shp"
    reader = shapefile.Reader(str(path))
    fields = [field[0] for field in reader.fields[1:]]
    records = pd.DataFrame(reader.records(), columns=fields)
    return records[["PLOTID", "stratum"]].drop_duplicates("PLOTID")


def proxy_summary(frame: pd.DataFrame, proxy: str):
    grouped = frame.groupby("stratum", observed=True)[proxy]
    return (
        grouped.mean().to_dict(),
        grouped.var(ddof=1).fillna(0).to_dict(),
        grouped.size().to_dict(),
    )


def add_result(rows, method, proxy, n_pred, lam, estimate, total_area):
    p, se, ci = estimate
    rows.append(
        {
            "method": method,
            "proxy": proxy,
            "n_predicted": n_pred,
            "lambda": lam,
            "proportion": p,
            "se": se,
            "ci95_low": ci[0],
            "ci95_high": ci[1],
            "area_km2": p * total_area,
            "ci95_width_km2": (ci[1] - ci[0]) * total_area,
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--recover-root", type=Path, default=DEFAULT_RECOVER)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    output = args.output or args.input_dir / "comparison.csv"

    lookup = build_lookup(args.recover_root)
    Nh = read_areas(
        args.recover_root,
        lookup,
        args.input_dir / "stratum_areas.csv",
    )
    total_area = sum(Nh.values())
    labelled = pd.read_csv(args.input_dir / "labelled_predictions.csv")
    if "stratum" not in labelled:
        labelled = labelled.merge(
            read_design_strata(args.recover_root),
            on="PLOTID",
            how="left",
            validate="one_to_one",
        )
    required = ["stratum", "r", *PROXIES]
    if "lc_2018" in labelled and "lc_2024" in labelled:
        required.extend(["lc_2018", "lc_2024"])
    labelled = labelled.dropna(subset=required).copy()
    labelled["stratum"] = labelled["stratum"].astype(str)
    labelled = labelled[labelled["stratum"].isin(Nh)].copy()
    y = truth_indicator(labelled)
    strata = labelled["stratum"].to_numpy()

    expected_hard = labelled["stratum"].str.endswith("_built_loss").astype(float)
    hard_disagreement = float(
        (labelled["yhat_hard"].astype(float) != expected_hard).mean()
    )
    print(
        f"Labelled points: {len(labelled):,}; strata: {labelled['stratum'].nunique()}; "
        f"historical-stratum/current-hard disagreement: {hard_disagreement:.2%}"
    )

    missing_labelled = sorted(set(Nh) - set(strata))
    if missing_labelled:
        raise ValueError(f"Area strata without labelled observations: {missing_labelled}")

    rows = []
    classical = stratified_prop(y, strata, Nh)
    add_result(rows, "Olofsson stratified", "none", 0, 0.0, classical, total_area)

    predicted_files = sorted(
        args.input_dir.glob("predicted_*.csv"),
        key=lambda p: int(p.stem.split("_")[-1]),
    )
    for path in predicted_files:
        predicted = pd.read_csv(path)
        predicted["stratum_id"] = predicted["stratum_id"].astype(int)
        predicted = predicted.merge(lookup, on="stratum_id", how="left", validate="many_to_one")
        n_pred = len(predicted)

        for proxy in PROXIES:
            means, variances, counts = proxy_summary(predicted, proxy)
            labelled_proxy = labelled[proxy].astype(float).to_numpy()

            fixed = stratified_ppi_prop(
                y,
                labelled_proxy,
                strata,
                Nh,
                means,
                variances,
                counts,
                lam=1.0,
            )
            add_result(rows, "PPI/PTD fixed", proxy, n_pred, 1.0, fixed, total_area)

            lam = optimal_lam_stratified(
                y,
                labelled_proxy,
                strata,
                Nh,
                variances,
                counts,
            )
            tuned = stratified_ppi_prop(
                y,
                labelled_proxy,
                strata,
                Nh,
                means,
                variances,
                counts,
                lam=lam,
            )
            add_result(rows, "PPI++ tuned", proxy, n_pred, lam, tuned, total_area)

            map_only = sum(Nh[h] / total_area * means[h] for h in Nh)
            add_result(
                rows,
                "Predicted-only diagnostic",
                proxy,
                n_pred,
                1.0,
                (map_only, float("nan"), (float("nan"), float("nan"))),
                total_area,
            )

    result = pd.DataFrame(rows)
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False)
    display = result.copy()
    for col in ["proportion", "se", "ci95_low", "ci95_high", "lambda"]:
        display[col] = display[col].map(lambda x: "" if pd.isna(x) else f"{x:.6f}")
    for col in ["area_km2", "ci95_width_km2"]:
        display[col] = display[col].map(lambda x: "" if pd.isna(x) else f"{x:,.0f}")
    print(display.to_string(index=False))
    print(f"\nWrote comparison -> {output}")


if __name__ == "__main__":
    main()
