"""Shared design-based helpers for reproducible RECOVER/HABLOSS analyses."""
from __future__ import annotations

import csv
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import shapefile


Z_95 = 1.959963984540054


def read_shapefile_table(path: Path) -> pd.DataFrame:
    """Read DBF attributes without requiring geopandas/GDAL."""
    reader = shapefile.Reader(str(path))
    fields = [field[0] for field in reader.fields[1:]]
    return pd.DataFrame(reader.records(), columns=fields)


def read_stratum_areas_parallel(
    area_dir: Path,
    id_to_label: dict[int, str],
    workers: int = 16,
) -> dict[str, float]:
    """Sum sharded GEE area CSVs and return labelled stratum areas in km2."""
    files = sorted(area_dir.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No area CSVs found in {area_dir}")

    def read_one(path: Path) -> dict[int, float]:
        totals: dict[int, float] = {}
        with path.open(newline="", encoding="utf-8-sig") as handle:
            for row in csv.DictReader(handle):
                if not row.get("stratum") or not row.get("area"):
                    continue
                h = int(float(row["stratum"]))
                totals[h] = totals.get(h, 0.0) + float(row["area"]) / 1e6
        return totals

    totals: dict[int, float] = {}
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for part in executor.map(read_one, files):
            for h, area in part.items():
                totals[h] = totals.get(h, 0.0) + area

    missing = sorted(set(totals) - set(id_to_label))
    if missing:
        raise ValueError(f"Area stratum IDs absent from lookup: {missing}")
    return {id_to_label[h]: area for h, area in totals.items()}


def stratified_ratio(
    frame: pd.DataFrame,
    stratum_areas: dict[str, float],
    numerator: np.ndarray | pd.Series,
    denominator: np.ndarray | pd.Series,
    stratum_col: str = "stratum",
) -> tuple[dict[str, float], pd.DataFrame]:
    """Combined ratio estimator and linearised design variance.

    ``numerator`` and ``denominator`` are paired 0/1 indicators on the same
    sampled records. Area weights are fixed and strata without observations
    are rejected rather than silently omitted.
    """
    data = frame.copy()
    data["_y"] = np.asarray(numerator, dtype=float)
    data["_x"] = np.asarray(denominator, dtype=float)
    data[stratum_col] = data[stratum_col].astype(str)
    data = data[data[stratum_col].isin(stratum_areas)].copy()

    missing = sorted(set(stratum_areas) - set(data[stratum_col]))
    if missing:
        raise ValueError(f"Area strata without observations: {missing}")

    rows = []
    for h, Ah in stratum_areas.items():
        sample = data[data[stratum_col] == h]
        y = sample["_y"].to_numpy()
        x = sample["_x"].to_numpy()
        nh = len(sample)
        rows.append(
            {
                "stratum": h,
                "area_km2": Ah,
                "n": nh,
                "ybar": y.mean(),
                "xbar": x.mean(),
                "sy2": y.var(ddof=1) if nh > 1 else 0.0,
                "sx2": x.var(ddof=1) if nh > 1 else 0.0,
                "sxy": np.cov(y, x, ddof=1)[0, 1] if nh > 1 else 0.0,
            }
        )
    by_stratum = pd.DataFrame(rows)
    Y = float((by_stratum["area_km2"] * by_stratum["ybar"]).sum())
    X = float((by_stratum["area_km2"] * by_stratum["xbar"]).sum())
    if X <= 0:
        raise ValueError("Estimated ratio denominator is zero")
    ratio = Y / X
    variance_terms = (
        by_stratum["area_km2"] ** 2
        * (
            by_stratum["sy2"]
            + ratio**2 * by_stratum["sx2"]
            - 2 * ratio * by_stratum["sxy"]
        )
        / by_stratum["n"]
    )
    variance = float(variance_terms.sum() / X**2)
    se = float(np.sqrt(max(0.0, variance)))
    result = {
        "estimate": ratio,
        "se": se,
        "ci95_low": ratio - Z_95 * se,
        "ci95_high": ratio + Z_95 * se,
        "margin95": Z_95 * se,
        "numerator_area_km2": Y,
        "denominator_area_km2": X,
        "raw_numerator": int(data["_y"].sum()),
        "raw_denominator": int(data["_x"].sum()),
        "n_usable": len(data),
        "n_strata": data[stratum_col].nunique(),
    }
    by_stratum["variance_term"] = variance_terms / X**2
    return result, by_stratum
