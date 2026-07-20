"""Reproduce the stratum decomposition of continuous-score PPI++ variance."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from compare_ppi_subsets import (  # noqa: E402
    DEFAULT_INPUT,
    DEFAULT_RECOVER,
    build_lookup,
    proxy_summary,
    read_areas,
    read_design_strata,
    truth_indicator,
)
from estimators import optimal_lam_stratified  # noqa: E402


DEFAULT_OUTPUT = Path("data/analysis_results")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--recover-root", type=Path, default=DEFAULT_RECOVER)
    parser.add_argument("--predicted-file", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    predicted_path = args.predicted_file or args.input_dir / "predicted_100000.csv"

    lookup = build_lookup(args.recover_root)
    areas = read_areas(
        args.recover_root,
        lookup,
        args.input_dir / "stratum_areas.csv",
    )
    total_area = float(sum(areas.values()))
    labelled = pd.read_csv(args.input_dir / "labelled_predictions.csv")
    if "stratum" not in labelled:
        labelled = labelled.merge(
            read_design_strata(args.recover_root),
            on="PLOTID",
            how="left",
            validate="one_to_one",
        )
    required = ["stratum", "r", "yhat_score"]
    if "lc_2018" in labelled and "lc_2024" in labelled:
        required.extend(["lc_2018", "lc_2024"])
    labelled = labelled.dropna(subset=required).copy()
    labelled["stratum"] = labelled["stratum"].astype(str)
    labelled = labelled[labelled["stratum"].isin(areas)].copy()
    y = truth_indicator(labelled)
    yhat = labelled["yhat_score"].astype(float).to_numpy()
    strata = labelled["stratum"].to_numpy()

    predicted = pd.read_csv(predicted_path)
    predicted["stratum_id"] = predicted["stratum_id"].astype(int)
    predicted = predicted.merge(
        lookup, on="stratum_id", how="left", validate="many_to_one"
    )
    means, predicted_vars, predicted_counts = proxy_summary(
        predicted, "yhat_score"
    )
    del means
    lam = optimal_lam_stratified(
        y, yhat, strata, areas, predicted_vars, predicted_counts
    )

    rows = []
    for h, Ah in areas.items():
        mask = strata == h
        nh = int(mask.sum())
        if nh == 0:
            raise ValueError(f"No labelled observations in stratum {h!r}")
        weight = Ah / total_area
        residual = y[mask] - lam * yhat[mask]
        label_variance = (
            weight**2 * residual.var(ddof=1) / nh if nh > 1 else 0.0
        )
        nu = int(predicted_counts.get(h, 0))
        predicted_variance = (
            weight**2 * lam**2 * float(predicted_vars.get(h, 0.0)) / nu
            if nu > 1
            else 0.0
        )
        rows.append(
            {
                "stratum": h,
                "area_km2": Ah,
                "area_weight": weight,
                "n_labelled": nh,
                "n_positive": int(y[mask].sum()),
                "residual_sd": residual.std(ddof=1) if nh > 1 else 0.0,
                "label_variance": label_variance,
                "predicted_variance": predicted_variance,
            }
        )
    result = pd.DataFrame(rows)
    total_label_variance = float(result["label_variance"].sum())
    total_predicted_variance = float(result["predicted_variance"].sum())
    total_variance = total_label_variance + total_predicted_variance
    result["share_label_variance"] = (
        result["label_variance"] / total_label_variance
    )
    result["share_total_variance"] = (
        (result["label_variance"] + result["predicted_variance"])
        / total_variance
    )
    result = result.sort_values("label_variance", ascending=False).reset_index(
        drop=True
    )
    result["cumulative_share_label_variance"] = result[
        "share_label_variance"
    ].cumsum()

    summary = pd.DataFrame(
        [
            {
                "proxy": "yhat_score",
                "predicted_file": str(predicted_path),
                "n_labelled": len(labelled),
                "n_predicted": len(predicted),
                "n_strata": len(areas),
                "lambda": lam,
                "total_area_km2": total_area,
                "label_variance": total_label_variance,
                "predicted_variance": total_predicted_variance,
                "predicted_share_total_variance": (
                    total_predicted_variance / total_variance
                ),
                "top3_share_label_variance": result.loc[
                    :2, "share_label_variance"
                ].sum(),
                "top3_share_total_variance": result.loc[
                    :2, "share_total_variance"
                ].sum(),
            }
        ]
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output_dir / "ppi_variance_by_stratum.csv", index=False)
    summary.to_csv(args.output_dir / "ppi_variance_summary.csv", index=False)
    print(summary.to_string(index=False))
    print(result.head(10).to_string(index=False))
    print(f"Wrote results to {args.output_dir}")


if __name__ == "__main__":
    main()
