"""Run a stratified percentile bootstrap for the RECOVER PPI++ estimates."""
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
    PROXIES,
    build_lookup,
    proxy_summary,
    read_areas,
    read_design_strata,
    truth_indicator,
)
from estimators import (  # noqa: E402
    optimal_lam_stratified,
    stratified_ppi_bootstrap,
    stratified_ppi_prop,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--recover-root", type=Path, default=DEFAULT_RECOVER)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--n-bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260718)
    args = parser.parse_args()
    output = args.output or args.input_dir / "bootstrap_comparison.csv"

    lookup = build_lookup(args.recover_root)
    Nh = read_areas(
        args.recover_root, lookup, args.input_dir / "stratum_areas.csv"
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
    labelled_strata = labelled["stratum"].to_numpy()

    rows = []
    predicted_files = sorted(
        args.input_dir.glob("predicted_*.csv"),
        key=lambda p: int(p.stem.split("_")[-1]),
    )
    for file_index, path in enumerate(predicted_files):
        predicted = pd.read_csv(path)
        predicted["stratum_id"] = predicted["stratum_id"].astype(int)
        predicted = predicted.merge(
            lookup, on="stratum_id", how="left", validate="many_to_one"
        )
        predicted = predicted.dropna(subset=["stratum", *PROXIES]).copy()
        predicted["stratum"] = predicted["stratum"].astype(str)
        predicted_strata = predicted["stratum"].to_numpy()

        for proxy_index, proxy in enumerate(PROXIES):
            means, variances, counts = proxy_summary(predicted, proxy)
            labelled_proxy = labelled[proxy].astype(float).to_numpy()
            predicted_proxy = predicted[proxy].astype(float).to_numpy()
            lam = optimal_lam_stratified(
                y,
                labelled_proxy,
                labelled_strata,
                Nh,
                variances,
                counts,
            )
            point, analytic_se, analytic_ci = stratified_ppi_prop(
                y,
                labelled_proxy,
                labelled_strata,
                Nh,
                means,
                variances,
                counts,
                lam=lam,
            )
            draws, lambdas = stratified_ppi_bootstrap(
                y,
                labelled_proxy,
                labelled_strata,
                Nh,
                predicted_proxy,
                predicted_strata,
                n_boot=args.n_bootstrap,
                seed=args.seed + file_index * 100 + proxy_index,
            )
            low, high = np.quantile(draws, [0.025, 0.975])
            rows.append(
                {
                    "method": "PPI++ tuned stratified bootstrap",
                    "proxy": proxy,
                    "n_predicted": len(predicted),
                    "n_bootstrap": args.n_bootstrap,
                    "lambda": lam,
                    "lambda_bootstrap_median": np.median(lambdas),
                    "lambda_bootstrap_low": np.quantile(lambdas, 0.025),
                    "lambda_bootstrap_high": np.quantile(lambdas, 0.975),
                    "area_km2": point * total_area,
                    "analytic_se_km2": analytic_se * total_area,
                    "analytic_ci95_low_km2": analytic_ci[0] * total_area,
                    "analytic_ci95_high_km2": analytic_ci[1] * total_area,
                    "bootstrap_se_km2": draws.std(ddof=1) * total_area,
                    "bootstrap_ci95_low_km2": low * total_area,
                    "bootstrap_ci95_high_km2": high * total_area,
                    "bootstrap_bias_km2": (draws.mean() - point) * total_area,
                }
            )
            print(
                f"{len(predicted):,} {proxy}: {point * total_area:,.0f} km2; "
                f"bootstrap 95% CI {low * total_area:,.0f}--{high * total_area:,.0f}; "
                f"lambda {lam:.3f}"
            )

    result = pd.DataFrame(rows)
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False)
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
